"""Wake word (subsystem 7): text matching, both detectors, the listener thread, config, console. No hardware."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.router import FastPathRouter
from nexus.ai.voice.base import STTUnavailable, SilentTTS, Transcript
from nexus.ai.voice.listener import VoiceInput
from nexus.ai.voice.recorder import Recorder, VADConfig
from nexus.ai.voice.speaker import Speaker
from nexus.ai.voice.stt import HybridSTT, STTOption
from nexus.ai.voice.wake import (OpenWakeWordDetector, WakeHit, WakeUnavailable, WhisperWakeDetector, find_wake,
                                 strip_wake)
from nexus.ai.voice.wake_listener import WakeEvent, WakeListener, WakeOptions, chime_clip
from nexus.ai.providers.selection import SelectionContext
from nexus.console import Console
from nexus.core.config import ConfigError, WakeSettings, load_ai_settings, load_wake_settings
from nexus.core.resources import ResourceManager
from nexus.core.runtime import Runtime
from nexus.system.audio.devices import MicrophoneManager
from tests.conftest import FakeProbe, ScriptedPlanner, plan
from tests.voice_fixtures import FakeCapture, FakeDevices, FakeStream, FakeSTT, mic, silence, voice

# --- is it addressed to NEXUS? -------------------------------------------------------------------


@pytest.mark.parametrize("text, rest", [
    ("Nexus, bhai GTA chala de.", "bhai GTA chala de."),
    ("Hey Nexus open chrome", "open chrome"),
    ("Oye nexus! time kya hai", "time kya hai"),
    ("next us open chrome", "open chrome"),       # how a small model may split it
    ("Nexas, what's the time?", "what's the time?"),
    ("Nexux GTA laga", "GTA laga"),               # one letter off
    ("नेक्सस क्रोम खोलो", "क्रोम खोलो"),
    ("ਨੈਕਸਸ, ਕਰੋਮ ਖੋਲ੍ਹੋ", "ਕਰੋਮ ਖੋਲ੍ਹੋ"),
    ("Nexus.", ""),
])
def test_wake_word_at_the_start_is_found(text, rest):
    assert find_wake(text) == (True, rest)
    assert strip_wake(text) == rest


@pytest.mark.parametrize("text", ["Next song please", "I love Nexus", "the nexus of it all", "Lexus is a car",
                                  "necklace", "Texas is big", "what's next", ""])
def test_other_speech_is_not_addressed_to_nexus(text):
    assert find_wake(text)[0] is False
    assert strip_wake(text) == text.strip()


def test_words_before_the_wake_word_can_be_skipped_for_audio_kept_from_before():
    assert find_wake("and then Nexus open chrome", skip=2) == (True, "open chrome")
    assert find_wake("and then Nexus open chrome")[0] is False


# --- the recorder continues an utterance ------------------------------------------------------------

def test_recorder_continues_speech_already_in_progress():
    lead = voice(800)
    stream = FakeStream(voice(600) + silence(1200))
    utt = Recorder(VADConfig()).record(stream, noise_floor=100.0, lead_in=lead)
    assert 1.3 < utt.clip.seconds < 2.0           # lead-in + the rest + the kept tail
    assert utt.clip.pcm16.startswith(lead) and utt.speech_seconds >= 1.3


def test_recorder_counts_silence_already_heard_after_the_lead_in():
    stream = FakeStream(silence(1500))
    utt = Recorder(VADConfig()).record(stream, noise_floor=100.0, lead_in=voice(700) + silence(450),
                                       lead_in_silence_ms=450)
    assert utt.clip.seconds < 1.3                 # ended ~450 ms later, not a full 900 ms


def test_recorder_with_a_known_noise_floor_needs_no_calibration():
    # Speech from the very first frame is caught: no 300 ms of calibration.
    stream = FakeStream(voice(600) + silence(1200))
    utt = Recorder(VADConfig()).record(stream, noise_floor=100.0)
    assert utt.speech_seconds >= 0.5


# --- detector 1: Whisper on bursts of speech ---------------------------------------------------------

def whisper_detector(*results, rm=None, **kw):
    rm = rm or ResourceManager(FakeProbe(), snapshot_ttl=0)
    stt = FakeSTT("tiny", results=list(results))
    rm.register(stt)
    return WhisperWakeDetector(stt, rm, **kw), stt, rm


def feed(detector, audio: bytes):
    frame = 16000 * detector.frame_ms // 1000 * 2
    hits = []
    for i in range(0, len(audio) - frame + 1, frame):
        hit = detector.process(audio[i:i + frame])
        if hit is not None:
            hits.append((i / 32000, hit))
    return hits


def test_burst_starting_with_nexus_wakes_with_the_audio_so_far():
    det, stt, _ = whisper_detector(Transcript("Nexus, GTA chala de", "en", 0.8))
    hits = feed(det, silence(600) + voice(2500) + silence(1000))
    assert len(hits) == 1 and len(stt.calls) == 1
    at, hit = hits[0]
    assert 2.0 < at < 2.5                          # 1.6 s into the burst, while still speaking
    assert not hit.alone and hit.heard == "Nexus, GTA chala de"
    assert 1.6 <= len(hit.lead_in) / 32000 <= 2.4  # speech from its start, with up to 750 ms before it
    assert stt.calls[0][1] == "en" and stt.loaded


def test_short_request_is_checked_at_the_pause():
    det, stt, _ = whisper_detector(Transcript("Nexus, open Chrome.", "en", 0.9))
    hits = feed(det, silence(600) + voice(1000) + silence(1000))
    assert len(hits) == 1 and not hits[0][1].alone and hits[0][1].silence_ms >= 450


def test_nexus_alone_then_a_pause():
    det, _, _ = whisper_detector(Transcript("Nexus.", "en", 0.9))
    hits = feed(det, silence(600) + voice(600) + silence(1000))
    assert len(hits) == 1 and hits[0][1].alone


def test_other_speech_costs_one_check_per_burst_and_never_wakes():
    det, stt, _ = whisper_detector(Transcript("so what's the weather like", "en", 0.9))
    assert feed(det, silence(600) + voice(4000) + silence(800) + voice(1000) + silence(800)) == []
    assert len(stt.calls) == 2 and det.checks == 2


def test_long_speech_is_split_so_nexus_inside_it_can_still_be_heard():
    det, stt, _ = whisper_detector(Transcript("blah blah", "en", 0.9), Transcript("Nexus stop", "en", 0.9),
                                   max_burst_s=3.0)
    hits = feed(det, silence(600) + voice(9000))  # a TV talking without a pause
    assert len(stt.calls) >= 2 and hits


def test_clicks_and_coughs_are_not_checked():
    det, stt, _ = whisper_detector()
    feed(det, silence(600) + voice(150) + silence(900))
    assert stt.calls == []


def test_checks_are_rate_limited():
    det, stt, _ = whisper_detector(Transcript("chatter", "en", 0.9), max_checks_per_minute=3)
    feed(det, silence(600) + (voice(700) + silence(800)) * 6)
    assert len(stt.calls) == 3 and det.skipped == 3


def test_unclear_transcript_does_not_wake():
    det, _, _ = whisper_detector(Transcript("Nexus", "en", 0.1))
    assert feed(det, silence(600) + voice(700) + silence(900)) == []


def test_model_that_cannot_run_is_reported_not_raised():
    det, stt, _ = whisper_detector(STTUnavailable("no memory"))
    assert feed(det, silence(600) + voice(700) + silence(900)) == []
    assert "no memory" in det.problem


def test_start_preloads_the_model_and_fails_clearly():
    det, stt, _ = whisper_detector()
    det.start()
    assert stt.loaded
    det2, stt2, _ = whisper_detector()
    stt2.is_available = False
    stt2.problem = "model not downloaded"
    with pytest.raises(WakeUnavailable, match="not downloaded"):
        det2.start()


# --- detector 2: openWakeWord (the networks replaced by fakes; the real ones are in test_wake_live.py) --

class FakeFeatures:
    def __init__(self):
        self.pushed = 0
        self.resets = 0

    def push(self, chunk):
        self.pushed += 1

    def window(self, frames):
        return None

    def reset(self):
        self.resets += 1


class FakeModel:
    frames = 16

    def __init__(self, scores):
        self.scores = list(scores)

    def score(self, window):
        return self.scores.pop(0) if self.scores else 0.0


def oww(scores, tmp_path, **kw):
    import numpy as np

    det = OpenWakeWordDetector(tmp_path / "nexus.onnx", tmp_path / "mel.onnx", tmp_path / "emb.onnx", **kw)
    det._np, det._features, det._model = np, FakeFeatures(), FakeModel(scores)
    return det


def chunks(audio: bytes):
    return [audio[i:i + 2560] for i in range(0, len(audio) - 2559, 2560)]


def test_oww_wakes_above_the_threshold_with_the_last_second_of_audio(tmp_path):
    det = oww([0.0] * 10 + [0.2, 0.9], tmp_path)
    hits = [h for c in chunks(voice(1200)) if (h := det.process(c))]
    assert len(hits) == 1 and hits[0].score == 0.9 and hits[0].heard == ""
    assert len(hits[0].lead_in) == 12 * 2560       # all 12 chunks (under the 1 s limit)...
    assert det._features.resets == 1               # ready for the next one


def test_oww_ignores_the_first_frames_and_respects_patience(tmp_path):
    det = oww([0.99] * 5 + [0.0], tmp_path)
    assert not [h for c in chunks(voice(800)) if det.process(c)]   # warm-up frames never count
    det = oww([0.0] * 6 + [0.8, 0.1, 0.8, 0.8], tmp_path, patience=2)
    hits = [i for i, c in enumerate(chunks(voice(1000))) if det.process(c)]
    assert hits == [9]


def test_oww_lead_in_is_at_most_one_second(tmp_path):
    det = oww([0.0] * 30 + [0.9], tmp_path)
    hit = [h for c in chunks(voice(3000)) if (h := det.process(c))][0]
    assert len(hit.lead_in) == 12 * 2560


def test_oww_missing_files_explain_what_to_do(tmp_path):
    det = OpenWakeWordDetector(tmp_path / "nexus.onnx", tmp_path / "melspectrogram.onnx",
                               tmp_path / "embedding_model.onnx")
    assert not det.available() and "train_wake_model.py" in det.problem
    (tmp_path / "nexus.onnx").write_bytes(b"x")
    assert not det.available() and "get_wake_model.py" in det.problem
    with pytest.raises(WakeUnavailable):
        det.start()


def test_oww_broken_model_file_is_unavailable_not_a_crash(tmp_path):
    for name in ("nexus.onnx", "melspectrogram.onnx", "embedding_model.onnx"):
        (tmp_path / name).write_bytes(b"not a model")
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0)
    det = OpenWakeWordDetector(tmp_path / "nexus.onnx", tmp_path / "melspectrogram.onnx",
                               tmp_path / "embedding_model.onnx", rm)
    with pytest.raises(WakeUnavailable, match="couldn't load"):
        det.start()


# --- the listener thread ----------------------------------------------------------------------------

class ScriptedDetector:
    """Wakes on the n-th frame(s); records what it saw."""

    frame_ms = 30

    def __init__(self, wake_at=(), *, alone=False, fail=None, name="wake:test"):
        from nexus.ai.providers.base import Locality, ProviderInfo

        self.info = ProviderInfo(name=name, locality=Locality.LOCAL)
        self.wake_at = set(wake_at)
        self.alone = alone
        self.fail = fail
        self.frames = 0
        self.problem = ""
        self.resets = 0
        self.started = self.stopped = False
        self.noise_floor = 100.0

    def available(self):
        return self.fail is None

    def start(self):
        if self.fail:
            raise WakeUnavailable(self.fail)
        self.started = True

    def stop(self):
        self.stopped = True

    def reset(self):
        self.resets += 1

    def process(self, frame):
        self.frames += 1
        if self.frames in self.wake_at:
            return WakeHit(self.info.name, 0.9, "Nexus" if self.alone else "", voice(600) if not self.alone else b"",
                           0, alone=self.alone, noise_floor=100.0)
        return None


class GappedStream(FakeStream):
    """Like a real microphone queue: each segment arrives only after the previous one was read and the
    queue ran empty once (so draining stale audio doesn't swallow what the user says next)."""

    def __init__(self, *segments, tail=None):
        super().__init__(b"", tail=tail)
        self._segments = list(segments)

    def read(self, timeout):
        if self._pos >= len(self._audio) and self._segments:
            self._audio, self._pos = self._segments.pop(0), 0
            return None
        return super().read(timeout)


def make_voice(devices, capture, *results):
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0)
    stt = FakeSTT("base", results=list(results))
    hybrid = HybridSTT([STTOption(stt, stt.spec, 1, game_mode_ok=True)], rm, lambda: SelectionContext())
    return VoiceInput(MicrophoneManager(devices), capture, hybrid, rm), stt, rm


class Run:
    """A WakeListener in its thread, collecting what it reports."""

    def __init__(self, vi, detectors, rm, *, answers=(), speaking=lambda: False, **opts):
        self.heard, self.events, self.chimes = [], [], 0
        self.answers = list(answers)
        self.done = threading.Event()

        def on_heard(heard, hit):
            self.heard.append((heard.text, hit))
            self.done.set()
            return self.answers.pop(0) if self.answers else False

        def chime():
            self.chimes += 1

        options = WakeOptions(**{"speaking_tail_s": 0.0, "retry_s": 0.05, "device_timeout_s": 0.3, **opts})
        self.listener = WakeListener(vi, detectors, rm, on_heard=on_heard, on_event=self.events.append,
                                     speaking=speaking, chime=chime, options=options)

    def __enter__(self):
        self.listener.start()
        return self

    def __exit__(self, *exc):
        self.listener.stop()

    def wait(self, pred, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            time.sleep(0.01)
        return False

    def kinds(self):
        return [e.kind for e in self.events]


def test_nexus_then_request_goes_to_the_conversation():
    devices = FakeDevices(mic("Mic (USB)"))
    capture = FakeCapture({"Mic (USB)": FakeStream(voice(900) + silence(1200), tail=silence(300))})
    vi, stt, rm = make_voice(devices, capture, Transcript("Nexus, GTA chala de", "hi", 0.9))
    det = ScriptedDetector(wake_at=[5])
    with Run(vi, [det], rm) as run:
        assert run.wait(lambda: run.heard)
    text, hit = run.heard[0]
    assert "GTA chala de" in text and not text.lower().startswith("nexus")
    assert hit.detector == "wake:test"
    assert run.kinds()[:2] == ["listening", "woke"] and run.chimes == 0
    assert run.listener.wakes == 1 and det.stopped
    clip = stt.calls[0][0]
    assert clip.pcm16.startswith(voice(600))  # the wake word's own audio leads the request


def test_nexus_alone_chimes_and_waits_for_the_request():
    devices = FakeDevices(mic("Mic"))
    capture = FakeCapture({"Mic": FakeStream(silence(600) + voice(900) + silence(1200), tail=silence(300))})
    vi, stt, rm = make_voice(devices, capture, Transcript("open chrome", "en", 0.9))
    with Run(vi, [ScriptedDetector(wake_at=[3], alone=True)], rm) as run:
        assert run.wait(lambda: run.heard)
    assert run.heard[0][0] == "open chrome" and run.chimes == 1
    assert any(e.kind == "follow_up" for e in run.events)


def test_nexus_heard_only_as_nexus_asks_for_the_request():
    # openWakeWord woke it, and the speech after it was only the wake word itself.
    devices = FakeDevices(mic("Mic"))
    capture = FakeCapture({"Mic": FakeStream(silence(1500) + voice(900) + silence(1200), tail=silence(300))})
    vi, stt, rm = make_voice(devices, capture, Transcript("Nexus.", "en", 0.9), Transcript("time kya hai", "hi", 0.9))
    with Run(vi, [ScriptedDetector(wake_at=[2])], rm) as run:
        assert run.wait(lambda: run.heard)
    assert run.heard[0][0] == "time kya hai" and run.chimes == 1


def test_silence_after_nexus_is_reported_and_nothing_runs():
    devices = FakeDevices(mic("Mic"))
    capture = FakeCapture({"Mic": FakeStream(silence(300), tail=silence(300))})
    vi, _, rm = make_voice(devices, capture)
    with Run(vi, [ScriptedDetector(wake_at=[3], alone=True)], rm, command_timeout_s=2) as run:
        assert run.wait(lambda: "nothing" in run.kinds())
    assert run.heard == []
    assert any("didn't hear a request" in e.message for e in run.events)


def test_question_from_nexus_is_answered_without_the_wake_word():
    devices = FakeDevices(mic("Mic"))
    stream = GappedStream(voice(900) + silence(1200), voice(500) + silence(1200), tail=silence(300))
    capture = FakeCapture({"Mic": stream})
    vi, stt, rm = make_voice(devices, capture, Transcript("Nexus delete the test folder", "en", 0.9),
                             Transcript("haan", "hi", 0.9))
    with Run(vi, [ScriptedDetector(wake_at=[5])], rm, answers=[True, False]) as run:
        assert run.wait(lambda: len(run.heard) == 2)
    assert run.heard[1] == ("haan", None)  # an answer, not a new wake
    assert run.listener.wakes == 1


def test_no_follow_up_when_disabled():
    devices = FakeDevices(mic("Mic"))
    capture = FakeCapture({"Mic": FakeStream(voice(900) + silence(1200) + voice(500) + silence(1200),
                                             tail=silence(300))})
    vi, _, rm = make_voice(devices, capture, Transcript("Nexus delete it", "en", 0.9), Transcript("haan", "hi", 0.9))
    with Run(vi, [ScriptedDetector(wake_at=[5])], rm, answers=[True], follow_up_s=0) as run:
        assert run.wait(lambda: run.heard)
        time.sleep(0.3)
    assert len(run.heard) == 1


def test_nexus_speaking_is_not_heard_as_a_wake_word():
    devices = FakeDevices(mic("Mic"))
    capture = FakeCapture({"Mic": FakeStream(voice(900), tail=silence(300))})
    vi, _, rm = make_voice(devices, capture)
    det = ScriptedDetector(wake_at=[1])
    with Run(vi, [det], rm, speaking=lambda: True) as run:
        time.sleep(0.3)
    assert det.frames == 0 and run.heard == []


def test_barge_in_listens_while_speaking():
    devices = FakeDevices(mic("Mic"))
    capture = FakeCapture({"Mic": FakeStream(voice(900) + silence(1200), tail=silence(300))})
    vi, _, rm = make_voice(devices, capture, Transcript("Nexus stop", "en", 0.9))
    stopped = []
    run = Run(vi, [ScriptedDetector(wake_at=[3])], rm, speaking=lambda: True, barge_in=True)
    run.listener.stop_speaking = lambda: stopped.append(True)
    with run:
        assert run.wait(lambda: run.heard)
    assert stopped


def test_test_mode_reports_and_does_not_act():
    devices = FakeDevices(mic("Mic"))
    capture = FakeCapture({"Mic": FakeStream(voice(900), tail=silence(300))})
    vi, _, rm = make_voice(devices, capture)
    run = Run(vi, [ScriptedDetector(wake_at=[3])], rm)
    run.listener.test(10)
    with run:
        assert run.wait(lambda: "test" in run.kinds())
    assert run.heard == [] and "test mode" in [e for e in run.events if e.kind == "test"][0].message


def test_no_microphone_then_one_is_plugged_in():
    devices = FakeDevices()
    capture = FakeCapture({"Mic": FakeStream(voice(900) + silence(1200), tail=silence(300))})
    vi, _, rm = make_voice(devices, capture, Transcript("Nexus hi", "en", 0.9))
    with Run(vi, [ScriptedDetector(wake_at=[5])], rm) as run:
        assert run.wait(lambda: "problem" in run.kinds())
        assert "no microphone" in run.listener.status()["problem"]
        devices.plug(mic("Mic"))
        assert run.wait(lambda: run.heard)


def test_lost_microphone_is_replaced_by_the_next():
    devices = FakeDevices(mic("Headset", comms=True), mic("Webcam mic"))
    capture = FakeCapture({"Headset": FakeStream(silence(90)),      # stops sending audio
                           "Webcam mic": FakeStream(voice(900) + silence(1200), tail=silence(300))})
    vi, _, rm = make_voice(devices, capture, Transcript("Nexus hi", "en", 0.9))
    with Run(vi, [ScriptedDetector(wake_at=[8])], rm) as run:
        assert run.wait(lambda: run.heard)
    assert capture.opened[:2] == ["Headset", "Webcam mic"]
    assert run.listener.mic.name == "Webcam mic"


def test_stream_that_ends_after_working_a_while_is_reopened_not_blamed():
    # e.g. the audio system re-initialised for new headphones: the same microphone is opened again.
    devices = FakeDevices(mic("Headset", comms=True), mic("Webcam mic"))
    first = FakeStream(silence(300))
    second = FakeStream(voice(900) + silence(1200), tail=silence(300))
    streams = iter([first, second])

    class Capture(FakeCapture):
        def open(self, m, **kw):
            self.opened.append(m.name)
            return next(streams)

    capture = Capture()
    vi, _, rm = make_voice(devices, capture, Transcript("Nexus hi", "en", 0.9))
    clock = [0.0]
    run = Run(vi, [ScriptedDetector(wake_at=[40])], rm)
    run.listener._clock = lambda: time.monotonic() + clock[0]
    real_lost = run.listener._lost

    def lost():
        clock[0] += 60.0  # it had been working for a minute
        real_lost()
    run.listener._lost = lost
    with run:
        assert run.wait(lambda: run.heard)
    assert capture.opened[:2] == ["Headset", "Headset"]


def test_microphone_privacy_switch_is_respected():
    devices = FakeDevices(mic("Mic"))
    vi, _, rm = make_voice(devices, FakeCapture({"Mic": FakeStream(silence(300), tail=silence(300))}))
    vi._permission = lambda: False
    with Run(vi, [ScriptedDetector()], rm) as run:
        assert run.wait(lambda: "problem" in run.kinds())
    assert "Privacy" in run.listener.status()["problem"]


def test_first_detector_that_starts_is_used_and_the_others_explained():
    devices = FakeDevices(mic("Mic"))
    capture = FakeCapture({"Mic": FakeStream(silence(300), tail=silence(300))})
    vi, _, rm = make_voice(devices, capture)
    broken = ScriptedDetector(fail="nexus.onnx not found - train it", name="wake:openwakeword-nexus")
    works = ScriptedDetector(name="wake:whisper-base")
    with Run(vi, [broken, works], rm) as run:
        assert run.wait(lambda: run.listener.status()["state"] == "listening")
    assert run.listener.detector is works and "train it" in run.listener.skipped[0]


def test_no_detector_at_all_is_reported():
    devices = FakeDevices(mic("Mic"))
    vi, _, rm = make_voice(devices, FakeCapture())
    with Run(vi, [ScriptedDetector(fail="model not downloaded")], rm) as run:
        assert run.wait(lambda: "problem" in run.kinds())
    assert "model not downloaded" in run.listener.status()["problem"]
    assert run.listener.status()["state"] == "unavailable"


def test_pause_off_and_game_mode_release_the_microphone():
    devices = FakeDevices(mic("Mic"))
    stream = FakeStream(silence(300), tail=silence(300))
    vi, _, rm = make_voice(devices, FakeCapture({"Mic": stream}))
    det = ScriptedDetector()
    with Run(vi, [det], rm, game_mode="off") as run:
        assert run.wait(lambda: run.listener.status()["state"] == "listening")
        with run.listener.paused("push-to-talk is listening"):
            assert run.wait(lambda: stream.closed)
            assert run.listener.status()["paused"] == "push-to-talk is listening"
        stream.closed = False
        assert run.wait(lambda: run.listener.status()["state"] == "listening")
        run.listener.set_enabled(False)
        assert run.wait(lambda: run.listener.status()["state"] == "off")
        run.listener.set_enabled(True)
        assert run.wait(lambda: run.listener.status()["state"] == "listening")
        rm.enter_game_mode()
        assert run.wait(lambda: run.listener.status()["paused"] == "a game is running")
        rm.exit_game_mode()


def test_chime_is_short_and_quiet():
    clip = chime_clip()
    assert 0.15 < clip.seconds < 0.25
    from array import array

    samples = array("h")
    samples.frombytes(clip.pcm16)
    assert max(abs(s) for s in samples) < 0.25 * 32767


# --- config -------------------------------------------------------------------------------------------

def test_wake_config_defaults_and_the_shipped_file(tmp_path):
    assert load_wake_settings(tmp_path / "missing.toml") == WakeSettings()
    shipped = Path(__file__).resolve().parents[1] / "config" / "nexus.toml"
    settings = load_wake_settings(shipped)
    assert settings.enabled and settings.detector == "auto" and settings.whisper_model == "base"
    load_ai_settings(shipped)  # [wake] is a known section


@pytest.mark.parametrize("line, error", [
    ('detector = "alexa"', "detector"), ("threshold = 1.5", "threshold"), ("follow_up_seconds = 99", "follow_up"),
    ('whisper_model = "huge"', "whisper_model"), ('model = "../evil"', "model"), ("volume = 3", "unknown"),
    ('game_mode = "maybe"', "game_mode"), ('enabled = "yes"', "enabled"),
])
def test_bad_wake_config_stops_startup(tmp_path, line, error):
    file = tmp_path / "nexus.toml"
    file.write_text(f"[wake]\n{line}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=error):
        load_wake_settings(file)


def test_integer_threshold_is_accepted(tmp_path):
    file = tmp_path / "nexus.toml"
    file.write_text("[wake]\nthreshold = 0.6\n", encoding="utf-8")
    assert load_wake_settings(file).threshold == 0.6


# --- console --------------------------------------------------------------------------------------------

class StubWake:
    def __init__(self):
        self.enabled = True
        self.started = False
        self.on_heard = self.on_event = None
        self.skipped = ["wake:openwakeword-nexus: nexus.onnx not found - train it"]
        self.tested = None

    def start(self):
        self.started = True

    def set_enabled(self, on):
        self.enabled = on

    def test(self, seconds):
        self.tested = seconds

    def status(self):
        return {"state": "listening" if self.enabled else "off", "detector": "wake:whisper-base",
                "microphone": "Mic", "problem": "", "paused": "", "wakes": 2, "testing": False}


class Assistant:
    def __init__(self, engine, wake):
        self.engine = engine
        self.chat = type("Chat", (), {"last_choice": None})()
        self.speaker = Speaker(SilentTTS())
        self.voice = None
        self.voice_note = self.speech_note = ""
        self.wake = wake
        self.wake_note = ""


def console(make_kernel, audit, fake_platform, lines, wake, planner=None):
    kernel = make_kernel()
    rt = Runtime(settings=None, platform=fake_platform, kernel=kernel, registry=kernel.registry,
                 policy=kernel.policy, audit=audit)
    engine = ConversationEngine(kernel, planner or ScriptedPlanner(plan("Hello!")), router=FastPathRouter(),
                                resources=ResourceManager(FakeProbe(), snapshot_ttl=0))
    out = []
    feed = iter(lines)
    c = Console(rt, Assistant(engine, wake), input_fn=lambda _: next(feed), output=out.append)
    c.run()
    return c, out


def test_console_starts_the_wake_word_and_wake_commands(make_kernel, audit, fake_platform):
    wake = StubWake()
    c, out = console(make_kernel, audit, fake_platform, ["/wake", "/wake off", "/wake test", "/wake on", "/quit"], wake)
    text = "\n".join(out)
    assert wake.started and wake.on_heard == c._wake_heard and wake.on_event == c._wake_event
    assert "wake word: listening" in text and "wake:whisper-base" in text and "train it" in text
    assert "Wake word off" in text and wake.tested == 20 and wake.enabled


def test_console_without_wake_word_explains(make_kernel, audit, fake_platform):
    _, out = console(make_kernel, audit, fake_platform, ["/wake", "/quit"], None)
    assert "isn't available" in "\n".join(out)


class FakeHeard:
    def __init__(self, text):
        self.text = self.original = text
        self.confidence = 0.95
        self.guessed = False
        self.changes = ()
        self.truncated = False
        self.transcript = Transcript(text, "en", 0.95)


def test_wake_request_runs_through_the_conversation_engine(make_kernel, audit, fake_platform):
    c, out = console(make_kernel, audit, fake_platform, ["/quit"], StubWake())
    c._voice_timings = None
    c.assistant.voice = type("V", (), {"last_timings": {"transcribing_s": 0.4}})()
    asked = c._wake_heard(FakeHeard("what time is it"), None)
    text = "\n".join(out)
    assert 'heard: "what time is it"' in text and "nexus> It's" in text and asked is False


def test_wake_request_reports_when_nexus_asks_back(make_kernel, audit, fake_platform):
    c, out = console(make_kernel, audit, fake_platform, ["/quit"], StubWake(),
                     planner=ScriptedPlanner(plan("Which one do you mean?")))
    c.assistant.voice = type("V", (), {"last_timings": {}})()
    assert c._wake_heard(FakeHeard("open the game"), None) is True


def test_wake_events_are_shown(make_kernel, audit, fake_platform):
    c, out = console(make_kernel, audit, fake_platform, ["/quit"], StubWake())
    c._wake_event(WakeEvent("listening", 'listening for "Nexus" with Mic'))
    c._wake_event(WakeEvent("woke", "", WakeHit("wake:whisper-base", 0.9, "Nexus, open chrome")))
    c._wake_event(WakeEvent("problem", "no microphone is connected"))
    text = "\n".join(out)
    assert '/wake off to stop' in text and '(Nexus ("Nexus, open chrome") - listening)' in text
    assert "wake word: no microphone" in text
