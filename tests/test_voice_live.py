"""Live speech-to-text: the real faster-whisper models on this PC (pytest -m voice).

The audio is synthesised offline by Windows' own voices (System.Speech) and fed
to NEXUS as if it came from a microphone - no speakers, no loopback device.
A real microphone is tested separately (tests/test_voice_mic.py, pytest -m mic).
"""

from __future__ import annotations

import re
import sys
import time

import psutil
import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.types import ActionReport, UserInput
from nexus.ai.providers.selection import SelectionContext
from nexus.ai.router import FastPathRouter
from nexus.ai.voice.base import NoSpeech, Unintelligible
from nexus.ai.voice.listener import VoiceInput
from nexus.ai.voice.stt import HybridSTT, STTOption
from nexus.ai.voice.whisper import WhisperSTT, model_folder
from nexus.core.kernel import Status
from nexus.core.resources import ResourceManager, ResourceState
from nexus.system.audio.devices import MicrophoneManager
from tests.voice_fixtures import FakeCapture, FakeDevices, FakeStream, mic, noise, read_wav, silence, synthesize

pytestmark = [pytest.mark.voice,
              pytest.mark.skipif(sys.platform != "win32", reason="synthetic speech uses Windows voices"),
              pytest.mark.skipif(not all((model_folder(m) / "model.bin").is_file() for m in ("base", "small")),
                                 reason="run scripts/get_stt_model.py base and small first")]

PHRASES = {"time": "What time is it?", "folder": "Open my downloads folder.",
           "remember": "Remember that my favourite game is Rocket League.", "yes": "Yes."}


def words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("favourite", "favorite")).split())


@pytest.fixture(scope="module")
def speech(tmp_path_factory):
    return synthesize(PHRASES, tmp_path_factory.mktemp("speech"))


@pytest.fixture(scope="module")
def small():
    stt = WhisperSTT("small", ram_mb=560, threads=6)
    yield stt
    stt.unload()


@pytest.mark.parametrize("name", list(PHRASES))
@pytest.mark.parametrize("voice", ["David", "Zira"])
def test_V01_clear_english_is_transcribed_exactly(speech, small, voice, name):
    if not small.is_loaded():
        small.load()
    t = small.transcribe(read_wav(speech[(voice, name)]))
    assert words(t.text) == words(PHRASES[name]), t.text
    assert t.language == "en" and t.confidence > 0.6
    assert t.seconds < 5  # measured ~1.2 s on the i5-12400F


def test_V02_silence_and_noise_produce_no_words(small):
    if not small.is_loaded():
        small.load()
    from nexus.ai.voice.base import AudioClip

    for pcm in (silence(3000), noise(3000, 150)):
        assert small.transcribe(AudioClip(pcm)).text == ""  # nothing invented from nothing


def test_V03_unload_ends_the_process_and_returns_its_memory():
    stt = WhisperSTT("base", ram_mb=320, threads=2)
    before = psutil.virtual_memory().available
    stt.load()
    pid, used = stt.pid, stt.memory_mb()
    assert 100 < used < 320, f"base worker holds {used} MB"
    stt.unload()
    time.sleep(0.5)
    assert not psutil.pid_exists(pid)
    assert stt.memory_mb() == 0
    # the OS got it back (loose: other programs move memory too)
    assert psutil.virtual_memory().available > before - 150 * 2**20


def pipeline(make_kernel, speech_audio: bytes, *, game_mode=False, clock=None):
    rm = ResourceManager(snapshot_ttl=0, **({"clock": clock} if clock else {}))
    small = WhisperSTT("small", ram_mb=560, threads=6, idle_unload_s=60)
    base = WhisperSTT("base", ram_mb=320, threads=2, idle_unload_s=60)
    stt = HybridSTT([STTOption(small, small.spec, 2), STTOption(base, base.spec, 1, game_mode_ok=True)], rm,
                    lambda: SelectionContext())
    if game_mode:
        rm.enter_game_mode()
    capture = FakeCapture({"Microphone (Test)": FakeStream(speech_audio)})
    router = FastPathRouter()
    vi = VoiceInput(MicrophoneManager(FakeDevices(mic("Microphone (Test)"))), capture, stt, rm,
                    fast_path=lambda text: router.route(text) is not None)

    class NoModel:
        def plan(self, *a, **k):
            raise AssertionError("the language model must not be needed")

    engine = ConversationEngine(make_kernel(), NoModel(), resources=rm, router=router)
    return vi, engine, rm, (small, base)


def test_V04_spoken_request_runs_through_the_kernel_without_a_language_model(make_kernel, speech):
    clip = read_wav(speech[("Zira", "time")])
    vi, engine, rm, models = pipeline(make_kernel, silence(400) + clip.pcm16 + silence(1500))
    try:
        heard = vi.listen()
        assert words(heard.text) == "what time is it"
        events = []
        engine.handle(UserInput(heard.text, interface="voice", confidence=heard.confidence), events.append)
        report = [e for e in events if isinstance(e, ActionReport)][0]
        assert report.tool == "get_datetime" and report.status is Status.DONE
        assert rm.state is ResourceState.IDLE
        assert vi.last_timings["transcribing_s"] < 5
    finally:
        for m in models:
            m.unload()


def test_V05_game_mode_uses_the_light_model(make_kernel, speech):
    clip = read_wav(speech[("David", "folder")])
    vi, engine, rm, (small, base) = pipeline(make_kernel, silence(400) + clip.pcm16 + silence(1500),
                                             game_mode=True)
    try:
        heard = vi.listen()
        assert words(heard.text) == "open my downloads folder"
        assert vi.stt.last_choice == "stt:whisper-base@cpu"
        assert base.is_loaded() and not small.is_loaded()
    finally:
        small.unload()
        base.unload()


def test_V06_idle_speech_model_is_unloaded(make_kernel, speech):
    now = [1000.0]
    clip = read_wav(speech[("Zira", "yes")])
    vi, engine, rm, (small, base) = pipeline(make_kernel, silence(400) + clip.pcm16 + silence(1500),
                                             clock=lambda: now[0])
    try:
        vi.listen()
        pid = small.pid
        assert small.is_loaded() and psutil.pid_exists(pid)
        now[0] += 61
        assert rm.release_idle() == ["stt:whisper-small@cpu"]
        time.sleep(0.3)
        assert not small.is_loaded() and not psutil.pid_exists(pid)
    finally:
        small.unload()
        base.unload()


def test_V07_background_noise_alone_never_becomes_a_request(make_kernel):
    vi, engine, rm, models = pipeline(make_kernel, noise(9000, 600))
    try:
        with pytest.raises((NoSpeech, Unintelligible)):
            vi.listen()
    finally:
        for m in models:
            m.unload()
