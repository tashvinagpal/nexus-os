"""Wake word with the real models (pytest -m wake). Speech is generated with Kokoro - no microphone, no speaker.

Needs: scripts/get_wake_model.py [--hey-jarvis], scripts/get_tts_model.py fp32, and for the "Nexus" tests
models/wake/nexus.onnx (scripts/train_wake_model.py); the Whisper test needs scripts/get_stt_model.py base.
Anything missing is skipped with the command that provides it.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from nexus.ai.voice.wake import OpenWakeWordDetector, WhisperWakeDetector
from nexus.core.config import PROJECT_ROOT
from nexus.core.resources import ResourceManager

pytestmark = pytest.mark.wake

WAKE = PROJECT_ROOT / "models" / "wake"
KOKORO = PROJECT_ROOT / "models" / "tts" / "kokoro-v1.0"
FEATURES = (WAKE / "melspectrogram.onnx", WAKE / "embedding_model.onnx")


def need(path: Path, how: str) -> None:
    if not path.is_file():
        pytest.skip(f"{path.name} missing - {how}")


@pytest.fixture(scope="module")
def say():
    need(KOKORO / "kokoro-v1.0.onnx", "scripts/get_tts_model.py fp32")
    if find_spec("kokoro_onnx") is None:
        pytest.skip("kokoro-onnx not installed (pip install -e \".[tts]\")")
    import numpy as np
    from kokoro_onnx import Kokoro

    kokoro = Kokoro(str(KOKORO / "kokoro-v1.0.onnx"), str(KOKORO / "voices-v1.0.bin"))

    def speak(text: str, voice: str = "am_michael", *, ipa: bool = False):
        samples, rate = kokoro.create(text, voice=voice, lang="en-us", is_phonemes=ipa)
        n = int(len(samples) * 16000 / rate)
        y = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
        return (np.clip(y, -1, 1) * 32767 * 0.5).astype(np.int16)
    return speak


def stream_of(*parts):
    import numpy as np

    gap = np.zeros(12800, np.int16)  # 0.8 s
    return np.concatenate([gap] + [p for part in parts for p in (part, gap)] + [gap, gap]).tobytes()


def hits(detector, pcm: bytes):
    frame = 16000 * detector.frame_ms // 1000 * 2
    found = []
    for i in range(0, len(pcm) - frame + 1, frame):
        hit = detector.process(pcm[i:i + frame])
        if hit is not None:
            found.append((round(i / 32000, 2), hit))
    return found


def test_features_match_the_openwakeword_package_exactly(say):
    for f in FEATURES:
        need(f, "scripts/get_wake_model.py")
    need(WAKE / "hey_jarvis_v0.1.onnx", "scripts/get_wake_model.py --hey-jarvis")
    if find_spec("openwakeword") is None:
        pytest.skip("the openwakeword package isn't installed (only needed for this comparison)")
    import numpy as np
    from openwakeword.model import Model

    from nexus.ai.voice.oww import CHUNK, WakeFeatures, WakeModel

    audio = np.frombuffer(stream_of(say("The weather is nice today."), say("Hey Jarvis, open the browser.")),
                          dtype=np.int16)
    ref = Model(wakeword_models=[str(WAKE / "hey_jarvis_v0.1.onnx")], inference_framework="onnx",
                melspec_model_path=str(FEATURES[0]), embedding_model_path=str(FEATURES[1]))
    ours, net = WakeFeatures(*FEATURES), WakeModel(WAKE / "hey_jarvis_v0.1.onnx")
    a, b = [], []
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        a.append(list(ref.predict(audio[i:i + CHUNK]).values())[0])
        ours.push(audio[i:i + CHUNK])
        b.append(net.score(ours.window(net.frames)))
    assert np.abs(np.array(a[30:]) - np.array(b[30:])).max() < 1e-4  # after both have 2.4 s of real audio
    assert max(b) > 0.9


def test_real_detector_hears_hey_jarvis_and_not_other_speech(say):
    need(WAKE / "hey_jarvis_v0.1.onnx", "scripts/get_wake_model.py --hey-jarvis")
    det = OpenWakeWordDetector(WAKE / "hey_jarvis_v0.1.onnx", *FEATURES, ResourceManager())
    det.start()
    try:
        found = hits(det, stream_of(say("I think we should leave a little earlier today."),
                                    say("Hey Jarvis, what time is it?", "af_heart"),
                                    say("Next stop is Connaught Place."), say("Hey Jarvis!", "bm_george")))
    finally:
        det.stop()
    assert len(found) == 2
    assert all(h.score >= 0.5 and len(h.lead_in) > 0 for _, h in found)


NEXUS_CASES = [("Nexus, open Chrome.", "af_nova", False), ("Hey Nexus!", "bm_lewis", False),
               ("nˈeːksəs", "hm_psi", True), ("Nexus, what time is it?", "am_echo", False)]
OTHER = [("Next stop is Connaught Place.", "af_nova"), ("My brother drives a Lexus.", "am_echo"),
         ("We visited Texas last summer.", "bm_lewis"), ("Put the necklace in the box.", "if_sara"),
         ("Alexa, what's the time?", "af_nova"), ("Bhai kya kar raha hai?", "hm_psi")]


def test_trained_nexus_model_on_voices_it_never_heard(say):
    """Voices kept out of training (scripts/train_wake_model.py HELD_OUT) saying "Nexus" and near-misses."""
    need(WAKE / "nexus.onnx", "train it: scripts/train_wake_model.py")
    det = OpenWakeWordDetector(WAKE / "nexus.onnx", *FEATURES, ResourceManager())
    det.start()
    try:
        heard = [bool(hits(det, stream_of(say(text, voice, ipa=ipa)))) for text, voice, ipa in NEXUS_CASES]
        det.reset()
        false = hits(det, stream_of(*(say(text, voice) for text, voice in OTHER)))
    finally:
        det.stop()
    assert sum(heard) >= 3, heard
    assert false == []


def test_whisper_detector_with_the_real_model(say):
    from nexus.ai.voice.whisper import WhisperSTT, model_folder

    need(model_folder("base") / "model.bin", "scripts/get_stt_model.py base")
    rm = ResourceManager()
    stt = WhisperSTT("base", threads=2, ram_mb=320)
    rm.register(stt)
    det = WhisperWakeDetector(stt, rm)
    det.start()
    try:
        found = hits(det, stream_of(say("Nexus, open Chrome."), say("Open Chrome please."),
                                    say("I think we should leave a little earlier today.")))
    finally:
        rm.unload_all()
    assert len(found) == 1 and found[0][0] < 3.0


def test_real_detector_through_the_listener_to_a_request(say):
    """Kokoro speech -> real openWakeWord detector -> WakeListener -> VoiceInput (speech-to-text faked)."""
    need(WAKE / "hey_jarvis_v0.1.onnx", "scripts/get_wake_model.py --hey-jarvis")
    import threading

    from nexus.ai.providers.selection import SelectionContext
    from nexus.ai.voice.base import Transcript
    from nexus.ai.voice.listener import VoiceInput
    from nexus.ai.voice.stt import HybridSTT, STTOption
    from nexus.ai.voice.wake_listener import WakeListener, WakeOptions
    from nexus.system.audio.devices import MicrophoneManager
    from tests.conftest import FakeProbe
    from tests.voice_fixtures import FakeCapture, FakeDevices, FakeStream, FakeSTT, mic, silence

    audio = stream_of(say("I think we should leave a little earlier today."),
                      say("Hey Jarvis, open Chrome.", "af_heart"))
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0)
    stt = FakeSTT("base", results=[Transcript("Hey Jarvis, open Chrome.", "en", 0.9)])
    vi = VoiceInput(MicrophoneManager(FakeDevices(mic("Mic"))),
                    FakeCapture({"Mic": FakeStream(audio, tail=silence(300))}),
                    HybridSTT([STTOption(stt, stt.spec, 1)], rm, lambda: SelectionContext()), rm)
    det = OpenWakeWordDetector(WAKE / "hey_jarvis_v0.1.onnx", *FEATURES, rm)
    heard, done = [], threading.Event()

    def on_heard(h, hit):
        heard.append((h.text, hit))
        done.set()
        return False

    listener = WakeListener(vi, [det], rm, on_heard=on_heard, options=WakeOptions(speaking_tail_s=0))
    listener.start()
    try:
        assert done.wait(20)
    finally:
        listener.stop()
    text, hit = heard[0]
    assert hit.score >= 0.5 and len(stt.calls) == 1
    assert stt.calls[0][0].seconds > 1.0  # the wake word's audio and the request, as one utterance
