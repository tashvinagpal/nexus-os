"""Live voice output: the real Kokoro model on this PC, played into a fake sink (pytest -m tts).

No speakers are used here; tests/test_tts_hardware.py plays through the real output.
"""

from __future__ import annotations

import re
import time

import psutil
import pytest

from nexus.ai.voice.base import TTSError
from nexus.ai.voice.kokoro import KOKORO_FILES, MODELS_DIR, KokoroTTS
from nexus.ai.voice.speaker import Speaker
from nexus.core.resources import ResourceManager
from tests.tts_fixtures import FakeSink

pytestmark = [pytest.mark.tts,
              pytest.mark.skipif(not all((MODELS_DIR / KOKORO_FILES[k][0]).is_file() for k in ("fp32", "voices")),
                                 reason="run scripts/get_tts_model.py fp32 first")]

PARAGRAPH = ("I found Outlook classic and Outlook. Which one do you want? I can open either of them for you, "
             "or neither if you changed your mind.")


@pytest.fixture(scope="module")
def kokoro():
    tts = KokoroTTS("fp32", threads=4, ram_mb=600)
    started = time.perf_counter()
    tts.load()
    tts.cold_load_s = time.perf_counter() - started
    yield tts
    tts.unload()


def test_T01_cold_load(kokoro):
    assert kokoro.cold_load_s < 10  # measured 1.7-3.8 s on the i5-12400F
    assert 150 < kokoro.memory_mb() < 900


def test_T02_short_sentence_is_fast(kokoro):
    kokoro.synthesize("Warm up.")
    started = time.perf_counter()
    clip = kokoro.synthesize("Opening Google Chrome.", language="en")
    took = time.perf_counter() - started
    assert clip.sample_rate == 24000 and 0.8 < clip.seconds < 3
    assert took < 2.0, f"{took:.2f} s"  # measured ~0.5 s


def test_T03_paragraph_faster_than_real_time(kokoro):
    started = time.perf_counter()
    clip = kokoro.synthesize(PARAGRAPH)
    took = time.perf_counter() - started
    assert took < clip.seconds  # measured ~0.3x real time


@pytest.mark.parametrize("text, language", [("क्रोम खुल गया है।", "hi"), ("अभी तीन बजकर बयालीस मिनट हुए हैं।", "hi"),
                                            ("हां भाई, एक second.", "hi"), ("It's 3:42 PM, bhai.", "en")])
def test_T04_hindi_and_mixed_text(kokoro, text, language):
    clip = kokoro.synthesize(text, language=language)
    assert clip.seconds > 0.5


def test_T05_bad_voice_is_an_error_and_the_engine_survives(kokoro):
    with pytest.raises(TTSError):
        kokoro.synthesize("Hello.", voice="xx_nobody")
    assert kokoro.is_loaded() and kokoro.synthesize("Still working.").seconds > 0.3


def test_T06_speaker_end_to_end_with_real_voice(kokoro):
    rm = ResourceManager(snapshot_ttl=0)
    speaker = Speaker(kokoro, rm, sink=FakeSink(instant=False, play_seconds=0.05))
    assert speaker.say(PARAGRAPH)
    assert speaker.last["chunks"] >= 2 and speaker.last["first_audio_s"] < 2.0
    assert speaker.say("Haan bhai, ek second.", language="hinglish")


def test_T07_stop_mid_reply(kokoro):
    rm = ResourceManager(snapshot_ttl=0)
    sink = FakeSink(instant=False, play_seconds=3)
    speaker = Speaker(kokoro, rm, sink=sink)
    speaker.say_async(PARAGRAPH)
    deadline = time.time() + 5
    while not sink.playbacks and time.time() < deadline:
        time.sleep(0.01)
    assert speaker.stop() and speaker.wait(5)
    assert len(sink.played) == 1 and kokoro.is_loaded()


def test_T08_unload_ends_the_process_and_reload_works():
    tts = KokoroTTS("fp32", threads=4)
    tts.load()
    pid = tts.pid
    started = time.perf_counter()
    tts.unload()
    assert time.perf_counter() - started < 3
    time.sleep(0.3)
    assert not psutil.pid_exists(pid)
    tts.load()
    assert tts.synthesize("Back again.").seconds > 0.3
    tts.unload()


def test_T09_what_it_says_can_be_understood():
    """Round trip: Kokoro speaks, Whisper (subsystem 5) transcribes. English must come back word for word."""
    from nexus.ai.voice.whisper import WhisperSTT, model_folder

    if not (model_folder("small") / "model.bin").is_file():
        pytest.skip("whisper small not downloaded")
    tts, stt = KokoroTTS("fp32", threads=4), WhisperSTT("small", ram_mb=600, threads=6)
    tts.load()
    stt.load()
    try:
        for text in ("Opening Google Chrome.", "Grand Theft Auto V Legacy is running.",
                     "I can't change security settings."):
            heard = stt.transcribe(tts.synthesize(text)).text
            norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).split()  # noqa: E731
            assert norm(heard) == norm(text), heard
    finally:
        tts.unload()
        stt.unload()
