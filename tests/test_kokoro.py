"""The Kokoro provider's process management and contract, with a fake worker (no model needed)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import psutil
import pytest

from nexus.ai.providers.base import Locality
from nexus.ai.voice.base import TTSError
from nexus.ai.voice.kokoro import KokoroTTS
from nexus.core.resources import Placement

WORKER = str(Path(__file__).with_name("fake_tts_worker.py"))


def fake(mode: str = "ok", **kw) -> KokoroTTS:
    return KokoroTTS("fp32", command=[sys.executable, WORKER, mode], **kw)


def test_initialization_and_spec():
    tts = fake()
    assert tts.info.name == "tts:kokoro@cpu" and tts.info.locality is Locality.LOCAL
    assert tts.spec.kind == "tts" and tts.spec.placement is Placement.CPU and tts.spec.vram_mb == 0
    assert not tts.is_loaded() and tts.available()


def test_load_synthesize_unload_ends_the_process():
    tts = fake()
    tts.load()
    pid = tts.pid
    clip = tts.synthesize("Opening Google Chrome.", language="en")
    assert clip.sample_rate == 24000 and clip.seconds == pytest.approx(0.22)
    assert tts.last_synthesis["characters"] == 22 and tts.last_synthesis["voice"] == "af_heart"
    tts.unload()
    time.sleep(0.2)
    assert not tts.is_loaded() and not psutil.pid_exists(pid)


def test_languages_pick_the_voice():
    tts = fake(voices={"en": "am_michael", "hi": "hm_omega"})
    tts.load()
    try:
        tts.synthesize("क्रोम खुल गया है।", language="hi")
        assert tts.last_synthesis["voice"] == "hm_omega" and tts.last_synthesis["language"] == "hi"
        tts.synthesize("Hello.", language="fr")  # not validated here: English
        assert tts.last_synthesis["language"] == "en"
    finally:
        tts.unload()


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_is_a_clear_error(text):
    tts = fake()
    with pytest.raises(TTSError) as err:
        tts.synthesize(text)
    assert "nothing to say" in str(err.value)


def test_synthesis_before_loading():
    with pytest.raises(TTSError):
        fake().synthesize("Hello.")


def test_engine_error_is_reported():
    tts = fake("error")
    tts.load()
    try:
        with pytest.raises(TTSError) as err:
            tts.synthesize("Hello.")
        assert "not found" in str(err.value)
        assert tts.is_loaded()  # a bad request doesn't end the engine
    finally:
        tts.unload()


def test_crash_is_reported_and_the_next_load_starts_fresh():
    tts = fake("crash")
    tts.load()
    with pytest.raises(TTSError) as err:
        tts.synthesize("Hello.")
    assert "restarts next time" in str(err.value) and not tts.is_loaded()
    tts.load()  # recovery: a new worker
    assert tts.is_loaded()
    tts.unload()


def test_timeout_stops_a_stuck_engine():
    tts = fake("slow", timeout_s=0.5)
    tts.load()
    started = time.perf_counter()
    with pytest.raises(TTSError):
        tts.synthesize("Hello.")
    assert time.perf_counter() - started < 3 and not tts.is_loaded()


def test_failed_load_is_unavailable_for_a_while():
    tts = fake("fail_load")
    with pytest.raises(TTSError) as err:
        tts.load()
    assert "damaged" in str(err.value) and not tts.available() and not tts.is_loaded()


def test_missing_model_is_explained(tmp_path):
    tts = KokoroTTS("fp32", models_dir=tmp_path)
    assert not tts.available()
    assert "get_tts_model.py fp32" in tts.problem or "kokoro-onnx" in tts.problem
    assert tts.health_check().ok is False
    with pytest.raises(TTSError):
        tts.load()


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        KokoroTTS("int4")


def test_capabilities_and_health():
    tts = fake()
    assert tts.health_check().ok and "loads on first use" in tts.health_check().detail
    tts.load()
    try:
        caps = tts.capabilities()
        assert caps.languages == ("en", "hi") and caps.sample_rates == (24000,)
        assert "af_heart" in caps.voices and "hf_alpha" in caps.voices
        assert not caps.supports_ssml
        health = tts.health_check()
        assert health.ok and "loaded" in health.detail
    finally:
        tts.unload()


def test_stop_never_kills_the_engine():
    tts = fake()
    tts.load()
    pid = tts.pid
    tts.stop()
    assert tts.is_loaded() and tts.pid == pid
    tts.unload()
