"""[voice] in config/nexus.toml: shipped defaults and fail-closed validation."""

from __future__ import annotations

import pytest

from nexus.core.config import PROJECT_ROOT, ConfigError, load_ai_settings, load_voice_settings


def write(tmp_path, text: str):
    f = tmp_path / "nexus.toml"
    f.write_text(text, encoding="utf-8")
    return f


def test_shipped_config():
    voice = load_voice_settings(PROJECT_ROOT / "config" / "nexus.toml")
    assert voice.enabled and voice.microphone == "auto"
    assert voice.languages == ("en", "hi", "pa") and voice.language == "auto"
    assert [o.model for o in voice.stt] == ["small", "base"]
    assert all(o.placement == "cpu" for o in voice.stt)  # no GPU unless you choose it
    light = [o for o in voice.stt if o.game_mode]
    assert [o.model for o in light] == ["base"] and light[0].threads <= 2
    assert voice.stt_idle_unload_seconds <= 300  # never kept loaded
    assert voice.max_utterance_seconds <= 30 and voice.no_speech_timeout_seconds <= 10
    assert voice.confirm_min_confidence >= voice.min_confidence
    load_ai_settings(PROJECT_ROOT / "config" / "nexus.toml")  # [voice] is an allowed section


def test_missing_file_gives_defaults(tmp_path):
    voice = load_voice_settings(tmp_path / "none.toml")
    assert voice.enabled and voice.stt == ()


@pytest.mark.parametrize("text, message", [
    ("[voice]\nmicrofone = 'auto'\n", "unknown setting 'microfone'"),
    ("[voice]\nenabled = 'yes'\n", "must be bool"),
    ("[voice]\nmax_utterance_seconds = 600\n", "between 3 and 60"),
    ("[voice]\nno_speech_timeout_seconds = 0\n", "between 2 and 30"),
    ("[voice]\nmin_confidence = 1.5\n", "between 0.0 and 1.0"),
    ("[voice]\nlanguages = []\n", "language codes"),
    ("[voice]\nlanguages = ['english']\n", "language codes"),
    ("[voice]\nlanguage = 'fr'\n", "one of voice.languages"),
    ("[[voice.stt]]\nmodel = 'smal'\nram_mb = 500\n", "must be one of"),
    ("[[voice.stt]]\nmodel = 'small'\n", "needs model, ram_mb"),
    ("[[voice.stt]]\nmodel = 'small'\nram_mb = 500\nplacement = 'tpu'\n", "placement (cpu|gpu)"),
    ("[[voice.stt]]\nmodel = 'small'\nram_mb = 500\nthreads = 99\n", "threads"),
    ("[[voice.stt]]\nmodel = 'small'\nram_mb = 500\nbeam = 5\n", "unknown setting 'beam'"),
])
def test_bad_settings_stop_startup(tmp_path, text, message):
    with pytest.raises(ConfigError) as err:
        load_voice_settings(write(tmp_path, text))
    assert message in str(err.value)


def test_gpu_option_is_accepted_when_asked_for(tmp_path):
    voice = load_voice_settings(write(tmp_path, "[[voice.stt]]\nmodel = 'large-v3-turbo'\nplacement = 'gpu'\n"
                                                "ram_mb = 900\nvram_mb = 1700\n"))
    assert voice.stt[0].placement == "gpu" and voice.stt[0].vram_mb == 1700
