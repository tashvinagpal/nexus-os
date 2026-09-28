"""[tts] in config/nexus.toml: shipped defaults and fail-closed validation."""

from __future__ import annotations

import pytest

from nexus.core.config import PROJECT_ROOT, ConfigError, load_ai_settings, load_tts_settings


def write(tmp_path, text: str):
    f = tmp_path / "nexus.toml"
    f.write_text(text, encoding="utf-8")
    return f


def test_shipped_config():
    tts = load_tts_settings(PROJECT_ROOT / "config" / "nexus.toml")
    assert tts.enabled and tts.provider == "kokoro" and tts.model == "fp32"
    assert tts.placement == "cpu"            # no GPU reserved for speech
    assert tts.game_mode == "quiet"          # never interferes with a game by default
    assert tts.idle_unload_seconds <= 300    # never kept loaded
    assert tts.romanized == "hindi"
    load_ai_settings(PROJECT_ROOT / "config" / "nexus.toml")  # [tts] is an allowed section


def test_missing_file_gives_defaults(tmp_path):
    assert load_tts_settings(tmp_path / "none.toml").enabled


@pytest.mark.parametrize("text, message", [
    ("[tts]\nvoise = 'af_heart'\n", "unknown setting 'voise'"),
    ("[tts]\nprovider = 'elevenlabs'\n", "must be one of kokoro"),
    ("[tts]\nmodel = 'huge'\n", "must be one of"),
    ("[tts]\nspeed = 5.0\n", "between 0.5 and 2.0"),
    ("[tts]\nvolume = 1.5\n", "between 0.0 and 1.0"),
    ("[tts]\ngame_mode = 'loud'\n", "quiet, speak"),
    ("[tts]\nromanized = 'klingon'\n", "english, hindi"),
    ("[tts]\nvoice = 'Robot Voice'\n", "voice name"),
    ("[tts]\nidle_unload_seconds = 1\n", "between 10 and 3600"),
    ("[tts]\nenabled = 'yes'\n", "must be bool"),
])
def test_bad_settings_stop_startup(tmp_path, text, message):
    with pytest.raises(ConfigError) as err:
        load_tts_settings(write(tmp_path, text))
    assert message in str(err.value)
