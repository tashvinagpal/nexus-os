import pytest

from nexus.core.config import PROJECT_ROOT, ConfigError, load_ai_settings
from nexus.system.device import DeviceRole, detect_capabilities


def write(tmp_path, text):
    f = tmp_path / "nexus.toml"
    f.write_text(text, encoding="utf-8")
    return f


def test_shipped_config_is_valid():
    ai = load_ai_settings(PROJECT_ROOT / "config" / "nexus.toml")
    assert ai.allow_cloud is False
    assert [o.placement for o in ai.llm] == ["gpu", "gpu", "cpu"]


def test_missing_file_gives_defaults(tmp_path):
    assert load_ai_settings(tmp_path / "none.toml").llm == ()


@pytest.mark.parametrize("text", [
    "[ai]\nallow_clowd = true\n",
    "[ai]\nallow_cloud = \"yes\"\n",
    "[gpu]\nx = 1\n",
    "[[ai.llm]]\nmodel = \"x\"\nplacement = \"tpu\"\nram_mb = 1\n",
    "[[ai.llm]]\nmodel = \"x\"\nplacement = \"gpu\"\nram_mb = true\n",
    "[ai]\nllm_idle_unload_seconds = 5\n",
])
def test_bad_config_stops_startup(tmp_path, text):
    with pytest.raises(ConfigError):
        load_ai_settings(write(tmp_path, text))


def test_detect_capabilities_on_this_machine():
    caps = detect_capabilities()
    assert caps.ram_mb > 1000 and caps.cpu_threads >= 1
    assert caps.role in (DeviceRole.DESKTOP, DeviceRole.LAPTOP)
    assert caps.can_execute_desktop_tools
