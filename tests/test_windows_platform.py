"""Checks against the real Windows machine. Nothing here modifies the system:
the policy only *evaluates* requests; no tool is executed."""

import os
from pathlib import Path

import pytest

from nexus.core.config import load_settings
from nexus.core.runtime import create_runtime
from nexus.security.permissions import Verdict

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows only")


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    from nexus.system.platform import current

    data = tmp_path_factory.mktemp("var")
    settings = load_settings({"NEXUS_DATA_DIR": str(data)})
    return create_runtime(settings, current())


def verdict(rt, path):
    tool = rt.registry.get("create_folder")
    return rt.policy.evaluate(tool, tool.validate({"path": path})).verdict


def test_platform_basics(runtime):
    p = runtime.platform
    assert p.name == "windows"
    assert isinstance(p.is_elevated(), bool)
    assert p.cpu_name()
    protected = [str(x).lower() for x in p.protected_paths()]
    assert os.environ["SystemRoot"].lower() in protected


def test_real_policy_decisions(runtime):
    windir = os.environ["SystemRoot"]
    drive = os.environ.get("SystemDrive", "C:") + "\\"
    assert verdict(runtime, os.path.join(windir, "System32", "nexus")) is Verdict.BLOCK
    assert verdict(runtime, os.path.join(os.environ["ProgramFiles"], "nexus")) is Verdict.BLOCK
    assert verdict(runtime, str(Path.home() / "AppData" / "Roaming" / "x")) is Verdict.BLOCK
    assert verdict(runtime, drive) is Verdict.BLOCK
    assert verdict(runtime, drive + "$Recycle.Bin") is Verdict.BLOCK  # a real model proposed this
    assert verdict(runtime, drive + "nexus-test") is Verdict.CONFIRM
    assert verdict(runtime, str(Path.home() / "Desktop" / "nexus-test")) is Verdict.ALLOW


def test_nexus_cannot_modify_itself(runtime):
    assert verdict(runtime, str(runtime.settings.project_root / "nexus" / "evil")) is Verdict.BLOCK
    assert verdict(runtime, str(runtime.settings.data_dir / "audit")) is Verdict.BLOCK
