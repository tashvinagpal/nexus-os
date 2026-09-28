import os
from pathlib import Path

import pytest

from nexus.security.permissions import Policy, PolicyError, Verdict, is_within
from nexus.tools.base import Param, Risk
from tests.conftest import make_tool

WRITER = make_tool("writer", params=(Param("path", "path", "p", writes=True),))
READER = make_tool("reader", params=(Param("path", "path", "p"),))


def decide(policy, tool, **args):
    return policy.evaluate(tool, tool.validate(args))


def test_safe_tool_allowed(policy):
    assert decide(policy, make_tool("info")).verdict is Verdict.ALLOW


def test_confirm_tool_needs_confirmation(policy):
    assert decide(policy, make_tool("rm", risk=Risk.CONFIRM)).verdict is Verdict.CONFIRM


def test_high_risk_always_blocked(policy):
    d = decide(policy, make_tool("format_drive", risk=Risk.HIGH))
    assert d.verdict is Verdict.BLOCK and "high-risk" in d.reason


def test_write_inside_safe_root_allowed(policy, sandbox):
    assert decide(policy, WRITER, path=str(sandbox / "home" / "new")).verdict is Verdict.ALLOW


def test_write_outside_safe_root_needs_confirmation(policy, sandbox):
    d = decide(policy, WRITER, path=str(sandbox / "outside" / "new"))
    assert d.verdict is Verdict.CONFIRM and "outside your safe folders" in d.reason


def test_write_into_protected_blocked(policy, sandbox):
    d = decide(policy, WRITER, path=str(sandbox / "system" / "evil"))
    assert d.verdict is Verdict.BLOCK and "protected" in d.reason


def test_protected_wins_over_safe_root(sandbox):
    p = Policy(protected=[sandbox / "home" / "keep"], safe_roots=[sandbox / "home"])
    assert decide(p, WRITER, path=str(sandbox / "home" / "keep" / "x")).verdict is Verdict.BLOCK


def test_dotdot_traversal_cannot_escape(policy, sandbox):
    sneaky = str(sandbox / "home" / ".." / "system" / "evil")
    assert decide(policy, WRITER, path=sneaky).verdict is Verdict.BLOCK


def test_drive_root_blocked(policy):
    root = Path.cwd().anchor or "/"
    assert decide(policy, WRITER, path=root).verdict is Verdict.BLOCK


def test_reading_protected_path_is_not_a_write(policy, sandbox):
    assert decide(policy, READER, path=str(sandbox / "system" / "file")).verdict is Verdict.ALLOW


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_case_insensitive_on_windows(policy, sandbox):
    upper = str(sandbox / "SYSTEM" / "evil")
    assert decide(policy, WRITER, path=upper).verdict is Verdict.BLOCK


@pytest.mark.skipif(os.name != "nt", reason="uses a Windows directory junction")
def test_junction_into_protected_is_followed(policy, sandbox):
    import _winapi

    link = sandbox / "home" / "innocent"
    _winapi.CreateJunction(str(sandbox / "system"), str(link))
    assert decide(policy, WRITER, path=str(link / "evil")).verdict is Verdict.BLOCK


def test_blocked_and_always_confirm_lists(sandbox):
    p = Policy(protected=[], safe_roots=[sandbox], blocked_tools=["nope"], always_confirm=["info"])
    assert decide(p, make_tool("nope")).verdict is Verdict.BLOCK
    assert decide(p, make_tool("info")).verdict is Verdict.CONFIRM


def test_is_within():
    base = Path.cwd()
    assert is_within(base / "a", base)
    assert is_within(base, base)
    assert not is_within(base.parent, base)
    assert not is_within(Path(str(base) + "-sibling"), base)  # prefix but not inside


# --- policy file ---------------------------------------------------------------

def load(tmp_path, text, protected=(), safe=()):
    f = tmp_path / "policy.toml"
    f.write_text(text, encoding="utf-8")
    return Policy.from_file(f, builtin_protected=list(protected), default_safe_roots=list(safe))


def test_missing_file_uses_defaults(tmp_path):
    p = Policy.from_file(tmp_path / "nope.toml", builtin_protected=[tmp_path / "s"],
                         default_safe_roots=[tmp_path / "h"])
    assert p.protected == ((tmp_path / "s").resolve(),)


def test_file_adds_protection_and_keeps_builtins(sandbox):
    p = load(sandbox, '[paths]\nprotected = ["%s"]\n' % str(sandbox / "home" / "keep").replace("\\", "\\\\"),
             protected=[sandbox / "system"], safe=[sandbox / "home"])
    assert decide(p, WRITER, path=str(sandbox / "home" / "keep" / "x")).verdict is Verdict.BLOCK
    assert decide(p, WRITER, path=str(sandbox / "system" / "x")).verdict is Verdict.BLOCK


def test_file_safe_roots_replace_default(sandbox):
    p = load(sandbox, '[paths]\nsafe_roots = ["%s"]\n' % str(sandbox / "outside").replace("\\", "\\\\"),
             safe=[sandbox / "home"])
    assert decide(p, WRITER, path=str(sandbox / "outside" / "x")).verdict is Verdict.ALLOW
    assert decide(p, WRITER, path=str(sandbox / "home" / "x")).verdict is Verdict.CONFIRM


@pytest.mark.parametrize("text", [
    "[paths]\nprotectd = []\n",           # typo must not silently weaken the policy
    "[danger]\nx = 1\n",
    "[tools]\nblocked = \"not a list\"\n",
    "this is not toml",
])
def test_bad_policy_file_fails_closed(tmp_path, text):
    with pytest.raises(PolicyError):
        load(tmp_path, text)
