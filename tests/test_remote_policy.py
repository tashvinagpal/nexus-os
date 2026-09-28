"""Remote-device rules in the policy and kernel (additions to the Action Kernel)."""

from nexus.core.kernel import Status, ToolRequest
from nexus.security.permissions import Policy, Verdict
from tests.conftest import Recorder, make_tool


def test_remote_request_needs_confirmation_by_default(policy):
    tool = make_tool("launch_game")
    assert policy.evaluate(tool, {}).verdict is Verdict.ALLOW
    d = policy.evaluate(tool, {}, device="nxd-phone")
    assert d.verdict is Verdict.CONFIRM and "another device" in d.reason


def test_remote_allowlist_and_blocklist(sandbox):
    p = Policy(protected=[], safe_roots=[sandbox], remote_allowed=["launch_game"], remote_blocked=["settings"])
    assert p.evaluate(make_tool("launch_game"), {}, device="nxd-phone").verdict is Verdict.ALLOW
    assert p.evaluate(make_tool("settings"), {}, device="nxd-phone").verdict is Verdict.BLOCK
    assert p.evaluate(make_tool("settings"), {}).verdict is Verdict.ALLOW


def test_remote_section_in_policy_file(tmp_path):
    f = tmp_path / "policy.toml"
    f.write_text('[remote]\nallow_without_confirmation = ["launch_game"]\nblocked = ["create_folder"]\n',
                 encoding="utf-8")
    p = Policy.from_file(f, builtin_protected=[], default_safe_roots=[tmp_path])
    assert p.remote_allowed == {"launch_game"} and p.remote_blocked == {"create_folder"}


def test_preview_runs_nothing_and_logs_nothing(make_kernel, audit):
    body = Recorder()
    k = make_kernel(make_tool("t", run=body))
    assert k.preview(ToolRequest("t")).verdict is Verdict.ALLOW
    assert k.preview(ToolRequest("missing")).verdict is Verdict.BLOCK
    assert k.preview(ToolRequest("t", device="nxd-phone")).verdict is Verdict.CONFIRM
    assert body.calls == [] and audit.tail(10) == []


def test_confirmation_bound_to_requesting_device(make_kernel, audit):
    body = Recorder()
    k = make_kernel(make_tool("t", run=body))
    out = k.submit(ToolRequest("t", device="nxd-phone"))
    assert out.status is Status.NEEDS_CONFIRMATION
    assert k.resolve(out.confirmation_id, True, device="nxd-laptop").status is Status.INVALID
    assert audit.tail(1)[0]["event"] == "confirmation_wrong_device"
    assert body.calls == []
    # still pending for the right device
    assert k.resolve(out.confirmation_id, True, device="nxd-phone").status is Status.DONE
    assert len(body.calls) == 1


def test_local_user_can_answer_a_remote_request(make_kernel):
    body = Recorder()
    k = make_kernel(make_tool("t", run=body))
    out = k.submit(ToolRequest("t", device="nxd-phone"))
    assert k.resolve(out.confirmation_id, True).status is Status.DONE


def test_device_is_recorded_in_audit(make_kernel, audit):
    make_kernel().submit(ToolRequest("get_system_info", {"section": "memory"}, device="nxd-phone"))
    assert audit.tail(100)[0]["device"] == "nxd-phone"
