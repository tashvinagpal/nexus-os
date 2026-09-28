import os

import pytest

from nexus.core.kernel import Status, ToolRequest
from nexus.security.permissions import Policy
from nexus.tools.base import Param, Risk, ToolResult
from tests.conftest import Recorder, make_tool


def events(audit):
    return [e["event"] for e in audit.tail(100)]


def test_unknown_tool_is_invalid_and_audited(make_kernel, audit):
    out = make_kernel().submit(ToolRequest("run_shell", {"cmd": "format C:"}))
    assert out.status is Status.INVALID
    assert events(audit) == ["request", "outcome"]
    assert audit.tail(2)[0]["args"] == {"cmd": "format C:"}


def test_bad_arguments_never_reach_the_tool(make_kernel):
    body = Recorder()
    k = make_kernel(make_tool("t", run=body, params=(Param("n", "int", "n"),)))
    assert k.submit(ToolRequest("t", {"n": "abc"})).status is Status.INVALID
    assert body.calls == []


def test_safe_tool_runs_once(make_kernel, audit):
    body = Recorder()
    out = make_kernel(make_tool("t", run=body)).submit(ToolRequest("t"))
    assert out.status is Status.DONE and out.result.verified is True
    assert len(body.calls) == 1
    assert events(audit) == ["request", "decision", "executed", "outcome"]


def test_confirmation_flow_approved(make_kernel, audit):
    body = Recorder()
    k = make_kernel(make_tool("risky", risk=Risk.CONFIRM, run=body))
    out = k.submit(ToolRequest("risky"))
    assert out.status is Status.NEEDS_CONFIRMATION and out.confirmation_id
    assert body.calls == []  # nothing happens before the user answers
    done = k.resolve(out.confirmation_id, approved=True)
    assert done.status is Status.DONE and len(body.calls) == 1
    assert "approved" in events(audit)


def test_confirmation_is_single_use(make_kernel):
    body = Recorder()
    k = make_kernel(make_tool("risky", risk=Risk.CONFIRM, run=body))
    cid = k.submit(ToolRequest("risky")).confirmation_id
    k.resolve(cid, approved=True)
    assert k.resolve(cid, approved=True).status is Status.INVALID
    assert len(body.calls) == 1


def test_forged_confirmation_id_rejected(make_kernel, audit):
    assert make_kernel().resolve("deadbeef", approved=True).status is Status.INVALID
    assert events(audit) == ["confirmation_unknown"]


def test_declined_does_nothing(make_kernel):
    body = Recorder()
    k = make_kernel(make_tool("risky", risk=Risk.CONFIRM, run=body))
    out = k.resolve(k.submit(ToolRequest("risky")).confirmation_id, approved=False)
    assert out.status is Status.DECLINED and body.calls == []
    assert "Nothing was changed" in out.message


def test_confirmation_expires(make_kernel, clock):
    body = Recorder()
    k = make_kernel(make_tool("risky", risk=Risk.CONFIRM, run=body))
    cid = k.submit(ToolRequest("risky")).confirmation_id
    clock.now += 61
    assert k.resolve(cid, approved=True).status in (Status.EXPIRED, Status.INVALID)
    assert body.calls == [] and k.pending_count() == 0


def test_high_risk_blocked(make_kernel):
    body = Recorder()
    out = make_kernel(make_tool("wipe", risk=Risk.HIGH, run=body)).submit(ToolRequest("wipe"))
    assert out.status is Status.BLOCKED and body.calls == []


def test_crashing_tool_reported_not_raised(make_kernel):
    k = make_kernel(make_tool("boom", run=Recorder(exc=RuntimeError("kaboom"))))
    out = k.submit(ToolRequest("boom"))
    assert out.status is Status.FAILED and "kaboom" in out.message
    assert k.submit(ToolRequest("get_system_info", {"section": "memory"})).status is Status.DONE


def test_success_with_failed_verification_is_a_failure(make_kernel):
    body = Recorder(result=ToolResult(True, "trust me", verified=False))
    out = make_kernel(make_tool("liar", run=body)).submit(ToolRequest("liar"))
    assert out.status is Status.FAILED and "verification failed" in out.message


def test_non_result_return_value_is_a_failure(make_kernel):
    out = make_kernel(make_tool("bad", run=lambda: "ok")).submit(ToolRequest("bad"))
    assert out.status is Status.FAILED


def test_sensitive_args_redacted_in_audit(make_kernel, audit):
    tool = make_tool("login", params=(Param("pin", "str", "p", sensitive=True),))
    make_kernel(tool).submit(ToolRequest("login", {"pin": "1234"}))
    assert "1234" not in audit.path.read_text(encoding="utf-8")


def test_request_source_is_audited(make_kernel, audit):
    make_kernel().submit(ToolRequest("get_system_info", {"section": "memory"}, source="ai"))
    assert audit.tail(100)[0]["source"] == "ai"


@pytest.mark.skipif(os.name != "nt", reason="uses a Windows directory junction")
def test_policy_rechecked_at_approval_time(make_kernel, sandbox):
    """A junction swapped in while the user reads the prompt must not redirect the write."""
    import _winapi

    k = make_kernel()
    target = sandbox / "outside" / "link" / "payload"
    out = k.submit(ToolRequest("create_folder", {"path": str(target)}))
    assert out.status is Status.NEEDS_CONFIRMATION
    _winapi.CreateJunction(str(sandbox / "system"), str(sandbox / "outside" / "link"))
    done = k.resolve(out.confirmation_id, approved=True)
    assert done.status is Status.BLOCKED
    assert not (sandbox / "system" / "payload").exists()
