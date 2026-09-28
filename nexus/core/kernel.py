"""The Action Kernel: the single path from "someone wants X" to X happening.

    request -> validate -> policy -> (confirm) -> execute -> verify -> audit

Every caller - the developer console, the conversation engine, the device
mesh gateway, later voice and the desktop shell - goes through submit() and
resolve(). Confirmation is answered by a user interface calling resolve(); it
is never exposed as a tool, so an AI model cannot approve its own requests.
A confirmation for a request from another device can only be answered by
that same device or on this computer.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from nexus.security.audit import AuditLog
from nexus.security.permissions import Decision, Policy, Verdict
from nexus.tools.base import Tool, ToolRegistry, ToolResult, ValidationError

log = logging.getLogger(__name__)

_AUDIT_TEXT_LIMIT = 500


class Status(str, enum.Enum):
    DONE = "done"
    FAILED = "failed"
    NEEDS_CONFIRMATION = "needs_confirmation"
    NEEDS_CLARIFICATION = "needs_clarification"  # e.g. "GTA V or GTA IV?" - nothing was done
    UNSUPPORTED = "unsupported"                  # not possible yet - nothing was done, nothing failed
    DECLINED = "declined"
    BLOCKED = "blocked"
    INVALID = "invalid"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    source: str = "console"    # which part of NEXUS asked: console | ai | voice | ui | mesh
    device: str | None = None  # None = this computer; else the id of the paired device that asked


@dataclass(frozen=True)
class Outcome:
    request_id: str
    status: Status
    message: str
    result: ToolResult | None = None
    confirmation_id: str | None = None


@dataclass(frozen=True)
class _Pending:
    request_id: str
    tool: Tool
    args: dict[str, Any]
    device: str | None
    expires_at: float


class ActionKernel:
    def __init__(self, registry: ToolRegistry, policy: Policy, audit: AuditLog, *,
                 confirm_ttl: float = 120.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.registry = registry
        self.policy = policy
        self.audit = audit
        self.confirm_ttl = confirm_ttl
        self._clock = clock
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    def preview(self, request: ToolRequest) -> Decision:
        """What submit() would decide, without running anything or writing to the audit log.

        Lets a front end avoid announcing "launching X" for a request that will be
        refused. submit() re-validates and re-checks everything; preview grants nothing.
        """
        tool = self.registry.get(request.tool)
        if tool is None:
            return Decision(Verdict.BLOCK, f"there is no tool called '{request.tool}'")
        try:
            args = tool.validate(request.args)
        except ValidationError as exc:
            return Decision(Verdict.BLOCK, str(exc))
        return self.policy.evaluate(tool, args, device=request.device)

    def submit(self, request: ToolRequest) -> Outcome:
        rid = uuid.uuid4().hex[:12]
        tool = self.registry.get(request.tool)
        self.audit.record("request", request_id=rid, tool=request.tool, source=request.source,
                          device=request.device, args=_redact(tool, request.args))
        if tool is None:
            return self._finish(rid, Status.INVALID, f"There is no tool called '{request.tool}'.")
        try:
            args = tool.validate(request.args)
        except ValidationError as exc:
            return self._finish(rid, Status.INVALID, str(exc))

        decision = self.policy.evaluate(tool, args, device=request.device)
        self.audit.record("decision", request_id=rid, verdict=decision.verdict.value, reason=decision.reason)
        if decision.verdict is Verdict.BLOCK:
            return self._finish(rid, Status.BLOCKED, f"Refused: {decision.reason}.")
        if decision.verdict is Verdict.CONFIRM:
            cid = uuid.uuid4().hex[:8]
            with self._lock:
                self._drop_expired()
                self._pending[cid] = _Pending(rid, tool, args, request.device,
                                              self._clock() + self.confirm_ttl)
            self.audit.record("confirmation_requested", request_id=rid, confirmation_id=cid)
            return Outcome(rid, Status.NEEDS_CONFIRMATION,
                           f"{tool.summary(args)}. Asking because {decision.reason}.",
                           confirmation_id=cid)
        return self._execute(rid, tool, args, request.device)

    def resolve(self, confirmation_id: str, approved: bool, *, device: str | None = None) -> Outcome:
        """Answer a pending confirmation. Must only be called from a user-facing interface.

        `device` is who is answering: None for this computer, else a paired device id.
        A request from a device can be answered by that device or locally, never by a third device.
        """
        with self._lock:
            pending = self._pending.get(confirmation_id)
            if pending is not None and device is not None and device != pending.device:
                pending = None
                wrong_device = True
            else:
                wrong_device = False
                self._pending.pop(confirmation_id, None)
        if pending is None:
            self.audit.record("confirmation_wrong_device" if wrong_device else "confirmation_unknown",
                              confirmation_id=confirmation_id, device=device)
            return Outcome("-", Status.INVALID, "That confirmation doesn't exist or was already answered.")
        rid = pending.request_id
        if self._clock() > pending.expires_at:
            return self._finish(rid, Status.EXPIRED, "That confirmation expired. Nothing was changed.")
        if not approved:
            return self._finish(rid, Status.DECLINED, "Cancelled. Nothing was changed.")

        self.audit.record("approved", request_id=rid, device=device)
        # things may have changed while waiting
        decision = self.policy.evaluate(pending.tool, pending.args, device=pending.device)
        if decision.verdict is Verdict.BLOCK:
            return self._finish(rid, Status.BLOCKED, f"Refused: {decision.reason}.")
        return self._execute(rid, pending.tool, pending.args, pending.device)

    def pending_count(self) -> int:
        with self._lock:
            self._drop_expired()
            return len(self._pending)

    def _execute(self, rid: str, tool: Tool, args: dict[str, Any], device: str | None) -> Outcome:
        started = time.perf_counter()
        call_args = {**args, "origin": device} if tool.receives_origin else args
        try:
            result = tool.run(**call_args)
        except Exception as exc:  # a broken tool must never take NEXUS down
            log.exception("tool %s crashed", tool.name)
            result = ToolResult(False, f"{tool.name} failed unexpectedly: {exc}")
        if not isinstance(result, ToolResult):
            result = ToolResult(False, f"{tool.name} returned an invalid result.")
        elif result.ok and result.verified is False:
            result = ToolResult(False, f"{tool.name} reported success but verification failed.",
                                result.data, verified=False)
        self.audit.record("executed", request_id=rid, tool=tool.name, ok=result.ok,
                          verified=result.verified, ms=round((time.perf_counter() - started) * 1000))
        if result.ok:
            status = Status.DONE
        elif result.options:
            status = Status.NEEDS_CLARIFICATION
        elif result.refused:
            status = Status.BLOCKED  # an intentional refusal by the tool, reported like a policy refusal
        elif result.unsupported:
            status = Status.UNSUPPORTED
        else:
            status = Status.FAILED
        return self._finish(rid, status, result.message, result)

    def _finish(self, rid: str, status: Status, message: str, result: ToolResult | None = None) -> Outcome:
        self.audit.record("outcome", request_id=rid, status=status.value,
                          message=message[:_AUDIT_TEXT_LIMIT])
        return Outcome(rid, status, message, result)

    def _drop_expired(self) -> None:
        now = self._clock()
        for cid in [c for c, p in self._pending.items() if now > p.expires_at]:
            self.audit.record("confirmation_expired", request_id=self._pending.pop(cid).request_id)


def _redact(tool: Tool | None, args: Mapping[str, Any]) -> dict[str, Any]:
    sensitive = {p.name for p in tool.params if p.sensitive} if tool else set()
    out: dict[str, Any] = {}
    for key, value in args.items():
        if key in sensitive:
            out[key] = "***"
        elif isinstance(value, str) and len(value) > _AUDIT_TEXT_LIMIT:
            out[key] = value[:_AUDIT_TEXT_LIMIT] + "..."
        else:
            out[key] = value
    return out
