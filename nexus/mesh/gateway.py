"""Gateway: where messages from your other devices enter NEXUS.

It authenticates (paired, not revoked, correct NEXUS and recipient, valid
signature, fresh, not replayed) and then hands the request to the same
conversation engine / Action Kernel a local user goes through - tagged with
the sending device, so the security policy applies its remote-device rules.
Being paired never grants more than the policy allows.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.types import ActionReport, ConfirmationRequest, TurnEvent, UserInput, event_to_dict
from nexus.core.kernel import ActionKernel, Status, ToolRequest
from nexus.core.resources import ResourceManager
from nexus.mesh.messages import PROTOCOL_VERSION, Envelope, MessageError, MessageKind, seal, signature_valid
from nexus.mesh.registry import DeviceRegistry
from nexus.mesh.transport import Transport
from nexus.security.identity import DeviceKey, b64d


class MeshGateway:
    def __init__(self, key: DeviceKey, nexus_id: str, registry: DeviceRegistry, kernel: ActionKernel, *,
                 engine: ConversationEngine | None = None, resources: ResourceManager | None = None,
                 clock: Callable[[], float] = time.time, max_skew_s: float = 120.0) -> None:
        self.key = key
        self.nexus_id = nexus_id
        self.registry = registry
        self.kernel = kernel
        self.engine = engine
        self.resources = resources
        self._clock = clock
        self.max_skew_s = max_skew_s
        self._seen: dict[str, float] = {}
        self._transport: Transport | None = None

    def attach(self, transport: Transport) -> None:
        self._transport = transport

    def watch_apps(self, apps) -> None:
        """Forward background results (e.g. "GTA V is running.") to the device that asked for them."""

        def forward(event) -> None:
            record = self.registry.get(event.origin) if event.origin else None
            if record is None or record.revoked or self._transport is None:
                return
            say = {"type": "say", "kind": "result" if event.ok else "error", "text": event.message,
                   "language": "en"}
            self._transport.send(seal(self.key, nexus_id=self.nexus_id, recipient=event.origin,
                                      kind=MessageKind.TURN_EVENT, body={"event": say}))

        apps.subscribe(forward)

    def receive(self, envelope: Envelope) -> None:
        """Transport callback: handle and send any replies back."""
        for reply in self.handle(envelope):
            if self._transport is not None:
                self._transport.send(reply)

    def handle(self, env: Envelope) -> list[Envelope]:
        problem = self._authenticate(env)
        if problem:
            # No reply to unauthenticated senders: don't tell a stranger anything.
            self.kernel.audit.record("mesh_rejected", sender=env.sender, kind=env.kind.value, reason=problem)
            return []
        self.kernel.audit.record("mesh_message", sender=env.sender, kind=env.kind.value, msg_id=env.msg_id)
        try:
            replies = self._dispatch(env)
        except MessageError as exc:
            replies = [(MessageKind.ERROR, {"error": str(exc)})]
        return [seal(self.key, nexus_id=self.nexus_id, recipient=env.sender, kind=kind,
                     body={**body, "in_reply_to": env.msg_id}) for kind, body in replies]

    # --- internals ---------------------------------------------------------------

    def _authenticate(self, env: Envelope) -> str | None:
        if env.v != PROTOCOL_VERSION:
            return f"unsupported protocol version {env.v}"
        if env.nexus_id != self.nexus_id:
            return "message belongs to a different NEXUS"
        if env.recipient != self.key.device_id:
            return "message is not addressed to this device"
        record = self.registry.get(env.sender)
        if record is None:
            return "unknown device (not paired)"
        if record.revoked:
            return "device access was revoked"
        if not signature_valid(env, b64d(record.public_key)):
            return "invalid signature"
        now = self._clock()
        if abs(now * 1000 - env.sent_at_ms) > self.max_skew_s * 1000:
            return "message is too old (or the device clock is wrong)"
        self._seen = {k: exp for k, exp in self._seen.items() if exp > now}
        if env.msg_id in self._seen:
            return "replayed message"
        self._seen[env.msg_id] = now + 2 * self.max_skew_s
        return None

    def _dispatch(self, env: Envelope) -> list[tuple[MessageKind, dict[str, Any]]]:
        body = env.body
        conversation = body.get("conversation_id", f"device:{env.sender}")
        user = UserInput(text=body.get("text", ""), interface="mesh", device=env.sender,
                         conversation_id=conversation)
        events: list[TurnEvent] = []

        if env.kind is MessageKind.USER_INPUT:
            if self.engine is None:
                raise MessageError("this device has no conversation engine")
            self.engine.handle(user, events.append)
        elif env.kind is MessageKind.ACTION_REQUEST:
            outcome = self.kernel.submit(ToolRequest(body["tool"], body["args"], source="mesh", device=env.sender))
            verified = outcome.result.verified if outcome.result else None
            events.append(ActionReport(body["tool"], outcome.status, verified, outcome.message, outcome.request_id))
            if outcome.status is Status.NEEDS_CONFIRMATION and outcome.confirmation_id:
                events.append(ConfirmationRequest(outcome.confirmation_id, outcome.message))
        elif env.kind is MessageKind.CONFIRM:
            if self.engine is not None:
                self.engine.answer(body["confirmation_id"], body["approved"], user, events.append)
            else:
                outcome = self.kernel.resolve(body["confirmation_id"], body["approved"], device=env.sender)
                verified = outcome.result.verified if outcome.result else None
                events.append(ActionReport("?", outcome.status, verified, outcome.message, outcome.request_id))
        elif env.kind is MessageKind.STATUS_REQUEST:
            # Includes this device's capabilities ("16 GB RAM, 12 GB GPU, runs llama3.1:8b").
            status = self.resources.status() if self.resources is not None else {}
            return [(MessageKind.STATUS, {"status": {**status, "device_name": self.key.identity.name,
                                                     "device_id": self.key.device_id}})]
        else:
            raise MessageError(f"{env.kind.value} is not a request")
        return [(MessageKind.TURN_EVENT, {"event": event_to_dict(e)}) for e in events]
