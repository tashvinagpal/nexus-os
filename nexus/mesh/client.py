"""MeshClient: how a phone (or laptop) app talks to a NEXUS core device.

The client only *sends requests* and shows what comes back. It runs no tools
and holds no authority: every request is authenticated and judged by the
core's gateway and Action Kernel. It pins the core's public key from pairing
and drops anything not signed by it.

This is the interface the mobile app will be built on; the phone's own UI,
voice and notifications sit on top of it and respect Android/iOS sandboxing.
"""

from __future__ import annotations

from typing import Any

from nexus.mesh.messages import Envelope, MessageKind, seal, signature_valid
from nexus.mesh.transport import LoopbackHub, Transport
from nexus.security.identity import DeviceKey, b64d


class MeshClient:
    def __init__(self, key: DeviceKey, *, nexus_id: str, core_id: str, core_public_key: str) -> None:
        self.key = key
        self.nexus_id = nexus_id
        self.core_id = core_id
        self._core_key = b64d(core_public_key)
        self._transport: Transport | None = None
        self.inbox: list[Envelope] = []
        self.dropped = 0

    def connect(self, hub: LoopbackHub) -> None:
        self._transport = hub.connect(self.key.device_id, self._receive)

    # --- requests ----------------------------------------------------------------

    def say(self, text: str, *, conversation_id: str | None = None) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"text": text}
        if conversation_id:
            body["conversation_id"] = conversation_id
        return self._events(self._send(MessageKind.USER_INPUT, body))

    def request(self, tool: str, args: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._events(self._send(MessageKind.ACTION_REQUEST, {"tool": tool, "args": args or {}}))

    def confirm(self, confirmation_id: str, approved: bool) -> list[dict[str, Any]]:
        return self._events(self._send(MessageKind.CONFIRM,
                                       {"confirmation_id": confirmation_id, "approved": approved}))

    def status(self) -> dict[str, Any]:
        replies = self._send(MessageKind.STATUS_REQUEST, {})
        return next((r.body["status"] for r in replies if r.kind is MessageKind.STATUS), {})

    # --- internals ---------------------------------------------------------------

    def _send(self, kind: MessageKind, body: dict[str, Any]) -> list[Envelope]:
        if self._transport is None:
            raise RuntimeError("not connected")
        env = seal(self.key, nexus_id=self.nexus_id, recipient=self.core_id, kind=kind, body=body)
        start = len(self.inbox)
        self._transport.send(env)  # loopback delivers replies synchronously; a network transport would await
        return [r for r in self.inbox[start:] if r.body.get("in_reply_to") == env.msg_id]

    def _receive(self, env: Envelope) -> None:
        if env.sender != self.core_id or env.recipient != self.key.device_id or not signature_valid(env, self._core_key):
            self.dropped += 1
            return
        self.inbox.append(env)

    @staticmethod
    def _events(replies: list[Envelope]) -> list[dict[str, Any]]:
        return [r.body["event"] for r in replies if r.kind is MessageKind.TURN_EVENT]
