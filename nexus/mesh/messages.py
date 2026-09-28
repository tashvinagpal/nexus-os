"""Signed message envelopes exchanged between NEXUS devices.

Every envelope names its NEXUS, sender and recipient, carries a unique id and
a timestamp (for replay protection), and is signed by the sender's device key
over a canonical JSON encoding. Confidentiality comes from the transport
(e.g. Noise/TLS on the LAN, end-to-end encryption through any relay); the
signature makes messages authentic even through an untrusted relay.
"""

from __future__ import annotations

import enum
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from nexus.security.identity import DeviceKey, b64d, b64e, verify_signature

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024


class MessageKind(str, enum.Enum):
    USER_INPUT = "user_input"          # client -> core: something the user typed or said
    ACTION_REQUEST = "action_request"  # client -> core: a structured tool request (e.g. an app button)
    CONFIRM = "confirm"                # client -> core: answer to a pending confirmation
    STATUS_REQUEST = "status_request"  # client -> core: what is NEXUS doing, resources
    TURN_EVENT = "turn_event"          # core -> client: say / action report / confirmation request
    STATUS = "status"                  # core -> client
    ERROR = "error"                    # core -> client


# kind -> {field: (type, required)}
_BODY: dict[MessageKind, dict[str, tuple[type, bool]]] = {
    MessageKind.USER_INPUT: {"text": (str, True), "conversation_id": (str, False)},
    MessageKind.ACTION_REQUEST: {"tool": (str, True), "args": (dict, True)},
    MessageKind.CONFIRM: {"confirmation_id": (str, True), "approved": (bool, True),
                          "conversation_id": (str, False)},
    MessageKind.STATUS_REQUEST: {},
    MessageKind.TURN_EVENT: {"event": (dict, True), "in_reply_to": (str, False)},
    MessageKind.STATUS: {"status": (dict, True), "in_reply_to": (str, False)},
    MessageKind.ERROR: {"error": (str, True), "in_reply_to": (str, False)},
}


class MessageError(ValueError):
    """Malformed envelope or body."""


def validate_body(kind: MessageKind, body: dict[str, Any]) -> None:
    spec = _BODY[kind]
    for key in body:
        if key not in spec:
            raise MessageError(f"{kind.value}: unexpected field '{key}'")
    for key, (typ, required) in spec.items():
        if key not in body:
            if required:
                raise MessageError(f"{kind.value}: missing field '{key}'")
        elif not isinstance(body[key], typ) or (typ is not bool and isinstance(body[key], bool)):
            raise MessageError(f"{kind.value}: field '{key}' must be {typ.__name__}")


@dataclass(frozen=True)
class Envelope:
    v: int
    msg_id: str
    nexus_id: str
    sender: str
    recipient: str
    sent_at_ms: int
    kind: MessageKind
    body: dict[str, Any] = field(default_factory=dict)
    sig: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    def signing_bytes(self) -> bytes:
        d = self.to_dict()
        d.pop("sig")
        return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Envelope:
        if len(text.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise MessageError("message too large")
        try:
            d = json.loads(text)
            env = cls(v=int(d["v"]), msg_id=str(d["msg_id"]), nexus_id=str(d["nexus_id"]),
                      sender=str(d["sender"]), recipient=str(d["recipient"]), sent_at_ms=int(d["sent_at_ms"]),
                      kind=MessageKind(d["kind"]), body=d["body"], sig=str(d["sig"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise MessageError(f"malformed envelope: {exc}") from None
        if not isinstance(env.body, dict):
            raise MessageError("body must be an object")
        validate_body(env.kind, env.body)
        return env


def seal(key: DeviceKey, *, nexus_id: str, recipient: str, kind: MessageKind, body: dict[str, Any],
         now_ms: int | None = None) -> Envelope:
    """Build and sign an envelope from this device."""
    validate_body(kind, body)
    env = Envelope(v=PROTOCOL_VERSION, msg_id=uuid.uuid4().hex, nexus_id=nexus_id, sender=key.device_id,
                   recipient=recipient, sent_at_ms=now_ms if now_ms is not None else int(time.time() * 1000),
                   kind=kind, body=body)
    return replace(env, sig=b64e(key.sign(env.signing_bytes())))


def signature_valid(env: Envelope, public_key: bytes) -> bool:
    try:
        sig = b64d(env.sig)
    except ValueError:
        return False
    return verify_signature(public_key, env.signing_bytes(), sig)
