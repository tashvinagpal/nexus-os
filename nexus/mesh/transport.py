"""Transports move envelopes between devices. Authentication does not depend on
the transport (envelopes are signed), so LAN, relay and loopback are
interchangeable."""

from __future__ import annotations

from typing import Callable, Protocol

from nexus.mesh.messages import Envelope

Handler = Callable[[Envelope], None]


class Transport(Protocol):
    def send(self, envelope: Envelope) -> None: ...


class LoopbackHub:
    """In-process 'network' for tests and single-machine use.

    Envelopes are serialized to JSON and parsed back on delivery, exactly as a
    real wire would, so parsing and signature checks are exercised.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def connect(self, device_id: str, on_receive: Handler) -> LoopbackTransport:
        self._handlers[device_id] = on_receive
        return LoopbackTransport(self)

    def deliver(self, envelope: Envelope) -> None:
        handler = self._handlers.get(envelope.recipient)
        if handler is not None:
            handler(Envelope.from_json(envelope.to_json()))


class LoopbackTransport:
    def __init__(self, hub: LoopbackHub) -> None:
        self._hub = hub

    def send(self, envelope: Envelope) -> None:
        self._hub.deliver(envelope)
