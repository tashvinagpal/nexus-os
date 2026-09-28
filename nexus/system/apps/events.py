"""Events from background app/game work (e.g. "GTA V is running" after an asynchronous launch)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppEvent:
    kind: str                 # "launch"
    app: str                  # display name
    ok: bool
    verified: bool | None
    message: str              # written by code from the verification result, never by a model
    origin: str | None = None  # the paired device that asked (None = this computer)


class EventHub:
    def __init__(self) -> None:
        self._listeners: list[Callable[[AppEvent], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, listener: Callable[[AppEvent], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def emit(self, event: AppEvent) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # a broken display must not break verification
                log.exception("app event listener failed")
