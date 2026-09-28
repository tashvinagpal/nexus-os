"""User-controlled, local-first memory.

Every device keeps its own memory database. A memory's scope decides where it
may ever go:

  device     stays on this device (the default)
  synced     may be shared with your paired devices, when sync exists (encrypted, device to device)
  private    never leaves this device and is never added to AI prompts automatically
  temporary  this device only, and expires

Records are shaped for sync from day one (UUIDs, timestamps, origin device,
tombstones) but only `synced` records are ever offered for sync. Nothing
requires a server.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence


class MemoryScope(str, enum.Enum):
    DEVICE = "device"
    SYNCED = "synced"
    PRIVATE = "private"
    TEMPORARY = "temporary"


@dataclass(frozen=True)
class MemoryRecord:
    id: str                    # UUID, unique across all devices
    text: str                  # "Chrome is my preferred browser" (erased when forgotten)
    created_at: str            # ISO 8601 UTC
    updated_at: str
    origin_device: str | None  # device that asked NEXUS to remember it (None = this device)
    deleted: bool = False      # tombstone for synced memories: "forget that" syncs as a deletion
    scope: MemoryScope = MemoryScope.DEVICE
    expires_at: str | None = None


class MemoryStore(Protocol):
    def remember(self, text: str, *, scope: MemoryScope = MemoryScope.DEVICE,
                 origin_device: str | None = None, ttl_s: float | None = None) -> MemoryRecord: ...
    def recall(self, query: str, *, limit: int = 5, include_private: bool = False) -> Sequence[MemoryRecord]: ...
    def all(self, *, include_private: bool = True) -> Sequence[MemoryRecord]: ...
    def best_match(self, query: str, *, include_private: bool = True) -> MemoryRecord | None: ...
    def forget(self, record_id: str) -> bool: ...
    def changes_since(self, cursor: str | None) -> tuple[Sequence[MemoryRecord], str | None]:
        """Synced-scope records (incl. tombstones) changed after `cursor`, and a new cursor."""
        ...
    def apply_changes(self, records: Iterable[MemoryRecord]) -> int:
        """Merge synced records from another of your devices (last writer wins)."""
        ...
