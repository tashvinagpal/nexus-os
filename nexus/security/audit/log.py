"""Append-only, hash-chained audit log (JSON Lines).

Every entry stores the SHA-256 of the previous entry, so editing or removing
a line in the middle breaks the chain and verify() reports where.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(entry: dict[str, Any]) -> str:
    body = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev = self._last_hash()

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                     "event": event, **fields, "prev": self._prev}
            entry = json.loads(json.dumps(entry, default=str))  # normalise Paths etc. to JSON types
            entry["hash"] = _digest(entry)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            self._prev = entry["hash"]
            return entry

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            lines = f.read().splitlines()
        return [json.loads(line) for line in lines[-n:] if line.strip()]

    def verify(self) -> tuple[bool, str]:
        """Recompute the hash chain. Returns (ok, explanation)."""
        if not self.path.exists():
            return True, "audit log is empty"
        prev = GENESIS
        count = 0
        with self.path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    stored = entry.pop("hash")
                except (json.JSONDecodeError, KeyError, AttributeError):
                    return False, f"line {lineno} is not a valid audit entry"
                if entry.get("prev") != prev:
                    return False, f"chain broken at line {lineno} (an entry before it was removed or changed)"
                if _digest(entry) != stored:
                    return False, f"line {lineno} was modified after it was written"
                prev = stored
                count += 1
        return True, f"{count} entries, hash chain intact"

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        with self.path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            last = f.read().rstrip(b"\r\n").rsplit(b"\n", 1)[-1]
        try:
            return json.loads(last)["hash"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Don't silently start a fresh chain on a damaged log - that would hide tampering.
            raise RuntimeError(f"audit log {self.path} ends with a damaged entry; "
                               "inspect it before starting NEXUS") from None
