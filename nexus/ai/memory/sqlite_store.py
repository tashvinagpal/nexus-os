"""SQLite memory store: one local file, no server.

Recall is keyword matching (language-agnostic, works for English, Hinglish,
Devanagari, Gurmukhi). The store sits behind MemoryStore so vector retrieval
can replace it later without touching the AI core.
"""

from __future__ import annotations

import sqlite3
import threading
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

from nexus.ai.memory import MemoryRecord, MemoryScope

DEFAULT_TEMPORARY_TTL_S = 24 * 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    scope         TEXT NOT NULL CHECK (scope IN ('device', 'synced', 'private', 'temporary')),
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    expires_at    TEXT,
    origin_device TEXT,
    deleted       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
-- User-defined names for apps ("browser" -> Google Chrome). Device-local unless the user shares them.
CREATE TABLE IF NOT EXISTS app_aliases (
    alias         TEXT PRIMARY KEY,
    app_id        TEXT NOT NULL,
    app_name      TEXT NOT NULL,
    scope         TEXT NOT NULL DEFAULT 'device' CHECK (scope IN ('device', 'synced')),
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    origin_device TEXT
);
"""

# Words that carry no meaning for matching (English + Hinglish).
_STOP = {
    "a", "an", "the", "is", "are", "was", "am", "i", "me", "my", "mine", "you", "your", "that", "this", "it",
    "to", "of", "in", "on", "for", "and", "or", "do", "does", "what", "which", "who", "about", "remember",
    "please", "nexus", "mera", "meri", "mere", "mujhe", "main", "hai", "hain", "tha", "thi", "ka", "ki", "ke",
    "ko", "se", "mein", "me", "kya", "kaun", "kaunsa", "konsa", "yaad", "bhai", "yaar", "tum", "tumhe",
    "aap", "aapko", "hoon", "ho", "ye", "yeh", "wo", "woh", "forget", "bhool", "ja", "jao",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds")


def tokens(text: str) -> set[str]:
    # Keep letters, combining marks (Devanagari/Gurmukhi vowel signs) and digits together.
    norm = unicodedata.normalize("NFC", text.lower())
    cleaned = "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in norm)
    return {w for w in cleaned.split() if w not in _STOP}


def _similar(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def _score(query: set[str], record: MemoryRecord) -> float:
    if not query:
        return 0.0
    words = tokens(record.text)
    return sum(1 for q in query if any(_similar(q, w) for w in words)) / len(query)


class SQLiteMemoryStore:
    def __init__(self, path: Path | None, *, clock: Callable[[], datetime] = _now) -> None:
        """`path=None` keeps the database in RAM (tests, evaluation)."""
        self.path = path
        self._clock = clock
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA secure_delete = ON")  # forgotten text is overwritten on disk
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    # --- writes ------------------------------------------------------------------

    def remember(self, text: str, *, scope: MemoryScope = MemoryScope.DEVICE,
                 origin_device: str | None = None, ttl_s: float | None = None) -> MemoryRecord:
        text = " ".join(text.split())
        if not text:
            raise ValueError("nothing to remember")
        now = self._clock()
        expires = None
        if scope is MemoryScope.TEMPORARY:
            expires = _iso(now + timedelta(seconds=ttl_s or DEFAULT_TEMPORARY_TTL_S))
        with self._lock:
            existing = self._db.execute(
                "SELECT id FROM memories WHERE deleted = 0 AND lower(text) = lower(?)", (text,)).fetchone()
            if existing is not None:  # same fact again: refresh instead of duplicating
                self._db.execute("UPDATE memories SET scope = ?, updated_at = ?, expires_at = ? WHERE id = ?",
                                 (scope.value, _iso(now), expires, existing["id"]))
                record_id = existing["id"]
            else:
                record_id = str(uuid.uuid4())
                self._db.execute(
                    "INSERT INTO memories (id, text, scope, created_at, updated_at, expires_at, origin_device)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record_id, text, scope.value, _iso(now), _iso(now), expires, origin_device))
            self._db.commit()
        record = self.get(record_id)
        assert record is not None
        return record

    def forget(self, record_id: str) -> bool:
        """Remove a memory. Synced ones leave a tombstone (id + time only) so other devices delete it too."""
        with self._lock:
            row = self._db.execute("SELECT scope FROM memories WHERE id = ? AND deleted = 0",
                                   (record_id,)).fetchone()
            if row is None:
                return False
            if row["scope"] == MemoryScope.SYNCED.value:
                self._db.execute("UPDATE memories SET text = '', deleted = 1, updated_at = ? WHERE id = ?",
                                 (_iso(self._clock()), record_id))
            else:
                self._db.execute("DELETE FROM memories WHERE id = ?", (record_id,))
            self._db.commit()
        return True

    # --- reads -------------------------------------------------------------------

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self._db.execute("SELECT * FROM memories WHERE id = ?", (record_id,)).fetchone()
        return _record(row) if row else None

    def all(self, *, include_private: bool = True) -> list[MemoryRecord]:
        self._purge_expired()
        rows = self._db.execute("SELECT * FROM memories WHERE deleted = 0 ORDER BY updated_at DESC").fetchall()
        return [r for r in map(_record, rows) if include_private or r.scope is not MemoryScope.PRIVATE]

    def recall(self, query: str, *, limit: int = 5, include_private: bool = False) -> list[MemoryRecord]:
        q = tokens(query)
        scored = [(s, r) for r in self.all(include_private=include_private) if (s := _score(q, r)) > 0]
        scored.sort(key=lambda sr: (sr[0], sr[1].updated_at), reverse=True)
        return [r for _, r in scored[:limit]]

    def best_match(self, query: str, *, include_private: bool = True) -> MemoryRecord | None:
        found = self.recall(query, limit=1, include_private=include_private)
        return found[0] if found else None

    # --- sync (synced scope only) ------------------------------------------------

    def changes_since(self, cursor: str | None) -> tuple[list[MemoryRecord], str | None]:
        rows = self._db.execute(
            "SELECT * FROM memories WHERE scope = 'synced' AND updated_at > ? ORDER BY updated_at",
            (cursor or "",)).fetchall()
        records = [_record(r) for r in rows]
        return records, (records[-1].updated_at if records else cursor)

    def apply_changes(self, records: Iterable[MemoryRecord]) -> int:
        applied = 0
        with self._lock:
            for rec in records:
                if rec.scope is not MemoryScope.SYNCED:
                    continue  # only synced memories may arrive from other devices
                row = self._db.execute("SELECT updated_at FROM memories WHERE id = ?", (rec.id,)).fetchone()
                if row is not None and row["updated_at"] >= rec.updated_at:
                    continue
                self._db.execute(
                    "INSERT OR REPLACE INTO memories"
                    " (id, text, scope, created_at, updated_at, expires_at, origin_device, deleted)"
                    " VALUES (?, ?, 'synced', ?, ?, NULL, ?, ?)",
                    (rec.id, "" if rec.deleted else rec.text, rec.created_at, rec.updated_at,
                     rec.origin_device, int(rec.deleted)))
                applied += 1
            self._db.commit()
        return applied

    # --- app aliases (user-controlled, visible in /aliases and "what do you remember") ----

    def get_alias(self, alias: str) -> tuple[str, str] | None:
        row = self._db.execute("SELECT app_id, app_name FROM app_aliases WHERE alias = ?", (alias,)).fetchone()
        return (row["app_id"], row["app_name"]) if row else None

    def set_alias(self, alias: str, app_id: str, app_name: str, *, origin_device: str | None = None) -> str | None:
        """Point `alias` at an app. Returns the app it pointed at before, if any."""
        now = _iso(self._clock())
        with self._lock:
            old = self._db.execute("SELECT app_name FROM app_aliases WHERE alias = ?", (alias,)).fetchone()
            self._db.execute(
                "INSERT INTO app_aliases (alias, app_id, app_name, created_at, updated_at, origin_device)"
                " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(alias) DO UPDATE SET app_id = excluded.app_id,"
                " app_name = excluded.app_name, updated_at = excluded.updated_at,"
                " origin_device = excluded.origin_device",
                (alias, app_id, app_name, now, now, origin_device))
            self._db.commit()
        return old["app_name"] if old else None

    def remove_alias(self, alias: str) -> bool:
        with self._lock:
            removed = self._db.execute("DELETE FROM app_aliases WHERE alias = ?", (alias,)).rowcount > 0
            self._db.commit()
        return removed

    def aliases(self) -> list[tuple[str, str]]:
        rows = self._db.execute("SELECT alias, app_name FROM app_aliases ORDER BY alias").fetchall()
        return [(r["alias"], r["app_name"]) for r in rows]

    def close(self) -> None:
        self._db.close()

    def _purge_expired(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                             (_iso(self._clock()),))
            self._db.commit()


def _record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(id=row["id"], text=row["text"], created_at=row["created_at"], updated_at=row["updated_at"],
                        origin_device=row["origin_device"], deleted=bool(row["deleted"]),
                        scope=MemoryScope(row["scope"]), expires_at=row["expires_at"])
