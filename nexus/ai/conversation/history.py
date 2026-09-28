"""Conversation history. In-memory for v0.1; the interface allows a shared,
synced store later so a conversation started on the phone continues on the PC."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol, Sequence

from nexus.ai.providers.base import ChatMessage


class HistoryStore(Protocol):
    def recent(self, conversation_id: str, limit: int = 8) -> Sequence[ChatMessage]: ...
    def append(self, conversation_id: str, message: ChatMessage) -> None: ...


class InMemoryHistory:
    def __init__(self, max_messages: int = 16) -> None:
        self._conversations: dict[str, deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=max_messages))

    def recent(self, conversation_id: str, limit: int = 8) -> Sequence[ChatMessage]:
        return list(self._conversations[conversation_id])[-limit:]

    def append(self, conversation_id: str, message: ChatMessage) -> None:
        self._conversations[conversation_id].append(message)
