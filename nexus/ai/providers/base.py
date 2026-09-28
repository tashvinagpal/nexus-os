"""Provider interfaces shared by every AI capability (chat/LLM, STT, TTS, vision).

A provider says *where* it runs (Locality) so NEXUS can prefer local, fall
back to another NEXUS device on the LAN, and use the cloud only if allowed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class Locality(str, enum.Enum):
    LOCAL = "local"  # on this device
    LAN = "lan"      # another of your NEXUS devices on the local network
    CLOUD = "cloud"  # a remote service; off unless you enable it


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    locality: Locality
    model: str = ""


class ProviderUnavailable(RuntimeError):
    """The provider can't serve requests right now (not running, model missing, no memory)."""


@dataclass(frozen=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str


@dataclass(frozen=True)
class ChatReply:
    text: str
    model: str
    seconds: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)


class ChatProvider(Protocol):
    """The AIProvider interface for language models."""

    info: ProviderInfo

    def available(self) -> bool: ...

    def chat(self, messages: Sequence[ChatMessage], *, schema: dict[str, Any] | None = None,
             temperature: float = 0.2) -> ChatReply:
        """One non-streaming completion. `schema` asks for JSON matching that JSON Schema."""
        ...
