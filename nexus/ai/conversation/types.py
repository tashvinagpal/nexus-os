"""The stages of a NEXUS turn, as distinct types.

  1. conversational reply  Say(REPLY)              AI-written, checked for false "done" claims
  2. action request        ActionStep -> ToolRequest  AI-proposed, judged by the Action Kernel
  3. tool result           ActionReport             the kernel's Outcome, passed through verbatim
  4. verified result       ActionReport.verified    True/False/None from the tool's own check
  5. final response        Say(RESULT/ERROR/...)    composed by code from 3+4, never by the AI

Plus Say(ANNOUNCE) ("Launching GTA V...") - progress wording written by the
tool, spoken before a slow action runs - and ConfirmationRequest.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Union

from nexus.core.kernel import Status


@dataclass(frozen=True)
class UserInput:
    text: str
    interface: str = "console"     # console | voice | ui | mesh
    device: str | None = None      # None = this device; else the paired device it came from
    conversation_id: str = "default"
    confidence: float | None = None  # voice only: how sure speech recognition was (0-1)
    inferred: bool = False           # voice only: some words were reconstructed from similar sounds


@dataclass(frozen=True)
class ActionStep:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    say: str                               # stage 1 (may be empty for pure actions)
    actions: tuple[ActionStep, ...] = ()   # stage 2
    language: str = "en"                   # en | hi | hinglish | pa | other
    acknowledge: bool = True               # False for instant fast-path answers (no "one moment")


class SayKind(str, enum.Enum):
    REPLY = "reply"        # conversation
    ANNOUNCE = "announce"  # before a slow action runs
    RESULT = "result"      # outcome of an action
    CONFIRM = "confirm"    # asking the user to approve
    CLARIFY = "clarify"    # asking which of several matches the user meant
    REFUSAL = "refusal"    # NEXUS intentionally won't do it (security) - not a failure
    UNSUPPORTED = "unsupported"  # not possible yet - not a failure
    ERROR = "error"        # it was attempted and failed


@dataclass(frozen=True)
class Say:
    kind: SayKind
    text: str
    language: str = "en"


@dataclass(frozen=True)
class ActionReport:
    tool: str
    status: Status
    verified: bool | None
    message: str
    request_id: str


@dataclass(frozen=True)
class ConfirmationRequest:
    confirmation_id: str
    prompt: str


TurnEvent = Union[Say, ActionReport, ConfirmationRequest]


def event_to_dict(event: TurnEvent) -> dict[str, Any]:
    """Wire/UI form of an event (used by the mesh gateway and, later, the HUD)."""
    if isinstance(event, Say):
        return {"type": "say", "kind": event.kind.value, "text": event.text, "language": event.language}
    if isinstance(event, ActionReport):
        return {"type": "action", "tool": event.tool, "status": event.status.value,
                "verified": event.verified, "message": event.message, "request_id": event.request_id}
    return {"type": "confirmation", "confirmation_id": event.confirmation_id, "prompt": event.prompt}
