"""Memory tools: remember, list, forget - all through the Action Kernel and audit log."""

from __future__ import annotations

import re

from nexus.ai.memory import MemoryRecord, MemoryScope, MemoryStore
from nexus.tools.base import Param, Risk, Tool, ToolResult

_SECRET = re.compile(
    r"\b(?:password|passwd|pwd|passcode|pin|otp|cvv|cvc|api[\s_-]?key|secret[\s_-]?key|private[\s_-]?key|"
    r"token|seed phrase|recovery phrase)\b|\b(?:\d[ -]?){13,19}\b", re.IGNORECASE)


def looks_like_secret(text: str) -> bool:
    return bool(_SECRET.search(text))


def _label(record: MemoryRecord) -> str:
    tags = {MemoryScope.SYNCED: " [shared with your devices]", MemoryScope.PRIVATE: " [private]",
            MemoryScope.TEMPORARY: " [temporary]"}
    return record.text + tags.get(record.scope, "")


def make_memory_tools(store: MemoryStore) -> list[Tool]:
    def remember(text: str, scope: str, origin: str | None) -> ToolResult:
        if looks_like_secret(text):
            return ToolResult(False, "I can't store passwords, PINs, OTPs, keys or card numbers - a password "
                                     "manager is the right place for those.", refused=True)
        record = store.remember(text, scope=MemoryScope(scope), origin_device=origin)
        stored = store.get(record.id)
        ok = stored is not None and not stored.deleted and stored.text == record.text
        return ToolResult(ok, f"Got it, I'll remember: {_label(record)}" if ok else "Saving that memory failed.",
                          {"id": record.id, "scope": record.scope.value}, verified=ok)

    def list_memories(query: str | None, origin: str | None) -> ToolResult:
        # Private memories never leave this device - not even to your own phone.
        include_private = origin is None
        records = (store.recall(query, limit=20, include_private=include_private) if query
                   else store.all(include_private=include_private))
        # Your names for apps are memory too - listed openly, never hidden context.
        aliases = store.aliases() if not query and hasattr(store, "aliases") else []
        if not records and not aliases:
            return ToolResult(True, "I don't have any memories matching that." if query
                              else "I don't remember anything about you yet.", {"count": 0}, verified=True)
        lines = [f"{i}. {_label(r)}" for i, r in enumerate(records, 1)]
        lines += [f"- you call {app} your {alias}" for alias, app in aliases]
        return ToolResult(True, "Here's what I remember:\n" + "\n".join(lines),
                          {"count": len(records), "ids": [r.id for r in records], "aliases": len(aliases)},
                          verified=True)

    def forget_memory(query: str, origin: str | None) -> ToolResult:
        record = store.best_match(query, include_private=origin is None)
        if record is None:
            return ToolResult(False, f"I couldn't find a memory matching \"{query}\".")
        store.forget(record.id)
        gone = store.get(record.id)
        ok = gone is None or gone.deleted
        return ToolResult(ok, f"Forgotten: {record.text}" if ok else "Forgetting that failed.",
                          {"id": record.id}, verified=ok)

    def describe_forget(args) -> str:
        record = store.best_match(args["query"])
        return (f"Forget \"{record.text}\"" if record
                else f"Forget a memory matching \"{args['query']}\" (none matches right now)")

    scopes = tuple(s.value for s in MemoryScope)
    return [
        Tool(
            name="remember",
            description="Save a fact or preference the user asks NEXUS to remember (stored on this device).",
            risk=Risk.SAFE,
            params=(Param("text", "str", "the fact, in the user's words, e.g. 'Chrome is my preferred browser'"),
                    Param("scope", "str", "device (default), synced, private or temporary", required=False,
                          default=MemoryScope.DEVICE.value, choices=scopes)),
            run=remember, receives_origin=True,
            describe=lambda a: f"Remember \"{a['text']}\"",
        ),
        Tool(
            name="list_memories",
            description="List what NEXUS remembers about the user, optionally only memories matching a query.",
            risk=Risk.SAFE,
            params=(Param("query", "str", "optional words to filter by", required=False),),
            run=list_memories, receives_origin=True,
        ),
        Tool(
            name="forget_memory",
            description="Forget (delete) the saved memory that best matches the query.",
            risk=Risk.CONFIRM,  # shows exactly which memory before deleting it
            params=(Param("query", "str", "words identifying the memory to forget"),),
            run=forget_memory, receives_origin=True, describe=describe_forget,
        ),
    ]
