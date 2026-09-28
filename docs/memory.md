# Subsystem 3: Memory

Code: `nexus/ai/memory/` (interface + SQLite store), `nexus/tools/memory.py`.
Data: `var/memory/memory.db` on this device - protected from all tools.

## Local-first scopes

| Scope | Where it can go | In AI prompts | Use |
|---|---|---|---|
| `device` (default) | this device only | when relevant | normal facts and preferences |
| `synced` | your paired devices, when sync exists (encrypted, device to device) | when relevant | things every device should know |
| `private` | never leaves this device - not even to your own phone | never automatically | sensitive facts |
| `temporary` | this device only, expires (default 24 h) | when relevant | "for today, I'm working from home" |

There is no server. `changes_since()` only ever returns `synced` records;
`apply_changes()` only ever accepts `synced` records (last writer wins).

## Using it

| You say | Path | What happens |
|---|---|---|
| "Remember that Chrome is my preferred browser" / "yaad rakhna ki ..." | fast path, no model | `remember` -> "Got it, I'll remember: ..." (verified by reading it back) |
| "remember privately that ..." / "for today ..." | fast path | private / temporary scope |
| "What do you remember about me?" / "mere baare mein kya yaad hai?" | fast path | `list_memories` |
| "Which browser do I prefer?" | model | relevant non-private memories are added to the prompt -> "You prefer Chrome." |
| "Forget my browser preference" | model -> `forget_memory` | **asks first, showing exactly which memory**; then deletes it |
| `/memory`, `/forget <id>` | keyboard, no AI | list with ids; delete one exact record |

## Safety and privacy

- Passwords, PINs, OTPs, keys, tokens and card numbers are **refused**, never stored.
- Forgetting really removes the text: rows are deleted (`secure_delete` overwrites
  them on disk); synced memories leave only an empty tombstone so other devices delete it too.
- Memory tools run through the Action Kernel (audited). From another device:
  confirmation by default, and private memories are never returned.
- Which device wrote a memory (`origin_device`) is set by the kernel from the
  authenticated request - it can't be supplied by the AI.

## Known limits (v0.1)

- Recall is keyword matching (works across English/Hinglish/Devanagari/Gurmukhi),
  not semantic; the `MemoryStore` interface allows vector retrieval later.
- Conflicting facts ("Chrome" then "Firefox") are both kept; newest is listed first.
- The database file is not encrypted at rest yet (OS-backed encryption comes with sync).
- Slang like "wo browser wali baat bhool ja" is not always understood as "forget".
