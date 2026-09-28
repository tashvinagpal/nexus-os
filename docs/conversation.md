# Conversation and responses

Code: `nexus/ai/conversation/`, `nexus/ai/planner/`.

## The stages of a turn (distinct types)

| # | Stage | Type | Written by |
|---|---|---|---|
| 1 | Conversational reply | `Say(REPLY)` | the model (conversation-only turns), checked |
| 2 | Action request | `ActionStep` -> `ToolRequest` | the model proposes; the kernel judges |
| 3 | Tool result | `ActionReport` | the Action Kernel (`Outcome`) |
| 4 | Verified result | `ActionReport.verified` | the tool's own post-check |
| 5 | Final response | `Say(RESULT / ERROR)` | code, from 3 + 4 |

Plus `Say(ANNOUNCE)` (the tool's own progress line before a slow action),
`Say(CONFIRM)` + `ConfirmationRequest` (id for buttons).

Trivial requests skip the model entirely (`nexus/ai/router.py`):
"GPU ka temperature?" -> `get_system_info` -> the real reading, no acknowledgement, no model loaded.

```
"Nexus bhai GTA chala de"
  -> strip wake word -> fast path? no -> planner (THINKING)
  -> Say(REPLY)    "Haan bhai, ek second."       code-written, mirrors bhai/yaar
  -> preview: ALLOW?
  -> Say(ANNOUNCE) "Launching GTA V..."          from the tool, not the model
  -> kernel.submit (EXECUTING) -> ActionReport(done, verified=True)
  -> Say(RESULT)   "GTA V is ready."             from the verified outcome
```

## How NEXUS avoids false claims

Found by evaluating real models: without these, they said things like
"Formatting C drive, please wait...", "Windows Defender disabled.",
"GTA chala diya", "C drive mein 465 GB jagah hai" - none of which was true.

1. **Structural:** when a turn has actions, the model's own text is not spoken.
2. **No early announcement:** the progress line is only spoken if `preview()` says
   the action will run without confirmation.
3. **Conversation-only replies are checked** (`responder.claims_unverified`) for
   completion claims ("done", "is ready", "ho gaya" - `claims_success`) and
   activity claims ("formatting...", "please wait", fake "are you sure?"), in
   English, Hinglish, Hindi and Punjabi, with negation handled ("I can't shut
   down your PC" is fine). Flagged replies are replaced by an honest "I haven't
   done anything yet...". Acceptance test A03 enforces that any completion
   claim anywhere follows a DONE + verified action.
4. **Made-up numbers:** hardware figures (GB, %, C, GHz) in a conversation-only
   reply are replaced unless the conversation already holds verified figures of
   the same kind (so "is 58 C ok?" follow-ups still work).
5. **Prompt:** lists what's not available yet and forbids substituting another tool.

Items 3-4 are safety nets and can miss things; items 1-2 are guarantees.

## Confirmations by voice/text

`parse_confirmation()` is fixed code (never the model): yes = yes/haan/theek
hai/kar do/हाँ/ਹਾਂ...; no = no/nahi/mat kar/rehne de/नहीं/ਨਹੀਂ...; anything
else (contradictory, laughter, a new request) is *not* a yes: the pending
action is cancelled and the text is handled as a new request. Pending
confirmations are keyed by (conversation, device).

## Evaluation

`scripts/eval_planner.py` runs `evals/planner_cases.json` against local models
without executing anything. Results: [architecture.md](architecture.md#models-on-this-pc-measured-2026-09-28-rtx-3060-12-gb-16-gb-ram).

```powershell
.\.venv\Scripts\python.exe scripts\eval_planner.py llama3.1:8b@gpu --catalog production
```

Known weaknesses (llama3.1:8b): with no `launch_app` tool yet, 3 of 17 app
requests open a related website instead of declining; about 1 in 54 unrelated
requests triggers a harmless unrequested action (e.g. creating a folder). The
kernel's confirmation rules are why this can't become harmful.
