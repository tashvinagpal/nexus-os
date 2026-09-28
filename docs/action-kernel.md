# Subsystem 1: Action Kernel

Code: `nexus/core/kernel.py`, `nexus/tools/`, `nexus/security/`.

## Request flow

```
ToolRequest(tool, args, source)
  -> registry lookup         unknown tool            -> INVALID
  -> Tool.validate(args)     bad/unknown/missing arg -> INVALID
  -> Policy.evaluate         BLOCK                   -> BLOCKED
                             CONFIRM                 -> NEEDS_CONFIRMATION (+ confirmation_id)
                                                          resolve(id, yes)  -> re-evaluate -> execute
                                                          resolve(id, no)   -> DECLINED
                                                          after 120 s       -> EXPIRED
                             ALLOW                   -> execute
  -> Tool.run                exception               -> FAILED (NEXUS keeps running)
  -> verification            ok but verified=False   -> FAILED
  -> audit + Outcome         DONE / FAILED
```

### Additions for conversation and cross-device (no behavior change for local requests)

- `ToolRequest.device` - `None` for this computer, else the paired device id.
  Passed to the policy and recorded in the audit log.
- `Policy.evaluate(..., device=...)` - requests from another device need
  confirmation unless the tool is in `[remote] allow_without_confirmation`
  (default: `get_system_info`); `[remote] blocked` tools are refused remotely.
- `resolve(id, approved, device=...)` - a remote request's confirmation can be
  answered by the same device or locally; a third device gets INVALID (audited
  as `confirmation_wrong_device`) and the confirmation stays pending.
- `preview(request)` - validate + policy only: no execution, no audit entry.
  Used so NEXUS doesn't announce "Launching X..." for something that won't run.
- `Tool.progress` - tool-written progress wording spoken before slow actions.

## Policy rules (in order)

1. Tool listed in `[tools] blocked` -> BLOCK
2. Tool risk HIGH -> BLOCK (until strong authentication exists)
3. Risk CONFIRM, or tool in `always_confirm` -> CONFIRM
4. URL params: only `http`/`https` (also enforced during validation)
5. Path params marked `writes=True`:
   - UNC/device path (`\\server\...`, `\\?\...`) -> BLOCK. These are rejected
     *before* resolving, because resolving a UNC path makes Windows send your
     login hash to that server.
   - resolved through `..`, symlinks and junctions; re-resolved at approval time
   - drive root -> BLOCK
   - inside a protected folder -> BLOCK
   - outside every safe root -> CONFIRM

Built-in protected folders: `%SystemRoot%`, Program Files (both),
`%ProgramData%`, `%USERPROFILE%\AppData`, `Recovery`, `System Volume
Information`, `$Recycle.Bin`, `$WinREAgent`, `Boot`, `Users\Default`, and
NEXUS's own project folder, data folder and config folder.

## Adding a tool

```python
Tool(
    name="set_volume",
    description="Set the master volume (0-100).",
    risk=Risk.SAFE,
    params=(Param("level", "int", "volume percent"),),
    run=set_volume,                 # returns ToolResult(ok, message, data, verified)
    describe=lambda a: f"Set volume to {a['level']}%",
)
```

Register it in `nexus/tools/__init__.py:default_tools`, and add tests for
success, bad input and verification. A tool must:

- do one thing, with typed parameters (no free-form command strings)
- mark any path it modifies with `writes=True`
- set `verified=True` only after checking the effect happened; use `None`
  when it can't be checked
- call OS-specific code through `Platform`

## Audit log

`var/audit/audit.jsonl`, one JSON object per line. Each entry has `prev`
(the previous entry's hash) and `hash` (SHA-256 of itself). `audit verify` in
the console, or `AuditLog.verify()`, recomputes the chain and reports the
first edited or missing line. The chain detects edits. It does not stop
someone with file access from deleting the whole log; off-machine backup of
the log is a v0.8 item.

## Known limits (v0.1)

- A small check-then-act window remains between the policy check and the
  filesystem call. It is narrowed by re-checking at approval, and closing it
  fully is a sandbox (v0.8) job.
- `open_url` can't verify the browser actually showed the page, so it reports
  `verified=None`.
- The audit log is read in full by `audit`; rotation is added when it grows.
- The Administrator refusal is implemented, but no test runs elevated.
