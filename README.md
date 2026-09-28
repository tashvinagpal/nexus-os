# NEXUS OS

A computer operating system with an AI-native control layer. Built in stages:
a normal Windows application first (v0.1), then a Linux-based environment
(v0.9), then a bootable OS (v1.0). See [docs/architecture.md](docs/architecture.md).

## Status: v0.1 - subsystems 1-4 done (Subsystem 4 frozen), acceptance criteria automated

- **Apps & games** - discovers what's installed (Start menu, Desktop, Store,
  Steam, Epic, Rockstar); "open Chrome" / "GTA chala de" launch the real program
  and NEXUS only says it's open/running after seeing its process; games start in
  the background and enter game mode first; "close Chrome" closes gracefully and
  verifies (force-close only with confirmation, never protected processes or raw
  PIDs); your own names ("Chrome is my browser"); shells, admin tools and loose
  .exe files are refused - and refusals are worded as refusals, not failures. ([docs/apps.md](docs/apps.md))

- **Action Kernel** - typed tools, deterministic security policy, confirmations,
  hash-chained audit log. No shell tool exists. ([docs/action-kernel.md](docs/action-kernel.md))
- **Conversation** - talks back in English/Hindi/Hinglish/Punjabi; never claims a
  result the kernel didn't verify. ([docs/conversation.md](docs/conversation.md))
- **Response paths** - trivial requests (time, hardware readings, memory) are
  answered with **no model loaded**; everything else goes to the local planner
  (llama3.1:8b via Ollama; 97.4% system-level on 78 test phrases, ~0.6 s).
- **Resources** - models load only if they fit, unload when idle, game mode
  frees the GPU (measured 6.6 -> 1.5 GB VRAM); idle NEXUS uses ~35 MB RAM.
- **Memory** - local-first SQLite with device / synced / private / temporary
  scopes; refuses passwords; "forget" shows exactly what it deletes. ([docs/memory.md](docs/memory.md))
- **Cross-device foundations** - device identities, signed messages, per-device
  permissions, phone client interface. No networking yet. ([docs/mesh.md](docs/mesh.md))
- **Acceptance criteria** - all are tests: 618 normal + 11 live hardware tests pass. ([docs/acceptance.md](docs/acceptance.md))

## Setup (Windows, normal non-admin PowerShell)

```powershell
cd C:\Users\yugar\Downloads\OS\nexus-os
py -3.12 -m venv .venv          # or: python -m venv .venv  (3.10+ works)
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Needs [Ollama](https://ollama.com) with `llama3.1:8b` and `llama3.2` for the
AI; everything else (tools, memory, fast path) works without it.

## Run

```powershell
.\.venv\Scripts\python.exe -m nexus            # --no-ai for tools and memory only
```

```
you> GPU ka temperature kya hai?                  (instant, no model loaded)
you> Remember that Chrome is my preferred browser
you> Which browser do I prefer?                   -> "You prefer Chrome."
you> open chrome                                  -> "Google Chrome is open." (process verified)
you> launch GTA V                                 -> "Starting ..." now, "... is running." once verified
you> Chrome is my browser                         -> then "open my browser"
you> /status   /apps games   /aliases   /memory   /forget <id>   /gamemode on|off   /unload   /tools   /audit verify
```

## Test and evaluate

```powershell
.\.venv\Scripts\python.exe -m pytest                  # default suite, simulated hardware
.\.venv\Scripts\python.exe -m pytest -m live          # real GPU + Ollama (not while gaming); briefly opens Character Map
.\.venv\Scripts\python.exe scripts\eval_planner.py llama3.1:8b@gpu --catalog production
.\.venv\Scripts\python.exe scripts\eval_apps.py       # discovery accuracy on this PC (read-only)
```

## Where things live

| Path | What |
|---|---|
| `nexus/` | all NEXUS code |
| `config/policy.toml` | security: protected/safe folders, blocked tools, remote and per-device rules |
| `config/nexus.toml` | AI models (best first, measured sizes), reserves, idle unload, keep-loaded |
| `evals/` | planner test phrases |
| `var/` | runtime data: audit log, memory.db, device identity, logs, eval results (git-ignored) |

Uninstall: delete this folder. NEXUS v0.1 writes nothing to the registry,
installs no services, and adds no startup entries.
