# NEXUS OS

A computer operating system with an AI-native control layer. Built in stages:
a normal Windows application first (v0.1), then a Linux-based environment
(v0.9), then a bootable OS (v1.0). See [docs/architecture.md](docs/architecture.md).

## Status: v0.1 - subsystems 1-7 done (4, 5 and 6 frozen; 7 awaiting a real-microphone test), acceptance criteria automated

- **Apps & games** - discovers what's installed (Start menu, Desktop, Store,
  Steam, Epic, Rockstar); "open Chrome" / "GTA chala de" launch the real program
  and NEXUS only says it's open/running after seeing its process; games start in
  the background and enter game mode first; "close Chrome" closes gracefully and
  verifies (force-close only with confirmation, never protected processes or raw
  PIDs); your own names ("Chrome is my browser"); shells, admin tools and loose
  .exe files are refused - and refusals are worded as refusals, not failures. ([docs/apps.md](docs/apps.md))

- **Voice input** - push-to-talk (Enter or `/listen`), local speech-to-text
  (faster-whisper, CPU): the spoken request goes through exactly the same engine,
  policy and confirmations as typed text. Any microphone is picked up automatically,
  including one plugged in later. Stereo Mix and phone audio are never used.
  Unclear speech asks before acting, and a misheard "yes" never approves anything. Tested
  with a real speaker: English, Hinglish, Hindi, Punjabi, Haryanvi. ([docs/voice.md](docs/voice.md))

- **Voice output** - replies are spoken through the Windows default output (Kokoro,
  local, CPU): first sound ~0.7 s, markdown/links/code not read out, Hindi and Hinglish
  through a Hindi voice; typing or talking interrupts it, quiet while you game, text
  always shown. Speech can never run anything. ([docs/tts.md](docs/tts.md))

- **Wake word** - say "Nexus, GTA chala de" with no key press. An always-on listener
  (in memory only, nothing saved or sent) wakes on "Nexus" and hears the rest of the
  sentence; "Nexus" on its own gets a chime and waits for the request; NEXUS's
  questions can be answered without "Nexus". Two detectors: a light Whisper check on
  bursts of speech (works now) and NEXUS's own trained "Nexus" model (openWakeWord
  format, ~2% of one CPU core), trained locally from Kokoro voices and your own
  recordings. ([docs/wake.md](docs/wake.md))

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
- **Acceptance criteria** - all are tests: 1078 normal + 11 live hardware + 14 live voice + 12 live speech tests pass. ([docs/acceptance.md](docs/acceptance.md))

## Setup (Windows, normal non-admin PowerShell)

```powershell
cd C:\Users\yugar\Downloads\OS\nexus-os
py -3.12 -m venv .venv          # or: python -m venv .venv  (3.10+ works)
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
# voice input (optional, ~275 MB of packages), then one or both speech models:
.\.venv\Scripts\python.exe -m pip install -e ".[voice]"
.\.venv\Scripts\python.exe scripts\get_stt_model.py small     # 463 MB, the default
.\.venv\Scripts\python.exe scripts\get_stt_model.py base      # 141 MB, used in game mode
# voice output (optional, ~10 MB of packages with [voice] installed) and its model:
.\.venv\Scripts\python.exe -m pip install -e ".[tts]"
.\.venv\Scripts\python.exe scripts\get_tts_model.py fp32      # 310 MB model + 26 MB voices
# wake word: works with the Whisper base model above; for the trained "Nexus" model (~20 min, CPU):
.\.venv\Scripts\python.exe -m pip install -e ".[tts,wake-train]"
.\.venv\Scripts\python.exe scripts\get_wake_model.py              # 2.4 MB shared feature networks
.\.venv\Scripts\python.exe scripts\train_wake_model.py            # -> models\wake\nexus.onnx
.\.venv\Scripts\python.exe scripts\record_wake_samples.py         # optional: your voice, then train with --samples
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
you>                                            (Enter on an empty line = push-to-talk; Ctrl+C cancels)
you> /mic                                        microphones found, which one is used and why
you> /tts   /tts off|on   /tts test   /stop       voice output: status, on/off, test sentence, stop speaking
     "Nexus, open chrome"                        wake word: no key needed
you> /wake   /wake off|on   /wake test            wake word: status, on/off, show detections without acting
you> /status   /apps games   /aliases   /memory   /forget <id>   /gamemode on|off   /unload   /tools   /audit verify
```

## Test and evaluate

```powershell
.\.venv\Scripts\python.exe -m pytest                  # default suite, simulated hardware
.\.venv\Scripts\python.exe -m pytest -m live          # real GPU + Ollama (not while gaming); briefly opens Character Map
.\.venv\Scripts\python.exe -m pytest -m voice         # real speech models on synthetic speech
.\.venv\Scripts\python.exe -m pytest -m mic           # the real microphone (skipped as PENDING without one)
.\.venv\Scripts\python.exe scripts\voice_check.py      # speak test phrases in each language (nothing is executed)
.\.venv\Scripts\python.exe -m pytest -m tts           # the real voice model (nothing played)
.\.venv\Scripts\python.exe -m pytest -m speaker       # plays through your speakers
.\.venv\Scripts\python.exe scripts\tts_check.py        # voice output report on your real output device
.\.venv\Scripts\python.exe -m pytest -m wake          # wake word with the real models (Kokoro speech, no mic)
.\.venv\Scripts\python.exe scripts\eval_planner.py llama3.1:8b@gpu --catalog production
.\.venv\Scripts\python.exe scripts\eval_apps.py       # discovery accuracy on this PC (read-only)
```

## Where things live

| Path | What |
|---|---|
| `nexus/` | all NEXUS code |
| `config/policy.toml` | security: protected/safe folders, blocked tools, remote and per-device rules |
| `config/nexus.toml` | AI models (best first, measured sizes), reserves, idle unload, keep-loaded, voice input, voice output |
| `evals/` | planner test phrases |
| `var/` | runtime data: audit log, memory.db, device identity, logs, eval results (git-ignored) |

Uninstall: delete this folder. NEXUS v0.1 writes nothing to the registry,
installs no services, and adds no startup entries.
