# NEXUS OS architecture

**NEXUS is a computer operating system with an AI-native control layer - not an
AI chatbot that happens to control a computer.** Keyboard, mouse, windows,
files, browser, settings and games always work normally; voice and AI are an
additional way to drive the same system, never a replacement for it.

## End state

```
PC firmware (UEFI) -> bootloader -> Linux kernel
  -> NEXUS system services   (core daemon, AI services, memory, mesh)  - unprivileged
  -> NEXUS desktop           (Qt/QML Wayland compositor = the NEXUS shell)
  -> apps, browser, terminal, games (Steam/Proton)
```

A Linux distribution with its own shell and AI services, the model of
SteamOS/ChromeOS/Android. Writing a new kernel is not a goal.

## One NEXUS, many devices

```
                 NEXUS identity (nexus_id)  - one per user
                           |
      +--------------------+--------------------+
      |                    |                    |
 Desktop PC (core)      Laptop (core)       Phone (client)
 full models, games     same software,      voice/chat, status,
 heavy automation       lighter models      confirmations, remote
      |                    |                    |
      +----- signed envelopes over LAN / optional relay -----+
                           |
     every device: own key pair, device id derived from it
     every request: judged by the receiving device's Action Kernel
```

- **Local first.** A single PC works fully offline. Devices talk directly on the
  LAN. A relay/cloud is optional, only forwards signed (later E2E-encrypted)
  envelopes, and is never required.
- **Device roles** (`nexus/system/device.py`): DESKTOP and LAPTOP run the same
  core and adapt through the resource manager; MOBILE is a client/control
  surface that respects Android/iOS sandboxing and never runs desktop tools.
- **Shared memory** is designed in (`nexus/ai/memory`): UUID records, per-record
  timestamps, origin device, tombstones, `changes_since()` - so devices can
  merge memory later without changing the AI core.
- **Pairing grants nothing by itself.** Requests from another device get
  stricter policy (see Security). Details: [mesh.md](mesh.md).

## Layers inside one NEXUS device

```
 front ends     console (text + push-to-talk voice) | HUD (later) | mesh gateway (other devices)
                    \          |            |            /
 conversation   ConversationEngine: fast path OR plan -> ack -> preview -> kernel -> verified result
                    |                              (docs/conversation.md)
 response path  FastPathRouter: time, hardware readings, memory -> tool directly, NO model loaded
                    |           everything else -> planner (language model)
 AI providers   HybridChat -> best of Local / LAN / Cloud that fits right now
                    |               ^
 resources      ResourceManager: states, admission, lazy load, idle unload, game mode
                    |
 core           ActionKernel: validate -> policy -> confirm -> execute -> verify -> audit
                    |                              (docs/action-kernel.md)
 tools          structured Tools (typed params, declared Risk, progress wording)
                    |
 system         OS services: info, device capabilities, apps (discovery/catalog/launch+verify),
                games (Steam/Epic/Rockstar providers, GameService, GameWatcher -> GAME_MODE)
                    |
 platform       WindowsPlatform (now) | LinuxPlatform (v0.9)
```

## Rules that do not bend

1. **No tool runs a shell command from AI text.** Missing capabilities can't be misused.
2. **The policy is deterministic code.** The AI only *proposes*; it never decides permission.
3. **Confirmation is a user-interface function.** Only a UI calls `kernel.resolve()`.
   Spoken/typed yes/no is parsed by fixed code, never by the model. A confirmation
   for a phone's request can be answered only by that phone or on the PC itself.
4. **NEXUS never claims what the kernel didn't verify.** With actions, the model's
   words are not spoken at all; results are worded from kernel outcomes.
5. **Fail closed.** Bad policy/config file or damaged audit log -> NEXUS doesn't start.
6. **Unprivileged core.** Refuses to run as Administrator.
7. **Low idle footprint.** Nothing heavy is loaded until needed; the game comes first.
8. **Platform seam.** Tools call `Platform`, never OS APIs directly.

## Acceptance criteria

The 12 criteria locked on 2026-09-28 (converse without tools, verified actions,
no unverified claims, unloading, game mode, no 8B for trivial requests, phone
can't bypass security, device vs user identity, local memory, Linux path,
mobile client, keyboard/mouse) are automated tests - see [acceptance.md](acceptance.md).

## Response paths

| Request | Path | Model loaded? |
|---|---|---|
| "what time is it?", "GPU ka temperature?", "kitni storage bachi hai?" | fast path -> tool | no |
| "remember that ...", "what do you remember about me?" | fast path -> memory tool | no |
| "open chrome", "GTA chala de", "open downloads", "go to youtube" | fast path -> launch_app / open_folder / open_url | no |
| "make a folder", "forget that", unknown app names | planner -> tool | yes (best that fits) |
| "yaar I'm bored", "what can you do?" | planner -> conversation | yes |
| complex reasoning | planner with the strongest option that fits | yes |

The fast path is high-precision (0 wrong on the evaluation set); anything it
isn't sure about goes to the planner. A stronger tier for complex requests
plugs into the same provider list when one exists (bigger local model or,
if you enable it, cloud).

## Resource management (`nexus/core/resources.py`)

States: `IDLE`, `LISTENING`, `THINKING`, `VISION`, `EXECUTING`, `SPEAKING`, and
`GAME_MODE` (which overrides the others while on).

The ResourceManager knows: total/available RAM, total/used/free VRAM, CPU %,
GPU %, battery (present, plugged in, %), network (a non-loopback interface is up -
checked locally, nothing is sent), NEXUS's own RAM, loaded models, and the
device's capabilities (type, CPU, RAM, GPU, VRAM, supported models).

- Models are **registered, not loaded**. First use loads them, only if the
  ResourceManager **admits** them: free RAM/VRAM minus a reserve kept for you
  (2 GB RAM, 1 GB VRAM by default, `config/nexus.toml`).
- If a model doesn't fit, **idle** NEXUS models are evicted (least recently used).
- Models **unload after idle timeout** (LLM: 300 s). Ollama's own keep-alive is
  set to the same value, so they unload even if NEXUS is closed.
- **Game mode** unloads every NEXUS model from the GPU immediately and refuses
  GPU models until it ends; the AI falls back to a CPU-only model.
- A model stays loaded permanently only if you set `keep_llm_loaded = true`
  (off by default); game mode still unloads it.
- Voice input (subsystem 5, [voice.md](voice.md)): the speech model loads when
  listening starts (LISTENING), in a worker process that is ended when the model
  unloads (60 s idle), so all of its memory really comes back. Game mode uses
  only the light model (base, 2 CPU threads). The model never uses the GPU
  unless configured.
- Voice output (`nexus/ai/voice/speaker.py`) loads its model only while
  SPEAKING and yields to games (text-only). Pipeline: (wake word, later) ->
  STT on demand -> fast path/planner -> kernel -> TTS on demand -> unload on idle.
- Idle cost measured: NEXUS process 33-40 MB RAM, 0 VRAM (38 MB with voice
  input on and the microphone check running).
- Voice input: whisper-small 391 MB RAM loaded (551 MB transcribing), ~1.2 s per
  phrase on the CPU; whisper-base 192/305 MB, ~0.6 s. Loaded and idle: 0 % CPU.

Provider choice per request (`nexus/ai/providers/selection.py`): best-first
list from `config/nexus.toml`; skip cloud unless enabled, skip LAN/cloud
without network, skip models this device can't run at all
(`DeviceCapabilities.supported_models`), skip heavy models while saving power
(laptop on battery), skip unavailable providers, skip anything the
ResourceManager won't admit right now.

## Models on this PC (measured 2026-09-28, RTX 3060 12 GB, 16 GB RAM)

| Model | Placement | VRAM | Host RAM | Cold load | Speed | Planner score* | Median latency |
|---|---|---|---|---|---|---|---|
| llama3.1:8b | GPU | 5.0 GB | 0.4 GB | 4.6-22 s** | 36 tok/s | 96-98% planned (54) / 89% production (64) | 0.6 s |
| llama3.2 (3B) | GPU | 2.4 GB | 0.7 GB | 3 s | 126 tok/s | 81% planned | 0.4 s |
| llama3.2 (3B) | CPU only | 0 | 2.4 GB | 3 s | 17 tok/s | 67% production (12 cases) | 3.0 s |

\* `scripts/eval_planner.py` on 54 of our own phrases (English, Hinglish, Hindi,
Punjabi, Haryanvi, small talk, unsafe requests). "planned" includes the
upcoming `launch_app`/`close_app` definitions; "production" uses today's tools
only. Run-to-run variance is 1-2 cases. Real-world accuracy will be lower;
grow `evals/planner_cases.json` with your real phrasing.
\** first load after boot is from disk; later loads come from the OS file cache.

Default: llama3.1:8b on GPU; llama3.2 on GPU when VRAM is tight; llama3.2 on
CPU in game mode. No other model has been downloaded. Any new model gets a
row here (size, VRAM, speed, score) before it is added to the config.

## Project layout

```
nexus-os/
  nexus/
    core/            kernel, resources, config, runtime + assistant wiring
    tools/           structured tools
    security/        permissions (policy), audit, identity (device keys)
    system/          platform/ (Windows now), info, device capabilities, audio/ (microphones)
    ai/
      providers/     ChatProvider interface, Ollama, HybridChat + selection
      planner/       language -> Plan (LLM, JSON-schema constrained)
      conversation/  turn types, engine, responder (honesty), history
      memory/        MemoryStore interface (SQLite impl = subsystem 3)
      voice/         speech-to-text (faster-whisper worker, recorder, listener), TTS interface
      vision/        VisionProvider interface (v0.5)
    mesh/            envelopes, device registry, gateway, transports
    console.py       text front end
  config/            policy.toml (security), nexus.toml (models, resources)
  evals/             planner test phrases
  scripts/           eval_planner.py
  docs/ tests/ models/ var/ (runtime data, git-ignored)
```

## v0.1 plan (Windows prototype)

| # | Subsystem | Status |
|---|---|---|
| 1 | Action Kernel - tools, policy, confirmation, audit | done |
| 1b | Architecture: conversation/response, resources, providers, identity, mesh | done |
| 2 | AI planner (text) with evaluation | done: 89-98% on 64 phrases, 0.6 s |
| 2b | Acceptance criteria as tests; fast path; capabilities; per-device permissions | done |
| 3 | Memory (SQLite, local-first scopes) - [memory.md](memory.md) | done |
| 4 | App & game discovery, verified launch/close, async game launch, aliases - [apps.md](apps.md) | done, FROZEN |
| 5 | Voice input: STT (push-to-talk) behind `STTProvider`, automatic microphones - [voice.md](voice.md) | done, real microphone + real speaker tested, FROZEN |
| 6 | TTS behind `TTSProvider` | |
| 7 | Wake word | |
| 8 | HUD (Qt/QML): orb, transcript, confirmation buttons | |

Later: LAN transport + pairing (mesh), phone client, automatic game-mode
detection, vision.

**v0.1 acceptance:** offline, by voice or text, "Nexus bhai GTA chala de" ->
"Haan bhai, ek second." -> "Launching GTA V..." -> GTA starts -> "GTA V is
ready." only after the process is verified; memory survives restarts; writes
outside safe folders ask first; protected locations are refused; every action
is audited; tests pass.
