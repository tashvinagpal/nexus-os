# NEXUS acceptance criteria

Locked 2026-09-28. Every later subsystem must keep these passing.

```powershell
.\.venv\Scripts\python.exe -m pytest                 # everything except live (simulated hardware)
.\.venv\Scripts\python.exe -m pytest -m live         # real GPU + Ollama: loads/unloads llama3.1:8b (not while gaming)
```

| # | Criterion | Default suite (`tests/test_acceptance.py`) | Live (`tests/test_acceptance_live.py`) |
|---|---|---|---|
| 1 | Converse without executing a tool | `test_A01_*` | `test_L02_*` (real 8B) |
| 2 | Execute an action and verify it | `test_A02_*` (folder checked independently) | `test_L02_*` |
| 3 | Never claim an unverified success | `test_A03_*` - 8 failure scenarios; any "done/ready/opened" must follow a DONE + verified action | |
| 4 | Large models unload when not needed | `test_A04_*` - idle unload, background housekeeping, keep-loaded only by explicit setting | `test_L04_*` - freed 5152 MB; Ollama timer unloads even if NEXUS is gone |
| 5 | Game mode frees significant VRAM | `test_A05_*` | `test_L05_*` - 6627 -> 1475 MB, AI continued on CPU |
| 6 | Simple requests don't load the 8B model | `test_A06_*` - 6 request types, zero model calls | `test_L06_*` - real stack, no model loaded |
| 7 | A paired phone cannot bypass PC security | `test_A07_*` - protected paths, confirmations, device-bound approval, per-device rules, private memory, strangers | |
| 8 | Device identity separate from user identity | `test_A08_*` | |
| 9 | Memory can remain local | `test_A09_*` - default scope local, nothing offered for sync, no network imports in memory code | |
| 10 | Architecture can run on Linux | `test_A10_*` - no Windows API outside `platform/windows.py`, runtime platform choice, core runs on a non-Windows platform | |
| 11 | Architecture supports a mobile client | `test_A11_*` - phone client via MeshClient, capability exchange, rejects impostor replies | |
| 12 | Usable with keyboard and mouse | `test_A12_*` - no input hooks/automation/autostart in code; runs without AI; exits cleanly | |

Resource requirements (same file, `test_R*`):

| | Requirement | Test |
|---|---|---|
| R1 | Resource manager knows RAM, VRAM, CPU, GPU, loaded models, device type, battery, network | `test_R1_*` |
| R2 | A model loads only if it fits the current budget | `test_R2_*` |
| R3 | Idle NEXUS is light (< 150 MB, no model loaded) | `test_R3_*` - real process: 33-40 MB measured |
| R4 | Voice output loads only while speaking, unloads when idle, yields to games | `test_R4_*` |
| R5 | Laptop adapts: battery -> lighter model; weak laptop -> CPU model | `test_R5_*` |
| R6 | Device capabilities exposed (cpu, ram, gpu, vram, battery, network, type, supported models) | `test_R6_*` |
| R7 | Local-first: works with cloud off and no network | `test_R7_*` |

## Subsystem 4 (apps & games) - `tests/test_acceptance_apps.py`, live L07-L09

| # | Criterion | Test |
|---|---|---|
| 1 | "Open Chrome" = the installed app, not a website | `test_S01_*`, plus the model-chose-a-website guard |
| 2 | App and game aliases | `test_S02_*` (12 phrasings incl. Hinglish) |
| 3 | Unknown names never run as shell commands | `test_S03_*` |
| 4 | Arbitrary .exe/.bat/.ps1/paths rejected | `test_S04_*` |
| 5 | Already-running apps detected (focused, not relaunched) | `test_S05_*`, live L08 |
| 6 | Duplicate matches ask | `test_S06_*` |
| 7 | Success requires process/window verification | `test_S07_*`, live L08 |
| 8 | Failed launches never produce a success response | `test_S08_*` |
| 9 | GTA V discovered if installed | `test_S09_*`, live L07 (real Steam install) |
| 10 | Steam/Epic/Rockstar provider interfaces through mocks | `test_S10_*`, `tests/test_game_providers.py` |
| 11 | Game launch uses GAME_MODE; AI reloads lazily after | `test_S11_*`, live L11 |
| 12 | High-confidence launches use the fast path | `test_S12_*`, live L09 |
| 13 | Launching doesn't load the 8B model | `test_S13_*`, live L09 |

### Subsystem 4 hardening

| Requirement | Tests |
|---|---|
| `close_app`: successful close, verified | `test_close_app.py::test_successful_close_is_verified`, live L10 |
| already-closed app | `test_already_closed`, live L10 |
| protected process refusal (Windows, other users/system, NEXUS itself) - also by force | `test_protected_processes_are_refused*`, `test_real_guard_*` |
| unknown app refusal | `test_unknown_apps_are_not_closed` |
| verification failure reported honestly, no automatic escalation | `test_app_that_refuses_to_close_*`, `test_no_window_to_ask` |
| no arbitrary PID / path termination | `test_no_arbitrary_pid_or_path_termination`, `test_a_pid_argument_is_rejected_by_the_kernel`, live L10 |
| no shell execution | `test_close_never_runs_a_shell` |
| force close only after confirmation, only matched processes | `test_force_close_needs_confirmation_*` |
| async game launch doesn't block; "is running" only after verification; timeout message | `test_console_stays_responsive_while_a_game_starts` (real threads), `test_game_launch_returns_before_verification`, `test_game_failure_is_reported_*`, live L11 |
| game mode kept through async launch, released on failure, ended on close, no 8B reload | `test_game_providers.py`, `test_closing_a_game_leaves_game_mode_*`, live L11 |
| aliases: creation, resolution, ambiguity, deletion, restart persistence, malicious/conflicting | `test_app_aliases.py` |
| SUCCESS / FAILURE / REFUSAL / CLARIFICATION / UNSUPPORTED wording | `test_response_semantics.py` |

## Final results (Subsystem 4 frozen, 2026-09-28)

| | |
|---|---|
| normal suite | 618 passed, 0 failed |
| app/game acceptance + hardening subset | 268 passed |
| live hardware suite | 11 passed, 0 skipped, 0 failed |
| idle NEXUS | 35 MB RAM, 0 VRAM, no model loaded |
| llama3.1:8b | +5152 MB VRAM while loaded; idle unload frees 5152 MB |
| game mode | 6610 -> 1458 MB VRAM (L05); async lifecycle 6609 -> 1459 MB, off after close, 8B not reloaded (L11) |
| async launch | `launch_game` returns in ~0.1 s; console turn 0.003 s |
| planner, 78 phrases | system 97.4%, model 96.2%, fast path 37/78 (0 wrong) |
| discovery | 153 entries in ~1.6 s; shortcuts 146/150; names 42/43 (+1 correct clarification); games 3/3 |
