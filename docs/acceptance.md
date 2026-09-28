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

## Subsystem 5 (voice input / speech-to-text) - details in [voice.md](voice.md)

```powershell
.\.venv\Scripts\python.exe -m pytest -m voice   # real faster-whisper models, synthetic speech (no microphone needed)
.\.venv\Scripts\python.exe -m pytest -m mic     # the real microphone path (PENDING until one is connected)
```

| Requirement | Tests |
|---|---|
| STT provider abstraction (local now, cloud-ready by locality, never used unless enabled + online) | `test_stt.py::test_cloud_stt_is_never_used_unless_enabled_and_online` |
| mock microphone and mock STT; silence; cancellation; STT errors (timeout, crash, load failure, missing model) | `test_recorder.py`, `test_stt.py`, `test_voice_input.py` |
| never records forever (no speech, endless speech, endless noise) | `test_recorder.py::test_never_records_forever_*`, `test_silence_times_out_with_no_speech` |
| resource-aware model choice: memory, game mode (light model only, heavy unloaded), GPU failure -> CPU | `test_stt.py`, live `test_V05_*` |
| model loaded only when needed, unloaded when idle, process (and its memory) gone after unload | `test_stt.py::test_idle_stt_model_unloads_*`, live `test_V03_*`, `test_V06_*` |
| microphones: enumeration, automatic choice (communications default first), fallback, disconnect/reconnect, hotplug without restart, no microphone | `test_audio_devices.py`, `test_voice_input.py` |
| loopback (Stereo Mix), a phone's audio and virtual devices are never chosen automatically | `test_audio_devices.py::test_classification`, `test_default_that_is_virtual_or_loopback_is_not_used` |
| capture wrapper: WASAPI 16 kHz conversion, format fallbacks, access denied -> privacy message | `test_capture.py` |
| languages: restriction to en/hi/pa, Devanagari/Gurmukhi romanized, fast path still reached | `test_voice_text.py`, `test_voice_input.py`, `test_voice_conversation.py` |
| voice goes through the same engine and kernel; confirmations still required; a misheard "yes" never approves; a phone can't answer | `test_voice_conversation.py` |
| console: push-to-talk, errors shown and NEXUS keeps running | `test_voice_conversation.py::test_console_*`, `test_unexpected_voice_failure_*` |
| real speech end to end: spoken request -> kernel, no language model loaded | live `test_V04_*` |
| real microphone: opens and delivers audio; spoken request end to end | `test_voice_mic.py` M01, M02 (passed with the USB microphone, see results) |
| uncertain speech (confidence < 0.6 or a sound-alike guess) asks before any action that changes something | `test_voice_conversation.py` |
| spoken yes/no in five languages; a no always cancels; an unclear or guessed yes never approves ("haan ji" heard as "anji") | `test_responder.py`, `test_voice_conversation.py` |
| real-speaker transcripts -> requests; English never rewritten | `test_voice_normalize.py` |
| bounded decoding; unsure language -> both tried | `test_whisper_worker.py` |

### Results (2026-09-28) - Subsystem 5 FROZEN

| | |
|---|---|
| normal suite | 955 passed, 0 failed (618 before Subsystem 5 + 337 voice) |
| confirmation tests | 233 passed (`test_responder.py`, `test_voice_conversation.py`, `test_voice_normalize.py`) |
| Subsystem 4 regression | 268 passed (unchanged) + live L07-L11 in the live suite |
| live hardware suite | 11 passed |
| live voice suite | 14 passed (real whisper-small/base) |
| real microphone | USB microphone (AB13X), chosen automatically (Windows default for calls). `pytest -m mic`: audio 1.98 s in 2 s at 16 kHz. Unplug/re-plug noticed within the 5 s check; "no microphone" reported; listening again without a restart |
| real speaker, same person, 20 phrases | English 3/3, Hinglish 3/3, Hindi 2/3, Punjabi 2/3, Haryanvi 2/3, yes/no phrases 5/5 (before the fixes: 3/3, 0/3, 2/3, 1/3, 0/3) |
| spoken yes / no | yes recognised 6/6 (2 approve directly; 4 heard at confidence 0.55-0.57 ask again), no recognised 4/4 (all cancel) |
| wrong actions | 0. Unclear requests ask first; misses go to the planner, which also asks first when unclear |
| latency (end of speech -> text) | median 1.34 s, max 1.51 s (before: two outliers of 10.7 s and 23.3 s) |
| whisper-small | load 2.2 s (while you speak); 390-412 MB RAM loaded, ~550 MB peak; 0 VRAM; unload 0.07 s, process gone; 0 % CPU when idle |
| idle NEXUS with voice on | 38 MB RAM, 0.13 % of one CPU core, 0 VRAM |
| planner (78 phrases, after the dialect hint) | system 97.4 %, the same two misses as at the Subsystem 4 freeze |

## Subsystem 6 (voice output / text-to-speech) - details in [tts.md](tts.md)

```powershell
.\.venv\Scripts\python.exe -m pytest -m tts       # the real Kokoro model, nothing played
.\.venv\Scripts\python.exe -m pytest -m speaker   # plays through the real output (you hear it)
```

| Requirement | Tests |
|---|---|
| TTS provider abstraction: synthesize, speak, stop, available, capabilities, health check; replaceable engine | `test_kokoro.py`, `test_speaker.py` (fake provider) |
| local, offline provider; model loaded lazily, reused, unloaded when idle, reloaded; failure recovery | `test_speaker.py`, `test_kokoro.py`, live `test_T01_*`, `test_T08_*` |
| text prepared for speech (markdown, code, URLs, paths, emoji, units, long replies, chunking); the reply itself unchanged | `test_speech_text.py`, `test_speaker.py::test_markdown_*` |
| languages: English, Hindi, Hinglish (converted for the Hindi voice), Punjabi (approximate) | `test_speech_text.py`, `test_speaker.py::test_voice_follows_*`, live `test_T04_*`, `test_T09_*` |
| default output device; no device; device change; stalled device; microphones never used as outputs | `test_playback.py` |
| stop at once; speak -> stop -> speak again repeatedly; queued replies; a new interaction interrupts | `test_speaker.py`, `test_tts_integration.py::test_typing_interrupts_speech`, hardware `test_H02_*` |
| never talks over the microphone; quiet in game mode (model freed); thread-safe resource state | `test_speaker.py` |
| keyboard -> assistant -> speech; voice -> assistant -> speech; speech failure never breaks the assistant | `test_tts_integration.py` |
| speech can never run anything (no path to the kernel, tools or processes) | `test_speaker.py::test_speech_can_never_run_anything` |
| real output: plays, stops, works from the speech thread after the microphone used audio | `test_tts_hardware.py` H01-H03 |

### Results (2026-09-28) - Subsystem 6, awaiting review (not frozen)

| | |
|---|---|
| normal suite | 1078 passed, 0 failed (955 before Subsystem 6 + 123 new) |
| live TTS suite | 12 passed (real Kokoro fp32; Kokoro -> Whisper round trip word for word in English) |
| real output (`-m speaker`) | 3 passed on Speakers (AB13X USB Audio): spoke, stopped 20 ms after `stop()`, background-thread playback 4/4 |
| Subsystem 4 regression | 268 passed + live suite 11 passed |
| Subsystem 5 regression | 419 passed (default) + live voice 14 passed + real microphone M01 passed |
| first sound | 0.62-0.75 s with the voice loaded; 2.3 s for the very first reply (1.6-1.8 s load) |
| resources | voice process 440-580 MB RAM while loaded, 0 VRAM, 0 % CPU when idle, unload 0.07 s; idle NEXUS 38 MB |
