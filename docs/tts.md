# Subsystem 6: Voice output (text-to-speech)

Code: `nexus/ai/voice/speaker.py` (the TTS manager), `kokoro.py` + `kokoro_worker.py`
(the local voice), `speech_text.py` (what is said), `hinglish.py` (Hinglish for the
ear), `worker_process.py`, `nexus/system/audio/playback.py` (audio output),
`nexus/system/platform/windows.py` (output devices), `nexus/console.py` (`/tts`, `/stop`).

NEXUS speaks its replies through the Windows default audio output, locally and
offline. The reply is always shown as text too. Speech is an **output layer
only**: it receives text that was already decided and shown, and has no way to
run anything.

```
User -> voice (subsystem 5) or keyboard -> NEXUS core -> fast path / planner -> kernel
     -> reply (Say event) -> shown on screen
                          -> Speaker: prepare text -> sentences -> Kokoro -> default output
```

## Using it

```
you> what time is it?
nexus> It's 9:41 PM.                 <- shown, and spoken
you> /tts                            <- status: voice, output device, last timings
you> /tts off   /tts on              <- text only / speak again
you> /tts test [text]                <- say something, with timings
you> /stop                           <- stop speaking (typing anything or push-to-talk also interrupts)
```

## Architecture

| Layer | What it does |
|---|---|
| `TTSProvider` (`base.py`) | `synthesize(text, language, voice)`, `speak`, `stop`, `available`, `capabilities()`, `health_check()`. Replaceable: Kokoro today; Piper or a cloud voice could implement the same interface (a cloud one would only run with `allow_cloud` - none exists) |
| `Speaker` (`speaker.py`) - the TTS manager | prepares the text, splits it into sentences, synthesizes the next sentence while the current one plays, stops at once, queues replies, never talks while you are recording, stays quiet in game mode, loads the voice while NEXUS thinks, never raises into the console |
| `KokoroTTS` (`kokoro.py`) | Kokoro-82M in a worker process (like speech-to-text): unloading ends the process, so all of its memory comes back; a crash or hang can't take NEXUS down |
| `PortAudioSink` (`playback.py`) | plays PCM on the OS default output (WASAPI), non-blocking, `stop()` aborts at once, re-reads the device list when the default output changes (never while a microphone records), detects a device that stops playing |
| `Platform.audio_outputs()` | read-only list of playback endpoints (Windows Core Audio, eRender) - microphones are never listed |

Lifecycle: `idle -> (waiting while you speak into the microphone) -> synthesizing -> playing -> idle`,
reported by `/tts` and `Speaker.state`. The ResourceManager state is SPEAKING
while it talks. The resource manager's activity tracking was made thread-safe for
this: speech runs in the background while you type.

## The voice: Kokoro-82M (chosen after measuring)

| | Kokoro (chosen) | Piper (not installed) |
|---|---|---|
| quality | the most natural of the two | clearly synthetic |
| voices / languages | one model: 54 voices, English + Hindi (plus es, fr, it, ja, pt, zh - not used) | one model file per voice |
| install | `kokoro-onnx` + espeak-ng loader (~10 MB); uses the onnxruntime/numpy already installed for speech-to-text | similar |
| licence | Apache-2.0 (model), espeak-ng GPL-3.0 (phonemes, bundled by espeakng-loader) | MIT / GPL-3.0 |

Model variants, measured on this PC (i5-12400F, 16 GB, CPU):

| Kokoro model | disk | load | short sentence | 7.5 s paragraph | real-time factor | voice process RAM |
|---|---|---|---|---|---|---|
| **fp32 (default)**, 4 threads | 310 MB + 26 MB voices | 1.6-1.8 s | **0.49-0.59 s** | 2.3 s | 0.35 | ~440-580 MB |
| int8, 4 threads | 88 MB + 26 MB | 2.1 s | 1.6-1.8 s | 6.7 s | 1.09 | ~370 MB |

int8 is 3x slower here (its quantized operators run single-threaded: 2 threads and 6
threads measured the same), so fp32 is the default; int8 remains an option for a
low-RAM machine (`[tts] model = "int8"`). 6 threads were no faster than 4.

**GPU:** not used. The installed onnxruntime is the CPU build; GPU speech would need
onnxruntime-gpu plus CUDA/cuDNN (over 1 GB) for a voice that already runs at 0.35x
real time on the CPU - not justified. `placement = "gpu"` falls back to the CPU with
a logged note. VRAM used by speech: 0 (measured).

## Measured on this PC (2026-09-28, real output: Speakers (AB13X USB Audio))

| | |
|---|---|
| cold model load | 1.6-1.8 s (loaded in the background while NEXUS thinks) |
| first reply, voice not loaded | first sound after 2.3 s |
| first sound, voice loaded ("first audio" latency) | 0.62-0.75 s (median 0.72 s) for English, Hindi, Hinglish, Punjabi |
| playback start (synthesized -> sound) | ~30 ms |
| short sentence synthesis | 0.47-0.72 s |
| paragraph (130 characters) | first sound 0.72 s (only its first sentence is waited for), all done as it plays |
| stop | silent 15-20 ms after `stop()`; speaking again 0.75 s later |
| voice process RAM | 442-583 MB peak while speaking |
| CPU | ~160-380 % of one core while synthesizing (4 threads), 0 % loaded and idle |
| VRAM | unchanged (0 MB for speech) |
| unload | 0.07-0.09 s; the process is gone |
| idle NEXUS with voice output on | 38 MB RAM (unchanged), 0.26 % of one core |
| NEXUS after it has spoken | 43 MB (+5 MB: the audio library, once) |

Commands: `scripts/tts_check.py` (real output), `scripts/tts_check.py --silent`.

## Languages

| Language | How | Intelligibility check* |
|---|---|---|
| English | English voice (`af_heart`) | word for word (3/3 sentences in the live test; 92 % overall, the rest is "3:42 PM" written back as "3.42 p.m.") |
| Hindi (Devanagari) | Hindi voice (`hf_alpha`), espeak-ng Hindi phonemes | 55 % of words |
| Hinglish (Latin letters) | converted to Devanagari for the Hindi voice (`hinglish.py`); English words ("Chrome", "GTA", "second", installed app names) stay English | 38 % - vs 15 % when read by the English voice ("Ekdum badhiya bhai" -> "Ectombatiabye") |
| Punjabi | Gurmukhi converted to Devanagari, Hindi voice - **an approximation**: Kokoro has no Punjabi voice | 55 % |
| Haryanvi | no Haryanvi voice exists; replies are Hinglish/Hindi text and are read as such | - |

\* Measured by a round trip - Kokoro speaks, Whisper small (subsystem 5) transcribes, words
compared. That checks intelligibility to a speech recogniser, not naturalness to a
person: Whisper's own Hindi is weak, so the Indian-language numbers understate
what a listener hears. Listen yourself: `scripts/tts_check.py`.

- The reply's text language decides the voice: Devanagari -> Hindi voice even if tagged
  English. `[tts] romanized = "english"` reads Hinglish with the English voice instead.
- Nothing is translated. Only the script of romanized Hindi changes, for the ear.
- espeak-ng marks English words inside Hindi with "(en)...(hi)"; the worker phonemizes
  Hindi itself with those marks removed, so English words keep their English sounds.
- Capabilities (`KokoroTTS.capabilities()`): languages `en, hi`, 54 voices, 24 kHz, no SSML,
  no streaming inside a sentence (sentence-by-sentence instead).

## What is said (`speech_text.py`) - the reply itself is never changed

| Reply | Spoken |
|---|---|
| `Sure! **Opening Chrome now.**` + `https://example.com/long/path` | "Sure! Opening Chrome now. example.com." |
| a ```` ``` ```` code block | "The code is on the screen." |
| `[YouTube](https://youtube.com)` | "YouTube" |
| `C:\Users\me\Downloads\mods` | "mods" |
| `- RAM: 16 GB` / `45 C` / `1.8 GHz` | "RAM: 16 gigabytes" / "45 degrees Celsius" / "1.8 gigahertz" |
| `Wow!!! 🎉` / `e.g.` / `&` / `NEXUS` | "Wow!" / "for example" / "and" / "Nexus" |
| a table | its cells, comma-separated |
| more than ~600 characters | whole sentences up to that, then "The rest is on the screen." (in the reply's language) |

Chunks are sentence-sized (up to 220 characters; a very long sentence splits at commas,
then between words). The first sentence goes alone so the first sound comes sooner.

## Resources, game mode, interruptions

- Loaded lazily (first reply, or in the background while NEXUS thinks), reused for
  bursts, unloaded after 120 s idle (`idle_unload_seconds`), admitted by the
  ResourceManager against free RAM like every model. Never on the GPU.
- Game mode: `game_mode = "quiet"` (default) - replies are text only and the voice
  model is unloaded; `"speak"` keeps talking on the CPU.
- Typing anything, a command or push-to-talk stops speech first; so does `/stop`.
  Stopping never kills the voice engine (it would need 2 s to reload); a sentence
  being synthesized just finishes and is discarded.
- While a microphone records (LISTENING), NEXUS waits instead of talking (a background
  result such as "GTA V is running." is spoken after you finish).

## Output devices

The Windows default output (speakers, USB/Bluetooth headphones, a monitor's speakers)
at the moment of speaking; nothing is hard-coded. If you change the default in Windows,
the next sentence goes to the new one. Unplugged/disabled devices are ignored; with no
output at all NEXUS stays text-only ("no audio output device is connected").
Microphones are input (subsystem 5) and never appear here.

Found in testing: WASAPI playback from NEXUS's speech thread failed to start
(`AUDCLNT_E_UNSUPPORTED_FORMAT`) once the microphone code had used the audio system
on the main thread - 6/6 attempts. The speech thread now initialises COM for itself
(`WindowsPlatform.prepare_audio_thread`): 8/8 (test H03).

## Failures - all non-fatal, the text is always shown

| Situation | Behaviour |
|---|---|
| no output device | text only; `/tts` says "no audio output device is connected" |
| model not downloaded / kokoro-onnx not installed | text only; `/tts` and `/status` say how to install it |
| model fails to load | text only, retried after 5 minutes |
| engine crashes or hangs (30 s) | that reply is text only; the next one starts a new engine |
| device unplugged while speaking | playback detected as stalled after ~2 s, stopped, logged |
| invalid or empty text | nothing is spoken (`TTSError` / nothing to say) |
| any unexpected error | logged; the conversation continues |

Logs (`var/logs/nexus.log`): provider, model load/unload, output device and changes,
utterance length and timings, stops, errors - never the text of the conversation.

## Configuration (`[tts]` in config/nexus.toml)

`enabled`, `provider` (kokoro), `model` (fp32 | int8), `voice`, `hindi_voice`,
`romanized` (hindi | english), `speed` (0.5-2.0), `volume` (0-1), `placement` (cpu | gpu),
`threads` (0 = automatic), `ram_mb`, `idle_unload_seconds`, `game_mode` (quiet | speak),
`max_spoken_chars`. Same fail-closed rules as the rest of the file.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[tts]"        # ~10 MB (with [voice] already installed)
.\.venv\Scripts\python.exe scripts\get_tts_model.py fp32      # 310 MB model + 26 MB voices, asks first
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest                # everything with fakes: no model, no speakers
.\.venv\Scripts\python.exe -m pytest -m tts         # the real Kokoro model, nothing played (~30 s)
.\.venv\Scripts\python.exe -m pytest -m speaker     # plays through the real output (you hear it)
.\.venv\Scripts\python.exe scripts\tts_check.py     # full hardware report
```

| Area | Tests |
|---|---|
| text for speech: markdown, code, URLs, paths, emoji, units, truncation, chunking, Unicode | `test_speech_text.py` |
| Hinglish for the ear | `test_speech_text.py` |
| manager: cold/warm/unload/reload, failure recovery, chunk pipeline, stop, speak-stop-speak x6, queue, languages, game mode, microphone, thread-safe state, no output, device errors, crash survival, safety | `test_speaker.py` |
| provider: init, availability, health, capabilities, empty/invalid text, crash, timeout, unload, missing model | `test_kokoro.py` (fake worker) |
| playback: default device, no device, fallbacks, stop, device change, stalled device, thread preparation | `test_playback.py` |
| keyboard/voice -> assistant -> speech, failures, interrupts, `/tts`, wiring | `test_tts_integration.py` |
| config | `test_tts_config.py` |
| real model (T01-T09, incl. Kokoro -> Whisper round trip) | `test_tts_live.py` |
| real output (H01-H03) | `test_tts_hardware.py` |

## Limitations

- Indian languages are intelligible but not natural: Kokoro's Hindi uses espeak-ng
  phonemes; Hinglish is read through a rule-based conversion; Punjabi uses the Hindi
  voice; there is no Haryanvi voice. A dedicated Indic TTS model could be added later
  behind the same provider interface.
- No SSML; no streaming inside a sentence; one reply at a time (queued).
- No echo cancellation: push-to-talk stops speech before listening, so NEXUS doesn't
  hear itself; an always-on wake word (subsystem 7) will need more than that.
- If the default output changes mid-sentence, that sentence ends; the next one uses
  the new device.
- The int8 model (88 MB) was downloaded for the comparison and is not used by default;
  it can be deleted from `models/tts/`.
