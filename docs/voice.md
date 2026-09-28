# Subsystem 5: Voice input (speech-to-text)

Code: `nexus/system/audio/` (microphones, capture), `nexus/ai/voice/` (recorder,
speech-to-text, listener), `nexus/system/platform/windows.py` (device list,
privacy switch), `nexus/console.py` (`/listen`, `/mic`).

Push-to-talk, local-first. A spoken request becomes ordinary text for the same
conversation engine, fast path, planner, Action Kernel, security policy and audit
log as typed text. There is no second command system, and voice grants nothing
by itself.

## Using it

```
you>                              <- Enter on an empty line (or /listen, /l) = push-to-talk
  (listening with Headset (AirPods Pro Hands-Free) - speak now; Ctrl+C cancels)
  heard: "कितने बजे हैं"   (hi, confidence 0.84, transcribed in 1.2s)
nexus> It's 9:41 PM.
you> /mic                         <- which microphones exist, which one is used and why others aren't
you> /mic use airpods             <- prefer one (part of its name); /mic auto goes back to automatic
```

Listening always ends by itself: at the first 0.9 s pause after you speak, after
15 s of speech, or after 6 s with no speech. Ctrl+C cancels. Nothing is recorded
to disk and nothing leaves the PC.

## Pipeline

```
/listen
  privacy switch allows the microphone?            (Windows: Settings > Privacy > Microphone)
  pick a microphone                                MicrophoneManager (device list read again every time)
  pick a speech model, start loading it            HybridSTT.prepare - loads while you speak
  LISTENING: record until the pause                Recorder (energy VAD, noise-adaptive)
      microphone won't open / disconnects  ->  next microphone
  transcribe                                       faster-whisper in a worker process
  clean, reject noise/unclear, romanize            nexus/ai/voice/text.py
  -> UserInput(text, interface="voice", confidence) -> ConversationEngine (unchanged path)
```

## Microphones: any suitable one, chosen automatically

No microphone is hard-coded. The device list comes from Windows Core Audio
(MMDevice API, read-only: `WindowsPlatform.audio_inputs()`, about 5 ms). Only
capture endpoints are listed, so speakers never appear. Recording uses PortAudio
(`sounddevice`) in WASAPI shared mode at 16 kHz mono; Windows converts the format.

Order: your pick (`/mic use`, or `[voice] microphone`) -> the Windows default
**communications** microphone -> the default microphone -> any other (headset >
microphone > unknown input > line in; USB > built-in > Bluetooth).

| Input | Used? | How it's recognised |
|---|---|---|
| USB / wired / headset / Bluetooth headset / laptop array microphones | yes | active capture endpoint |
| "Stereo Mix", "What U Hear", "Monitor of ..." | **never** - it records what the PC plays (a video could "speak" commands) | built-in codec input with no microphone form factor, or its name |
| A phone connected for calls/music ("... HF Audio", "A2DP SNK") | **never** - that is the phone's audio, not a microphone here | Bluetooth hands-free role / audio-source service |
| Virtual devices (iVCam, virtual cables) | only if you pick one with `/mic use` | root-enumerated (software) driver |
| Disabled or unplugged endpoints | no, and `/mic` says so | endpoint state |

Hotplug: the list is re-read every 5 s in the background (`device_poll_seconds`),
on every `/listen`, and when a microphone fails. Plugging one in prints "Microphone
connected: X. NEXUS will listen with X." with no restart needed.
- **A microphone that won't open** (busy, format refused) is skipped for 30 s and
  the next one is tried.
- **Disconnect before you speak:** NEXUS switches microphones and keeps listening.
- **Disconnect mid-sentence:** "X stopped while you were speaking. Switched to Y -
  please say it again." Half a sentence is never acted on.

Linux later: the same `MicrophoneManager`; `PortAudioCapture.list_inputs()`
already works there as a device list, and a `LinuxPlatform.audio_inputs()` can
add PipeWire/PulseAudio details (monitor sources = loopback).

## Recording

Energy-based voice-activity detection. It needs no model and ~0 CPU.
- **Noise floor:** measured during the first 300 ms, capped at about -35 dBFS so
  that talking straight away isn't taken for noise.
- **Tracking the room:** the floor follows the quietest level, moving down at once
  and up slowly, so a fan is learned but pauses between syllables never are.
- **Lead-in and blips:** 750 ms of lead-in is kept so the first word isn't cut;
  bursts under 200 ms (clicks, coughs) are ignored.
- **Silent microphone:** exact digital silence gets its own message ("muted, or
  volume at zero").
- **Device gone:** no audio from the device for 2 s means it's lost.

## Speech-to-text

faster-whisper (CTranslate2, int8 on CPU) runs in a **worker process**
(`nexus/ai/voice/whisper_worker.py`) that the ResourceManager loads and unloads:
- **Unloading ends the process.** All of its RAM (model *and* libraries) and, on
  a GPU, all of its VRAM goes back. Measured: the worker is gone 0.05-0.07 s
  after unload.
- **A stuck transcription is stopped** (timeout 30 s), and a native crash can't
  take NEXUS down. The next request simply starts a new worker.
- **Loading starts when listening starts,** so it overlaps your speech.
- **Unused models unload** after 60 s idle.
- **Nothing is downloaded automatically.** `scripts/get_stt_model.py <name>`
  shows the size and asks first. Models live in `models/stt/`, which git ignores.
- **Silero voice-activity filtering runs first.** Silence produces no words at
  all, so nothing is invented from noise.
- **Language detection and transcription share one encoder pass,** which halved
  latency.
- **Idle worker threads sleep instead of spinning** (`OMP_WAIT_POLICY=PASSIVE`).
  Otherwise each transcription cost about 1 s of 100 % CPU afterwards, with no
  speed difference.
- **Decoding is bounded.** In the first real test, two unclear Hindi phrases took
  10.7 s and 23.3 s because Whisper retried the decode at six temperatures and
  looped on repeated words up to 448 tokens. Now:
  - one retry at most (temperatures 0.0, 0.3);
  - a length cap that fits the speech: 32 + 32 tokens per second of speech.
    Gurmukhi needs ~23 tokens/s, Hindi ~12, English ~3. A 2 s phrase gets 96
    tokens.
  - A/B on the same clips: identical latency on normal speech (1.33 vs 1.34 s).
- **Unsure language: both are tried.** Below 50 % detection confidence, the two
  most likely languages are decoded and the words the model is surer of win. In
  the real test, Haryanvi "हाँ, कर दे" was detected as English (31 %) and came out
  *translated* ("I'll do it."). The encoder pass is shared, and the runner-up gets
  one greedy decode only (~0.2-0.6 s). This happens only when unsure; clear
  English never triggers it.

Model choice per request (`HybridSTT`), with the same rules as the chat models:
- **Best first.** An option is skipped if it isn't installed, doesn't fit the free
  RAM/VRAM (reserves kept), or is LAN/cloud with cloud off or no network.
- **Game mode:** only options marked `game_mode = true` (base, 2 threads, CPU),
  and an idle heavier model is unloaded at once.
- **A GPU option that fails to load** is skipped for 5 minutes and the CPU option
  is used. Tested with the real libraries: "cublas64_12.dll is not found" ->
  answered by whisper-base on CPU.
- **Cloud STT:** the `STTProvider` interface has a locality. A cloud provider would
  only run with `allow_cloud = true` and a network, but **none exists**.

| Model (CPU int8) | Disk | RAM loaded / while transcribing | Load | Latency, 1-4 s phrase | 10 s of speech | CPU while transcribing |
|---|---|---|---|---|---|---|
| **small**, 6 threads (default) | 463 MB | 391 / 551 MB | 2.1-2.5 s | 1.21 s median, 1.31 s max | 1.55 s | ~390 % of one core |
| **base**, 2 threads (game mode) | 141 MB | 192 / 305 MB | 1.2 s | 0.62 s median, 0.69 s max | 0.74 s | ~170 % |

Measured 2026-09-28 with `scripts/measure_stt.py` on the i5-12400F, 16 GB, CPU only.
- **VRAM:** unchanged (0 MB used by STT; ±40 MB noise from other apps).
- **Loaded but idle:** 0 % CPU.
- **After unload:** the worker process is gone and free RAM returns to its
  before-load level.
- **Cold start (model not loaded):** 1.4-1.55 s (small) and 0.72-0.76 s (base)
  from end of speech to text, because loading ran while you spoke. Add the 0.9 s
  pause NEXUS waits for.
- **NEXUS idle with voice on:** 38 MB RAM (was 35-37 MB), 0.13 % of one core
  including the 5 s microphone check, and 0 VRAM. PortAudio adds ~7 MB on the
  first `/listen` with a microphone.

GPU faster-whisper would need NVIDIA's cuBLAS 12 + cuDNN 9 libraries (~1.1 GB of
pip packages). They are not installed, and the GPU isn't needed for these latencies.

## Languages

Whisper detects the language of each utterance, restricted to `languages = ["en",
"hi", "pa"]`. When Hindi is labelled Urdu (the same spoken language in another
script), it counts as Hindi. There is no Haryanvi model and none is faked:
Haryanvi comes out as Hindi, and the pipeline below is made tolerant of it.

### From transcript to request (`nexus/ai/voice/normalize.py`)

A real speaker showed what Whisper actually writes:
- Hindi in Devanagari;
- Punjabi mostly in **Devanagari** too ("किन्ने वाजे, ने");
- Hinglish either in Devanagari ("जीटिय चलादे") or as English-looking guesses
  ("Chrome, Coldo." for "Chrome khol do").

The fast path and the planner were built and evaluated on romanized Hinglish, so
the transcript is brought to that form in steps, from exact to guessing:

| Step | Examples | Kind |
|---|---|---|
| 1. script | Devanagari/Gurmukhi -> Latin letters ("चला दे" -> "chala de"); kept as written if the fast path already understands it ("कितने बजे हैं") | exact |
| 2. dialect | fused words split ("chalade" -> "chala de", "bandkaro" -> "band karo"); Punjabi "dio" -> "do", "kinne vaje" -> "kitne baje", "hoya" -> "hua", "dasso" -> "batao"; Haryanvi "koni" -> "nahi", "mhara" -> "mera"; questions "ke/ki ... sai/se" -> "kya ... hai" | exact rules |
| 3. sounds | words that sound like a command word: "Coldo" -> "khol do", "ताम" -> "time", "बही" -> "bhai" | see below |
| 4. names | words that sound like an **installed** app's name (from the app catalog, nothing hard-coded): "क्रोम" -> chrome, "जी टी ए" / "जीटिय" -> gta, "ओ बी एस" -> obs | see below |

Rules for the guessing steps (3-4):
- **Short utterances only.**
- **Used only if the fast path then understands the whole request,** or it is a
  clear yes/no answer. Otherwise the planner gets the exact text from steps 1-2,
  never a guess.
- **No word is ever dropped to make a command fit.** "don't open chrome" can never
  become "open chrome", and a garbled wake word stays in the text.
- **Spellings are not guesses.** An English name written in Devanagari that sounds
  exactly like the name ("क्रोम") is the same word, and so is a filler ("bahi" ->
  "bhai").
- **Everything else is marked as a guess,** and NEXUS asks before acting on it (see
  Safety). That covers English-looking mishearings ("Coldo") and near misses
  ("जीटिय").

The planner prompt also says that spoken messages may contain Haryanvi and Punjabi
words ("ke" = "kya", "sai" = "hai", "koni" = "nahi", "dasso" = "batao"...). Planner
evaluation afterwards: system 97.4 % on 78 phrases, the same two misses as when
Subsystem 4 was frozen, so there was no regression. Haryanvi small talk ("ke haal se
bhai?") is still one of those two misses.

### First real-speaker test (2026-09-28, USB microphone, whisper-small)

Before these fixes, 15 phrases, same speaker:

| Group | Result | What happened |
|---|---|---|
| English | 3/3 | exact words |
| Hinglish | 0/3 | "GTA chala de" -> "अग्स्छ जीटिय चलादे"; "Chrome khol do" -> "Chrome, Coldo."; "Haan, kar do" -> "अ, कर दो." |
| Hindi | 2/3 | time and folder right; "नहीं, रहने दो" heard exactly but **not recognised as a no** |
| Punjabi | 1/3 | time right (written in Devanagari); "Chrome khol dio" -> "Chrome coldio" (confidence 0.37); "ਹਾਂ ਜੀ" -> "अंदी" |
| Haryanvi | 0/3 | "के ताम हो रहा से"; "जी टी ए चलाडे बही"; "हाँ, कर दे" -> translated to "I'll do it." |

The same transcripts, through today's pipeline (`tests/test_voice_normalize.py`):
- **Now handled:**
  - "Chrome, Coldo." / "Chrome coldio" -> launch Chrome, **asked first** (guess).
  - "के ताम हो रहा से" -> the time.
  - "जी टी ए चलाडे बही" -> launch GTA, asked first (confidence 0.57).
  - "नहीं रहने दो" -> no.
  - "अ, कर दो" -> yes, asked again (confidence 0.50).
- **Still not recoverable from the text:**
  - "अग्स्छ जीटिय चलादे": the garbled wake word stays.
  - "अंदी" (for "ਹਾਂ ਜੀ").
  - "I'll do it." The two-language decode is meant to prevent this at the source;
    it needs the retest to confirm.

### Real-speaker retest after the fixes (2026-09-28, same speaker and microphone)

Re-scored with the final parser, which added "haan ji" heard as "अंजी"/"anji":

| Group | Result | Examples (what Whisper wrote -> what NEXUS would do, nothing executed) |
|---|---|---|
| English | 3/3 | "Open my Downloads folder" -> open the folder |
| Hinglish | 3/3 | "नेक्सस जीटिय चलादे" -> start GTA, **asks first** (a guess); "क्रोम खोल्दो" -> open Chrome, asks first (0.59); "आ, कर दो" -> yes, asked again (0.57) |
| Hindi | 2/3 | time right; "नहीं रहेने दो" -> no; "डाउनलोड्स फ़ोल्डर खोलो" came out *translated* ("Downloads folder, open it", 0.50) -> the planner, which would ask first |
| Punjabi | 2/3 | "क्रोम पोल दियो" -> open Chrome, asks first; "अंजी" -> yes, asked again (0.57); time came out as "किनने वजने" -> the planner |
| Haryanvi | 2/3 | "जीतिये चलादे बही" -> start GTA (0.63, spelled letter by letter: acts directly); "आ, करे दे" -> yes, asked again; time came out as "के ताई मोगरे लेए से" -> the planner |
| yes/no phrases | 5/5 | all four no's cancel; "अंजी" at 0.64 approves |

- **Yes and no:** yes recognised 6/6. Two approve directly (confidence 0.64 and 0.73);
  the other four were heard at 0.55-0.57, so NEXUS asks again. No recognised 4/4,
  and all cancel.
- **Wrong actions:** none.
- **Latency:** median 1.34 s, max 1.51 s.
- **Model:** loaded in 2.2 s while you spoke; worker RAM up to 412 MB; VRAM
  unchanged; unloaded in 0.07 s and the process was gone.
- **Hotplug:** unplugging and re-plugging were both noticed, and NEXUS listened
  again without a restart.

`scripts/voice_check.py` repeats this with a microphone.
- It prompts 15 phrases in the five groups plus 5 yes/no answers.
- It prints what was heard, how NEXUS normalized it, and what NEXUS would do (act,
  **ask first**, yes/no, or the planner). It never executes anything.
- It measures load time, RAM and VRAM, then runs an unplug/re-plug check.
- `--save-audio` keeps the recordings on this PC, and `--replay` re-runs them
  through a changed pipeline or model without speaking again.

## Safety

- **Voice is not a permission.** Spoken requests pass the same policy. Confirmations
  are still required, BLOCKED stays blocked, and a paired phone still can't answer
  the PC's questions.
- **Misheard speech never approves an action.** A spoken "yes" counts only if it was
  heard with confidence >= 0.6 (`confirm_min_confidence`) and was not reconstructed
  from similar sounds. Otherwise: "I didn't hear that clearly. Say yes or no again -
  or type it." The question stays open.
- **A spoken no always cancels, and no wins.** The recognised forms cover all five
  languages, in Latin letters, Devanagari and Gurmukhi:
  - no / nope / nahi / nahin / नहीं
  - rehne do / rahne do / रहने दो / ਰਹਿਣ ਦਿਓ
  - chhod do / छोड़ दो
  - nahi chahiye, koni, na / ना
  - never mind / leave it
  Any answer containing a no is a no ("haan... nahi, rehne do"), and so is a short
  answer with a misheard word in it ("nahi renny do"). Anything that isn't a clear
  yes cancels the pending action anyway.
- **Uncertain speech asks before it acts.** A spoken request asks first ("I'm not
  sure I heard you right ("..."). Should I open Google Chrome? (yes/no)") if it was
  heard with confidence below 0.6, or needed a sound-alike guess. This applies to
  anything that changes something: open, start, close, write, remember, create.
  Only read-only answers (time, system info, lists) are given directly. Typed text
  and clearly heard speech act as before.
- **Unclear speech isn't acted on.** Below confidence 0.4 (`min_confidence`), NEXUS
  shows what it heard and asks you to repeat it.
- **Known Whisper inventions are ignored** when the model wasn't sure ("Thanks for
  watching!", "[Music]").
- **What never counts as a microphone:** loopback inputs (the PC's own audio) and a
  phone's call/music audio.
- **No input hooks.** Push-to-talk is Enter or `/listen` in the console. There are
  no global keyboard hooks (acceptance test A12).

## What NEXUS says when something goes wrong (it keeps running in every case)

| Situation | Message (shortened) |
|---|---|
| no microphone | "No microphone is connected, so voice input is off for now (typing still works) ... Plug one in and NEXUS will find it - no restart needed." |
| privacy switch off / access denied | "Microphone access is turned off. Allow it in Windows Settings > Privacy & security > Microphone" |
| microphone busy / won't open | tries the next one; if none: "No microphone could be opened: ..." |
| unplugged while listening | switches, or "X stopped while you were speaking. Switched to Y - please say it again." |
| silence | "I didn't hear anything in 6 seconds." (digital silence: "may be muted") |
| unclear | "I couldn't make that out clearly (I heard "..."). Please say it again." |
| model not downloaded / no memory / game mode without a light model | "I can't transcribe speech right now: ..." (said *before* recording) |
| GPU libraries missing | falls back to CPU silently (logged) |
| transcription too slow | "Transcribing took longer than 30 seconds, so I stopped it." |
| worker crashed | "The speech engine stopped unexpectedly (it restarts next time)." |
| Ctrl+C | "(cancelled)" |
| anything unexpected | "Voice input failed unexpectedly (...). Typing still works; details are in var/logs/nexus.log." |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest                   # default suite: all voice logic with fakes (no hardware)
.\.venv\Scripts\python.exe -m pytest -m voice          # real faster-whisper models on synthetic speech (~25 s)
.\.venv\Scripts\python.exe -m pytest -m mic            # the real microphone (skipped as PENDING without one)
$env:NEXUS_MIC_SPEAK=1; .\.venv\Scripts\python.exe -m pytest -m mic   # ...and say "What time is it?"
.\.venv\Scripts\python.exe scripts\voice_check.py      # real microphone, all languages, nothing executed
.\.venv\Scripts\python.exe scripts\measure_stt.py      # load time, latency, RAM/VRAM/CPU per model
.\.venv\Scripts\python.exe scripts\eval_stt.py         # accuracy on synthetic speech
```

| Area | Tests |
|---|---|
| enumeration, classification (loopback/remote/virtual/disabled), ranking, default communications mic | `test_audio_devices.py` |
| fallback, disconnect/reconnect, hotplug watcher, no microphone, picking one, real Windows list | `test_audio_devices.py`, `test_voice_input.py` |
| capture wrapper: WASAPI conversion, format fallbacks, downmix, access denied, re-scan | `test_capture.py` |
| VAD: speech/pause, never forever, silence, noise, clicks, cancel, device lost | `test_recorder.py` |
| STT worker: load/unload, timeout, crash, cancel, load failure, missing model | `test_stt.py` (fake worker process) |
| model choice: best first, memory, game mode, GPU -> CPU, cloud rules, preload, idle unload | `test_stt.py` |
| listener: every error path, languages, romanizing | `test_voice_input.py`, `test_voice_text.py` |
| conversation: same kernel, confirmations, low-confidence yes, phone can't answer, console | `test_voice_conversation.py` |
| uncertain speech asks first; guessed yes asked again; no cancels; read-only answers direct | `test_voice_conversation.py` |
| real-speaker transcripts -> requests; English never rewritten; spellings vs guesses; dialect forms | `test_voice_normalize.py` |
| yes/no in all five languages (the real test's "नहीं रहने दो" and the retest phrases) | `test_responder.py` |
| worker decoding: length cap, one retry, unsure language -> both tried, no speech -> nothing | `test_whisper_worker.py` (fake model) |
| config validation | `test_voice_config.py` |
| real models (V01-V07), real microphone (M01-M02) | `test_voice_live.py`, `test_voice_mic.py` |

**Subsystem 5 is FROZEN** (2026-09-28): no further feature additions; only fixes.

## Not in Subsystem 5

- **Real-microphone test with a USB microphone (AB13X), 2026-09-28:**
  - detected automatically;
  - `pytest -m mic` passed;
  - unplug and re-plug were noticed within the 5 s check, with no restart;
  - it listened again straight away.
  Language accuracy after the fixes: see "Real-speaker retest" above.
- Punjabi recognition by Whisper is weak ("ਹਾਂ ਜੀ" -> "अंदी"), and a garbled
  "Nexus" can't be recovered. With push-to-talk, "Nexus" isn't needed.
- Wake word, TTS, GPU speech-to-text, a cloud STT provider, the HUD, and
  hold-to-talk. Hold-to-talk would need a global key hook, which the design
  forbids; the future HUD will have its own button.
