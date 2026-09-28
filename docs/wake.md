# Subsystem 7: Wake word ("Nexus")

Code: `nexus/ai/voice/wake.py` (detectors, "is this addressed to NEXUS?"), `oww.py`
(openWakeWord on onnxruntime), `wake_listener.py` (the always-on listener),
`recorder.py` / `listener.py` (continuing an utterance after the wake word),
`nexus/console.py` (`/wake`), `config/nexus.toml` `[wake]`. Scripts:
`get_wake_model.py`, `train_wake_model.py`, `record_wake_samples.py`.

Say **"Nexus"** and your request, with no key press:

```
you>
  (Nexus - listening)
  heard: "Nexus bhai GTA chala de" -> "bhai GTA chala de"   (hi, confidence 0.91, transcribed in 1.1s)
nexus> Haan bhai, ek second.
```

The request after "Nexus" goes through **exactly** the same pipeline as push-to-talk
and typed text: speech-to-text, fast path, planner, Action Kernel, security
policy, confirmations, audit log. The wake word grants nothing by itself; a
misheard "yes" still never approves anything.

```
Microphone (always open while the wake word is on)
  -> WakeDetector, frame by frame, in memory only
       openwakeword: every 80 ms -> score        whisper: speech bursts -> first 1.6 s -> text
  -> "Nexus" -> NEXUS stops talking; the same utterance keeps recording (Recorder)
       "Nexus, GTA chala de"   -> speech-to-text -> wake word removed -> conversation engine
       "Nexus" ... pause       -> chime, "yes? - listening" -> the request is recorded
  -> NEXUS asked something (yes/no, which one?) -> the answer is heard without "Nexus"
  -> back to waiting
```

## Privacy

- The microphone stays open while the wake word is on, so Windows shows its
  microphone-in-use icon. The audio is examined **in memory, on this PC**:
  openWakeWord keeps the last ~10 s of speech *features* (numbers, not audio) and
  ~1 s of audio; the Whisper detector keeps the current burst of speech (at most 6 s).
  Nothing is written to disk, sent anywhere, or logged except "a wake word was heard".
- Only after "Nexus" is anything transcribed and acted on - and then only that request.
- The Windows microphone privacy switch is respected. `/wake off` releases the
  microphone (push-to-talk still works); `enabled = false` in `[wake]` turns it off for good.

## Using it

```
/wake              state, detector, microphone, how often it woke, why a detector isn't used
/wake off | on     stop / start listening for "Nexus"
/wake test         20 s: every detection is shown ("score 0.93") and nothing is acted on
```

- **Pauses by itself** while push-to-talk records, while NEXUS is speaking (so it
  can't wake itself up - see `barge_in`), and in game mode if `game_mode = "off"`.
- **Follow-up**: when NEXUS asks something ("Should I delete ...?", "Which one?"),
  it listens for the answer for `follow_up_seconds` without "Nexus", after it has
  finished speaking.
- Microphones: the same automatic choice as push-to-talk (`/mic`). A microphone
  plugged in later is used, a better one is switched to within 2 s, one that
  disappears is replaced by the next. If the audio system restarts (new
  headphones), the stream is reopened.

## Two detectors

| | `openwakeword` (trained "Nexus") | `whisper` (works without training) |
|---|---|---|
| how | three small ONNX networks on every 80 ms: mel spectrogram -> Google's speech embedding -> NEXUS's own classifier | energy detector finds bursts of speech; the first 1.6 s of each burst goes to a light Whisper model; the text must *start* with "Nexus" |
| CPU | ~1.6 ms per 80 ms = **~2% of one core**, all the time (measured) | nothing in silence; one short transcription per burst of speech (at most `max_checks_per_minute`) |
| RAM | ~60 MB | the Whisper worker (base ~320 MB, loads when speech is heard, unloads when idle) |
| reacts | ~0.1-0.3 s after "Nexus" ends | after 1.6 s of speech, or at the pause after a short request |
| needs | `get_wake_model.py` (2.4 MB) + `nexus.onnx` from `train_wake_model.py` | a downloaded Whisper model (`base` is already there for game mode) |
| packages | none new: onnxruntime + numpy come with `[voice]` | none new |

`detector = "auto"` uses openwakeword when `models/wake/nexus.onnx` exists and the
Whisper detector otherwise; `/wake` says which one runs and why the other doesn't.
Any openWakeWord-format model works (`model = "hey_jarvis_v0.1"` after
`get_wake_model.py --hey-jarvis` to try the pipeline before training).

`nexus/ai/voice/oww.py` re-implements openWakeWord's streaming feature pipeline on
onnxruntime alone (no scipy, scikit-learn or TFLite). Its scores equal the
`openwakeword` package's exactly (max difference 0.0 on the same audio, `test_wake_live.py`).

### What counts as "Nexus" in text (Whisper detector, and removing it before the request)

`find_wake()`: "Nexus", spellings a speech model produces ("Nexas", "Nexis", "next
us", one letter off like "Nexux"), Devanagari and Gurmukhi ("नेक्सस", "ਨੈਕਸਸ"),
after up to two fillers ("hey", "oye", "arre", "ok"). It must be at the start:
"Next song", "I love Nexus", "Lexus", "Texas" and "necklace" don't count.

## Training the "Nexus" model

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[tts,wake-train]"
.\.venv\Scripts\python.exe scripts\get_wake_model.py
.\.venv\Scripts\python.exe scripts\train_wake_model.py                 # synthetic voices only
.\.venv\Scripts\python.exe scripts\record_wake_samples.py --background 120
.\.venv\Scripts\python.exe scripts\train_wake_model.py --samples var\wake\samples   # + your voice and room
```

Everything runs locally on the CPU; nothing is downloaded but the two shared
networks. `train_wake_model.py`:

1. **Speech from Kokoro** (the voice NEXUS speaks with): "Nexus" by 45 voices plus
   90 blends of two voices (new-sounding speakers), at speeds 0.8-1.3, as English
   text and as phonemes (the Indian "nˈeːksəs" by the Hindi voices), alone and
   followed by requests ("Nexus, GTA chala de"). The same speakers say ordinary
   English/Hinglish/Hindi sentences and near-misses ("next", "Lexus", "Texas",
   "necklace", "Alexa") - `evals/wake_phrases.json`.
2. **Rooms and microphones**: synthetic reverb (RT60 0.15-0.8 s), microphone
   frequency response, white/pink/brown noise, mains hum, other people talking,
   your own background recordings; loudness from a quiet voice across the room to close-up.
3. **Classifier**: 16 frames x 96 features -> 64 -> 64 -> 1, trained with numpy
   (hard negatives weighted up each round), saved as ONNX in openWakeWord's format.
4. **Test on voices it never heard** (9 voices + 24 blends kept out of training),
   through the real runtime pipeline: a long stream of their speech with "Nexus"
   in it. The threshold is the lowest with no false wake-up on the first half of
   the stream, then moved to 0.5 so `[wake] threshold = 0.5` is that point. The
   second half is measured and written to `models/wake/nexus.json`.

Your own recordings matter most: half train the model, half test it ("your
voice: n of m heard" in the report).

### Measured (this build, synthetic voices only)

RESULTS_PLACEHOLDER

## Settings (`[wake]` in config/nexus.toml)

| setting | default | |
|---|---|---|
| `enabled` | true | `/wake off` for the session |
| `detector` | auto | auto / openwakeword / whisper |
| `model` | nexus | `models/wake/<model>.onnx` |
| `threshold`, `patience` | 0.5, 1 | openwakeword: higher = fewer false wake-ups, more misses |
| `whisper_model`, `max_checks_per_minute` | base, 20 | Whisper detector |
| `command_timeout_seconds` | 6 | after "Nexus" on its own |
| `follow_up_seconds` | 6 | answering NEXUS's question without "Nexus"; 0 = off |
| `chime` | true | two short tones after "Nexus" on its own |
| `barge_in` | false | listen while NEXUS talks (it may hear itself; use headphones) |
| `game_mode` | listen | listen / off while a game runs |

Unknown or out-of-range values stop NEXUS from starting, like the rest of the file.

## Tests

| | |
|---|---|
| wake word in text: spellings, scripts, fillers, near-misses | `test_wake.py` |
| Recorder continuing an utterance (lead-in, known noise floor) | `test_wake.py` |
| Whisper detector: bursts, check timing, "Nexus" alone, one check per burst, long speech split, rate limit, clicks, unclear text, model failure | `test_wake.py` |
| openWakeWord detector: threshold, warm-up, patience, lead-in, missing/broken files | `test_wake.py` |
| listener: request, "Nexus" alone + chime, silence, follow-up answer, speaking, barge-in, test mode, no mic -> plugged in, lost mic, stream reset, privacy switch, detector fallback, pause/off/game mode | `test_wake.py` |
| config, console (`/wake`, events, requests through the engine, follow-up detection) | `test_wake.py` |
| trainer: resampling, trimming, augmentation, classifier, ONNX export = numpy, scoring | `test_wake_training.py` |
| real models (`pytest -m wake`): features = openwakeword package; real detector on "Hey Jarvis"; trained "Nexus" on unseen voices vs near-misses; Whisper detector | `test_wake_live.py` |

## Limitations

- **Not yet tested with a real microphone and a real speaker** - this subsystem was
  built in a cloud container. Everything was measured on Kokoro-generated speech.
  Real voices, rooms and microphones are harder; recording your own samples and
  retraining is the fix, and `/wake test` shows the scores live.
- The Whisper detector could not be run against the real Whisper model in the build
  environment (model download blocked); its logic is tested with simulated speech,
  and `test_wake_live.py::test_whisper_detector_with_the_real_model` checks it on your PC.
- Synthetic training speakers: Kokoro's 54 voices and their blends are far fewer than
  real people. There is no Punjabi or Haryanvi voice (the Hindi voices stand in).
- No echo cancellation: while NEXUS speaks, the wake word isn't listened for
  (unless `barge_in`); typing or `/stop` still interrupts it.
- One wake word ("Nexus"); no speaker identification - anyone saying "Nexus" wakes it
  (security still applies: confirmations, protected paths, and so on).
