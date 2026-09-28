"""Record yourself saying "Nexus" (and, optionally, your room) to train the wake word on your voice.

    .venv\\Scripts\\python.exe scripts\\record_wake_samples.py                  # 30 x "Nexus", ~3 minutes
    .venv\\Scripts\\python.exe scripts\\record_wake_samples.py --background 120  # + 2 min of your room/TV/game
    .venv\\Scripts\\python.exe scripts\\train_wake_model.py --samples var\\wake\\samples

Uses the same microphone NEXUS picks (or --mic <part of a name>). Each "Nexus" is cut at the pause
after it and saved as a WAV in var/wake/samples/positive/ (git-ignored, never uploaded). Recording
more later adds to the folder; delete a file to drop a bad take. Half of the takes train the
model, the other half test it.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus.core.config import PROJECT_ROOT  # noqa: E402

FOLDER = PROJECT_ROOT / "var" / "wake" / "samples"
# Say it in different ways, so the model learns you - not one recording.
STYLES = ["normally", "normally", "a bit quieter", "a bit louder", "quickly", "slowly", "like a question",
          "the way you'd say it in Hindi", "from further away (lean back)", "turned away from the microphone",
          "then keep talking: \"Nexus, open Chrome\"", "then keep talking: \"Nexus bhai GTA chala de\""]


def save(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=30, help="how many times to say \"Nexus\"")
    parser.add_argument("--background", type=int, default=0, metavar="SECONDS",
                        help="also record this many seconds of normal sound (no \"Nexus\"): talking, TV, a game")
    parser.add_argument("--mic", default="auto", help="part of the microphone's name (default: as NEXUS picks)")
    args = parser.parse_args()

    from nexus.ai.voice.base import NoSpeech
    from nexus.ai.voice.recorder import Recorder, VADConfig
    from nexus.system import platform as nexus_platform
    from nexus.system.audio import MicrophoneManager, PortAudioCapture

    plat = nexus_platform.current()
    capture = PortAudioCapture()
    mics = MicrophoneManager(getattr(plat, "audio_inputs", None) or capture.list_inputs, preferred=args.mic)
    scan = mics.scan()
    if not scan.candidates:
        print("No microphone found.")
        return 1
    mic = scan.chosen
    stream = capture.open(mic, sample_rate=16000, block_ms=30)
    print(f"Microphone: {mic.name}")
    positive = FOLDER / "positive"
    start = len(list(positive.glob("*.wav"))) if positive.is_dir() else 0
    recorder = Recorder(VADConfig(end_silence_ms=500, max_utterance_s=4, no_speech_timeout_s=8))
    try:
        saved = 0
        while saved < args.count:
            style = STYLES[saved % len(STYLES)]
            input(f"\n[{saved + 1}/{args.count}] Press Enter, then say \"Nexus\" {style}.")
            try:
                utterance = recorder.record(stream)
            except NoSpeech as exc:
                print(f"  {exc} Again.")
                continue
            path = positive / f"nexus_{start + saved + 1:03d}.wav"
            save(path, utterance.clip.pcm16, utterance.clip.sample_rate)
            print(f"  saved {path.name} ({utterance.clip.seconds:.1f} s)")
            saved += 1
        if args.background:
            input(f"\nNow {args.background} s of normal sound WITHOUT \"Nexus\": talk, play a video or a game. "
                  "Press Enter to start.")
            chunks, end = [], time.monotonic() + args.background
            while time.monotonic() < end:
                block = stream.read(0.2)
                if block:
                    chunks.append(block)
            path = FOLDER / "background" / f"room_{int(time.time())}.wav"
            save(path, b"".join(chunks), stream.sample_rate)
            print(f"  saved {path}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.close()
    print(f"\nRecordings are in {FOLDER}. Next:\n"
          f"  .venv\\Scripts\\python.exe scripts\\train_wake_model.py --samples {FOLDER.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
