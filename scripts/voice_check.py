"""Check voice input with a real microphone, in each language. NOTHING IS EXECUTED.

    .venv\\Scripts\\python.exe scripts\\voice_check.py              # say the prompted phrases, then a hotplug check
    .venv\\Scripts\\python.exe scripts\\voice_check.py --list       # only show the microphones
    .venv\\Scripts\\python.exe scripts\\voice_check.py --only hi,pa --no-hotplug
    .venv\\Scripts\\python.exe scripts\\voice_check.py --save-audio # also keep the recordings (var/eval/audio/)
    .venv\\Scripts\\python.exe scripts\\voice_check.py --replay var\\eval\\audio\\<run>   # re-run saved recordings

For each phrase it prints the microphone, what the model wrote, the language it
detected, its confidence, how loud the speech was, how NEXUS normalized it, and
what NEXUS WOULD do - act, ask first, treat it as a yes/no answer, or hand it
to the planner - without running anything. It also measures model load time,
RAM, VRAM and unloading. Results are appended to var/eval/voice_check.jsonl.
Recordings are kept only with --save-audio (they stay on this PC, in var/).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

from nexus.ai.conversation.engine import _READ_ONLY_TOOLS, strip_wake_word  # noqa: E402
from nexus.ai.conversation.responder import parse_confirmation  # noqa: E402
from nexus.ai.providers.selection import SelectionContext  # noqa: E402
from nexus.ai.router import FastPathRouter  # noqa: E402
from nexus.ai.voice.base import AudioClip, VoiceInputError  # noqa: E402
from nexus.ai.voice.listener import Heard, VoiceInput  # noqa: E402
from nexus.ai.voice.recorder import level  # noqa: E402
from nexus.ai.voice.stt import HybridSTT, STTOption  # noqa: E402
from nexus.ai.voice.text import romanize  # noqa: E402
from nexus.ai.voice.whisper import WhisperSTT  # noqa: E402
from nexus.core.config import PROJECT_ROOT, load_voice_settings  # noqa: E402
from nexus.core.resources import ResourceManager  # noqa: E402
from nexus.system import info  # noqa: E402
from nexus.system import platform as nexus_platform  # noqa: E402
from nexus.system.apps.service import AppService  # noqa: E402
from nexus.system.audio import AudioInputError, MicrophoneManager, PortAudioCapture  # noqa: E402

# (group, what to say, reference text, expected: a tool name, or "yes"/"no" for answers to a question)
PROMPTS = [
    ("en", "What time is it?", "What time is it?", "get_datetime"),
    ("en", "Open my downloads folder.", "Open my downloads folder.", "open_folder"),
    ("en", "Yes, go ahead.", "Yes, go ahead.", "yes"),
    ("hinglish", "Nexus, GTA chala de.", "Nexus, GTA chala de.", "launch_app"),
    ("hinglish", "Chrome khol do.", "Chrome khol do.", "launch_app"),
    ("hinglish", "Haan, kar do.", "Haan, kar do.", "yes"),
    ("hi", "कितने बजे हैं?  (kitne baje hain)", "कितने बजे हैं", "get_datetime"),
    ("hi", "डाउनलोड्स फ़ोल्डर खोलो  (downloads folder kholo)", "डाउनलोड्स फ़ोल्डर खोलो", "open_folder"),
    ("hi", "नहीं, रहने दो  (nahin, rehne do)", "नहीं, रहने दो", "no"),
    ("pa", "ਕਿੰਨੇ ਵਜੇ ਨੇ?  (kinne vaje ne)", "ਕਿੰਨੇ ਵਜੇ ਨੇ", "get_datetime"),
    ("pa", "ਕ੍ਰੋਮ ਖੋਲ੍ਹ ਦਿਓ  (Chrome khol dio)", "ਕ੍ਰੋਮ ਖੋਲ੍ਹ ਦਿਓ", "launch_app"),
    ("pa", "ਹਾਂ ਜੀ  (haan ji)", "ਹਾਂ ਜੀ", "yes"),
    ("haryanvi", "के टाइम हो रह्या सै?  (ke time ho rhya se)", "के टाइम हो रह्या सै", "get_datetime"),
    ("haryanvi", "जीटीए चला दे भाई  (GTA chala de bhai)", "जीटीए चला दे भाई", "launch_app"),
    ("haryanvi", "हाँ, कर दे  (haan, kar de)", "हाँ, कर दे", "yes"),
    # answers to "should I...?" - recognition only; nothing is ever pending or executed here
    ("confirm", "haan, kar do", "haan, kar do", "yes"),
    ("confirm", "nahi, rehne do", "nahi, rehne do", "no"),
    ("confirm", "nahin, rahne do", "nahin, rahne do", "no"),
    ("confirm", "haan ji", "haan ji", "yes"),
    ("confirm", "nahi chahiye", "nahi chahiye", "no"),
]
GROUPS = ("en", "hinglish", "hi", "pa", "haryanvi", "confirm")


def dbfs(rms: float) -> float:
    return 20 * math.log10(max(rms, 1.0) / 32768)


def speech_level(clip: AudioClip | None) -> float | None:
    if clip is None:
        return None
    pcm, frame = clip.pcm16, clip.sample_rate * 30 // 1000 * 2
    levels = sorted(level(pcm[i:i + frame]) for i in range(0, len(pcm) - frame, frame))
    return levels[int(len(levels) * 0.9)] if levels else None  # the loud 10%: the speech itself


def vram_used() -> int | None:
    gpus = info.gpus()
    return int(gpus[0]["vram_used_mb"]) if gpus and gpus[0].get("vram_used_mb") is not None else None


def _words(text: str) -> list[str]:
    text = romanize(strip_wake_word(text)).lower()
    return re.sub(r"[^\w\s]", " ", text).split()


def word_accuracy(heard: str, reference: str) -> float:
    """1 - word error rate, both sides romanized (so Punjabi in Devanagari vs Gurmukhi compares)."""
    h, r = _words(heard), _words(reference)
    if not r:
        return 1.0 if not h else 0.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return max(0.0, 1 - prev[-1] / len(r))


def decide(router: FastPathRouter, heard: Heard, expected: str, confirm_min: float) -> tuple[str, str | None]:
    """What NEXUS would do - the same rules as ConversationEngine - described, never executed."""
    unclear = heard.confidence is None or heard.confidence < confirm_min or heard.guessed
    why = "a guess" if heard.guessed else f"confidence {heard.confidence or 0:.2f} < {confirm_min}"
    if expected in ("yes", "no"):
        answer = parse_confirmation(heard.text)
        if answer is False:
            return "a NO: cancels the pending question", "no"
        if answer is True and unclear:
            return f"a YES, not clear enough to approve ({why}): NEXUS would ask again", "yes (asks again)"
        if answer is True:
            return "a YES: approves the pending question", "yes"
        return "not a yes/no: the pending question is cancelled, then it's a new request", None
    plan = router.route(strip_wake_word(heard.text))
    if plan is None:
        tail = f" - and ask before acting ({why})" if unclear else ""
        return f"not understood by the fast path -> the planner (language model) would decide{tail}", None
    if not plan.actions:
        return f"fast path reply: {plan.say!r}", None
    step = plan.actions[0]
    args = ", ".join(f"{k}={v!r}" for k, v in step.args.items())
    if unclear and step.tool not in _READ_ONLY_TOOLS:
        return f"would ASK FIRST ({why}): {step.tool}({args})  [NOT executed]", step.tool
    return f"{step.tool}({args})  [NOT executed]", step.tool


def show(heard: Heard, described: str, listener: VoiceInput, clip: AudioClip | None, stt: WhisperSTT) -> dict:
    t = listener.last_timings
    speech = speech_level(clip)
    print(f"   microphone   : {heard.microphone}")
    print(f'   model wrote  : "{heard.original}"  [{heard.transcript.language}, '
          f"confidence {heard.confidence or 0:.2f}]")
    if heard.text != heard.original:
        print(f'   NEXUS reads  : "{heard.text}"   ({"; ".join(heard.changes)})')
    print(f"   NEXUS would  : {described}")
    print(f"   latency      : {t.get('transcribing_s', 0):.2f} s after you stopped (audio {t.get('audio_s', 0):.1f} s)"
          f"; speech level {dbfs(speech or 1):.0f} dBFS; speech model RAM {stt.memory_mb()} MB")
    return {"speech_dbfs": round(dbfs(speech or 1), 1)}


def hotplug_check(mics: MicrophoneManager, listener: VoiceInput, poll_s: float) -> dict:
    """Unplug / re-plug the microphone while NEXUS's own background check watches."""
    events: list[tuple[float, str]] = []
    started = time.monotonic()

    def on_change(change) -> None:
        events.append((time.monotonic(), change.describe()))
        print(f"\n   [{time.monotonic() - started:5.1f} s] NEXUS noticed: {change.describe()}")

    mics.subscribe(on_change)
    mics.scan()
    mics.start_watching(poll_s)
    result: dict = {"poll_seconds": poll_s}
    try:
        print(f"\n=== Hotplug check (NEXUS checks for devices every {poll_s:g} s; no restart) ===")
        before = mics.last_scan.chosen.name if mics.last_scan and mics.last_scan.chosen else None
        answer = input("Unplug the microphone now. When it's out, press Enter (or type s + Enter to skip): ")
        if answer.strip().lower() == "s":
            return {"skipped": True}
        deadline = time.monotonic() + poll_s + 3
        while time.monotonic() < deadline and not any("disconnected" in e[1] for e in events):
            time.sleep(0.1)
        now_chosen = mics.scan().chosen
        result["after_unplug_using"] = now_chosen.name if now_chosen else None
        result["unplug_noticed"] = any("disconnected" in e[1] for e in events)
        print(f"   After unplugging NEXUS would use: {result['after_unplug_using'] or 'no microphone'}")
        if now_chosen is None:
            try:
                listener.listen()
            except (VoiceInputError, AudioInputError) as exc:
                print(f"   /listen now says: {exc}")
                result["listen_without_mic"] = str(exc)
        input("Plug the microphone back in. Press Enter once it's in: ")
        deadline = time.monotonic() + poll_s + 8
        while time.monotonic() < deadline and not (events and "Microphone connected" in events[-1][1]):
            time.sleep(0.1)
        chosen = mics.scan().chosen
        result["after_replug_using"] = chosen.name if chosen else None
        result["replug_noticed"] = any("Microphone connected" in e[1] for e in events)
        result["same_as_before"] = chosen is not None and chosen.name == before
        print(f"   After plugging back in NEXUS uses: {result['after_replug_using'] or 'no microphone'}")
        if chosen is not None:
            input("Press Enter, then say:  What time is it?")
            try:
                heard = listener.listen(on_status=lambda s: print(f"   ({s})"))
                print(f'   heard: "{heard.original}" with {heard.microphone} - no restart needed')
                result["heard_after_replug"] = heard.original
            except (VoiceInputError, AudioInputError) as exc:
                print(f"   -> {type(exc).__name__}: {exc}")
                result["heard_after_replug_error"] = str(exc)
        result["events"] = [e[1] for e in events]
        return result
    finally:
        mics.stop_watching()


def save_wav(path: Path, clip: AudioClip) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(clip.sample_rate)
        w.writeframes(clip.pcm16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="only list microphones")
    parser.add_argument("--model", default="small", help="whisper model (default: small)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--only", default="", help="comma-separated: " + ",".join(GROUPS))
    parser.add_argument("--no-hotplug", action="store_true", help="skip the unplug/re-plug check")
    parser.add_argument("--save-audio", action="store_true", help="keep the recordings in var/eval/audio/")
    parser.add_argument("--replay", default="", help="re-run recordings saved with --save-audio (no microphone)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    plat = nexus_platform.current()
    mics = MicrophoneManager(getattr(plat, "audio_inputs", None) or PortAudioCapture().list_inputs)
    if not args.replay:
        print("Microphones (found automatically):")
        for line in mics.describe():
            print(line)
        if args.list:
            return 0
        if mics.scan().chosen is None:
            print("\nNo usable microphone - connect one and run this again (no restart of anything needed).")
            return 1

    print("\nReading the installed apps (read-only) so the 'would do' column is accurate...")
    apps = AppService(plat)
    router = FastPathRouter(apps=apps)
    voice = load_voice_settings(PROJECT_ROOT / "config" / "nexus.toml")
    rm = ResourceManager()
    stt = WhisperSTT(args.model, ram_mb=600, threads=args.threads, languages=voice.languages)
    listener = VoiceInput(mics, PortAudioCapture(), HybridSTT([STTOption(stt, stt.spec, 1)], rm,
                                                             lambda: SelectionContext()),
                          rm, permission=getattr(plat, "microphone_allowed", None), min_confidence=0.0,
                          fast_path=lambda text: router.route(text) is not None,
                          names=lambda: [a for r in apps.catalog().records for a in r.aliases])
    system = {"ram_free_before_mb": psutil.virtual_memory().available // 2**20, "vram_used_before_mb": vram_used(),
              "model": stt.info.name}
    print(f"Before loading the speech model: {system['ram_free_before_mb']} MB RAM free, "
          f"VRAM used {system['vram_used_before_mb']} MB.")
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    audio_dir = PROJECT_ROOT / "var" / "eval" / "audio" / stamp
    results, hotplug = [], None
    prompts = [p for p in PROMPTS if not wanted or p[0] in wanted]
    if args.replay:
        index = json.loads((Path(args.replay) / "index.json").read_text(encoding="utf-8"))
        prompts = [(e["group"], e["prompt"], e["reference"], e["expected"], Path(args.replay) / e["file"])
                   for e in index if not wanted or e["group"] in wanted]
    else:
        print("\nNothing you say is executed. Speak normally; NEXUS stops listening at your pause.")
    try:
        if args.replay:  # live runs load the model while you speak; replay loads it first
            with rm.use(stt.spec.name):
                system["model_load_s"] = round(stt.last_load_seconds or 0, 2)
                system["vram_used_loaded_mb"] = vram_used()
        for n, item in enumerate(prompts, 1):
            group, prompt, reference, expected = item[:4]
            clip = None
            try:
                if args.replay:
                    print(f"\n[{group}] {prompt}")
                    with wave.open(str(item[4]), "rb") as w:
                        clip = AudioClip(w.readframes(w.getnframes()), w.getframerate())
                    t0 = time.perf_counter()
                    with rm.use(stt.spec.name):
                        transcript = stt.transcribe(clip)
                    listener.last_timings = {"transcribing_s": time.perf_counter() - t0, "audio_s": clip.seconds}
                    original = listener._accept(transcript)
                    u = listener.normalizer(original)
                    heard = Heard(u.text, original, transcript, "(saved recording)", 0.0, guessed=u.guessed,
                                  changes=u.changes)
                else:
                    input(f"\n[{group}] Press Enter, then say:  {prompt}")
                    heard = listener.listen(on_status=lambda s: print(f"   ({s})"))
                    clip = listener.last_utterance.clip if listener.last_utterance else None
            except (VoiceInputError, AudioInputError) as exc:
                print(f"   -> {type(exc).__name__}: {exc}")
                results.append({"group": group, "reference": reference, "expected": expected, "error": str(exc),
                                "error_kind": type(exc).__name__})
                continue
            if "model_load_s" not in system and stt.last_load_seconds is not None:
                system["model_load_s"] = round(stt.last_load_seconds, 2)
                system["vram_used_loaded_mb"] = vram_used()
            described, got = decide(router, heard, expected, voice.confirm_min_confidence)
            extra = show(heard, described, listener, clip, stt)
            row = {"group": group, "reference": reference, "expected": expected, "heard": heard.original,
                   "acted": heard.text, "changes": list(heard.changes), "guessed": heard.guessed,
                   "language": heard.transcript.language, "language_probability": heard.transcript.language_probability,
                   "confidence": heard.confidence, "would_do": described, "got": got,
                   "understood": got is not None and got.split(" ")[0] == expected,
                   "word_accuracy": round(word_accuracy(heard.original, reference), 3),
                   "transcribing_s": round(listener.last_timings.get("transcribing_s", 0), 3),
                   "audio_s": round(listener.last_timings.get("audio_s", 0), 2), **extra,
                   "microphone": heard.microphone, "model": stt.info.name, "worker_ram_mb": stt.memory_mb()}
            if args.save_audio and clip is not None and not args.replay:
                name = f"{n:02d}_{group}.wav"
                save_wav(audio_dir / name, clip)
                row["audio_file"] = str(audio_dir / name)
            results.append(row)
        if not args.no_hotplug and not args.replay:
            hotplug = hotplug_check(mics, listener, float(voice.device_poll_seconds or 5))
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        pid = stt.pid
        system["worker_ram_peak_mb"] = max((r.get("worker_ram_mb", 0) for r in results), default=0)
        t0 = time.perf_counter()
        rm.unload_all()
        stt.unload()
        system["unload_s"] = round(time.perf_counter() - t0, 3)
        time.sleep(1.0)
        system["worker_gone_after_unload"] = pid is None or not psutil.pid_exists(pid)
        system["ram_free_after_mb"] = psutil.virtual_memory().available // 2**20
        system["vram_used_after_mb"] = vram_used()

    if args.save_audio and not args.replay:
        index = [{"file": Path(r["audio_file"]).name, "group": r["group"], "reference": r["reference"],
                  "expected": r["expected"], "prompt": next(p[1] for p in PROMPTS if p[2] == r["reference"])}
                 for r in results if r.get("audio_file")]
        (audio_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nRecordings saved in {audio_dir} (replay: --replay {audio_dir})")

    done = [r for r in results if "heard" in r]
    print("\n=== Summary ===")
    for group in GROUPS:
        rows = [r for r in results if r["group"] == group]
        if rows:
            ok = sum(r.get("understood", False) for r in rows)
            acc = statistics.mean(r.get("word_accuracy", 0) for r in rows)
            print(f"  {group:<9} understood {ok}/{len(rows)}   words right {acc:.0%}")
    if done:
        print(f"  median latency after you stop speaking: "
              f"{statistics.median(r['transcribing_s'] for r in done):.2f} s "
              f"(max {max(r['transcribing_s'] for r in done):.2f} s)")
    print(f"  model load {system.get('model_load_s')} s, speech model RAM up to {system['worker_ram_peak_mb']} MB, "
          f"VRAM {system['vram_used_before_mb']} -> {system.get('vram_used_loaded_mb')} -> "
          f"{system['vram_used_after_mb']} MB, unloaded in {system['unload_s']} s "
          f"(process gone: {system['worker_gone_after_unload']})")
    out = PROJECT_ROOT / "var" / "eval" / "voice_check.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    at = datetime.now(timezone.utc).isoformat()
    with out.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"at": at, "replay": bool(args.replay), **r}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"at": at, "system": system, "hotplug": hotplug}, ensure_ascii=False) + "\n")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
