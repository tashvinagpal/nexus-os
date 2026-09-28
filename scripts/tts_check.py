"""Check voice output on this PC's real audio output. NOTHING IS EXECUTED - NEXUS only speaks.

    .venv\\Scripts\\python.exe scripts\\tts_check.py            # you will hear a few short sentences
    .venv\\Scripts\\python.exe scripts\\tts_check.py --list     # only show the output devices
    .venv\\Scripts\\python.exe scripts\\tts_check.py --silent   # everything except playing (fake output)

Reports: output devices and the one used, model load time, synthesis time,
time to first sound, playback start, stop latency, RAM/VRAM/CPU, unload.
Results are appended to var/eval/tts_check.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

from nexus.ai.voice.kokoro import KokoroTTS  # noqa: E402
from nexus.ai.voice.speaker import Speaker, SpeechSettings  # noqa: E402
from nexus.core.config import PROJECT_ROOT, load_tts_settings  # noqa: E402
from nexus.core.resources import ResourceManager  # noqa: E402
from nexus.system import info  # noqa: E402
from nexus.system import platform as nexus_platform  # noqa: E402
from nexus.system.audio.playback import PortAudioSink  # noqa: E402

PHRASES = [
    ("en", "Opening Google Chrome."),
    ("en", "It's 3:42 PM."),
    ("en", "I found Outlook classic and Outlook. Which one do you want? I can open either of them for you, "
           "or neither if you changed your mind."),
    ("hi", "क्रोम खुल गया है। अभी तीन बजकर बयालीस मिनट हुए हैं।"),
    ("hinglish", "Haan bhai, ek second. GTA chala raha hoon."),
    ("pa", "ਮੈਂ ਹਾਲੇ ਕੁਝ ਨਹੀਂ ਕੀਤਾ।"),
]


def vram_used() -> int | None:
    gpus = info.gpus()
    return int(gpus[0]["vram_used_mb"]) if gpus and gpus[0].get("vram_used_mb") is not None else None


class Sampler:
    """Peak RAM and CPU of the voice process while it works."""

    def __init__(self, tts: KokoroTTS) -> None:
        self.tts, self.peak_mb, self.cpu, self._stop = tts, 0, [], threading.Event()

    def __enter__(self):
        def run() -> None:
            procs = []
            while not self._stop.wait(0.1):
                pid = self.tts.pid
                if pid is None:
                    continue
                try:
                    if not procs:
                        root = psutil.Process(pid)
                        procs = [root, *root.children(recursive=True)]
                        for p in procs:
                            p.cpu_percent(None)
                        continue
                    self.peak_mb = max(self.peak_mb, sum(p.memory_info().rss for p in procs) // 2**20)
                    self.cpu.append(sum(p.cpu_percent(None) for p in procs))
                except psutil.Error:
                    procs = []

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--silent", action="store_true", help="don't play (fake output)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    plat = nexus_platform.current()
    outputs = getattr(plat, "audio_outputs", lambda: [])()
    print("Audio outputs (found automatically; microphones are never listed here):")
    for o in outputs:
        mark = "*" if o.default and o.state == "active" else ("+" if o.state == "active" else "-")
        print(f"  {mark} {o.name}  [{o.form_factor}, {o.bus}, {o.state}]" + ("  (Windows default)" if o.default else ""))
    if args.list:
        return 0
    settings = load_tts_settings(PROJECT_ROOT / "config" / "nexus.toml")
    if args.silent:
        from tests.tts_fixtures import FakeSink

        sink = FakeSink(device="(silent - nothing played)", instant=False, play_seconds=0.05)
    else:
        sink = PortAudioSink(getattr(plat, "audio_outputs", None),
                             prepare_thread=getattr(plat, "prepare_audio_thread", None))
    ready, device = sink.ready()
    print(f"\nNEXUS speaks through: {device if ready else 'NOTHING - ' + device}")
    if not ready:
        print("Connect speakers or headphones and run this again (no restart of anything needed).")
        return 1

    rm = ResourceManager()
    tts = KokoroTTS(settings.model, voices={"en": settings.voice, "hi": settings.hindi_voice}, speed=settings.speed,
                    volume=settings.volume, threads=settings.threads or 4, ram_mb=settings.ram_mb)
    speaker = Speaker(tts, rm, sink=sink, settings=SpeechSettings(romanized=settings.romanized))
    report: dict = {"device": device, "model": tts.info.model, "idle_nexus_ram_mb":
                    psutil.Process().memory_info().rss // 2**20, "vram_before_mb": vram_used(),
                    "ram_free_before_mb": psutil.virtual_memory().available // 2**20, "phrases": []}
    print(f"Before: {report['ram_free_before_mb']} MB RAM free, VRAM used {report['vram_before_mb']} MB; "
          f"this process {report['idle_nexus_ram_mb']} MB\n")
    with Sampler(tts) as sampler:
        for n, (language, text) in enumerate(PHRASES):
            ok = speaker.say(text, language=language)
            last = dict(speaker.last)
            if n == 0:
                report["vram_loaded_mb"] = vram_used()
            row = {"language": language, "characters": len(text), "spoken": ok, **last,
                   "synthesis": dict(tts.last_synthesis)}
            report["phrases"].append(row)
            cold = f", model load {last['load_s']} s" if last.get("load_s") else ""
            print(f"  [{language:<8}] {'spoke' if ok else 'NOT spoken: ' + speaker.problem}: "
                  f"first sound after {last.get('first_audio_s')} s{cold}; first sentence synthesized in "
                  f"{last.get('first_synthesis_s')} s; {last.get('chunks')} chunk(s); total {last.get('total_s')} s")
        # stop: start a long reply, stop it after a second
        speaker.say_async("This is a long answer that NEXUS will be interrupted in the middle of. " * 3)
        deadline = time.time() + 10
        while speaker.state != "playing" and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(1.0)
        t = time.perf_counter()
        stopped = speaker.stop()
        report["stop_s"] = round(time.perf_counter() - t, 4)
        speaker.wait(10)
        print(f"\n  stop: {'stopped' if stopped else 'was not speaking'}; silent after {report['stop_s'] * 1000:.1f} ms")
        speaker.say("And I can speak again right after.")
        report["speak_after_stop_s"] = speaker.last.get("first_audio_s")
    report["worker_peak_ram_mb"] = sampler.peak_mb
    report["cpu_percent_while_speaking"] = round(statistics.mean(sampler.cpu)) if sampler.cpu else None
    time.sleep(1.5)
    report["cpu_percent_loaded_idle"] = None
    pid = tts.pid
    if pid:
        procs = [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]
        for p in procs:
            p.cpu_percent(None)
        time.sleep(1.0)
        report["cpu_percent_loaded_idle"] = sum(p.cpu_percent(None) for p in procs)
    t = time.perf_counter()
    tts.unload()
    report["unload_s"] = round(time.perf_counter() - t, 3)
    time.sleep(1.0)
    report["process_gone"] = pid is None or not psutil.pid_exists(pid)
    report["vram_after_mb"] = vram_used()
    report["ram_free_after_mb"] = psutil.virtual_memory().available // 2**20

    first = [r["first_audio_s"] for r in report["phrases"][1:] if r.get("first_audio_s")]
    print("\n=== Summary ===")
    print(f"  output: {device}")
    print(f"  model load (cold): {report['phrases'][0].get('load_s')} s; first sound on the very first reply: "
          f"{report['phrases'][0].get('first_audio_s')} s")
    if first:
        print(f"  first sound, model loaded: median {statistics.median(first):.2f} s (min {min(first):.2f}, "
              f"max {max(first):.2f})")
    print(f"  voice process RAM peak {report['worker_peak_ram_mb']} MB; CPU {report['cpu_percent_while_speaking']}% "
          f"of one core while speaking, {report['cpu_percent_loaded_idle']}% loaded and idle")
    print(f"  VRAM {report['vram_before_mb']} -> {report.get('vram_loaded_mb')} -> {report['vram_after_mb']} MB")
    print(f"  stop: silent after {report['stop_s'] * 1000:.1f} ms; speaking again: first sound after "
          f"{report['speak_after_stop_s']} s")
    print(f"  unload {report['unload_s']} s, process gone: {report['process_gone']}; RAM free "
          f"{report['ram_free_before_mb']} -> {report['ram_free_after_mb']} MB")
    out = PROJECT_ROOT / "var" / "eval" / "tts_check.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "silent": args.silent, **report},
                           ensure_ascii=False) + "\n")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
