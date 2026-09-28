"""Measure speech-to-text cost on this PC: load time, latency, RAM/VRAM before/during/after, CPU.

    .venv\\Scripts\\python.exe scripts\\measure_stt.py             # base and small, CPU int8
    .venv\\Scripts\\python.exe scripts\\measure_stt.py small:6 base:2

Speech is synthesised offline with Windows' own voices (no microphone, no
speakers, no loopback). Results are printed and appended to var/eval/stt_measure.jsonl.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

from nexus.ai.voice.base import AudioClip  # noqa: E402
from nexus.ai.voice.whisper import WhisperSTT  # noqa: E402
from nexus.core.config import PROJECT_ROOT  # noqa: E402
from nexus.system import info  # noqa: E402
from tests.voice_fixtures import read_wav, synthesize  # noqa: E402

PHRASES = {"time": "What time is it?", "folder": "Open my downloads folder.", "yes": "Yes.",
           "remember": "Remember that my favourite game is Rocket League.",
           "long": "Nexus, please remember that my favourite game is Rocket League and that I usually play "
                   "in the evening after college, and remind me to update my graphics driver tomorrow."}


def vram_used() -> int | None:
    gpus = info.gpus()
    return int(gpus[0]["vram_used_mb"]) if gpus and gpus[0].get("vram_used_mb") is not None else None


def ram_free() -> int:
    return psutil.virtual_memory().available // 2**20


def measure(model: str, threads: int, clips: dict[str, AudioClip]) -> dict:
    stt = WhisperSTT(model, ram_mb=1, threads=threads)
    before = {"ram_free_mb": ram_free(), "vram_used_mb": vram_used()}
    started = time.perf_counter()
    stt.load()
    load_s = time.perf_counter() - started
    loaded = {"worker_ram_mb": stt.memory_mb(), "ram_free_mb": ram_free(), "vram_used_mb": vram_used()}
    root = psutil.Process(stt.pid)
    procs = [root, *root.children(recursive=True)]
    peak, cpu, stop = [loaded["worker_ram_mb"]], [], threading.Event()

    def sample() -> None:
        for p in procs:
            p.cpu_percent(None)
        while not stop.wait(0.1):
            try:
                peak.append(sum(p.memory_info().rss for p in procs) // 2**20)
                cpu.append(sum(p.cpu_percent(None) for p in procs))
            except psutil.Error:
                pass

    sampler = threading.Thread(target=sample)
    sampler.start()
    latencies, texts = {}, {}
    for name, clip in clips.items():
        t = time.perf_counter()
        texts[name] = stt.transcribe(clip).text
        latencies[name] = round(time.perf_counter() - t, 3)
    stop.set()
    sampler.join()
    idle_cpu = []
    for p in procs:
        p.cpu_percent(None)
    time.sleep(1.0)
    try:
        idle_cpu.append(sum(p.cpu_percent(None) for p in procs))
    except psutil.Error:
        pass
    pid = stt.pid
    t = time.perf_counter()
    stt.unload()
    unload_s = time.perf_counter() - t
    time.sleep(1.0)
    after = {"ram_free_mb": ram_free(), "vram_used_mb": vram_used(), "worker_alive": psutil.pid_exists(pid)}
    short = [v for k, v in latencies.items() if k != "long"]
    return {"model": model, "threads": threads, "load_s": round(load_s, 2), "unload_s": round(unload_s, 2),
            "before": before, "loaded": loaded, "peak_worker_ram_mb": max(peak),
            "cpu_percent_while_transcribing": round(statistics.mean(cpu)) if cpu else None,
            "cpu_percent_loaded_idle": idle_cpu[0] if idle_cpu else None,
            "latency_short_median_s": statistics.median(short), "latency_short_max_s": max(short),
            "latency_long_s": latencies.get("long"), "long_audio_s": round(clips["long"].seconds, 1),
            "after_unload": after, "texts": texts}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    targets = [(t.split(":")[0], int(t.split(":")[1]) if ":" in t else 4) for t in sys.argv[1:]] or \
        [("base", 2), ("base", 4), ("small", 6), ("small", 4)]
    with tempfile.TemporaryDirectory() as tmp:
        wavs = synthesize(PHRASES, tmp, voices=("Microsoft Zira Desktop",))
        clips = {name: read_wav(path) for (_, name), path in wavs.items()}
    results = []
    for model, threads in targets:
        r = measure(model, threads, clips)
        results.append(r)
        print(f"\n{model} ({threads} threads): load {r['load_s']} s, unload {r['unload_s']} s")
        print(f"  RAM: worker {r['loaded']['worker_ram_mb']} MB loaded, {r['peak_worker_ram_mb']} MB peak; "
              f"free {r['before']['ram_free_mb']} -> {r['loaded']['ram_free_mb']} -> "
              f"{r['after_unload']['ram_free_mb']} MB after unload")
        print(f"  VRAM used: {r['before']['vram_used_mb']} -> {r['loaded']['vram_used_mb']} -> "
              f"{r['after_unload']['vram_used_mb']} MB")
        print(f"  latency: {r['latency_short_median_s']} s median / {r['latency_short_max_s']} s max for 1-4 s "
              f"phrases, {r['latency_long_s']} s for {r['long_audio_s']} s of speech")
        print(f"  CPU: {r['cpu_percent_while_transcribing']}% of one core while transcribing, "
              f"{r['cpu_percent_loaded_idle']}% loaded and idle; worker gone after unload: "
              f"{not r['after_unload']['worker_alive']}")
    out = PROJECT_ROOT / "var" / "eval" / "stt_measure.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **r}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
