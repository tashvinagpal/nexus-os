"""Text-to-speech worker: `python -m nexus.ai.voice.kokoro_worker --model <file> --voices <file> ...`

Kokoro-82M (ONNX) in its own process, so unloading returns all of its memory
and a crash can't take NEXUS down. Protocol (nexus/ai/voice/worker_process.py):
one JSON line per request, one JSON line per reply followed by `bytes` bytes
of mono 16-bit PCM. Nothing is downloaded and no text is logged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kokoro_worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--voices", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    args = parser.parse_args(argv)

    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)  # anything a library prints goes to the log, never into the protocol
    sys.stdout = sys.stderr
    requests = sys.stdin.buffer

    def send(message: dict, payload: bytes = b"") -> None:
        if payload:
            message = {**message, "bytes": len(payload)}
        out.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8") + payload)

    started = time.perf_counter()
    try:
        import numpy as np
        import onnxruntime as rt
        import phonemizer
        from kokoro_onnx import Kokoro

        options = rt.SessionOptions()
        options.intra_op_num_threads = max(1, args.threads)
        options.inter_op_num_threads = 1
        available = rt.get_available_providers()
        providers = ["CPUExecutionProvider"]
        note = ""
        if args.device == "gpu":
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                note = "GPU asked for, but this onnxruntime has no CUDA support: using the CPU"
        session = rt.InferenceSession(args.model, sess_options=options, providers=providers)
        kokoro = Kokoro.from_session(session, args.voices)
        voices = sorted(kokoro.get_voices())
        kokoro.create("Ready.", voice=voices[0], lang="en-us")  # warm up before saying "ready"
    except Exception as exc:
        send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    send({"ok": True, "event": "ready", "load_seconds": round(time.perf_counter() - started, 3),
          "voices": voices, "providers": session.get_providers(), "note": note})

    while True:
        line = requests.readline()
        if not line:
            return 0
        try:
            req = json.loads(line)
        except ValueError:
            send({"ok": False, "error": "bad request"})
            continue
        op = req.get("op")
        if op == "quit":
            return 0
        if op == "ping":
            send({"ok": True})
            continue
        if op != "synthesize":
            send({"ok": False, "error": f"unknown request {op!r}"})
            continue
        try:
            t0 = time.perf_counter()
            text, lang = str(req["text"]), req.get("lang", "en-us")
            if lang == "hi":
                # espeak reads English words inside Hindi the English way ("GTA", "Chrome") but marks them
                # "(en)...(hi)"; Kokoro would say those marks, so they are removed here.
                phonemes = phonemizer.phonemize(text, "hi", preserve_punctuation=True, with_stress=True,
                                                language_switch="remove-flags")
                samples, rate = kokoro.create(phonemes, voice=req["voice"], speed=float(req.get("speed", 1.0)),
                                              is_phonemes=True)
            else:
                samples, rate = kokoro.create(text, voice=req["voice"], speed=float(req.get("speed", 1.0)), lang=lang)
            volume = min(1.0, max(0.0, float(req.get("volume", 1.0))))
            pcm = (np.clip(samples * volume, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            send({"ok": True, "rate": int(rate), "seconds": round(time.perf_counter() - t0, 3),
                  "audio_seconds": round(len(samples) / rate, 3)}, pcm)
        except Exception as exc:
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    sys.exit(main())
