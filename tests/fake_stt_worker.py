"""A stand-in for nexus.ai.voice.whisper_worker that speaks the same protocol (no model).

    python fake_stt_worker.py <mode>
      ok         answers every request with a transcript ("heard N bytes")
      slow       waits 30 s before answering (timeout/cancel tests)
      crash      exits when asked to transcribe
      fail_load  reports a load error (like missing CUDA libraries)
      hang_load  never says it is ready
"""

import json
import os
import sys
import time


def main() -> int:
    mode = sys.argv[1]
    out = os.fdopen(os.dup(1), "wb", buffering=0)

    def send(message):
        out.write((json.dumps(message) + "\n").encode())

    if mode == "fail_load":
        send({"ok": False, "error": "RuntimeError: Library cublas64_12.dll is not found"})
        return 1
    if mode == "hang_load":
        time.sleep(60)
        return 1
    send({"ok": True, "event": "ready", "load_seconds": 0.01})
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        req = json.loads(line)
        if req["op"] == "quit":
            return 0
        pcm = stdin.read(req["bytes"])
        if mode == "crash":
            return 3
        if mode == "slow":
            time.sleep(30)
        send({"ok": True, "text": f"heard {len(pcm)} bytes", "language": req.get("language") or "en",
              "language_probability": 0.9, "confidence": 0.8, "no_speech_probability": 0.01,
              "audio_seconds": len(pcm) / 2 / req.get("rate", 16000), "seconds": 0.01,
              "echo_languages": req.get("languages")})


if __name__ == "__main__":
    sys.exit(main())
