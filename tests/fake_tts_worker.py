"""A stand-in for nexus.ai.voice.kokoro_worker speaking the same protocol (no model, no audio library).

    python fake_tts_worker.py <mode>
      ok         answers with 10 ms of 24 kHz audio per character
      slow       waits 30 s before answering (timeouts)
      crash      exits when asked to synthesize
      error      answers every synthesis with an error
      fail_load  reports a load error
"""

import json
import os
import sys
import time


def main() -> int:
    mode = sys.argv[1]
    out = os.fdopen(os.dup(1), "wb", buffering=0)

    def send(message, payload=b""):
        if payload:
            message = {**message, "bytes": len(payload)}
        out.write((json.dumps(message) + "\n").encode() + payload)

    if mode == "fail_load":
        send({"ok": False, "error": "NoSuchFile: kokoro-v1.0.onnx is damaged"})
        return 1
    send({"ok": True, "event": "ready", "load_seconds": 0.01, "voices": ["af_heart", "hf_alpha"],
          "providers": ["CPUExecutionProvider"]})
    for line in sys.stdin.buffer:
        req = json.loads(line)
        if req["op"] == "quit":
            return 0
        if req["op"] == "ping":
            send({"ok": True})
            continue
        if mode == "crash":
            return 3
        if mode == "slow":
            time.sleep(30)
        if mode == "error":
            send({"ok": False, "error": f"ValueError: Voice {req['voice']} not found"})
            continue
        send({"ok": True, "rate": 24000, "seconds": 0.01, "audio_seconds": len(req["text"]) / 100,
              "echo": {"lang": req["lang"], "voice": req["voice"], "speed": req["speed"]}},
             b"\x01\x00" * 240 * len(req["text"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
