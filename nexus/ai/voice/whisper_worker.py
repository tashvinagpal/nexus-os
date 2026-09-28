"""Speech-to-text worker: `python -m nexus.ai.voice.whisper_worker --model <folder> ...`

faster-whisper runs in its own process so that
  - unloading really returns all of its memory (model *and* libraries),
  - a stuck transcription can be stopped (the process is ended),
  - a crash in native code can't take NEXUS down.

Protocol over the pipes: the parent writes one JSON line per request - for
"transcribe" followed by `bytes` bytes of mono PCM16 - and the worker answers
with one JSON line. The model is loaded from a local folder only; nothing is
downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from importlib.util import find_spec
from pathlib import Path


def _read_exact(stream, n: int) -> bytes:
    chunks, left = [], n
    while left > 0:
        chunk = stream.read(left)
        if not chunk:
            raise EOFError("pipe closed")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def _cuda_dll_dirs() -> None:
    """GPU only: NVIDIA's pip packages (cuBLAS/cuDNN) keep their DLLs in their own folders."""
    for package in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
        try:
            spec = find_spec(package)
        except ModuleNotFoundError:
            continue
        for location in (spec.submodule_search_locations or []) if spec else []:
            for sub in ("bin", "lib"):
                folder = Path(location) / sub
                if folder.is_dir() and hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(folder))
                    os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")


def resample(np, audio, src: int, dst: int = 16000):
    if src == dst:
        return audio
    if src > dst:  # low-pass below the new Nyquist frequency first, so nothing aliases
        taps = 63
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * (0.45 * dst / src) * n) * np.hamming(taps)
        audio = np.convolve(audio, h / h.sum(), mode="same")
    positions = np.arange(int(len(audio) * dst / src)) * (src / dst)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


class EncoderCache:
    """Remembers the last encoder output so language detection and transcription share one pass.

    Whisper's encoder always processes a 30-second window, so detecting the language first used to
    double the time. Detection is run on exactly the features transcription will encode; if a
    future faster-whisper builds them differently this just misses (correct, only slower).
    """

    def __init__(self, encode) -> None:
        self._encode = encode
        self._key: bytes | None = None
        self._value = None
        self.hits = 0

    def __call__(self, features):
        digest = hashlib.blake2b(features.tobytes(), digest_size=16)
        digest.update(repr(features.shape).encode())
        key = digest.digest()
        if key == self._key:
            self.hits += 1
            return self._value
        self._value, self._key = self._encode(features), key
        return self._value

    def clear(self) -> None:
        self._key = self._value = None


# Decoding limits. Whisper retries a doubtful decode at higher temperatures and can loop on
# repeated words up to 448 tokens; both made real Hindi utterances take 10-23 s. Three
# temperatures and a length cap that fits the speech (Gurmukhi needs ~23 tokens/s, Hindi ~12,
# English ~3) keep the worst case short without cutting real sentences.
TEMPERATURES = (0.0, 0.3)  # greedy, then at most one retry
UNSURE_LANGUAGE = 0.5  # below this, the two most likely languages are both tried


def max_tokens(speech_seconds: float) -> int:
    return min(440, int(32 + 32 * speech_seconds))


def _decode(model, speech, language: str | None, req: dict, *, retry: bool = True) -> dict:
    started = time.perf_counter()
    segments, info = model.transcribe(
        speech, language=language, task="transcribe", beam_size=int(req.get("beam_size", 1)),
        vad_filter=False, condition_on_previous_text=False, without_timestamps=True,
        initial_prompt=req.get("prompt") or None, temperature=TEMPERATURES if retry else 0.0,
        max_new_tokens=max_tokens(len(speech) / 16000))
    segments = list(segments)
    weights = [max(s.end - s.start, 0.01) for s in segments]
    logprob = (sum(s.avg_logprob * w for s, w in zip(segments, weights)) / sum(weights)) if segments else None
    return {
        "text": " ".join(s.text.strip() for s in segments).strip(),
        "language": info.language,
        "logprob": logprob,
        "confidence": math.exp(logprob) if logprob is not None else None,
        "no_speech_probability": max((s.no_speech_prob for s in segments), default=None),
        "compression_ratio": max((s.compression_ratio for s in segments), default=None),
        "temperature": max((s.temperature or 0.0 for s in segments), default=0.0),
        "segments": len(segments),
        "decode_seconds": round(time.perf_counter() - started, 3),
    }


def speech_only(np, audio):
    """Silero voice-activity detection: the speech parts of the audio, or None if there is none."""
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

    chunks = get_speech_timestamps(audio, VadOptions())
    if not chunks:
        return None
    pieces, _ = collect_chunks(audio, chunks)
    return np.concatenate(pieces, axis=0)


def transcribe(model, np, pcm: bytes, req: dict, *, vad=speech_only) -> dict:
    started = time.perf_counter()
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio = resample(np, audio, int(req.get("rate", 16000)))
    result = {"text": "", "language": None, "language_probability": None, "confidence": None,
              "no_speech_probability": None, "compression_ratio": None, "segments": 0,
              "audio_seconds": round(len(audio) / 16000, 3), "speech_seconds": 0.0, "shared_encoder": False,
              "tried_languages": []}
    # Only the speech goes to Whisper: no speech -> no words, nothing guessed from noise.
    speech = vad(np, audio)
    if speech is None:
        return {**result, "seconds": round(time.perf_counter() - started, 3)}
    cache = model.encode if isinstance(model.encode, EncoderCache) else None
    hits_before = cache.hits if cache else 0
    language, language_probability = req.get("language"), None
    candidates = [language]
    allowed = [lang for lang in req.get("languages") or [] if isinstance(lang, str)]
    if language is None and allowed:
        # Whisper's detector knows ~100 languages; only the configured ones count. Spoken Hindi that
        # it labels Urdu (the same spoken language, another script) counts as Hindi.
        features = model.feature_extractor(speech)[..., :-1]  # what transcribe() will encode first
        _, _, probabilities = model.detect_language(features=features)
        scores = dict(probabilities)
        if "hi" in allowed and "ur" not in allowed:
            scores["hi"] = scores.get("hi", 0.0) + scores.get("ur", 0.0)
        ranked = sorted(allowed, key=lambda lang: -scores.get(lang, 0.0))
        language, language_probability = ranked[0], scores.get(ranked[0], 0.0)
        # Unsure (Hinglish, Haryanvi, Punjabi said like Hindi): try the runner-up too. Decoding Hindi
        # speech as English *translates* it ("हाँ, कर दे" -> "I'll do it."), so the words the model is
        # surer of win. The encoder pass is shared, so this costs one extra decode only when unsure.
        candidates = ranked[:2] if language_probability < UNSURE_LANGUAGE else ranked[:1]
    try:
        # The runner-up only needs a plain greedy decode to show whether it is surer: no retries.
        decodes = [_decode(model, speech, lang, req, retry=i == 0) for i, lang in enumerate(candidates)]
    finally:
        if cache is not None:
            result["shared_encoder"] = cache.hits > hits_before
            cache.clear()
    best = max(decodes, key=lambda d: d["logprob"] if d["logprob"] is not None else -99.0)
    if best is not decodes[0] and language_probability is not None:
        language_probability = None  # the detector preferred the other language; report the decode's
    return {
        **result,
        **{k: v for k, v in best.items() if k != "logprob"},
        "language_probability": language_probability,
        "tried_languages": [{"language": d["language"], "text": d["text"],
                             "confidence": d["confidence"], "seconds": d["decode_seconds"]} for d in decodes],
        "speech_seconds": round(len(speech) / 16000, 3),
        "seconds": round(time.perf_counter() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whisper_worker")
    parser.add_argument("--model", required=True, help="folder with a CTranslate2 Whisper model")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)  # anything a library prints goes to the log, never into the protocol
    sys.stdout = sys.stderr
    requests = sys.stdin.buffer

    def send(message: dict) -> None:
        out.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))

    os.environ["HF_HUB_OFFLINE"] = "1"
    started = time.perf_counter()
    try:
        if args.device == "cuda":
            _cuda_dll_dirs()
        import numpy as np
        from faster_whisper import WhisperModel

        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type,
                             cpu_threads=args.threads, num_workers=1, local_files_only=True)
        model.encode = EncoderCache(model.encode)
        # Warm up (and on a GPU prove the CUDA libraries really load) before saying "ready".
        model.detect_language(np.zeros(16000, dtype=np.float32))
    except Exception as exc:
        send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    send({"ok": True, "event": "ready", "load_seconds": round(time.perf_counter() - started, 3)})

    while True:
        line = requests.readline()
        if not line:
            return 0
        try:
            req = json.loads(line)
        except ValueError:
            send({"ok": False, "error": "bad request"})
            continue
        if req.get("op") == "quit":
            return 0
        if req.get("op") != "transcribe":
            send({"ok": False, "error": f"unknown request {req.get('op')!r}"})
            continue
        try:
            pcm = _read_exact(requests, int(req["bytes"]))
        except EOFError:
            return 0
        try:
            send({"ok": True, **transcribe(model, np, pcm, req)})
        except Exception as exc:
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    sys.exit(main())
