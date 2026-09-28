"""openWakeWord inference on onnxruntime alone: no scipy, scikit-learn or TFLite.

An openWakeWord model is three small ONNX networks run on 80 ms chunks of 16 kHz audio:

    audio (1280 samples) -> melspectrogram.onnx -> 32 mel bands per 10 ms
    last 76 mel frames   -> embedding_model.onnx (Google's speech_embedding) -> one 96-value frame
    last 16 embeddings   -> <wake word>.onnx -> score 0..1

The first two are shared by every wake word; only the last one is trained for
"Nexus". This file mirrors openwakeword.utils.AudioFeatures (v0.6) exactly -
tests/test_wake_oww.py checks the scores against the openwakeword package - so any
model made with openWakeWord's own training pipeline works here unchanged, and
so does one made by scripts/train_wake_model.py.

onnxruntime and numpy are already installed with [voice]; the three model files
(about 3.5 MB) come from scripts/get_wake_model.py.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

CHUNK = 1280          # samples per step (80 ms at 16 kHz)
_MEL_CONTEXT = 480    # extra samples the mel network needs before a chunk (3 hops of 160)
_WINDOW = 76          # mel frames per embedding
_MEL_KEEP = 970       # ~10 s of mel frames
_FEATURES_KEEP = 120  # ~10 s of embeddings


def _session(path: Path, threads: int) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class WakeFeatures:
    """Streaming speech embeddings: push 80 ms chunks, read the latest frames."""

    def __init__(self, melspec: Path, embedding: Path, *, threads: int = 1) -> None:
        import numpy as np

        self._np = np
        self._mel = _session(melspec, threads)
        self._embed = _session(embedding, threads)
        self._mel_input = self._mel.get_inputs()[0].name
        self._embed_input = self._embed.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        np = self._np
        self._raw: deque[int] = deque(maxlen=16000 * 10)
        self._mels = np.ones((_WINDOW, 32), dtype=np.float32)
        # openWakeWord starts from the embeddings of 4 s of quiet random noise, so a model never sees
        # an empty history; a fixed seed keeps NEXUS deterministic.
        rng = np.random.default_rng(0)
        self._features = self.embed_clip(rng.integers(-1000, 1000, 16000 * 4).astype(np.int16))

    # --- the two shared networks ---------------------------------------------------------

    def melspectrogram(self, samples: Any) -> Any:
        np = self._np
        x = np.asarray(samples, dtype=np.float32)[None, :]
        spec = np.squeeze(self._mel.run(None, {self._mel_input: x})[0])
        return (spec / 10 + 2).astype(np.float32)  # openWakeWord's fixed scaling

    def embeddings(self, mels: Any) -> Any:
        """(n, 76, 32) mel windows -> (n, 96) embeddings."""
        out = self._embed.run(None, {self._embed_input: mels[..., None].astype(self._np.float32)})[0]
        return out.reshape(len(mels), 96)

    def embed_clip(self, samples: Any) -> Any:
        """All embeddings of a whole clip (one per 80 ms once 76 mel frames exist) - for training."""
        np = self._np
        spec = self.melspectrogram(samples)
        windows = [spec[i:i + _WINDOW] for i in range(0, spec.shape[0], 8) if i + _WINDOW <= spec.shape[0]]
        if not windows:
            return np.zeros((0, 96), dtype=np.float32)
        return self.embeddings(np.stack(windows))

    # --- streaming -------------------------------------------------------------------------

    def push(self, chunk: Any) -> None:
        """One chunk of exactly CHUNK int16 samples."""
        np = self._np
        self._raw.extend(chunk.tolist())
        recent = list(self._raw)[-(CHUNK + _MEL_CONTEXT):]
        self._mels = np.vstack((self._mels, self.melspectrogram(recent)))[-_MEL_KEEP:]
        self._features = np.vstack((self._features, self.embeddings(self._mels[None, -_WINDOW:])))[-_FEATURES_KEEP:]

    def window(self, frames: int) -> Any:
        return self._features[-frames:][None, :, :].astype(self._np.float32)


class WakeModel:
    """One openWakeWord classifier (input [1, frames, 96] -> score)."""

    def __init__(self, path: Path, *, threads: int = 1) -> None:
        self._session = _session(path, threads)
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self.frames = int(inp.shape[1])

    def score(self, window: Any) -> float:
        return float(self._session.run(None, {self._input: window})[0].reshape(-1)[0])
