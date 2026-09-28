"""Train NEXUS's own "Nexus" wake-word model (openWakeWord format) - on this PC, CPU only, no downloads.

    .venv\\Scripts\\python.exe -m pip install -e ".[tts,wake-train]"     # Kokoro (voice output) + onnx
    .venv\\Scripts\\python.exe scripts\\get_wake_model.py                  # the two shared feature networks
    .venv\\Scripts\\python.exe scripts\\train_wake_model.py                # ~15-25 min on a 6-core CPU
    .venv\\Scripts\\python.exe scripts\\train_wake_model.py --samples var\\wake\\samples   # + your own voice

How it works (all local):
  1. Speech is generated with Kokoro, the voice NEXUS already speaks with: "Nexus" said by all 54
     voices (English, Hindi, and other accents saying the English word) plus blends of two voices,
     which sound like new people - at different speeds, the Indian pronunciation too. The same
     speakers also say ordinary sentences (English, Hinglish, Hindi) and near-misses like "next",
     "Lexus", "Texas", "necklace" - what must NOT wake NEXUS (evals/wake_phrases.json).
  2. Each clip is made to sound like a real room: reverb, microphone colour, fan/hum/babble noise,
     loudness from whisper to close-up. "Nexus" is placed right before other words ("Nexus, GTA
     chala de") as well as on its own.
  3. openWakeWord's shared networks turn audio into speech embeddings (nexus/ai/voice/oww.py); a
     small classifier (1536 -> 64 -> 64 -> 1) is trained on them with numpy.
  4. Voices never used for training are the test: a long stream of their speech with "Nexus" in it
     is run through the real detector. The threshold is set to the lowest value with no false
     wake-up on half of that stream, and the model is shifted so that value becomes 0.5 (the
     [wake] threshold default). The other half is measured and reported, honestly.

Your own recordings (scripts/record_wake_samples.py) make the biggest difference for your voice and
your microphone: half train the model, the other half are its test ("your voice: n of m heard").
Folders of any other WAV files (--negatives: your room, TV, games) teach it what to ignore.

Writes models/wake/nexus.onnx and models/wake/nexus.json (how it was made and what it measured).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from nexus.core.config import PROJECT_ROOT  # noqa: E402

RATE = 16000
CLIP_S = 2.24                 # long enough for 16 embedding frames (~1.3 s of context) plus margin
FRAMES = 16
WAKE_DIR = PROJECT_ROOT / "models" / "wake"
KOKORO_DIR = PROJECT_ROOT / "models" / "tts" / "kokoro-v1.0"
WORK_DIR = PROJECT_ROOT / "var" / "wake" / "train"
PHRASES = PROJECT_ROOT / "evals" / "wake_phrases.json"
# Voices kept out of training entirely: the test speakers (one or more per accent group).
HELD_OUT = ("af_nova", "am_echo", "bf_lily", "bm_lewis", "hm_psi", "if_sara", "zm_yunxi", "pm_alex", "jf_nezumi")


def log(message: str) -> None:
    print(message, flush=True)


# --- audio helpers (numpy only) ---------------------------------------------------------------------

def resample(x: np.ndarray, src: int, dst: int = RATE) -> np.ndarray:
    """Band-limited resampling through the FFT (clips are short)."""
    if src == dst or not len(x):
        return x.astype(np.float32)
    n = int(round(len(x) * dst / src))
    spec = np.fft.rfft(x)
    keep = n // 2 + 1
    spec = spec[:keep] if len(spec) >= keep else np.pad(spec, (0, keep - len(spec)))
    return (np.fft.irfft(spec, n) * (n / len(x))).astype(np.float32)


def trim(x: np.ndarray, db: float = 35.0) -> np.ndarray:
    if not len(x):
        return x
    frame = 160
    n = len(x) // frame
    if n == 0:
        return x
    energy = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.where(energy > energy.max() * 10 ** (-db / 20))[0]
    if not len(loud):
        return x
    return x[max(0, loud[0] - 2) * frame:min(len(x), (loud[-1] + 3) * frame)]


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[:len(x)].astype(np.float32)


def room(rng: np.random.Generator) -> np.ndarray:
    """A synthetic room impulse response: direct sound, then exponentially decaying reflections."""
    rt60 = rng.uniform(0.15, 0.8)
    n = int(rt60 * RATE)
    t = np.arange(n) / RATE
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * rng.uniform(0.2, 0.7)
    h[: int(0.002 * RATE)] = 0
    h[0] = 1.0
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def colour(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A microphone's frequency response: some low cut, some high cut."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / RATE)
    low, high = rng.uniform(60, 350), rng.uniform(3000, 7800)
    gain = 1 / np.sqrt(1 + (low / np.maximum(f, 1)) ** 2) / np.sqrt(1 + (f / high) ** 4)
    return np.fft.irfft(spec * gain, len(x)).astype(np.float32)


def noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "hum":
        t = np.arange(n) / RATE
        base = rng.choice([50.0, 60.0])
        x = sum(rng.uniform(0.2, 1) / k * np.sin(2 * np.pi * base * k * t + rng.uniform(0, 6)) for k in range(1, 6))
        return (x + 0.05 * rng.standard_normal(n)).astype(np.float32)
    white = rng.standard_normal(n)
    if kind == "white":
        return white.astype(np.float32)
    spec = np.fft.rfft(white)
    f = np.maximum(np.fft.rfftfreq(n, 1 / RATE), 20)
    shaped = spec / (np.sqrt(f) if kind == "pink" else f)  # pink 1/f power, brown 1/f^2 (fans, rumble)
    return np.fft.irfft(shaped, n).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


class Augmenter:
    def __init__(self, rng: np.random.Generator, babble: list[np.ndarray], extra_noise: list[np.ndarray]) -> None:
        self.rng = rng
        self.babble = babble          # other speech, mixed quietly behind
        self.extra_noise = extra_noise  # the user's own recordings of their room/TV/games

    def background(self, n: int) -> np.ndarray:
        rng = self.rng
        pick = rng.random()
        if pick < 0.2:
            return np.zeros(n, np.float32)
        if pick < 0.45 and self.babble:
            x = np.zeros(n, np.float32)
            for _ in range(rng.integers(1, 4)):
                clip = self.babble[rng.integers(len(self.babble))]
                start = rng.integers(0, max(1, n - len(clip) + 1)) if len(clip) < n else 0
                piece = clip[:n - start] if len(clip) >= n - start else clip
                x[start:start + len(piece)] += piece * rng.uniform(0.3, 1.0)
            return x
        if pick < 0.6 and self.extra_noise:
            clip = self.extra_noise[rng.integers(len(self.extra_noise))]
            if len(clip) > n:
                start = rng.integers(0, len(clip) - n)
                return clip[start:start + n].copy()
            return np.pad(clip, (0, n - len(clip)))
        return noise(str(rng.choice(["white", "pink", "brown", "hum"])), n, rng)

    def __call__(self, x: np.ndarray, *, heavy: bool = True) -> np.ndarray:
        rng = self.rng
        if rng.random() < (0.6 if heavy else 0.3):
            x = fft_convolve(x, room(rng))
        if rng.random() < 0.5:
            x = colour(x, rng)
        if rng.random() < (0.85 if heavy else 0.5):
            bg = self.background(len(x))
            if rms(bg) > 1e-6:
                snr = rng.uniform(0, 30)
                x = x + bg * (rms(x) / rms(bg)) / (10 ** (snr / 20))
        peak = np.max(np.abs(x)) + 1e-9
        level = 10 ** (rng.uniform(-32, -3) / 20)   # a quiet voice across the room .. close and loud
        return np.clip(x / peak * level, -1, 1).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path.name}: only 16-bit WAV files are supported")
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return resample(x, rate)


# --- speech from Kokoro ------------------------------------------------------------------------------

class Voices:
    def __init__(self, threads: int) -> None:
        try:
            import onnxruntime as rt
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise SystemExit(f"Kokoro isn't installed ({exc}): pip install -e \".[tts,wake-train]\"") from None
        model, voices = KOKORO_DIR / "kokoro-v1.0.onnx", KOKORO_DIR / "voices-v1.0.bin"
        if not model.is_file():
            raise SystemExit("The Kokoro voice model is missing: scripts/get_tts_model.py fp32")
        options = rt.SessionOptions()
        options.intra_op_num_threads = threads
        session = rt.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(session, str(voices))
        self.names = sorted(self.kokoro.get_voices())

    def style(self, name: str) -> np.ndarray:
        return self.kokoro.get_voice_style(name)

    def say(self, text: str, style: np.ndarray, speed: float, *, kind: str = "text") -> np.ndarray:
        if kind == "hindi":
            import phonemizer

            text = phonemizer.phonemize(text, "hi", preserve_punctuation=True, with_stress=True,
                                        language_switch="remove-flags")
            kind = "ipa"
        samples, rate = self.kokoro.create(text, voice=style, speed=speed, lang="en-us", is_phonemes=kind == "ipa")
        return trim(resample(np.asarray(samples, dtype=np.float32), rate))


def speakers(voices: Voices, names: list[str], blends: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    out = [(n, voices.style(n)) for n in names]
    for _ in range(blends):
        a, b = rng.choice(names, 2, replace=False)
        w = rng.uniform(0.25, 0.75)
        out.append((f"{a}+{b}", w * voices.style(a) + (1 - w) * voices.style(b)))
    return out


def synthesize(voices: Voices, group: list[tuple[str, np.ndarray]], phrases: dict, rng: np.random.Generator,
               *, positives: int, negatives: int, confusables: int, label: str) -> dict[str, list[np.ndarray]]:
    data: dict[str, list[np.ndarray]] = {"wake": [], "followers": [], "confusable": [], "speech": []}
    hindi_voice = lambda name: name.startswith(("hf_", "hm_")) or "+h" in name  # noqa: E731
    started = time.perf_counter()
    for i, (name, style) in enumerate(group):
        def speak(text: str, kind: str = "text") -> None | np.ndarray:
            try:
                x = voices.say(text, style, float(rng.uniform(0.8, 1.3)), kind=kind)
            except Exception as exc:  # a phoneme Kokoro can't say for this voice: skip the clip
                log(f"    ({name}: couldn't say {text!r}: {exc})")
                return None
            return x if len(x) > 1600 else None

        for _ in range(positives):
            if rng.random() < 0.35 or hindi_voice(name):
                clip = speak(str(rng.choice(phrases["positive_ipa"])), "ipa")
            else:
                clip = speak(str(rng.choice(phrases["positive_text"])))
            if clip is not None:
                data["wake"].append(clip)
        for text in rng.choice(phrases["followers"], 2, replace=False):
            clip = speak(str(text))
            if clip is not None:
                data["followers"].append(clip)
        for _ in range(confusables):
            if rng.random() < 0.3:
                clip = speak(str(rng.choice(phrases["confusable_ipa"])), "ipa")
            else:
                clip = speak(str(rng.choice(phrases["confusable_text"])))
            if clip is not None:
                data["confusable"].append(clip)
        for _ in range(negatives):
            if rng.random() < 0.2:
                clip = speak(str(rng.choice(phrases["negative_hindi"])), "hindi")
            else:
                clip = speak(str(rng.choice(phrases["negative_text"])))
            if clip is not None:
                data["speech"].append(clip)
        if (i + 1) % 10 == 0 or i + 1 == len(group):
            log(f"  {label}: {i + 1}/{len(group)} speakers ({time.perf_counter() - started:.0f} s)")
    return data


# --- examples ---------------------------------------------------------------------------------------

def place(word: np.ndarray, aug: Augmenter, rng: np.random.Generator, follower: np.ndarray | None,
          *, offset_s: float) -> np.ndarray:
    """A CLIP_S window that ends `offset_s` after `word` ends, with context before and maybe words after."""
    n = int(CLIP_S * RATE)
    tail_n = int(offset_s * RATE)
    tail = np.zeros(tail_n, np.float32)
    if follower is not None and tail_n:
        gap = int(rng.uniform(0, 0.12) * RATE)
        piece = follower[: max(0, tail_n - gap)] * rng.uniform(0.6, 1.2)
        tail[gap:gap + len(piece)] = piece
    word = word[-(n - tail_n):]
    before = aug.background(n - len(word) - tail_n) * rng.uniform(0.0, 0.3)
    return np.concatenate([before, word, tail])


class Embedder:
    def __init__(self, threads: int) -> None:
        from nexus.ai.voice.oww import WakeFeatures

        mel, emb = WAKE_DIR / "melspectrogram.onnx", WAKE_DIR / "embedding_model.onnx"
        if not (mel.is_file() and emb.is_file()):
            raise SystemExit("The shared feature networks are missing: scripts/get_wake_model.py")
        self.features = WakeFeatures(mel, emb, threads=threads)

    def last(self, x: np.ndarray) -> np.ndarray | None:
        e = self.features.embed_clip(to_int16(x))
        return e[-FRAMES:] if len(e) >= FRAMES else None

    def windows(self, x: np.ndarray, stride: int = 2) -> list[np.ndarray]:
        e = self.features.embed_clip(to_int16(x))
        return [e[i - FRAMES:i] for i in range(FRAMES, len(e) + 1, stride)]


def build_examples(data: dict[str, list[np.ndarray]], embed: Embedder, aug: Augmenter, rng: np.random.Generator,
                   *, copies: int, own: list[np.ndarray] = (), own_copies: int = 0,
                   extra_negatives: list[np.ndarray] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []

    def add(window, label, weight):
        if window is not None:
            xs.append(window)
            ys.append(label)
            ws.append(weight)

    followers = data["followers"] + data["speech"]
    for words, label, weight, n_copies in ((data["wake"], 1, 1.0, copies), (list(own), 1, 2.0, own_copies),
                                           (data["confusable"], 0, 3.0, copies)):
        for word in words:
            for _ in range(n_copies):
                follower = followers[rng.integers(len(followers))] if rng.random() < 0.5 and followers else None
                clip = place(word, aug, rng, follower, offset_s=float(rng.uniform(0.0, 0.24)))
                add(embed.last(aug(clip)), label, weight)
    # Conversations: several sentences (and near-misses) in a row, every 80 ms of them an example.
    talk = data["speech"] + data["confusable"] + data["followers"]
    for _ in range(len(data["speech"]) * copies // 2):
        parts = []
        for _ in range(rng.integers(3, 7)):
            parts.append(talk[rng.integers(len(talk))] * rng.uniform(0.5, 1.2))
            parts.append(np.zeros(int(rng.uniform(0.05, 0.6) * RATE), np.float32))
        for window in embed.windows(aug(np.concatenate(parts), heavy=False), stride=1):
            add(window, 0, 1.0)
    min_n = int(CLIP_S * RATE)
    for clip in extra_negatives:
        for _ in range(max(1, copies // 2)):
            padded = np.concatenate([clip, np.zeros(max(0, min_n - len(clip)), np.float32)])
            for window in embed.windows(aug(padded, heavy=False), stride=1):
                add(window, 0, 1.0)
    for kind in ("white", "pink", "brown", "hum", "silence"):
        for _ in range(max(2, copies)):
            x = np.zeros(4 * RATE, np.float32) if kind == "silence" else noise(kind, 4 * RATE, rng)
            for window in embed.windows(aug(x, heavy=False), stride=4):
                add(window, 0, 1.0)
    return np.array(xs, np.float32), np.array(ys, np.float32), np.array(ws, np.float32)


# --- the classifier (numpy) -----------------------------------------------------------------------

class MLP:
    def __init__(self, n_in: int, hidden: int, rng: np.random.Generator) -> None:
        he = lambda a, b: rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2 / a)  # noqa: E731
        self.params = [he(n_in, hidden), np.zeros(hidden, np.float32), he(hidden, hidden),
                       np.zeros(hidden, np.float32), he(hidden, 1) * 0.1, np.zeros(1, np.float32)]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x: np.ndarray):
        w1, b1, w2, b2, w3, b3 = self.params
        a1 = x @ w1 + b1
        h1 = np.maximum(a1, 0)
        a2 = h1 @ w2 + b2
        h2 = np.maximum(a2, 0)
        z = (h2 @ w3 + b3)[:, 0]
        return z, (x, a1, h1, a2, h2)

    def score(self, x: np.ndarray, batch: int = 4096) -> np.ndarray:
        return np.concatenate([1 / (1 + np.exp(-self.forward(x[i:i + batch])[0])) for i in range(0, len(x), batch)])

    def step(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, lr: float, l2: float = 1e-4) -> float:
        z, (x, a1, h1, a2, h2) = self.forward(x)
        p = 1 / (1 + np.exp(-z))
        eps = 1e-7
        loss = float(np.sum(w * -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / np.sum(w))
        dz = (w * (p - y) / np.sum(w))[:, None].astype(np.float32)
        w1, b1, w2, b2, w3, b3 = self.params
        g_w3, g_b3 = h2.T @ dz, dz.sum(0)
        dh2 = dz @ w3.T * (a2 > 0)
        g_w2, g_b2 = h1.T @ dh2, dh2.sum(0)
        dh1 = dh2 @ w2.T * (a1 > 0)
        g_w1, g_b1 = x.T @ dh1, dh1.sum(0)
        grads = [g_w1 + l2 * w1, g_b1, g_w2 + l2 * w2, g_b2, g_w3 + l2 * w3, g_b3]
        self.t += 1
        for i, g in enumerate(grads):  # Adam
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            m_hat = self.m[i] / (1 - 0.9 ** self.t)
            v_hat = self.v[i] / (1 - 0.999 ** self.t)
            self.params[i] = self.params[i] - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return loss


def train(x: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator, *, epochs: int,
          hidden: int = 64, batch: int = 256) -> tuple[MLP, np.ndarray, np.ndarray]:
    flat = x.reshape(len(x), -1)
    mean, std = flat.mean(0), flat.std(0) + 1e-4
    flat = (flat - mean) / std
    pos = y == 1
    w = w.copy()
    w[pos] *= (w[~pos].sum() / w[pos].sum()) * 0.5   # the wake word counts half as much as everything else
    model = MLP(flat.shape[1], hidden, rng)
    base = w.copy()
    for epoch in range(epochs):
        lr = 1e-3 * (0.5 * (1 + math.cos(math.pi * epoch / epochs)) * 0.9 + 0.1)
        order = rng.permutation(len(flat))
        losses = [model.step(flat[order[i:i + batch]], y[order[i:i + batch]], w[order[i:i + batch]], lr)
                  for i in range(0, len(order), batch)]
        scores = model.score(flat)
        # Hard negatives: what still looks like "Nexus" counts more in the next round.
        w = base * np.where(~pos & (scores > 0.2), 4.0, 1.0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            recall = float(np.mean(scores[pos] >= 0.5))
            false = float(np.mean(scores[~pos] >= 0.5))
            log(f"  epoch {epoch + 1:>2}/{epochs}: loss {np.mean(losses):.4f}, training recall {recall:.3f}, "
                f"false {false:.4f}")
    return model, mean, std


def export(model: MLP, mean: np.ndarray, std: np.ndarray, path: Path, *, shift: float = 0.0) -> None:
    """Write an openWakeWord-compatible ONNX model: [1, 16, 96] -> [1, 1]. The input scaling is folded
    into the first layer; `shift` moves the decision point (logit) to 0.5."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w1, b1, w2, b2, w3, b3 = model.params
    w1f = (w1 / std[:, None]).astype(np.float32)
    b1f = (b1 - (mean / std) @ w1).astype(np.float32)
    b3f = (b3 - shift).astype(np.float32)
    inits = [numpy_helper.from_array(np.array([-1, FRAMES * 96], np.int64), "shape"),
             numpy_helper.from_array(w1f, "w1"), numpy_helper.from_array(b1f, "b1"),
             numpy_helper.from_array(w2.astype(np.float32), "w2"), numpy_helper.from_array(b2.astype(np.float32), "b2"),
             numpy_helper.from_array(w3.astype(np.float32), "w3"), numpy_helper.from_array(b3f, "b3")]
    nodes = [helper.make_node("Reshape", ["x", "shape"], ["flat"]),
             helper.make_node("Gemm", ["flat", "w1", "b1"], ["a1"]), helper.make_node("Relu", ["a1"], ["h1"]),
             helper.make_node("Gemm", ["h1", "w2", "b2"], ["a2"]), helper.make_node("Relu", ["a2"], ["h2"]),
             helper.make_node("Gemm", ["h2", "w3", "b3"], ["z"]), helper.make_node("Sigmoid", ["z"], ["score"])]
    graph = helper.make_graph(nodes, "nexus_wake_word",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, FRAMES, 96])],
                              [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])], inits)
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                                   producer_name="nexus-os train_wake_model.py")
    onnx_model.ir_version = 8  # readable by older onnxruntime versions too
    onnx.checker.check_model(onnx_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(path))


# --- evaluation on unseen speakers, through the real detector -------------------------------------------

def test_stream(data: dict[str, list[np.ndarray]], aug: Augmenter, rng: np.random.Generator,
                own: list[np.ndarray] = ()) -> tuple[np.ndarray, list[tuple[float, str]], float]:
    """Speech of held-out speakers with "Nexus" (and requests) in it. Returns the audio, where each
    "Nexus" ends (seconds, whose), and how many seconds of it are other speech (for false wake-ups)."""
    parts: list[np.ndarray] = []
    ends: list[tuple[float, str]] = []
    t = 0.0
    events = [("speech", c) for c in data["speech"]] + [("confusable", c) for c in data["confusable"]]
    events += [("wake", c) for c in data["wake"]] + [("own", c) for c in own]
    order = rng.permutation(len(events))
    followers = data["followers"] + data["speech"]
    wake_s = 0.0
    for i in order:
        kind, clip = events[i]
        gap = aug.background(int(rng.uniform(0.6, 2.0) * RATE)) * 0.05
        piece = clip
        if kind in ("wake", "own"):
            if rng.random() < 0.6 and followers:
                pause = np.zeros(int(rng.uniform(0.05, 0.3) * RATE), np.float32)
                piece = np.concatenate([clip, pause, followers[rng.integers(len(followers))]])
            ends.append((t + (len(gap) + len(clip)) / RATE, kind))
            wake_s += len(piece) / RATE + 1.5
        segment = aug(np.concatenate([gap, piece]), heavy=False)
        parts.append(segment)
        t += len(segment) / RATE
    audio = np.concatenate(parts)
    return audio, ends, max(0.0, t - wake_s)


def run_detector(model: Path, audio: np.ndarray) -> np.ndarray:
    """Per-80 ms scores of the real runtime pipeline (without resets)."""
    from nexus.ai.voice.oww import CHUNK, WakeFeatures, WakeModel

    features = WakeFeatures(WAKE_DIR / "melspectrogram.onnx", WAKE_DIR / "embedding_model.onnx")
    net = WakeModel(model)
    pcm = to_int16(audio)
    scores = []
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        features.push(pcm[i:i + CHUNK])
        scores.append(net.score(features.window(net.frames)))
    scores = np.array(scores)
    scores[:5] = 0
    return scores


def detections(scores: np.ndarray, threshold: float, ends: list[tuple[float, str]]):
    """Hits (with a 1 s pause after each, like the listener), matched to the "Nexus" in the stream."""
    hits = []
    i = 0
    while i < len(scores):
        if scores[i] >= threshold:
            hits.append((i + 1) * 0.08)
            i += int(1.0 / 0.08)
        else:
            i += 1
    found = {k: 0 for k in ("wake", "own")}
    total = {k: sum(1 for _, kind in ends if kind == k) for k in found}
    used = set()
    for end, kind in ends:
        match = next((h for h in hits if end - 0.3 <= h <= end + 1.2 and h not in used), None)
        if match is not None:
            used.add(match)
            found[kind] += 1
    false = [h for h in hits if h not in used]
    return found, total, false


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, help="folder of your own recordings of \"Nexus\" (WAV)")
    parser.add_argument("--negatives", type=Path, action="append", default=[],
                        help="folder of WAV files that must NOT wake NEXUS (your room, TV, games); repeatable")
    parser.add_argument("--quick", action="store_true", help="a small model in a few minutes (to try the pipeline)")
    parser.add_argument("--out", type=Path, default=WAKE_DIR / "nexus.onnx")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = automatic)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()

    import os

    threads = args.threads or max(1, min(8, (os.cpu_count() or 2) - 1))
    rng = np.random.default_rng(args.seed)
    phrases = json.loads(PHRASES.read_text(encoding="utf-8"))
    size = dict(blends=12, positives=4, negatives=3, confusables=2, copies=2, eval_blends=4) if args.quick else \
        dict(blends=90, positives=10, negatives=8, confusables=5, copies=4, eval_blends=24)
    epochs = args.epochs or (15 if args.quick else 40)
    started = time.perf_counter()

    own_train: list[np.ndarray] = []
    own_test: list[np.ndarray] = []
    if args.samples:
        files = sorted(p for p in args.samples.rglob("*.wav") if "negative" not in p.parts
                       and "background" not in p.parts)
        clips = [trim(read_wav(p)) for p in files]
        clips = [c for c in clips if len(c) > 0.2 * RATE]
        own_train, own_test = clips[0::2], clips[1::2]
        log(f"Your recordings: {len(clips)} of \"Nexus\" ({len(own_train)} to train, {len(own_test)} to test)")
    extra_negatives: list[np.ndarray] = []
    for folder in args.negatives + ([args.samples / "negative", args.samples / "background"] if args.samples else []):
        for p in sorted(folder.rglob("*.wav")) if folder.is_dir() else []:
            x = read_wav(p)
            extra_negatives += [x[i:i + 10 * RATE] for i in range(0, len(x), 10 * RATE) if len(x[i:i + 10 * RATE]) > RATE]
    if extra_negatives:
        log(f"Other sounds (must not wake NEXUS): {len(extra_negatives)} pieces of up to 10 s")

    voices = Voices(threads)
    train_names = [n for n in voices.names if n not in HELD_OUT]
    test_names = [n for n in voices.names if n in HELD_OUT]
    key = hashlib.sha256(json.dumps([phrases, size, args.seed]).encode()).hexdigest()[:12]
    cache = WORK_DIR / f"speech-{key}.npz"
    if cache.is_file():
        log(f"Using the speech generated before ({cache.name})")
        stored = np.load(cache, allow_pickle=True)
        train_data, test_data = stored["train"].item(), stored["test"].item()
    else:
        log(f"Generating speech with Kokoro: {len(train_names)} voices + {size['blends']} blends to train, "
            f"{len(test_names)} other voices + {size['eval_blends']} blends to test")
        train_group = speakers(voices, train_names, size["blends"], rng)
        test_group = speakers(voices, test_names, size["eval_blends"], rng)
        train_data = synthesize(voices, train_group, phrases, rng, positives=size["positives"],
                                negatives=size["negatives"], confusables=size["confusables"], label="training")
        test_data = synthesize(voices, test_group, phrases, rng, positives=max(2, size["positives"] // 2),
                               negatives=size["negatives"], confusables=size["confusables"], label="test")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, train=np.array(train_data, dtype=object), test=np.array(test_data, dtype=object))
    log("  training: " + ", ".join(f"{len(v)} {k}" for k, v in train_data.items()))

    log("Making examples (rooms, microphones, noise) and computing speech features")
    embed = Embedder(threads)
    aug = Augmenter(rng, babble=train_data["speech"], extra_noise=extra_negatives)
    x, y, w = build_examples(train_data, embed, aug, rng, copies=size["copies"], own=own_train,
                             own_copies=size["copies"] * 6, extra_negatives=extra_negatives)
    log(f"  {int(y.sum())} wake-word examples, {int((1 - y).sum())} other examples")

    log(f"Training ({epochs} rounds)")
    model, mean, std = train(x, y, w, rng, epochs=epochs)

    log("Testing on voices it never heard, through the real detector")
    test_aug = Augmenter(np.random.default_rng(args.seed + 1), babble=test_data["speech"], extra_noise=extra_negatives)
    audio, ends, other_s = test_stream(test_data, test_aug, np.random.default_rng(args.seed + 2), own_test)
    half = len(audio) // 2
    raw = args.out.with_suffix(".uncalibrated.onnx")
    export(model, mean, std, raw)
    scores = run_detector(raw, audio)
    cut = half // 1280
    calib_ends = [(e, k) for e, k in ends if e * RATE < half]
    test_ends = [(e - cut * 0.08, k) for e, k in ends if e * RATE >= half + RATE]
    calib_other = other_s * half / len(audio)
    chosen = 0.95
    for threshold in np.arange(0.30, 0.96, 0.01):
        _, _, false = detections(scores[:cut], float(threshold), calib_ends)
        if not false:
            chosen = float(threshold)
            break
    shift = math.log(chosen / (1 - chosen))
    export(model, mean, std, args.out, shift=shift)
    raw.unlink(missing_ok=True)

    final = run_detector(args.out, audio)[cut:]
    found, total, false = detections(final, 0.5, test_ends)
    test_other_min = (other_s - calib_other) / 60
    heard = f"{found['wake']} of {total['wake']}"
    report = {
        "word": "nexus",
        "made": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by": "scripts/train_wake_model.py" + (" --quick" if args.quick else ""),
        "threshold": 0.5,
        "calibrated_from": round(chosen, 2),
        "training": {"voices": len(train_names), "blends": size["blends"], "wake_examples": int(y.sum()),
                     "other_examples": int((1 - y).sum()), "own_recordings": len(own_train),
                     "other_sound_pieces": len(extra_negatives), "epochs": epochs},
        "test": {"voices": test_names, "blends": size["eval_blends"], "nexus_heard": heard,
                 "recall": round(found["wake"] / max(1, total["wake"]), 3),
                 "own_voice_heard": f"{found['own']} of {total['own']}" if total["own"] else None,
                 "false_wakeups": len(false), "other_speech_minutes": round(test_other_min, 1)},
        "minutes_to_train": round((time.perf_counter() - started) / 60, 1),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("")
    log(f"Saved {args.out} ({args.out.stat().st_size // 1024} KB) and {args.out.with_suffix('.json').name}")
    log(f"Unseen voices: \"Nexus\" heard {heard} times ({report['test']['recall']:.0%}); "
        f"{len(false)} false wake-up(s) in {duration(test_other_min)} of other speech and near-misses.")
    if total["own"]:
        log(f"Your own recordings (not used for training): {found['own']} of {total['own']} heard.")
    log("NEXUS uses it the next time it starts ([wake] detector = \"auto\"); try it with /wake test.")
    return 0


def duration(minutes: float) -> str:
    return f"{minutes:.1f} minutes" if minutes >= 1 else f"{minutes * 60:.0f} seconds"


if __name__ == "__main__":
    sys.exit(main())
