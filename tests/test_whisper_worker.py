"""The worker's decoding choices with a fake Whisper model: bounded decoding, unsure languages."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from nexus.ai.voice import whisper_worker as worker  # noqa: E402

PCM = (np.sin(np.arange(16000 * 2) / 5) * 8000).astype(np.int16).tobytes()  # 2 s


class FakeModel:
    """Scripted detection probabilities and one transcript (with avg log-prob) per language."""

    def __init__(self, probs: dict[str, float], decodes: dict[str, tuple[str, float]]) -> None:
        self.probs = probs
        self.decodes = decodes
        self.calls: list[dict] = []
        self.encodes = 0
        self.encode = worker.EncoderCache(self._encode)
        self.feature_extractor = lambda audio: np.zeros((80, 201), dtype=np.float32)

    def _encode(self, features):
        self.encodes += 1
        return "encoded"

    def detect_language(self, features=None, **_):
        self.encode(worker_pad(features))
        ranked = sorted(self.probs.items(), key=lambda kv: -kv[1])
        return ranked[0][0], ranked[0][1], ranked

    def transcribe(self, speech, *, language, temperature, max_new_tokens, **_):
        self.calls.append({"language": language, "temperature": temperature, "max_new_tokens": max_new_tokens})
        self.encode(worker_pad(np.zeros((80, 200), dtype=np.float32)))
        text, logprob = self.decodes[language]
        segment = SimpleNamespace(text=text, avg_logprob=logprob, start=0.0, end=2.0, no_speech_prob=0.01,
                                  compression_ratio=1.1, temperature=0.0)
        return iter([segment]), SimpleNamespace(language=language, language_probability=self.probs.get(language, 0))


def worker_pad(features):
    return np.zeros((80, 3000), dtype=np.float32) if features is not None else None


def run(model, req=None):
    return worker.transcribe(model, np, PCM, {"languages": ["en", "hi", "pa"], **(req or {})},
                             vad=lambda np_, audio: audio)


def test_length_cap_fits_the_speech():
    assert worker.max_tokens(2.0) == 96          # a command: never 448 tokens of looping
    assert worker.max_tokens(15.0) == 440        # the longest utterance still fits Gurmukhi's ~23 tokens/s
    assert worker.max_tokens(0.5) >= 32


def test_sure_language_is_decoded_once_with_one_retry_at_most():
    model = FakeModel({"en": 0.93, "hi": 0.05}, {"en": ("What time is it?", -0.1)})
    result = run(model)
    assert result["text"] == "What time is it?" and result["language"] == "en"
    assert len(model.calls) == 1 and model.calls[0]["temperature"] == worker.TEMPERATURES
    assert len(worker.TEMPERATURES) <= 2 and model.calls[0]["max_new_tokens"] == 96
    assert result["shared_encoder"] is True and model.encodes == 1  # detection and decoding: one encoder pass


def test_unsure_language_tries_the_runner_up_and_keeps_the_surer_words():
    # Real case: Haryanvi "हाँ, कर दे" detected as English (0.31) came out translated: "I'll do it."
    model = FakeModel({"en": 0.31, "hi": 0.29, "pa": 0.1},
                      {"en": ("I'll do it.", -0.89), "hi": ("हाँ, कर दे", -0.35)})
    result = run(model)
    assert result["text"] == "हाँ, कर दे" and result["language"] == "hi"
    assert math.isclose(result["confidence"], math.exp(-0.35))
    assert [c["language"] for c in model.calls] == ["en", "hi"]
    assert model.calls[1]["temperature"] == 0.0  # the runner-up gets a plain greedy decode only
    assert [t["language"] for t in result["tried_languages"]] == ["en", "hi"]
    assert model.encodes == 1


def test_unsure_but_the_first_language_was_right():
    model = FakeModel({"en": 0.45, "hi": 0.40}, {"en": ("Open my downloads folder", -0.2), "hi": ("अपन में", -1.3)})
    result = run(model)
    assert result["text"] == "Open my downloads folder" and result["language"] == "en"
    assert result["language_probability"] == pytest.approx(0.45)


def test_urdu_counts_as_hindi():
    model = FakeModel({"ur": 0.5, "hi": 0.3, "en": 0.2}, {"hi": ("कितने बजे हैं", -0.2), "en": ("x", -2.0)})
    result = run(model)
    assert result["language"] == "hi" and result["language_probability"] == pytest.approx(0.8)
    assert len(model.calls) == 1  # 0.8 is sure: no second decode


def test_a_fixed_language_skips_detection():
    model = FakeModel({"en": 0.1, "hi": 0.9}, {"en": ("hello", -0.1)})
    result = run(model, {"language": "en"})
    assert result["text"] == "hello" and [c["language"] for c in model.calls] == ["en"]


def test_no_speech_means_no_decoding():
    model = FakeModel({"en": 1.0}, {"en": ("invented", -0.1)})
    result = worker.transcribe(model, np, PCM, {"languages": ["en"]}, vad=lambda np_, audio: None)
    assert result["text"] == "" and model.calls == []
