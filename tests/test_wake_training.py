"""scripts/train_wake_model.py building blocks: audio helpers, the classifier, its ONNX export."""

from __future__ import annotations

import importlib.util
import math
from importlib.util import find_spec
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_wake_model.py"
_spec = importlib.util.spec_from_file_location("train_wake_model", _SCRIPT)
train_wake_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_wake_model)


def test_resample_keeps_duration_and_tone():
    t = np.arange(24000) / 24000
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = train_wake_model.resample(x, 24000)
    assert len(y) == 16000
    spectrum = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(spectrum) - 440) <= 1


def test_trim_cuts_silence_around_speech():
    x = np.concatenate([np.zeros(8000), np.ones(4000) * 0.3, np.zeros(8000)]).astype(np.float32)
    y = train_wake_model.trim(x)
    assert 4000 <= len(y) <= 4000 + 5 * 160


def test_augmentation_stays_in_range_and_keeps_length():
    rng = np.random.default_rng(0)
    aug = train_wake_model.Augmenter(rng, babble=[rng.standard_normal(8000).astype(np.float32) * 0.1], extra_noise=[])
    x = np.sin(np.arange(16000) / 5).astype(np.float32) * 0.5
    for _ in range(20):
        y = aug(x)
        assert len(y) == len(x) and np.max(np.abs(y)) <= 1.0 and np.isfinite(y).all()


def test_classifier_learns_and_the_onnx_export_matches(tmp_path):
    if find_spec("onnx") is None or find_spec("onnxruntime") is None:
        pytest.skip("onnx/onnxruntime not installed (pip install -e \".[voice,wake-train]\")")
    import onnxruntime as ort

    rng = np.random.default_rng(1)
    n = 600
    x = rng.standard_normal((n, 16, 96)).astype(np.float32)
    y = (x[:, -1, 0] + x[:, -2, 5] > 0).astype(np.float32)      # a learnable rule
    model, mean, std = train_wake_model.train(x, y, np.ones(n, np.float32), rng, epochs=25)
    flat = (x.reshape(n, -1) - mean) / std
    scores = model.score(flat)
    assert np.mean((scores >= 0.5) == (y == 1)) > 0.9

    path = tmp_path / "nexus.onnx"
    shift = math.log(0.8 / 0.2)                                     # threshold 0.8 becomes 0.5
    train_wake_model.export(model, mean, std, path, shift=shift)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    assert inp.shape == [1, 16, 96]
    onnx_scores = np.array([session.run(None, {inp.name: x[i:i + 1]})[0][0, 0] for i in range(50)])
    z = np.log(scores[:50] / (1 - scores[:50]))
    expected = 1 / (1 + np.exp(-(z - shift)))
    assert np.allclose(onnx_scores, expected, atol=1e-4)
    assert np.array_equal(onnx_scores >= 0.5, scores[:50] >= 0.8)


def test_detections_match_wake_words_and_count_the_rest_as_false():
    scores = np.zeros(200)
    scores[20] = 0.9          # 1.68 s: the "Nexus" that ends at 1.6 s
    scores[21] = 0.9          # same one - skipped for 1 s after the hit
    scores[120] = 0.7         # 9.68 s: nothing was said there
    found, total, false = train_wake_model.detections(scores, 0.5, [(1.6, "wake"), (5.0, "wake")])
    assert found["wake"] == 1 and total["wake"] == 2 and false == [pytest.approx(9.68)]
