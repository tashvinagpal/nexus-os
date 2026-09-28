"""Speech-to-text: the worker-process provider and resource-aware model selection.

The worker is replaced by tests/fake_stt_worker.py (same protocol, no model), so
these tests run anywhere; tests/test_voice_live.py uses the real models.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from nexus.ai.providers.base import Locality
from nexus.ai.providers.selection import SelectionContext
from nexus.ai.voice.base import (AudioClip, ListeningCancelled, STTFailed, STTTimeout, STTUnavailable,
                                 Transcript)
from nexus.ai.voice.stt import HybridSTT, STTOption
from nexus.ai.voice.whisper import WhisperSTT, model_folder
from nexus.core.resources import Placement, ResourceManager
from tests.conftest import FakeProbe
from tests.voice_fixtures import FakeSTT

WORKER = str(Path(__file__).with_name("fake_stt_worker.py"))
CLIP = AudioClip(b"\x01\x00" * 16000)


def fake_whisper(mode: str = "ok", **kw) -> WhisperSTT:
    return WhisperSTT("base", ram_mb=300, command=[sys.executable, WORKER, mode], **kw)


# --- the worker process --------------------------------------------------------------

def test_load_transcribe_unload_ends_the_process():
    stt = fake_whisper(languages=("en", "hi"))
    assert not stt.is_loaded()
    stt.load()
    pid = stt.pid
    assert stt.is_loaded() and psutil.pid_exists(pid)
    t = stt.transcribe(CLIP)
    assert t.text == "heard 32000 bytes" and t.language == "en" and t.confidence == 0.8
    assert t.model == "stt:whisper-base@cpu"
    stt.unload()
    assert not stt.is_loaded()
    time.sleep(0.2)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_language_hint_is_passed_through():
    stt = fake_whisper()
    stt.load()
    try:
        assert stt.transcribe(CLIP, language_hint="hi").language == "hi"
    finally:
        stt.unload()


def test_timeout_stops_the_worker_and_the_next_request_can_reload():
    stt = fake_whisper("slow", timeout_s=0.5)
    stt.load()
    started = time.perf_counter()
    with pytest.raises(STTTimeout):
        stt.transcribe(CLIP)
    assert time.perf_counter() - started < 3
    assert not stt.is_loaded()  # the stuck process is gone


def test_crash_is_reported_not_raised_as_a_crash():
    stt = fake_whisper("crash")
    stt.load()
    with pytest.raises(STTFailed) as err:
        stt.transcribe(CLIP)
    assert "stopped unexpectedly" in str(err.value)
    assert not stt.is_loaded()


def test_cancel_during_transcription():
    stt = fake_whisper("slow")
    stt.load()
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    with pytest.raises(ListeningCancelled):
        stt.transcribe(CLIP, cancel=cancel)
    assert not stt.is_loaded()


def test_failed_load_is_unavailable_for_a_while():
    stt = fake_whisper("fail_load")
    with pytest.raises(STTUnavailable) as err:
        stt.load()
    assert "cublas64_12.dll" in str(err.value)
    assert not stt.is_loaded() and not stt.available() and "cublas" in stt.problem


def test_worker_that_never_gets_ready_times_out():
    stt = fake_whisper("hang_load", load_timeout_s=0.5)
    with pytest.raises(STTUnavailable):
        stt.load()
    assert not stt.is_loaded()


def test_transcribe_without_a_running_worker_or_audio():
    stt = fake_whisper()
    with pytest.raises(STTFailed):
        stt.transcribe(CLIP)
    with pytest.raises(STTFailed):
        stt.transcribe(AudioClip(b""))


def test_missing_model_or_package_is_explained(tmp_path):
    stt = WhisperSTT("small", ram_mb=500, models_dir=tmp_path)
    assert not stt.available()
    assert "get_stt_model.py small" in stt.problem or "faster-whisper" in stt.problem
    with pytest.raises(STTUnavailable):
        stt.load()
    folder = model_folder("small", tmp_path)
    assert folder == tmp_path / "faster-whisper-small"


def test_spec_is_a_managed_model():
    stt = WhisperSTT("base", ram_mb=320, idle_unload_s=60, placement=Placement.CPU)
    assert stt.spec.kind == "stt" and stt.spec.ram_mb == 320 and stt.spec.vram_mb == 0
    assert stt.info.locality is Locality.LOCAL


# --- choosing a model ------------------------------------------------------------------

def hybrid(*options, probe=None, context=None) -> tuple[HybridSTT, ResourceManager]:
    rm = ResourceManager(probe or FakeProbe(), snapshot_ttl=0)
    return HybridSTT(list(options), rm, context or (lambda: SelectionContext())), rm


def opt(provider, quality, game=False):
    return STTOption(provider, provider.spec, quality, game_mode_ok=game)


def test_best_available_model_is_used_and_loaded_on_demand():
    small, base = FakeSTT("small", ram_mb=560), FakeSTT("base", ram_mb=320)
    stt, rm = hybrid(opt(small, 2), opt(base, 1, game=True))
    assert not small.loaded and not base.loaded  # registered, not loaded
    stt.transcribe(CLIP)
    assert small.loads == 1 and base.loads == 0 and stt.last_choice == "stt:small@cpu"


def test_game_mode_uses_only_light_models_and_unloads_heavy_ones():
    small, base = FakeSTT("small", ram_mb=560), FakeSTT("base", ram_mb=320)
    stt, rm = hybrid(opt(small, 2), opt(base, 1, game=True))
    stt.transcribe(CLIP)
    assert small.loaded
    rm.enter_game_mode()
    stt.transcribe(CLIP)
    assert stt.last_choice == "stt:base@cpu"
    assert not small.loaded and small.unloads == 1  # freed right away, not after the idle timeout


def test_game_mode_without_a_light_model_says_so():
    small = FakeSTT("small", ram_mb=560)
    stt, rm = hybrid(opt(small, 1))
    rm.enter_game_mode()
    with pytest.raises(STTUnavailable) as err:
        stt.transcribe(CLIP)
    assert "too heavy while a game runs" in str(err.value)


def test_gpu_model_is_refused_in_game_mode_and_cpu_is_used():
    gpu = FakeSTT("small", placement=Placement.GPU, ram_mb=300, vram_mb=800)
    cpu = FakeSTT("base", ram_mb=320)
    stt, rm = hybrid(opt(gpu, 2, game=True), opt(cpu, 1, game=True))
    rm.enter_game_mode()
    stt.transcribe(CLIP)
    assert stt.last_choice == "stt:base@cpu" and gpu.loads == 0


def test_gpu_failure_falls_back_to_cpu():
    gpu = FakeSTT("small", placement=Placement.GPU, ram_mb=300, vram_mb=800, fail_load=True)
    cpu = FakeSTT("small", ram_mb=560)
    stt, rm = hybrid(opt(gpu, 2), opt(cpu, 1))
    t = stt.transcribe(CLIP)
    assert t.text == "hello" and stt.last_choice == "stt:small@cpu"
    stt.transcribe(CLIP)  # the broken GPU option isn't retried every time
    assert gpu.load_started.is_set() and cpu.loads == 1


def test_not_enough_memory_uses_the_lighter_model():
    small, base = FakeSTT("small", ram_mb=560), FakeSTT("base", ram_mb=320)
    probe = FakeProbe(ram_free=2048 + 400)  # 400 MB above the reserve
    stt, rm = hybrid(opt(small, 2), opt(base, 1), probe=probe)
    stt.transcribe(CLIP)
    assert stt.last_choice == "stt:base@cpu" and small.loads == 0


def test_nothing_can_run_explains_why():
    small = FakeSTT("small", ram_mb=560, is_available=False)
    small.problem = "model not downloaded - run: scripts\\get_stt_model.py small"
    stt, rm = hybrid(opt(small, 1))
    with pytest.raises(STTUnavailable) as err:
        stt.transcribe(CLIP)
    assert "get_stt_model.py small" in str(err.value)


def test_cloud_stt_is_never_used_unless_enabled_and_online():
    cloud = FakeSTT("cloud", locality=Locality.CLOUD, results=[Transcript("from the cloud", "en", 0.9)])
    local = FakeSTT("base", ram_mb=320)
    options = [STTOption(cloud, None, 9), opt(local, 1)]
    stt, rm = hybrid(*options)
    stt.transcribe(CLIP)
    assert stt.last_choice == "stt:base@cpu" and cloud.calls == []
    stt, rm = hybrid(*options, context=lambda: SelectionContext(allow_cloud=True, network_available=False))
    stt.transcribe(CLIP)
    assert cloud.calls == []  # no network
    stt, rm = hybrid(*options, context=lambda: SelectionContext(allow_cloud=True, network_available=True))
    assert stt.transcribe(CLIP).text == "from the cloud"  # the interface is ready for one


def test_prepare_starts_loading_in_the_background():
    small = FakeSTT("small", ram_mb=560)
    stt, rm = hybrid(opt(small, 1))
    assert stt.prepare() is not None
    assert small.load_started.wait(2)
    deadline = time.time() + 2
    while not small.loaded and time.time() < deadline:
        time.sleep(0.01)
    assert small.loaded
    assert rm.status()["models"][0]["in_use"] is False  # loaded, not held


def test_prepare_returns_none_when_nothing_can_run():
    small = FakeSTT("small", is_available=False)
    stt, _ = hybrid(opt(small, 1))
    assert stt.prepare() is None


def test_idle_stt_model_unloads_and_never_stays_forever():
    small = FakeSTT("small", ram_mb=560)
    now = [1000.0]
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0, clock=lambda: now[0])
    stt = HybridSTT([opt(small, 1)], rm, lambda: SelectionContext())
    stt.transcribe(CLIP)
    assert small.loaded
    now[0] += 61  # stt_idle_unload_seconds = 60
    assert rm.release_idle() == ["stt:small@cpu"] and not small.loaded


def test_transcription_errors_other_than_loading_are_not_retried_elsewhere():
    from nexus.ai.voice.base import NoSpeech

    a = FakeSTT("small", results=[NoSpeech("nothing")])
    b = FakeSTT("base")
    stt, _ = hybrid(opt(a, 2), opt(b, 1))
    with pytest.raises(NoSpeech):
        stt.transcribe(CLIP)
    assert b.calls == []
