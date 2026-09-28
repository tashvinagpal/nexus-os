import pytest

from nexus.core.resources import (Placement, ResourceManager, ResourceState, ResourceUnavailable,
                                  probe_system)
from tests.conftest import FakeBackend, FakeProbe


def manager(clock, **probe_kwargs):
    probe = FakeProbe(**probe_kwargs)
    return ResourceManager(probe, clock=clock, snapshot_ttl=0), probe


def test_nothing_is_loaded_at_start(clock):
    rm, probe = manager(clock)
    llm = FakeBackend("llm", probe=probe)
    rm.register(llm)
    assert not llm.loaded
    assert rm.state is ResourceState.IDLE


def test_lazy_load_on_first_use_and_reuse(clock):
    rm, probe = manager(clock)
    llm = FakeBackend("llm", probe=probe)
    rm.register(llm)
    with rm.use("llm"):
        assert llm.loaded
    with rm.use("llm"):
        pass
    assert llm.loads == 1


def test_refuses_when_vram_is_short(clock):
    rm, probe = manager(clock, vram_used=8000)  # 12288 total -> 4288 free
    rm.register(FakeBackend("big", vram_mb=5000, probe=probe))
    adm = rm.admit(rm._slots["big"].backend.spec)
    assert not adm.ok and "VRAM" in adm.reason
    with pytest.raises(ResourceUnavailable):
        with rm.use("big"):
            pass


def test_refuses_when_ram_is_short(clock):
    rm, probe = manager(clock, ram_free=3000)  # reserve 2048 -> only ~950 usable
    rm.register(FakeBackend("cpu", placement=Placement.CPU, ram_mb=2500, probe=probe))
    adm = rm.admit(rm._slots["cpu"].backend.spec)
    assert not adm.ok and "RAM" in adm.reason


def test_no_gpu_means_no_gpu_models(clock):
    rm, probe = manager(clock, vram_total=None, vram_used=None)
    rm.register(FakeBackend("gpu", probe=probe))
    assert "no usable GPU" in rm.admit(rm._slots["gpu"].backend.spec).reason


def test_idle_models_unload_after_timeout(clock):
    rm, probe = manager(clock)
    llm = FakeBackend("llm", idle_unload_s=300, probe=probe)
    rm.register(llm)
    with rm.use("llm"):
        pass
    clock.now += 299
    assert rm.release_idle() == [] and llm.loaded
    clock.now += 2
    assert rm.release_idle() == ["llm"] and not llm.loaded


def test_model_in_use_is_never_unloaded(clock):
    rm, probe = manager(clock)
    llm = FakeBackend("llm", idle_unload_s=1, probe=probe)
    rm.register(llm)
    with rm.use("llm"):
        clock.now += 100
        assert rm.release_idle() == [] and llm.loaded
        assert rm.unload_all() == []


def test_idle_model_evicted_to_make_room(clock):
    rm, probe = manager(clock, vram_used=1000)  # 11288 free, 1024 reserve
    old = FakeBackend("old", vram_mb=6000, probe=probe)
    new = FakeBackend("new", vram_mb=6000, probe=probe)
    rm.register(old)
    rm.register(new)
    with rm.use("old"):
        pass
    with rm.use("new"):
        assert new.loaded and not old.loaded


def test_game_mode_unloads_gpu_models_and_keeps_cpu(clock):
    rm, probe = manager(clock)
    gpu = FakeBackend("gpu", probe=probe)
    cpu = FakeBackend("cpu", placement=Placement.CPU, ram_mb=2500, probe=probe)
    rm.register(gpu)
    rm.register(cpu)
    with rm.use("gpu"), rm.use("cpu"):
        pass
    assert rm.enter_game_mode() == ["gpu"]
    assert rm.state is ResourceState.GAME_MODE
    assert not gpu.loaded and cpu.loaded
    assert not rm.admit(gpu.spec).ok and "game mode" in rm.admit(gpu.spec).reason
    with pytest.raises(ResourceUnavailable):
        with rm.use("gpu"):
            pass
    rm.exit_game_mode()
    assert rm.admit(gpu.spec).ok


def test_gpu_model_in_use_unloads_when_released_in_game_mode(clock):
    rm, probe = manager(clock)
    gpu = FakeBackend("gpu", probe=probe)
    rm.register(gpu)
    with rm.use("gpu"):
        rm.enter_game_mode()
        assert gpu.loaded  # not ripped out mid-answer
    assert not gpu.loaded


def test_busy_states_nest_and_restore(clock):
    rm, _ = manager(clock)
    with rm.busy(ResourceState.THINKING):
        assert rm.state is ResourceState.THINKING
        with rm.busy(ResourceState.EXECUTING):
            assert rm.state is ResourceState.EXECUTING
        assert rm.state is ResourceState.THINKING
    assert rm.state is ResourceState.IDLE
    with pytest.raises(ValueError):
        with rm.busy(ResourceState.GAME_MODE):
            pass


def test_game_mode_dominates_reported_state(clock):
    rm, _ = manager(clock)
    rm.enter_game_mode()
    with rm.busy(ResourceState.THINKING):
        assert rm.state is ResourceState.GAME_MODE
        assert rm.activity is ResourceState.THINKING


def test_status_report(clock):
    rm, probe = manager(clock)
    rm.register(FakeBackend("llm", probe=probe))
    with rm.use("llm"):
        pass
    st = rm.status()
    assert st["state"] == "idle" and st["models"][0]["loaded"] is True
    assert st["models"][0]["unloads_in_s"] == 300
    assert st["resources"]["ram_total_mb"] == 16000


def test_real_probe_reads_this_machine():
    snap = probe_system()
    assert snap.ram_total_mb > 1000 and 0 < snap.ram_available_mb <= snap.ram_total_mb
    assert snap.nexus_rss_mb > 0
