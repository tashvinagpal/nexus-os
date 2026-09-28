import pytest

from nexus.ai.providers import (ChatMessage, HybridChat, Locality, ProviderOption, ProviderUnavailable,
                                SelectionContext, select_provider)
from nexus.core.resources import Placement, ResourceManager
from tests.conftest import FakeChat, FakeProbe


@pytest.fixture
def setup(clock):
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    big = FakeChat("llm:8b@gpu", vram_mb=5100, ram_mb=500, probe=probe)
    small = FakeChat("llm:3b@gpu", vram_mb=2500, ram_mb=800, probe=probe)
    cpu = FakeChat("llm:3b@cpu", placement=Placement.CPU, ram_mb=2500, probe=probe)
    options = [ProviderOption(big, big.spec, 3), ProviderOption(small, small.spec, 2),
               ProviderOption(cpu, cpu.spec, 1)]
    for o in options:
        rm.register(o.provider)
    return rm, probe, options, (big, small, cpu)


def choose(options, rm, **ctx):
    option, skipped = select_provider(options, rm, SelectionContext(**ctx))
    return (option.provider.info.name if option else None), skipped


def test_best_option_when_everything_fits(setup):
    rm, _, options, _ = setup
    assert choose(options, rm)[0] == "llm:8b@gpu"


def test_falls_back_to_smaller_model_when_vram_is_busy(setup):
    rm, probe, options, _ = setup
    probe.base["vram_used"] = 7000  # e.g. a game is using VRAM: 5288 free
    name, skipped = choose(options, rm)
    assert name == "llm:3b@gpu"
    assert any("8b" in s and "VRAM" in s for s in skipped)


def test_game_mode_uses_cpu_model(setup):
    rm, _, options, _ = setup
    rm.enter_game_mode()
    name, skipped = choose(options, rm)
    assert name == "llm:3b@cpu"
    assert sum("game mode" in s for s in skipped) == 2


def test_nothing_fits_returns_reasons(setup):
    rm, probe, options, _ = setup
    rm.enter_game_mode()
    probe.base["ram_free"] = 3000
    name, skipped = choose(options, rm)
    assert name is None and len(skipped) == 3


def test_power_saving_skips_heavy_models(setup):
    rm, _, options, _ = setup
    name, skipped = choose(options, rm, power_saving=True)
    assert name == "llm:3b@gpu" and "too heavy" in skipped[0]


def test_unavailable_provider_skipped(setup):
    rm, _, options, (big, _, _) = setup
    big._available = False
    assert choose(options, rm)[0] == "llm:3b@gpu"


def test_cloud_only_when_allowed_and_online(setup):
    rm, _, options, _ = setup
    cloud = FakeChat("cloud", locality=Locality.CLOUD)
    opts = [ProviderOption(cloud, None, 10), *options]
    assert choose(opts, rm)[0] == "llm:8b@gpu"
    assert choose(opts, rm, allow_cloud=True)[0] == "llm:8b@gpu"  # still no network
    assert choose(opts, rm, allow_cloud=True, network_available=True)[0] == "cloud"


def test_lan_core_needs_network(setup):
    rm, _, options, _ = setup
    pc = FakeChat("lan:pc", locality=Locality.LAN)
    opts = [ProviderOption(pc, None, 10), *options]
    assert choose(opts, rm)[0] == "llm:8b@gpu"
    assert choose(opts, rm, network_available=True)[0] == "lan:pc"


def test_hybrid_chat_loads_chosen_model_lazily(clock):
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    chat = FakeChat("llm", probe=probe)
    hybrid = HybridChat([ProviderOption(chat, chat.spec, 1)], rm, SelectionContext)
    assert not chat.loaded
    reply = hybrid.chat([ChatMessage("user", "hi")])
    assert reply.text and chat.loaded and hybrid.last_choice == "llm"


def test_hybrid_chat_explains_when_nothing_can_run(clock):
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    chat = FakeChat("llm", probe=probe)
    hybrid = HybridChat([ProviderOption(chat, chat.spec, 1)], rm, SelectionContext)
    rm.enter_game_mode()
    with pytest.raises(ProviderUnavailable, match="game mode"):
        hybrid.chat([ChatMessage("user", "hi")])
