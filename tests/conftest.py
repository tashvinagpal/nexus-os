from __future__ import annotations

from pathlib import Path

import pytest

from nexus.core.kernel import ActionKernel
from nexus.security.audit import AuditLog
from nexus.security.permissions import Policy
from nexus.tools import ToolRegistry, default_tools
from nexus.tools.base import Param, Risk, Tool, ToolResult


class FakePlatform:
    """Records browser requests instead of opening anything."""

    name = "fake"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.opened: list[str] = []

    def protected_paths(self) -> list[Path]:
        return [self.root / "system"]

    def safe_roots(self) -> list[Path]:
        return [self.root / "home"]

    def open_uri(self, uri: str) -> None:
        self.opened.append(uri)

    def is_elevated(self) -> bool:
        return False

    def cpu_name(self) -> str:
        return "Fake CPU"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Recorder:
    """A test tool body that counts calls."""

    def __init__(self, result: ToolResult | None = None, exc: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.result = result or ToolResult(True, "did it", verified=True)
        self.exc = exc

    def __call__(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.result


def make_tool(name: str, risk: Risk = Risk.SAFE, run=None, params=()) -> Tool:
    return Tool(name=name, description=f"test tool {name}", risk=risk,
                params=tuple(params), run=run or Recorder())


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    for sub in ("home", "system", "outside"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def fake_platform(sandbox: Path) -> FakePlatform:
    return FakePlatform(sandbox)


@pytest.fixture
def policy(fake_platform: FakePlatform) -> Policy:
    return Policy(protected=fake_platform.protected_paths(), safe_roots=fake_platform.safe_roots())


@pytest.fixture
def audit(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "var" / "audit.jsonl")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_kernel(fake_platform, policy, audit, clock):
    def build(*extra: Tool, policy_override: Policy | None = None, memory=None) -> ActionKernel:
        registry = ToolRegistry([*default_tools(fake_platform, memory), *extra])
        return ActionKernel(registry, policy_override or policy, audit, confirm_ttl=60, clock=clock)
    return build


@pytest.fixture
def memory_store(tmp_path):
    from nexus.ai.memory.sqlite_store import SQLiteMemoryStore

    store = SQLiteMemoryStore(tmp_path / "memory" / "memory.db")
    yield store
    store.close()


# --- AI-layer fakes -------------------------------------------------------------------

from nexus.ai.conversation.types import ActionStep, Plan  # noqa: E402
from nexus.ai.providers.base import ChatReply, Locality, ProviderInfo  # noqa: E402
from nexus.core.resources import ModelSpec, Placement, ResourceManager, ResourceSnapshot  # noqa: E402


def snapshot(ram_free=8000, vram_total=12288, vram_used=1000, on_battery=None) -> ResourceSnapshot:
    return ResourceSnapshot(ram_total_mb=16000, ram_available_mb=ram_free, cpu_percent=5.0,
                            vram_total_mb=vram_total, vram_used_mb=vram_used, gpu_percent=3.0,
                            on_battery=on_battery, battery_percent=None if on_battery is None else 50.0,
                            nexus_rss_mb=60)


class FakeProbe:
    """Simulated machine: loading/unloading FakeBackends changes free memory."""

    def __init__(self, **kwargs) -> None:
        self.base = kwargs
        self.backends: list[FakeBackend] = []

    def __call__(self) -> ResourceSnapshot:
        snap = snapshot(**self.base)
        ram = sum(b.spec.ram_mb for b in self.backends if b.loaded)
        vram = sum(b.spec.vram_mb for b in self.backends if b.loaded)
        return ResourceSnapshot(**{**snap.__dict__, "ram_available_mb": snap.ram_available_mb - ram,
                                   "vram_used_mb": (snap.vram_used_mb or 0) + vram})


class FakeBackend:
    def __init__(self, name, placement=Placement.GPU, ram_mb=500, vram_mb=5000, idle_unload_s=300,
                 probe: FakeProbe | None = None) -> None:
        self.spec = ModelSpec(name=name, kind="llm", placement=placement, ram_mb=ram_mb,
                              vram_mb=vram_mb if placement is Placement.GPU else 0, idle_unload_s=idle_unload_s)
        self.loaded = False
        self.loads = 0
        if probe is not None:
            probe.backends.append(self)

    def load(self) -> None:
        self.loaded = True
        self.loads += 1

    def unload(self) -> None:
        self.loaded = False

    def is_loaded(self) -> bool:
        return self.loaded


class FakeChat(FakeBackend):
    """A ChatProvider + ModelBackend that answers with canned text."""

    def __init__(self, name, *, reply='{"language":"en","actions":[],"say":"hi"}', available=True,
                 locality=Locality.LOCAL, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.info = ProviderInfo(name=name, locality=locality, model=name)
        self.reply = reply
        self._available = available
        self.calls: list = []

    def available(self) -> bool:
        return self._available

    def chat(self, messages, *, schema=None, temperature=0.2) -> ChatReply:
        self.calls.append((list(messages), schema))
        return ChatReply(text=self.reply, model=self.info.model, seconds=0.01)


class ScriptedPlanner:
    """Returns pre-written plans (or raises) in order, recording what it was asked."""

    def __init__(self, *plans) -> None:
        self.plans = list(plans)
        self.seen: list[tuple[str, list]] = []
        self.memories_seen: list[list] = []

    def plan(self, text, *, history=(), memories=()) -> Plan:
        self.seen.append((text, list(history)))
        self.memories_seen.append(list(memories))
        item = self.plans.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def plan(say="", *actions, language="en") -> Plan:
    return Plan(say=say, actions=tuple(ActionStep(t, dict(a)) for t, a in actions), language=language)


@pytest.fixture
def resources(clock) -> ResourceManager:
    return ResourceManager(FakeProbe(), clock=clock, snapshot_ttl=0)


# Re-exported for tests
__all__ = ["FakePlatform", "Recorder", "make_tool", "Param", "FakeProbe", "FakeBackend", "FakeChat",
           "ScriptedPlanner", "plan", "snapshot"]
