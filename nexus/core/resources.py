"""Resource manager: keeps NEXUS light while idle and out of the way of games.

- Models are registered, not loaded. They load on first use, only if they
  fit in the RAM/VRAM that is actually free (keeping a reserve for you).
- Idle models are unloaded after their idle timeout. If a model doesn't fit,
  idle models NEXUS loaded earlier are evicted first.
- GAME_MODE unloads every NEXUS model from the GPU and refuses to load new
  ones there until game mode ends.

States: IDLE, LISTENING, THINKING, VISION, EXECUTING, SPEAKING describe what
NEXUS is doing; GAME_MODE overrides them while a game has priority.

A model is only kept loaded forever (idle_unload_s = inf) if the user chose
that in config/nexus.toml; game mode unloads it anyway.
"""

from __future__ import annotations

import enum
import logging
import math
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Protocol

import psutil

from nexus.system import info

if TYPE_CHECKING:
    from nexus.system.device import DeviceCapabilities

log = logging.getLogger(__name__)


class ResourceState(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    VISION = "vision"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    GAME_MODE = "game_mode"


class Placement(str, enum.Enum):
    GPU = "gpu"
    CPU = "cpu"


@dataclass(frozen=True)
class ModelSpec:
    name: str              # unique, e.g. "llm:llama3.1:8b@gpu"
    kind: str              # llm | stt | tts | vision | wakeword
    placement: Placement
    ram_mb: int            # host RAM while loaded (measured, see docs/architecture.md)
    vram_mb: int = 0       # VRAM while loaded
    idle_unload_s: float = 300.0


class ModelBackend(Protocol):
    spec: ModelSpec

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...


class ResourceUnavailable(RuntimeError):
    """A model can't be loaded right now (not enough memory, or game mode)."""


@dataclass(frozen=True)
class ResourceSnapshot:
    ram_total_mb: int
    ram_available_mb: int
    cpu_percent: float
    vram_total_mb: int | None
    vram_used_mb: int | None
    gpu_percent: float | None
    on_battery: bool | None
    battery_percent: float | None
    nexus_rss_mb: int
    network_up: bool | None = None  # a non-loopback interface is up (LAN); no traffic is sent to check

    @property
    def vram_free_mb(self) -> int | None:
        if self.vram_total_mb is None or self.vram_used_mb is None:
            return None
        return self.vram_total_mb - self.vram_used_mb


def network_up() -> bool:
    """True if any real network interface is up with an address. Checked locally, nothing is sent."""
    stats = psutil.net_if_stats()
    for name, addrs in psutil.net_if_addrs().items():
        if not stats.get(name) or not stats[name].isup:
            continue
        for a in addrs:
            if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                return True
            if a.family == socket.AF_INET6 and not (a.address == "::1" or a.address.lower().startswith("fe80")):
                return True
    return False


def probe_system() -> ResourceSnapshot:
    vm = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    gpus = info.gpus()
    gpu = gpus[0] if gpus else {}
    return ResourceSnapshot(
        ram_total_mb=vm.total // 2**20,
        ram_available_mb=vm.available // 2**20,
        cpu_percent=psutil.cpu_percent(interval=None),
        vram_total_mb=_int(gpu.get("vram_total_mb")),
        vram_used_mb=_int(gpu.get("vram_used_mb")),
        gpu_percent=gpu.get("usage_percent"),
        on_battery=None if battery is None else not battery.power_plugged,
        battery_percent=None if battery is None else battery.percent,
        nexus_rss_mb=psutil.Process().memory_info().rss // 2**20,
        network_up=network_up(),
    )


def _int(value: Any) -> int | None:
    return None if value is None else int(value)


@dataclass(frozen=True)
class Admission:
    ok: bool
    reason: str = ""


@dataclass
class _Slot:
    backend: ModelBackend
    last_used: float
    in_use: int = 0


class ResourceManager:
    def __init__(self, probe: Callable[[], ResourceSnapshot] = probe_system, *,
                 ram_reserve_mb: int = 2048, vram_reserve_mb: int = 1024,
                 clock: Callable[[], float] = time.monotonic, snapshot_ttl: float = 2.0,
                 capabilities: DeviceCapabilities | None = None) -> None:
        self._probe = probe
        self.capabilities = capabilities
        self.ram_reserve_mb = ram_reserve_mb
        self.vram_reserve_mb = vram_reserve_mb
        self._clock = clock
        self._snapshot_ttl = snapshot_ttl
        self._snapshot: tuple[float, ResourceSnapshot] | None = None
        self._slots: dict[str, _Slot] = {}
        # What NEXUS is doing, newest last. A list rather than one saved value because activities
        # can overlap on different threads (speaking in the background while you type).
        self._activities: list[tuple[object, ResourceState]] = []
        self._activity_lock = threading.Lock()  # never the model lock: loading a model takes seconds
        self._game_mode = False
        self._lock = threading.RLock()
        self._housekeeper: threading.Thread | None = None

    # --- state -------------------------------------------------------------------

    @property
    def state(self) -> ResourceState:
        return ResourceState.GAME_MODE if self._game_mode else self.activity

    @property
    def activity(self) -> ResourceState:
        activities = self._activities
        return activities[-1][1] if activities else ResourceState.IDLE

    @property
    def game_mode(self) -> bool:
        return self._game_mode

    @contextmanager
    def busy(self, state: ResourceState) -> Iterator[None]:
        """Mark what NEXUS is doing for the duration of a block."""
        if state is ResourceState.GAME_MODE:
            raise ValueError("use enter_game_mode() for game mode")
        token = object()
        with self._activity_lock:
            self._activities.append((token, state))
        try:
            yield
        finally:
            with self._activity_lock:
                self._activities = [a for a in self._activities if a[0] is not token]

    def enter_game_mode(self) -> list[str]:
        """Give the GPU to the game: unload every idle NEXUS model on the GPU."""
        with self._lock:
            self._game_mode = True
            unloaded = [name for name, slot in self._slots.items()
                        if slot.backend.spec.placement is Placement.GPU and slot.in_use == 0
                        and self._unload(slot)]
        log.info("game mode on; unloaded %s", unloaded)
        return unloaded

    def exit_game_mode(self) -> None:
        self._game_mode = False
        log.info("game mode off")

    # --- measurement -------------------------------------------------------------

    def snapshot(self, *, fresh: bool = False) -> ResourceSnapshot:
        now = self._clock()
        if fresh or self._snapshot is None or now - self._snapshot[0] > self._snapshot_ttl:
            self._snapshot = (now, self._probe())
        return self._snapshot[1]

    # --- models ------------------------------------------------------------------

    def register(self, backend: ModelBackend) -> None:
        name = backend.spec.name
        if name in self._slots:
            raise ValueError(f"model {name!r} is already registered")
        self._slots[name] = _Slot(backend, last_used=self._clock())

    def admit(self, spec: ModelSpec) -> Admission:
        """Would loading this model be allowed right now?"""
        if self._game_mode and spec.placement is Placement.GPU:
            return Admission(False, "game mode is on, so NEXUS keeps the GPU free")
        slot = self._slots.get(spec.name)
        if slot is not None and slot.backend.is_loaded():
            return Admission(True, "already loaded")
        snap = self.snapshot(fresh=True)
        if snap.ram_available_mb - spec.ram_mb < self.ram_reserve_mb:
            return Admission(False, f"needs ~{spec.ram_mb} MB RAM, {snap.ram_available_mb} MB free "
                                    f"(keeping {self.ram_reserve_mb} MB for your apps)")
        if spec.placement is Placement.GPU:
            free = snap.vram_free_mb
            if free is None:
                return Admission(False, "no usable GPU detected")
            if free - spec.vram_mb < self.vram_reserve_mb:
                return Admission(False, f"needs ~{spec.vram_mb} MB VRAM, {free} MB free "
                                        f"(keeping {self.vram_reserve_mb} MB spare)")
        return Admission(True)

    @contextmanager
    def use(self, name: str) -> Iterator[ModelBackend]:
        """Load the model if needed (and allowed), hold it while the block runs."""
        slot = self._slots.get(name)
        if slot is None:
            raise ValueError(f"unknown model {name!r}")
        spec = slot.backend.spec
        with self._lock:
            admission = self.admit(spec)
            if not admission.ok and not (self._game_mode and spec.placement is Placement.GPU):
                # Make room by unloading idle models, least recently used first.
                for victim in self._idle_slots(except_name=name):
                    if self._unload(victim):
                        admission = self.admit(spec)
                        if admission.ok:
                            break
            if not admission.ok:
                raise ResourceUnavailable(f"{name}: {admission.reason}")
            if not slot.backend.is_loaded():
                log.info("loading %s", name)
                slot.backend.load()
            slot.in_use += 1
        try:
            yield slot.backend
        finally:
            with self._lock:
                slot.in_use -= 1
                slot.last_used = self._clock()
                if self._game_mode and slot.backend.spec.placement is Placement.GPU and slot.in_use == 0:
                    self._unload(slot)

    def release_idle(self) -> list[str]:
        """Unload models that have been idle longer than their timeout."""
        now = self._clock()
        with self._lock:
            return [name for name, slot in self._slots.items()
                    if slot.in_use == 0 and now - slot.last_used >= slot.backend.spec.idle_unload_s
                    and self._unload(slot)]

    def unload(self, name: str) -> bool:
        """Unload one model now, unless it is in use. True if it was loaded and is now unloaded."""
        slot = self._slots.get(name)
        if slot is None:
            return False
        with self._lock:
            return slot.in_use == 0 and self._unload(slot)

    def unload_all(self) -> list[str]:
        with self._lock:
            return [name for name, slot in self._slots.items() if slot.in_use == 0 and self._unload(slot)]

    def start_housekeeping(self, interval_s: float = 30.0) -> None:
        """Background thread that unloads idle models."""
        if self._housekeeper is not None:
            return

        def loop() -> None:
            while True:
                time.sleep(interval_s)
                try:
                    unloaded = self.release_idle()
                    if unloaded:
                        log.info("unloaded idle models: %s", unloaded)
                except Exception:  # never let housekeeping die silently
                    log.exception("resource housekeeping failed")

        self._housekeeper = threading.Thread(target=loop, name="nexus-resources", daemon=True)
        self._housekeeper.start()

    def status(self) -> dict[str, Any]:
        now = self._clock()
        models = []
        for name, slot in self._slots.items():
            spec = slot.backend.spec
            loaded = slot.backend.is_loaded()
            idle = now - slot.last_used
            keep = math.isinf(spec.idle_unload_s)
            models.append({
                "name": name, "kind": spec.kind, "placement": spec.placement.value,
                "loaded": loaded, "in_use": slot.in_use > 0, "ram_mb": spec.ram_mb, "vram_mb": spec.vram_mb,
                "kept_loaded": keep,
                "unloads_in_s": (max(0, round(spec.idle_unload_s - idle))
                                 if loaded and not slot.in_use and not keep else None),
            })
        return {"state": self.state.value, "activity": self.activity.value, "game_mode": self._game_mode,
                "resources": asdict(self.snapshot()), "models": models,
                "device": self.capabilities.to_dict() if self.capabilities else None}

    # --- internals ---------------------------------------------------------------

    def _unload(self, slot: _Slot) -> bool:
        if not slot.backend.is_loaded():
            return False
        try:
            slot.backend.unload()
        except Exception:
            log.exception("failed to unload %s", slot.backend.spec.name)
            return False
        self._snapshot = None  # memory changed
        return True

    def _idle_slots(self, except_name: str) -> list[_Slot]:
        return sorted((s for n, s in self._slots.items() if n != except_name and s.in_use == 0),
                      key=lambda s: s.last_used)
