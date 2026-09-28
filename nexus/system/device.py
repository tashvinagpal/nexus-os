"""What kind of device NEXUS is running on, and what it can do.

The same NEXUS software runs on a desktop, a laptop and (as a client) a
phone; behaviour adapts to these capabilities instead of being hard-coded.
Devices share their capabilities with each other (mesh STATUS), so a phone
knows "the PC can run the 8B model, I can't".
"""

from __future__ import annotations

import enum
import platform as _stdlib_platform
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable

import psutil

from nexus.system import info

if TYPE_CHECKING:
    from nexus.core.resources import ModelSpec


class DeviceRole(str, enum.Enum):
    DESKTOP = "desktop"  # most powerful node: local models, games, heavy automation
    LAPTOP = "laptop"    # same software, adapts to battery / thermals / smaller models
    MOBILE = "mobile"    # client and control surface only; never runs desktop tools
    OTHER = "other"


@dataclass(frozen=True)
class DeviceCapabilities:
    role: DeviceRole
    cpu_threads: int
    ram_mb: int
    gpu_name: str | None
    vram_mb: int | None
    has_battery: bool
    cpu_name: str = ""
    on_battery: bool | None = None
    network_up: bool | None = None
    supported_models: tuple[str, ...] = ()  # models whose full size fits this device at all

    @property
    def can_run_local_llm(self) -> bool:
        """Enough memory for at least a small (~3B) local model."""
        return (self.vram_mb or 0) >= 4000 or self.ram_mb >= 12000

    @property
    def can_execute_desktop_tools(self) -> bool:
        return self.role in (DeviceRole.DESKTOP, DeviceRole.LAPTOP)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        d["supported_models"] = list(self.supported_models)
        return d


def supported_models(specs: Iterable[ModelSpec], *, ram_mb: int, vram_mb: int | None,
                     ram_reserve_mb: int = 2048, vram_reserve_mb: int = 1024) -> tuple[str, ...]:
    """Models that could fit this device even when nothing else is running.

    A static check on total memory ("this phone can't run the 8B model").
    Whether a model fits *right now* is the ResourceManager's admission check.
    """
    from nexus.core.resources import Placement

    ok = []
    for spec in specs:
        if ram_mb - ram_reserve_mb < spec.ram_mb:
            continue
        if spec.placement is Placement.GPU and (vram_mb is None or vram_mb - vram_reserve_mb < spec.vram_mb):
            continue
        ok.append(spec.name)
    return tuple(ok)


def detect_capabilities(*, cpu_name: str = "", model_specs: Iterable[ModelSpec] = (),
                        ram_reserve_mb: int = 2048, vram_reserve_mb: int = 1024) -> DeviceCapabilities:
    from nexus.core.resources import network_up

    battery = psutil.sensors_battery()
    gpus = info.gpus()
    gpu = gpus[0] if gpus else None
    ram_mb = psutil.virtual_memory().total // 2**20
    vram_mb = int(gpu["vram_total_mb"]) if gpu and gpu["vram_total_mb"] is not None else None
    return DeviceCapabilities(
        role=DeviceRole.LAPTOP if battery is not None else DeviceRole.DESKTOP,
        cpu_threads=psutil.cpu_count(logical=True) or 1,
        ram_mb=ram_mb,
        gpu_name=gpu["name"] if gpu else None,
        vram_mb=vram_mb,
        has_battery=battery is not None,
        cpu_name=cpu_name or _stdlib_platform.processor(),
        on_battery=None if battery is None else not battery.power_plugged,
        network_up=network_up(),
        supported_models=supported_models(model_specs, ram_mb=ram_mb, vram_mb=vram_mb,
                                          ram_reserve_mb=ram_reserve_mb, vram_reserve_mb=vram_reserve_mb),
    )
