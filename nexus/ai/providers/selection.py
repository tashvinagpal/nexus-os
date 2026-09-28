"""Pick the best provider that can actually run right now.

Options are tried best-first. An option is skipped if its locality isn't
allowed (cloud off, no network), the device is saving power and the option
is heavy, the provider isn't available, or the ResourceManager won't admit
its model (not enough free RAM/VRAM, or game mode and it needs the GPU).
This is what makes the same NEXUS code run a desktop, a laptop or (later) a
phone client well: they just have different option lists and resources.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from nexus.ai.providers.base import ChatMessage, ChatProvider, ChatReply, Locality, ProviderUnavailable
from nexus.core.resources import ModelSpec, ResourceManager, ResourceUnavailable

HEAVY_MODEL_MB = 4096  # skipped while saving power (e.g. laptop on low battery)


@dataclass(frozen=True)
class ProviderOption:
    provider: ChatProvider
    spec: ModelSpec | None  # None when the model runs elsewhere (LAN/cloud)
    quality: int            # higher is better


@dataclass(frozen=True)
class SelectionContext:
    allow_cloud: bool = False
    network_available: bool = False
    power_saving: bool = False
    # Models this device can run at all (DeviceCapabilities.supported_models); None = don't check.
    supported: frozenset[str] | None = None


def select_provider(options: Sequence[ProviderOption], resources: ResourceManager,
                    ctx: SelectionContext) -> tuple[ProviderOption | None, list[str]]:
    """Returns (chosen option or None, reasons each better option was skipped)."""
    skipped: list[str] = []
    for option in sorted(options, key=lambda o: -o.quality):
        name, locality = option.provider.info.name, option.provider.info.locality
        if locality is Locality.CLOUD and not ctx.allow_cloud:
            skipped.append(f"{name}: cloud AI is turned off in settings")
            continue
        if locality is not Locality.LOCAL and not ctx.network_available:
            skipped.append(f"{name}: no network")
            continue
        if option.spec is not None and ctx.supported is not None and option.spec.name not in ctx.supported:
            skipped.append(f"{name}: this device can't run it")
            continue
        if ctx.power_saving and option.spec and option.spec.ram_mb + option.spec.vram_mb > HEAVY_MODEL_MB:
            skipped.append(f"{name}: too heavy while saving power")
            continue
        if not option.provider.available():
            skipped.append(f"{name}: not available")
            continue
        if option.spec is not None:
            admission = resources.admit(option.spec)
            if not admission.ok:
                skipped.append(f"{name}: {admission.reason}")
                continue
        return option, skipped
    return None, skipped


class HybridChat:
    """The HybridProvider: a ChatProvider that routes each call to the best option available."""

    def __init__(self, options: Sequence[ProviderOption], resources: ResourceManager,
                 context: Callable[[], SelectionContext]) -> None:
        self.options = list(options)
        self.resources = resources
        self.context = context
        self.last_choice: str | None = None
        for option in self.options:
            if option.spec is not None:
                resources.register(option.provider)  # type: ignore[arg-type]  # local providers are ModelBackends

    def chat(self, messages: Sequence[ChatMessage], *, schema: dict[str, Any] | None = None,
             temperature: float = 0.2) -> ChatReply:
        option, skipped = select_provider(self.options, self.resources, self.context())
        if option is None:
            raise ProviderUnavailable("no AI model can run right now: " + "; ".join(skipped))
        self.last_choice = option.provider.info.name
        hold = self.resources.use(option.spec.name) if option.spec else nullcontext()
        try:
            with hold:
                return option.provider.chat(messages, schema=schema, temperature=temperature)
        except ResourceUnavailable as exc:
            raise ProviderUnavailable(str(exc)) from None
