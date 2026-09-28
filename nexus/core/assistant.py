"""Wires the AI layer (resources, providers, router, planner, conversation, voice) onto a Runtime."""

from __future__ import annotations

import math
import platform as _stdlib_platform
from dataclasses import dataclass

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.planner import LLMPlanner
from nexus.ai.providers import HybridChat, ProviderOption, SelectionContext
from nexus.ai.providers.ollama import OllamaChat
from nexus.ai.router import FastPathRouter
from nexus.ai.voice import SilentTTS, TTSProvider
from nexus.ai.voice.speaker import Speaker
from nexus.core.config import AISettings
from nexus.core.resources import Placement, ResourceManager
from nexus.core.runtime import Runtime
from nexus.security.identity import DeviceKey, NexusIdentity
from nexus.system.device import DeviceCapabilities, detect_capabilities


@dataclass(frozen=True)
class Assistant:
    settings: AISettings
    capabilities: DeviceCapabilities
    device: DeviceKey
    nexus: NexusIdentity
    resources: ResourceManager
    chat: HybridChat
    engine: ConversationEngine
    speaker: Speaker


def create_assistant(runtime: Runtime, ai: AISettings, *, tts: TTSProvider | None = None) -> Assistant:
    idle = math.inf if ai.keep_llm_loaded else ai.llm_idle_unload_seconds
    providers = [OllamaChat(o.model, placement=Placement(o.placement), ram_mb=o.ram_mb, vram_mb=o.vram_mb,
                            idle_unload_s=idle, base_url=ai.ollama_url) for o in ai.llm]  # best first

    caps = detect_capabilities(cpu_name=runtime.platform.cpu_name(), model_specs=[p.spec for p in providers],
                               ram_reserve_mb=ai.ram_reserve_mb, vram_reserve_mb=ai.vram_reserve_mb)
    ident_dir = runtime.settings.identity_dir
    device = DeviceKey.load_or_create(ident_dir / "device.json",
                                      name=_stdlib_platform.node() or "this-pc", role=caps.role)
    nexus = NexusIdentity.load_or_create(ident_dir / "nexus.json")

    resources = runtime.resources or ResourceManager(ram_reserve_mb=ai.ram_reserve_mb,
                                                     vram_reserve_mb=ai.vram_reserve_mb)
    resources.capabilities = caps
    options = [ProviderOption(p, p.spec, quality=len(providers) - rank) for rank, p in enumerate(providers)]
    supported = frozenset(caps.supported_models)

    def context() -> SelectionContext:
        snap = resources.snapshot()
        return SelectionContext(allow_cloud=ai.allow_cloud, network_available=bool(snap.network_up),
                                power_saving=bool(snap.on_battery) and ai.power_saving_on_battery,
                                supported=supported)

    def app_context() -> str:
        # Verified facts from discovery and the user's own app names (also shown in /aliases),
        # so the model can say "GTA V is installed" or "your browser is Chrome" truthfully.
        if runtime.apps is None:
            return ""
        lines = []
        games = [g.display_name for g in runtime.apps.catalog().games()][:15]
        if games:
            lines.append("Installed games on this PC: " + ", ".join(games) + ".")
        if runtime.memory is not None:
            named = [f'"{alias}" = {app}' for alias, app in runtime.memory.aliases()][:15]
            if named:
                lines.append("The user's own names for apps: " + ", ".join(named) + ".")
        return "\n".join(lines)

    chat = HybridChat(options, resources, context)
    planner = LLMPlanner(chat, list(runtime.registry), context=app_context)
    engine = ConversationEngine(runtime.kernel, planner, resources=resources, memory=runtime.memory,
                                router=FastPathRouter(apps=runtime.apps))
    speaker = Speaker(tts or SilentTTS(), resources)
    resources.start_housekeeping()
    return Assistant(ai, caps, device, nexus, resources, chat, engine, speaker)
