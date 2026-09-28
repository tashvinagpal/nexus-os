"""Wires the AI layer (resources, providers, router, planner, conversation, voice) onto a Runtime."""

from __future__ import annotations

import math
import os
import platform as _stdlib_platform
from dataclasses import dataclass

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.planner import LLMPlanner
from nexus.ai.providers import HybridChat, ProviderOption, SelectionContext
from nexus.ai.providers.ollama import OllamaChat
from nexus.ai.router import FastPathRouter
from nexus.ai.voice import SilentTTS, TTSProvider
from nexus.ai.voice.speaker import Speaker
from nexus.core.config import AISettings, VoiceSettings
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
    voice: object = None      # VoiceInput, or None when voice input is turned off
    voice_note: str = ""      # why voice input is off


def _stt_backends(runtime: Runtime, voice: VoiceSettings | None) -> list:
    if voice is None or not voice.enabled:
        return []
    from nexus.ai.voice.whisper import WhisperSTT

    auto_threads = max(1, min(6, (os.cpu_count() or 2) // 2))
    return [WhisperSTT(o.model, placement=Placement(o.placement), compute_type=o.compute_type,
                       threads=o.threads or auto_threads, ram_mb=o.ram_mb, vram_mb=o.vram_mb,
                       idle_unload_s=voice.stt_idle_unload_seconds, languages=voice.languages,
                       timeout_s=voice.transcribe_timeout_seconds,
                       log_file=runtime.settings.data_dir / "logs" / "stt-worker.log") for o in voice.stt]


def _app_names(runtime: Runtime):
    """Installed apps' names and the user's own names for them, for recognising them by sound."""
    if runtime.apps is None:
        return None

    def names() -> list[str]:
        found = [alias for record in runtime.apps.catalog().records for alias in record.aliases]
        if runtime.memory is not None:
            found += [alias for alias, _ in runtime.memory.aliases()]
        return found
    return names


def _voice_input(runtime: Runtime, voice: VoiceSettings, stt_backends: list, resources: ResourceManager,
                 context, *, fast_path=None) -> object:
    from nexus.ai.voice.listener import VoiceInput
    from nexus.ai.voice.recorder import VADConfig
    from nexus.ai.voice.stt import HybridSTT, STTOption
    from nexus.system.audio import MicrophoneManager, PortAudioCapture

    options = [STTOption(b, b.spec, quality=len(stt_backends) - rank, game_mode_ok=o.game_mode)
               for rank, (b, o) in enumerate(zip(stt_backends, voice.stt))]
    capture = PortAudioCapture()
    # The platform's own device list (Windows Core Audio) knows defaults, form factors and
    # loopback/virtual devices; PortAudio's list is the fallback on other platforms.
    list_inputs = getattr(runtime.platform, "audio_inputs", None) or capture.list_inputs
    mics = MicrophoneManager(list_inputs, preferred=voice.microphone)
    vad = VADConfig(end_silence_ms=voice.end_silence_ms, no_speech_timeout_s=voice.no_speech_timeout_seconds,
                    max_utterance_s=voice.max_utterance_seconds)
    return VoiceInput(mics, capture, HybridSTT(options, resources, context), resources, vad=vad,
                      permission=getattr(runtime.platform, "microphone_allowed", None),
                      min_confidence=voice.min_confidence, romanize=voice.romanize,
                      language=None if voice.language == "auto" else voice.language,
                      device_poll_s=voice.device_poll_seconds, fast_path=fast_path, names=_app_names(runtime))


def create_assistant(runtime: Runtime, ai: AISettings, *, tts: TTSProvider | None = None,
                     voice: VoiceSettings | None = None) -> Assistant:
    idle = math.inf if ai.keep_llm_loaded else ai.llm_idle_unload_seconds
    providers = [OllamaChat(o.model, placement=Placement(o.placement), ram_mb=o.ram_mb, vram_mb=o.vram_mb,
                            idle_unload_s=idle, base_url=ai.ollama_url) for o in ai.llm]  # best first
    stt_backends = _stt_backends(runtime, voice)

    caps = detect_capabilities(cpu_name=runtime.platform.cpu_name(),
                               model_specs=[p.spec for p in providers] + [b.spec for b in stt_backends],
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
    router = FastPathRouter(apps=runtime.apps)
    engine = ConversationEngine(runtime.kernel, planner, resources=resources, memory=runtime.memory,
                                router=router,
                                voice_confirm_min_confidence=voice.confirm_min_confidence if voice else 0.6)
    speaker = Speaker(tts or SilentTTS(), resources)
    voice_input, voice_note = None, "voice input is turned off in config/nexus.toml"
    if voice is not None and voice.enabled:
        if not stt_backends:
            voice_note = "no speech-to-text model is configured ([[voice.stt]] in config/nexus.toml)"
        else:
            voice_input, voice_note = _voice_input(runtime, voice, stt_backends, resources, context,
                                                   fast_path=lambda text: router.route(text) is not None), ""
    resources.start_housekeeping()
    return Assistant(ai, caps, device, nexus, resources, chat, engine, speaker, voice_input, voice_note)
