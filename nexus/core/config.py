from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path     # runtime data: audit log, logs, identity, (later) memory database
    policy_file: Path
    config_file: Path = PROJECT_ROOT / "config" / "nexus.toml"

    @property
    def audit_file(self) -> Path:
        return self.data_dir / "audit" / "audit.jsonl"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "logs" / "nexus.log"

    @property
    def identity_dir(self) -> Path:
        return self.data_dir / "identity"


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    data_dir = Path(env["NEXUS_DATA_DIR"]) if env.get("NEXUS_DATA_DIR") else PROJECT_ROOT / "var"
    policy = (Path(env["NEXUS_POLICY_FILE"]) if env.get("NEXUS_POLICY_FILE")
              else PROJECT_ROOT / "config" / "policy.toml")
    config = (Path(env["NEXUS_CONFIG_FILE"]) if env.get("NEXUS_CONFIG_FILE")
              else PROJECT_ROOT / "config" / "nexus.toml")
    return Settings(project_root=PROJECT_ROOT, data_dir=data_dir.resolve(), policy_file=policy.resolve(),
                    config_file=config.resolve())


# --- AI / resource settings (config/nexus.toml) --------------------------------------

class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class LLMOption:
    model: str
    placement: str  # gpu | cpu
    ram_mb: int
    vram_mb: int = 0


@dataclass(frozen=True)
class AISettings:
    ollama_url: str = "http://127.0.0.1:11434"
    allow_cloud: bool = False
    power_saving_on_battery: bool = True
    llm_idle_unload_seconds: int = 300
    keep_llm_loaded: bool = False  # only if the user explicitly wants it; game mode still unloads
    llm: tuple[LLMOption, ...] = field(default_factory=tuple)  # best first
    ram_reserve_mb: int = 2048
    vram_reserve_mb: int = 1024


_AI_KEYS = {"ollama_url": str, "allow_cloud": bool, "power_saving_on_battery": bool,
            "llm_idle_unload_seconds": int, "keep_llm_loaded": bool, "llm": list}
_LLM_KEYS = {"model": str, "placement": str, "ram_mb": int, "vram_mb": int}
_RES_KEYS = {"ram_reserve_mb": int, "vram_reserve_mb": int}


def load_ai_settings(file: Path) -> AISettings:
    """Read config/nexus.toml. Unknown or mistyped settings stop startup instead of being ignored."""
    if not file.exists():
        return AISettings()
    try:
        with file.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {file}: {exc}") from exc
    for section in data:
        if section not in ("ai", "resources", "apps", "voice", "tts", "wake"):
            raise ConfigError(f"{file}: unknown section [{section}]")
    ai = _checked(data.get("ai", {}), _AI_KEYS, "ai", file)
    res = _checked(data.get("resources", {}), _RES_KEYS, "resources", file)
    options = []
    for i, raw in enumerate(ai.pop("llm", [])):
        opt = _checked(raw, _LLM_KEYS, f"ai.llm[{i}]", file)
        if "model" not in opt or "ram_mb" not in opt or opt.get("placement") not in ("gpu", "cpu"):
            raise ConfigError(f"{file}: ai.llm[{i}] needs model, placement (gpu|cpu) and ram_mb")
        options.append(LLMOption(**opt))
    if ai.get("llm_idle_unload_seconds", 300) < 30:
        raise ConfigError(f"{file}: llm_idle_unload_seconds must be at least 30")
    return AISettings(**ai, **res, llm=tuple(options))


_APP_KEYS = {"app_launch_timeout_seconds": int, "game_launch_timeout_seconds": int, "auto_game_mode": bool,
             "watch_games": bool, "store_apps": bool}


def load_app_settings(file: Path):
    """The [apps] section of config/nexus.toml."""
    from nexus.system.apps.service import AppSettings

    if not file.exists():
        return AppSettings()
    try:
        with file.open("rb") as f:
            raw = tomllib.load(f).get("apps", {})
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {file}: {exc}") from exc
    apps = _checked(raw, _APP_KEYS, "apps", file)
    for key in ("app_launch_timeout_seconds", "game_launch_timeout_seconds"):
        if key in apps and not 5 <= apps[key] <= 900:
            raise ConfigError(f"{file}: apps.{key} must be between 5 and 900")
    return AppSettings(
        app_launch_timeout_s=float(apps.get("app_launch_timeout_seconds", 20)),
        game_launch_timeout_s=float(apps.get("game_launch_timeout_seconds", 180)),
        auto_game_mode=apps.get("auto_game_mode", True), watch_games=apps.get("watch_games", True),
        store_apps=apps.get("store_apps", True))


# --- voice input ([voice] in config/nexus.toml) ----------------------------------------

@dataclass(frozen=True)
class STTModelOption:
    model: str                 # faster-whisper model name: tiny | base | small | medium | large-v3 | ...
    placement: str = "cpu"     # cpu | gpu
    compute_type: str = "int8"
    threads: int = 4           # 0 = automatic (half the logical CPUs, at most 6)
    ram_mb: int = 500          # measured, see docs/voice.md
    vram_mb: int = 0
    game_mode: bool = False    # light enough to use while a game runs


@dataclass(frozen=True)
class VoiceSettings:
    enabled: bool = True
    microphone: str = "auto"
    languages: tuple[str, ...] = ("en", "hi", "pa")
    language: str = "auto"
    romanize: bool = True
    stt: tuple[STTModelOption, ...] = field(default_factory=tuple)  # best first
    stt_idle_unload_seconds: int = 60
    transcribe_timeout_seconds: int = 30
    max_utterance_seconds: int = 15
    no_speech_timeout_seconds: int = 6
    end_silence_ms: int = 900
    min_confidence: float = 0.4
    confirm_min_confidence: float = 0.6
    device_poll_seconds: int = 5


_VOICE_KEYS = {"enabled": bool, "microphone": str, "languages": list, "language": str, "romanize": bool,
               "stt": list, "stt_idle_unload_seconds": int, "transcribe_timeout_seconds": int,
               "max_utterance_seconds": int, "no_speech_timeout_seconds": int, "end_silence_ms": int,
               "min_confidence": float, "confirm_min_confidence": float, "device_poll_seconds": int}
_STT_KEYS = {"model": str, "placement": str, "compute_type": str, "threads": int, "ram_mb": int, "vram_mb": int,
             "game_mode": bool}
_VOICE_RANGES = {"stt_idle_unload_seconds": (10, 3600), "transcribe_timeout_seconds": (5, 300),
                 "max_utterance_seconds": (3, 60), "no_speech_timeout_seconds": (2, 30),
                 "end_silence_ms": (300, 3000), "min_confidence": (0.0, 1.0), "confirm_min_confidence": (0.0, 1.0),
                 "device_poll_seconds": (0, 60)}


def load_voice_settings(file: Path) -> VoiceSettings:
    """The [voice] section of config/nexus.toml (same fail-closed rules as the rest of the file)."""
    if not file.exists():
        return VoiceSettings()
    try:
        with file.open("rb") as f:
            raw = tomllib.load(f).get("voice", {})
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {file}: {exc}") from exc
    voice = _checked(raw, _VOICE_KEYS, "voice", file)
    for key, (low, high) in _VOICE_RANGES.items():
        if key in voice and not low <= voice[key] <= high:
            raise ConfigError(f"{file}: voice.{key} must be between {low} and {high}")
    options = []
    for i, entry in enumerate(voice.pop("stt", [])):
        opt = _checked(entry, _STT_KEYS, f"voice.stt[{i}]", file)
        if "model" not in opt or "ram_mb" not in opt or opt.get("placement", "cpu") not in ("cpu", "gpu"):
            raise ConfigError(f"{file}: voice.stt[{i}] needs model, ram_mb and placement (cpu|gpu)")
        from nexus.ai.voice.whisper import WHISPER_MODELS

        if opt["model"] not in WHISPER_MODELS:
            raise ConfigError(f"{file}: voice.stt[{i}].model must be one of {', '.join(WHISPER_MODELS)}")
        if not 0 <= opt.get("threads", 4) <= 64:
            raise ConfigError(f"{file}: voice.stt[{i}].threads must be between 0 (automatic) and 64")
        options.append(STTModelOption(**opt))
    languages = voice.pop("languages", ["en", "hi", "pa"])
    if not languages or not all(isinstance(x, str) and re.fullmatch(r"[a-z]{2,3}", x) for x in languages):
        raise ConfigError(f"{file}: voice.languages must be a list of language codes like \"en\", \"hi\"")
    language = voice.pop("language", "auto")
    if language != "auto" and language not in languages:
        raise ConfigError(f"{file}: voice.language must be \"auto\" or one of voice.languages")
    return VoiceSettings(**voice, languages=tuple(languages), language=language, stt=tuple(options))


# --- voice output ([tts] in config/nexus.toml) ------------------------------------------

@dataclass(frozen=True)
class TTSSettings:
    enabled: bool = True
    provider: str = "kokoro"
    model: str = "fp32"                # kokoro model file: fp32 | int8 (see docs/tts.md for the difference)
    voice: str = "af_heart"            # English voice
    hindi_voice: str = "hf_alpha"
    romanized: str = "hindi"           # Hinglish/Punjabi in Latin letters: hindi (converted, Hindi voice) | english
    speed: float = 1.0
    volume: float = 1.0
    placement: str = "cpu"
    threads: int = 0                   # 0 = automatic
    ram_mb: int = 700                  # measured, see docs/tts.md
    idle_unload_seconds: int = 120
    game_mode: str = "quiet"           # quiet: text only while a game runs | speak
    max_spoken_chars: int = 600


_TTS_KEYS = {"enabled": bool, "provider": str, "model": str, "voice": str, "hindi_voice": str, "romanized": str,
             "speed": float, "volume": float, "placement": str, "threads": int, "ram_mb": int,
             "idle_unload_seconds": int, "game_mode": str, "max_spoken_chars": int}
_TTS_CHOICES = {"provider": ("kokoro",), "model": ("fp32", "int8", "fp16"), "romanized": ("english", "hindi"),
                "placement": ("cpu", "gpu"), "game_mode": ("quiet", "speak")}
_TTS_RANGES = {"speed": (0.5, 2.0), "volume": (0.0, 1.0), "threads": (0, 64), "ram_mb": (50, 8192),
               "idle_unload_seconds": (10, 3600), "max_spoken_chars": (50, 5000)}


def load_tts_settings(file: Path) -> TTSSettings:
    """The [tts] section of config/nexus.toml (same fail-closed rules as the rest of the file)."""
    if not file.exists():
        return TTSSettings()
    try:
        with file.open("rb") as f:
            raw = tomllib.load(f).get("tts", {})
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {file}: {exc}") from exc
    tts = _checked(raw, _TTS_KEYS, "tts", file)
    for key, choices in _TTS_CHOICES.items():
        if key in tts and tts[key] not in choices:
            raise ConfigError(f"{file}: tts.{key} must be one of {', '.join(choices)}")
    for key, (low, high) in _TTS_RANGES.items():
        if key in tts and not low <= tts[key] <= high:
            raise ConfigError(f"{file}: tts.{key} must be between {low} and {high}")
    for key in ("voice", "hindi_voice"):
        if key in tts and not re.fullmatch(r"[a-z]{2}_[a-z]+", tts[key]):
            raise ConfigError(f"{file}: tts.{key} must be a voice name like \"af_heart\"")
    return TTSSettings(**tts)


def _checked(values: Any, spec: dict[str, type], name: str, file: Path) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ConfigError(f"{file}: [{name}] must be a table")
    for key, value in values.items():
        if key not in spec:
            raise ConfigError(f"{file}: unknown setting '{key}' in [{name}]")
        typ = spec[key]
        if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
            raise ConfigError(f"{file}: '{name}.{key}' must be {typ.__name__}")
    return dict(values)


# --- wake word ([wake] in config/nexus.toml) ----------------------------------------------

@dataclass(frozen=True)
class WakeSettings:
    enabled: bool = True
    detector: str = "auto"             # auto (openwakeword if its model exists, else whisper) | openwakeword | whisper
    model: str = "nexus"               # models/wake/<model>.onnx
    threshold: float = 0.5
    patience: int = 1                  # frames in a row above the threshold
    whisper_model: str = "base"
    max_checks_per_minute: int = 20
    command_timeout_seconds: int = 6
    follow_up_seconds: int = 6
    barge_in: bool = False
    game_mode: str = "listen"          # listen | off
    chime: bool = True


_WAKE_KEYS = {"enabled": bool, "detector": str, "model": str, "threshold": float, "patience": int,
              "whisper_model": str, "max_checks_per_minute": int, "command_timeout_seconds": int,
              "follow_up_seconds": int, "barge_in": bool, "game_mode": str, "chime": bool}
_WAKE_CHOICES = {"detector": ("auto", "openwakeword", "whisper"), "game_mode": ("listen", "off")}
_WAKE_RANGES = {"threshold": (0.05, 0.99), "patience": (1, 10), "max_checks_per_minute": (1, 120),
                "command_timeout_seconds": (2, 30), "follow_up_seconds": (0, 30)}


def load_wake_settings(file: Path) -> WakeSettings:
    """The [wake] section of config/nexus.toml (same fail-closed rules as the rest of the file)."""
    if not file.exists():
        return WakeSettings()
    try:
        with file.open("rb") as f:
            raw = tomllib.load(f).get("wake", {})
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {file}: {exc}") from exc
    if isinstance(raw, dict) and isinstance(raw.get("threshold"), int) and not isinstance(raw["threshold"], bool):
        raw = {**raw, "threshold": float(raw["threshold"])}
    wake = _checked(raw, _WAKE_KEYS, "wake", file)
    for key, choices in _WAKE_CHOICES.items():
        if key in wake and wake[key] not in choices:
            raise ConfigError(f"{file}: wake.{key} must be one of {', '.join(choices)}")
    for key, (low, high) in _WAKE_RANGES.items():
        if key in wake and not low <= wake[key] <= high:
            raise ConfigError(f"{file}: wake.{key} must be between {low} and {high}")
    if "model" in wake and not re.fullmatch(r"[A-Za-z0-9_.-]+", wake["model"]):
        raise ConfigError(f"{file}: wake.model must be a file name in models/wake/ without .onnx")
    from nexus.ai.voice.whisper import WHISPER_MODELS

    if wake.get("whisper_model", "base") not in WHISPER_MODELS:
        raise ConfigError(f"{file}: wake.whisper_model must be one of {', '.join(WHISPER_MODELS)}")
    return WakeSettings(**wake)
