from __future__ import annotations

import os
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
        if section not in ("ai", "resources", "apps"):
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
