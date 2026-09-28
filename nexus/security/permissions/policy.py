"""Permission policy: decides whether a validated tool call may run.

This is plain, deterministic code. No AI model is consulted, and nothing an
AI says can change a decision: the policy only sees the tool's declared risk,
the validated arguments, the built-in protected folders and the user's policy
file. The policy file can add restrictions; it cannot remove built-in ones.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from nexus.tools.base import Risk, Tool, is_network_or_device_path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class Verdict(str, enum.Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    BLOCK = "block"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str = ""


class PolicyError(Exception):
    """The policy file is missing required structure or contains unknown settings."""


def _norm(path: Path | str) -> Path:
    return Path(os.path.expanduser(str(path))).resolve(strict=False)


def is_within(path: Path, root: Path) -> bool:
    """True if path is root or inside it (case-insensitive on Windows)."""
    p, r = os.path.normcase(str(path)), os.path.normcase(str(root))
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:  # different drives
        return False


# Read-only tools another paired device (e.g. your phone) may use without a
# confirmation. Everything else requested from another device asks first.
DEFAULT_REMOTE_ALLOWED = ("get_system_info", "get_datetime")


@dataclass(frozen=True)
class DeviceRules:
    """Per-device permissions, set by the user on *this* device (pairing grants nothing)."""

    allow_without_confirmation: frozenset[str] | None = None  # None = use the [remote] default
    blocked: frozenset[str] = frozenset()


class Policy:
    def __init__(self, *, protected: Iterable[Path | str], safe_roots: Iterable[Path | str],
                 blocked_tools: Iterable[str] = (), always_confirm: Iterable[str] = (),
                 remote_allowed: Iterable[str] = DEFAULT_REMOTE_ALLOWED,
                 remote_blocked: Iterable[str] = (),
                 remote_devices: Mapping[str, DeviceRules] | None = None) -> None:
        self.protected = tuple(_norm(p) for p in protected)
        self.safe_roots = tuple(_norm(p) for p in safe_roots)
        self.blocked_tools = frozenset(blocked_tools)
        self.always_confirm = frozenset(always_confirm)
        self.remote_allowed = frozenset(remote_allowed)
        self.remote_blocked = frozenset(remote_blocked)
        self.remote_devices = dict(remote_devices or {})

    @classmethod
    def from_file(cls, file: Path, *, builtin_protected: Iterable[Path],
                  default_safe_roots: Iterable[Path]) -> Policy:
        """Load the user's policy file on top of the built-in protections.

        Fails closed: a malformed file or unknown setting raises PolicyError
        instead of silently falling back to a weaker policy.
        """
        if not file.exists():
            return cls(protected=builtin_protected, safe_roots=default_safe_roots)
        try:
            with file.open("rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise PolicyError(f"cannot read policy file {file}: {exc}") from exc

        allowed = {"paths": {"protected", "safe_roots"}, "tools": {"blocked", "always_confirm"},
                   "remote": {"allow_without_confirmation", "blocked", "devices"}}
        for section, values in data.items():
            if section not in allowed or not isinstance(values, dict):
                raise PolicyError(f"{file}: unknown section [{section}]")
            for key in values:
                if key not in allowed[section]:
                    raise PolicyError(f"{file}: unknown setting '{key}' in [{section}]")

        paths, tools, remote = data.get("paths", {}), data.get("tools", {}), data.get("remote", {})
        safe = (_str_list(paths["safe_roots"], "paths.safe_roots", file)
                if "safe_roots" in paths else list(default_safe_roots))
        remote_allowed = (_str_list(remote["allow_without_confirmation"],
                                    "remote.allow_without_confirmation", file)
                          if "allow_without_confirmation" in remote else DEFAULT_REMOTE_ALLOWED)
        devices = remote.get("devices", {})
        if not isinstance(devices, dict):
            raise PolicyError(f"{file}: [remote.devices] must be a table")
        rules = {}
        for device_id, cfg in devices.items():
            name = f"remote.devices.{device_id}"
            if not isinstance(cfg, dict) or set(cfg) - {"allow_without_confirmation", "blocked"}:
                raise PolicyError(f"{file}: [{name}] may only set allow_without_confirmation and blocked")
            rules[device_id] = DeviceRules(
                allow_without_confirmation=(frozenset(_str_list(cfg["allow_without_confirmation"], name, file))
                                            if "allow_without_confirmation" in cfg else None),
                blocked=frozenset(_str_list(cfg.get("blocked", []), name, file)))
        return cls(
            protected=[*builtin_protected, *_str_list(paths.get("protected", []), "paths.protected", file)],
            safe_roots=safe,
            blocked_tools=_str_list(tools.get("blocked", []), "tools.blocked", file),
            always_confirm=_str_list(tools.get("always_confirm", []), "tools.always_confirm", file),
            remote_allowed=remote_allowed,
            remote_blocked=_str_list(remote.get("blocked", []), "remote.blocked", file),
            remote_devices=rules,
        )

    def evaluate(self, tool: Tool, args: Mapping[str, Any], *, device: str | None = None) -> Decision:
        """Judge a validated call. `device` is None for this computer, else the paired device that asked."""
        if tool.name in self.blocked_tools:
            return Decision(Verdict.BLOCK, f"'{tool.name}' is blocked by your policy file")
        if tool.risk >= Risk.HIGH:
            return Decision(Verdict.BLOCK, "high-risk actions are disabled until strong authentication exists")
        rules = self.remote_devices.get(device, DeviceRules()) if device is not None else None
        if rules is not None and (tool.name in self.remote_blocked or tool.name in rules.blocked):
            return Decision(Verdict.BLOCK, f"'{tool.name}' can't be used from that device ({device})")

        needs_confirm = tool.risk >= Risk.CONFIRM or tool.name in self.always_confirm
        reasons = [f"'{tool.name}' always asks first"] if needs_confirm else []
        if rules is not None:
            allowed = (rules.allow_without_confirmation if rules.allow_without_confirmation is not None
                       else self.remote_allowed)
            if tool.name not in allowed:
                needs_confirm = True
                reasons.append(f"it was requested from another device ({device})")

        for param in tool.params:
            value = args.get(param.name)
            if value is None:
                continue
            if param.kind == "url" and urlsplit(str(value)).scheme.lower() not in ("http", "https"):
                return Decision(Verdict.BLOCK, "only http/https web addresses may be opened")
            if param.kind == "path" and param.writes:
                if is_network_or_device_path(str(value)):
                    return Decision(Verdict.BLOCK, "network and device paths are not allowed")
                path = _norm(value)  # re-resolve: links may have changed since validation
                if path.parent == path:
                    return Decision(Verdict.BLOCK, f"NEXUS will not modify the root of a drive ({path})")
                for root in self.protected:
                    if is_within(path, root):
                        return Decision(Verdict.BLOCK, f"{path} is inside a protected location ({root})")
                if not any(is_within(path, root) for root in self.safe_roots):
                    needs_confirm = True
                    reasons.append(f"{path} is outside your safe folders")

        if needs_confirm:
            return Decision(Verdict.CONFIRM, "; ".join(reasons))
        return Decision(Verdict.ALLOW)


def _str_list(value: Any, name: str, file: Path) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicyError(f"{file}: '{name}' must be a list of strings")
    return value
