"""Structured tools: the only way NEXUS can act on the computer.

The AI never runs shell commands. It picks a registered Tool and supplies
arguments; the arguments are validated here, judged by the security policy,
and only then does the tool run.
"""

from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
from urllib.parse import urlsplit

MAX_TEXT_LEN = 2000
_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_KINDS = ("str", "int", "float", "bool", "path", "url")
_JSON_TYPES = {"str": "string", "path": "string", "url": "string",
               "int": "integer", "float": "number", "bool": "boolean"}
_TRUE = {"true", "yes", "y", "1", "on"}
_FALSE = {"false", "no", "n", "0", "off"}


class Risk(enum.IntEnum):
    """Declared risk of a tool. The policy may raise it, never lower it."""

    SAFE = 0     # runs without asking
    CONFIRM = 1  # runs only after the user says yes
    HIGH = 2     # needs strong authentication; refused until that exists


class ValidationError(ValueError):
    """Tool arguments were missing, unknown or malformed."""


def is_network_or_device_path(text: str) -> bool:
    r"""True for UNC (\\server\share) and device (\\?\, \\.\) paths.

    These are rejected *before* any filesystem call: merely resolving a UNC
    path makes Windows connect to that server and offer your login hash.
    """
    if os.name == "nt":
        return text.replace("/", "\\").startswith("\\\\")
    return text.startswith("//")


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    description: str
    required: bool = True
    default: Any = None
    choices: tuple[str, ...] | None = None
    writes: bool = False     # kind="path": the tool creates/modifies this path
    sensitive: bool = False  # value is masked in the audit log

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown param kind {self.kind!r}")

    def coerce(self, raw: Any) -> Any:
        if self.kind == "bool":
            value = self._to_bool(raw)
        elif self.kind in ("int", "float"):
            value = self._to_number(raw)
        else:
            value = self._to_text(raw)
            if self.kind == "path":
                value = self._to_path(value)
            elif self.kind == "url":
                self._check_url(value)
        if self.choices is not None and value not in self.choices:
            raise ValidationError(f"{self.name}: must be one of {', '.join(self.choices)}")
        return value

    def _to_bool(self, raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValidationError(f"{self.name}: expected true or false")

    def _to_number(self, raw: Any) -> int | float:
        if isinstance(raw, bool):
            raise ValidationError(f"{self.name}: expected a number")
        try:
            return int(raw) if self.kind == "int" else float(raw)
        except (TypeError, ValueError):
            raise ValidationError(f"{self.name}: expected a number") from None

    def _to_text(self, raw: Any) -> str:
        if not isinstance(raw, str):
            raise ValidationError(f"{self.name}: expected text")
        text = raw.strip()
        if not text:
            raise ValidationError(f"{self.name}: must not be empty")
        if len(text) > MAX_TEXT_LEN or "\x00" in text:
            raise ValidationError(f"{self.name}: value is too long or contains invalid characters")
        return text

    def _to_path(self, text: str) -> Path:
        if is_network_or_device_path(text):
            raise ValidationError(f"{self.name}: network and device paths are not supported")
        path = Path(os.path.expanduser(text))
        if not path.is_absolute():
            raise ValidationError(
                f"{self.name}: needs a full path such as C:\\Users\\you\\Desktop\\folder "
                f"(or ~\\Desktop\\folder), got {text!r}")
        # Resolve "..", symlinks and junctions so the policy judges the real target.
        return path.resolve(strict=False)

    def _check_url(self, text: str) -> None:
        parts = urlsplit(text)
        if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
            raise ValidationError(f"{self.name}: only http:// and https:// web addresses are allowed")


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    # True: the tool checked its effect happened. False: it checked and it did not.
    # None: the effect cannot be checked (e.g. a browser tab opening), so NEXUS
    # must not claim more than "I asked the system to do it".
    verified: bool | None = None
    # Set (with ok=False) when the request matched several things: nothing was done, the user must choose.
    options: tuple[str, ...] = ()
    # ok=False because NEXUS intentionally won't do it (a security refusal) - not a failed attempt.
    refused: bool = False
    # ok=False because this is not supported yet (e.g. a launcher NEXUS can't drive) - not a failed attempt.
    unsupported: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: Risk
    params: tuple[Param, ...]
    run: Callable[..., ToolResult]
    describe: Callable[[Mapping[str, Any]], str] | None = None  # human wording for confirmations
    # Progress wording spoken *before* a slow action runs ("launching GTA V").
    # Written by the tool, never by the AI, so it can't claim success. None = quick tool, no announcement.
    progress: Callable[[Mapping[str, Any]], str] | None = None
    # If True the kernel passes origin=<device id or None> to run(): who asked. Set by the kernel
    # from the authenticated request, never an argument the AI can supply.
    receives_origin: bool = False

    def __post_init__(self) -> None:
        if not _TOOL_NAME.fullmatch(self.name):
            raise ValueError(f"invalid tool name {self.name!r}")

    def validate(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        known = {p.name for p in self.params}  # "origin" is never a param, so it can't be supplied
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValidationError(f"{self.name} does not take: {', '.join(unknown)}")
        args: dict[str, Any] = {}
        for p in self.params:
            if raw.get(p.name) is not None:
                args[p.name] = p.coerce(raw[p.name])
            elif p.required:
                raise ValidationError(f"{self.name} needs '{p.name}' ({p.description})")
            else:
                args[p.name] = p.default
        return args

    def summary(self, args: Mapping[str, Any]) -> str:
        if self.describe is not None:
            return self.describe(args)
        sensitive = {p.name for p in self.params if p.sensitive}
        shown = ", ".join(f"{k}={'***' if k in sensitive else v}"
                          for k, v in args.items() if v is not None)
        return f"Run {self.name}({shown})"

    def json_schema(self) -> dict[str, Any]:
        """Function-calling schema (OpenAI/Ollama format) for the AI planner."""
        props: dict[str, Any] = {}
        for p in self.params:
            prop: dict[str, Any] = {"type": _JSON_TYPES[p.kind], "description": p.description}
            if p.choices:
                prop["enum"] = list(p.choices)
            props[p.name] = prop
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": [p.name for p in self.params if p.required],
                },
            },
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self) -> Iterator[Tool]:
        return iter(sorted(self._tools.values(), key=lambda t: t.name))

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.json_schema() for tool in self]
