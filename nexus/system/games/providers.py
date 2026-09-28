"""The GameProvider interface, plus finding a game's real process inside its install folder."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol

from nexus.system.apps.names import aliases_for, normalize
from nexus.system.apps.records import GameRecord
from nexus.system.platform.base import LaunchSpec


class GameProvider(Protocol):
    name: str  # "steam" | "epic" | "rockstar"

    def discover(self) -> list[GameRecord]:
        """Installed games (read from the launcher's own files/registry, never by running it)."""
        ...

    def launch_spec(self, game: GameRecord) -> LaunchSpec:
        """How this launcher starts the game (URI or the game's own exe)."""
        ...


# Processes that mean "the game itself is running" (not its launcher), where we know them.
KNOWN_GAME_PROCESSES = {
    "grand theft auto v": ("gta5.exe", "gta5_enhanced.exe"),
    "red dead redemption 2": ("rdr2.exe",),
    "grand theft auto iv": ("gtaiv.exe",),
    "valorant": ("valorant-win64-shipping.exe",),
    "rocket league": ("rocketleague.exe",),
    "counter strike 2": ("cs2.exe",),
    "fortnite": ("fortniteclient-win64-shipping.exe",),
}

_SKIP_DIRS = {"redistributables", "redist", "_commonredist", "commonredist", "directx", "dotnet", "vcredist",
              "support", "installer", "installers", "__installer", "battleye", "easyanticheat", "eac",
              "crashreporter", "crashpad", "cc_bin", "prereq", "prerequisites", "tools", "engine", "thirdparty"}
_SKIP_TOKENS = {"be", "eac", "setup", "install", "installer", "unins", "uninstall", "redist", "launcher", "updater",
                "update", "crash", "reporter", "helper", "uploader", "ffmpeg", "benchmark", "server", "config",
                "settings", "diag", "diagnostics", "webhelper", "cef", "service"}
_SKIP_SUBSTRINGS = ("crashreport", "unins", "redist", "setup", "install", "launcher", "anticheat", "dxsetup")


def _stem_tokens(stem: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    return [t.lower() for t in re.split(r"[\s_\-.]+", spaced) if t]


def _is_helper(stem: str) -> bool:
    low = stem.lower()
    return any(t in _SKIP_TOKENS for t in _stem_tokens(stem)) or any(s in low for s in _SKIP_SUBSTRINGS)


def game_process_names(title: str, install_dir: Path | None, *, max_depth: int = 3,
                       max_entries: int = 4000) -> frozenset[str]:
    """The exe that is the game itself. Known titles first; otherwise the exe in the install folder whose
    name best matches the title (not the biggest file: setup and ffmpeg binaries are often bigger)."""
    core = " ".join(t for t in normalize(title).split() if t not in ("legacy", "enhanced", "remastered"))
    for known, names in KNOWN_GAME_PROCESSES.items():
        if core.startswith(known):
            return frozenset(names)
    if install_dir is None or not install_dir.is_dir():
        return frozenset()
    aliases = aliases_for(title)
    title_tokens = set(core.split())
    candidates: list[tuple[float, int, int, str]] = []
    seen = 0
    root_depth = len(install_dir.parts)
    for dirpath, dirnames, filenames in os.walk(install_dir):
        depth = len(Path(dirpath).parts) - root_depth
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIRS and depth < max_depth]
        for f in filenames:
            seen += 1
            if not f.lower().endswith(".exe") or _is_helper(Path(f).stem):
                continue
            stem = normalize(Path(f).stem.replace("_", " "))
            similarity = 3.0 if (stem in aliases or stem.replace(" ", "") in aliases) else 0.0
            similarity += len(set(stem.split()) & title_tokens)
            try:
                size = os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                size = 0
            candidates.append((similarity, -depth, size, f.lower()))
        if seen > max_entries:
            break
    if not candidates:
        return frozenset()
    candidates.sort(reverse=True)
    return frozenset({candidates[0][3]})
