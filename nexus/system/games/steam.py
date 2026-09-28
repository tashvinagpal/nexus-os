"""Steam: installed games from the library manifests. Launch with steam://rungameid/<appid>."""

from __future__ import annotations

import re
from pathlib import Path

from nexus.system.apps.names import aliases_for
from nexus.system.apps.records import AppKind, GameRecord
from nexus.system.games.providers import game_process_names
from nexus.system.platform import Platform
from nexus.system.platform.base import LaunchMethod, LaunchSpec

_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}])')
# Steam installs these as "apps" but they are not games.
_NOT_GAMES = re.compile(r"redistributable|steam linux runtime|proton|steamworks|dedicated server|sdk\b", re.I)


def parse_vdf(text: str) -> dict:
    """Valve KeyValues text -> nested dicts."""
    root: dict = {}
    stack = [root]
    key = None
    for match in _TOKEN.finditer(text):
        string, brace = match.groups()
        if brace == "{":
            child: dict = {}
            stack[-1][key] = child
            stack.append(child)
            key = None
        elif brace == "}":
            if len(stack) > 1:
                stack.pop()
        elif key is None:
            key = string.replace("\\\\", "\\")
        else:
            stack[-1][key] = string.replace("\\\\", "\\")
            key = None
    return root


class SteamProvider:
    name = "steam"

    def __init__(self, platform: Platform, *, steam_dir: Path | None = None) -> None:
        self.platform = platform
        self._steam_dir = steam_dir

    def steam_dir(self) -> Path | None:
        if self._steam_dir is not None:
            return self._steam_dir
        for key, value in ((r"HKCU\Software\Valve\Steam", "SteamPath"),
                           (r"HKLM\SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath")):
            found = self.platform.read_registry(key, value)
            if found and Path(found).is_dir():
                return Path(found)
        return None

    def libraries(self) -> list[Path]:
        steam = self.steam_dir()
        if steam is None:
            return []
        libs = [steam]
        vdf = steam / "steamapps" / "libraryfolders.vdf"
        if vdf.is_file():
            data = parse_vdf(vdf.read_text(encoding="utf-8", errors="replace"))
            folders = data.get("libraryfolders") or data.get("LibraryFolders") or {}
            for value in folders.values():
                path = value.get("path") if isinstance(value, dict) else value
                if path and Path(path).is_dir():
                    libs.append(Path(path))
        unique: dict[str, Path] = {}
        for lib in libs:
            unique.setdefault(str(lib.resolve()).lower(), lib)
        return list(unique.values())

    def discover(self) -> list[GameRecord]:
        games = []
        for lib in self.libraries():
            for manifest in sorted((lib / "steamapps").glob("appmanifest_*.acf")):
                state = parse_vdf(manifest.read_text(encoding="utf-8", errors="replace")).get("AppState", {})
                appid, name, installdir = state.get("appid"), state.get("name"), state.get("installdir")
                try:
                    installed = int(state.get("StateFlags", "0")) & 4
                except ValueError:
                    installed = 0
                if not (appid and name and installdir and installed) or _NOT_GAMES.search(name):
                    continue
                folder = lib / "steamapps" / "common" / installdir
                games.append(GameRecord(
                    id=f"steam:{appid}", display_name=name, aliases=aliases_for(name), kind=AppKind.GAME,
                    source="steam", launch_method=LaunchMethod.URI, target=f"steam://rungameid/{appid}",
                    install_dir=str(folder), process_names=game_process_names(name, folder),
                    publisher=None, launcher="steam", game_id=str(appid)))
        return games

    def launch_spec(self, game: GameRecord) -> LaunchSpec:
        return LaunchSpec(LaunchMethod.URI, f"steam://rungameid/{game.game_id}")
