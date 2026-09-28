"""Rockstar Games Launcher titles (registry). Titles bought on Steam/Epic are left to those providers."""

from __future__ import annotations

from pathlib import Path

from nexus.system.apps.names import aliases_for, normalize
from nexus.system.apps.records import AppKind, GameRecord
from nexus.system.games.providers import game_process_names
from nexus.system.platform import Platform
from nexus.system.platform.base import LaunchMethod, LaunchSpec

ROOT = r"HKLM\SOFTWARE\WOW6432Node\Rockstar Games"
_NOT_GAMES = {"launcher", "rockstar games social club", "social club", "steam", "epic"}
# The exe in the install folder that starts each title through the Rockstar launcher.
LAUNCH_EXES = {
    "grand theft auto v": "PlayGTAV.exe",
    "grand theft auto v enhanced": "PlayGTAV.exe",
    "red dead redemption 2": "RDR2.exe",
    "red dead redemption": "RDR.exe",
    "grand theft auto iv": "GTAIV.exe",
    "l a noire": "LANoire.exe",
    "max payne 3": "MaxPayne3.exe",
}


class RockstarProvider:
    name = "rockstar"

    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    def discover(self) -> list[GameRecord]:
        games = []
        for title in self.platform.registry_subkeys(ROOT):
            if title.lower() in _NOT_GAMES:
                continue
            # "InstallFolder" = installed through the Rockstar launcher. Steam/Epic copies register
            # InstallFolderSteam/InstallFolderEpic instead and are discovered by those providers.
            folder = self.platform.read_registry(f"{ROOT}\\{title}", "InstallFolder")
            if not folder or not Path(folder).is_dir():
                continue
            exe_name = LAUNCH_EXES.get(normalize(title))
            exe = Path(folder) / exe_name if exe_name else None
            if exe is None or not exe.is_file():
                continue
            games.append(GameRecord(
                id=f"rockstar:{title}", display_name=title, aliases=aliases_for(title), kind=AppKind.GAME,
                source="rockstar", launch_method=LaunchMethod.EXECUTABLE, target=str(exe),
                install_dir=str(Path(folder)), process_names=game_process_names(title, Path(folder)),
                publisher="Rockstar Games", launcher="rockstar", game_id=title))
        return games

    def launch_spec(self, game: GameRecord) -> LaunchSpec:
        return LaunchSpec(LaunchMethod.EXECUTABLE, game.target, working_dir=game.install_dir)
