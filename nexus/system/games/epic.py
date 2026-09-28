"""Epic Games: installed games from the launcher's manifest files."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from nexus.system.apps.names import aliases_for
from nexus.system.apps.records import AppKind, GameRecord
from nexus.system.games.providers import game_process_names
from nexus.system.platform import Platform
from nexus.system.platform.base import LaunchMethod, LaunchSpec


class EpicProvider:
    name = "epic"

    def __init__(self, platform: Platform, *, manifests_dir: Path | None = None) -> None:
        self.platform = platform
        self._manifests = manifests_dir

    def manifests_dir(self) -> Path | None:
        if self._manifests is not None:
            return self._manifests
        data = self.platform.known_folders().get("program_data")
        return data / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests" if data else None

    def discover(self) -> list[GameRecord]:
        folder = self.manifests_dir()
        if folder is None or not folder.is_dir():
            return []
        games = []
        for item in sorted(folder.glob("*.item")):
            try:
                m = json.loads(item.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            name, location, exe = m.get("DisplayName"), m.get("InstallLocation"), m.get("LaunchExecutable")
            if not (name and location and m.get("AppName") and m.get("CatalogNamespace") and m.get("CatalogItemId")):
                continue
            if m.get("bIsIncompleteInstall") or "games" not in [c.lower() for c in m.get("AppCategories", ["games"])]:
                continue  # half-installed, or an engine/plugin rather than a game
            install = Path(location)
            processes = frozenset({Path(exe).name.lower()}) if exe else game_process_names(name, install)
            uri = ("com.epicgames.launcher://apps/" + quote(f"{m['CatalogNamespace']}:{m['CatalogItemId']}:"
                                                             f"{m['AppName']}", safe="") + "?action=launch&silent=true")
            games.append(GameRecord(
                id=f"epic:{m['AppName']}", display_name=name.replace("®", "").replace("™", "").strip(),
                aliases=aliases_for(name), kind=AppKind.GAME, source="epic", launch_method=LaunchMethod.URI,
                target=uri, install_dir=str(install), process_names=processes, launcher="epic",
                game_id=m["AppName"]))
        return games

    def launch_spec(self, game: GameRecord) -> LaunchSpec:
        return LaunchSpec(LaunchMethod.URI, game.target)
