from __future__ import annotations

import enum
from dataclasses import dataclass, field

from nexus.system.platform.base import LaunchMethod, LaunchSpec


class AppKind(str, enum.Enum):
    APP = "app"      # a normal desktop or Store application
    GAME = "game"
    SHELL = "shell"  # command shells, terminals, script hosts, bare interpreters: never launched by NEXUS
    ADMIN = "admin"  # system administration tools (Registry Editor, .msc consoles): never launched by NEXUS


@dataclass(frozen=True)
class AppRecord:
    id: str
    display_name: str
    aliases: frozenset[str]
    kind: AppKind
    source: str                      # "start_menu", "desktop", "app_paths", "store", "steam", ... (merged: "a+b")
    launch_method: LaunchMethod
    target: str                      # exe path | .lnk path | URI | AppUserModelID
    arguments: str = ""              # from the shortcut itself
    working_dir: str | None = None
    publisher: str | None = None
    install_dir: str | None = None
    process_names: frozenset[str] = frozenset()  # lowercase exe names that mean "it is running"
    process_paths: frozenset[str] = frozenset()  # normcased exe paths that mean "it is running"
    package_family: str | None = None            # Store apps: running processes live under WindowsApps\<family>
    mediated: bool = False           # started through another launcher (Riot, BlueStacks, a Chrome profile):
                                     # the launcher's process alone doesn't prove the app itself started
    requires_elevation: bool | None = None       # None = checked at launch time
    confidence: float = 1.0          # how sure discovery is that this is a real, launchable app
    launcher: str | None = None      # games: steam | epic | rockstar | riot ...
    game_id: str | None = None       # the launcher's id (Steam appid, Epic AppName)
    unsupported_reason: str | None = None  # found, but NEXUS can't start it yet (e.g. an unsupported launcher)

    @property
    def spec(self) -> LaunchSpec:
        return LaunchSpec(self.launch_method, self.target, self.arguments, self.working_dir)

    @property
    def verifiable(self) -> bool:
        """Can NEXUS tell from running processes whether this started?"""
        return bool(self.process_names or self.process_paths or self.package_family)


@dataclass(frozen=True)
class GameRecord(AppRecord):
    """A game: launcher, launcher id, install location and the processes that mean it is running."""

    def __post_init__(self) -> None:
        if self.kind is not AppKind.GAME:
            raise ValueError("a GameRecord must have kind GAME")


@dataclass(frozen=True)
class RawEntry:
    """What a discovery source saw, before classification and merging."""

    name: str
    source: str
    shortcut: str | None = None      # the .lnk/.url file it came from
    target: str | None = None        # exe path
    arguments: str = ""
    working_dir: str | None = None
    url: str | None = None
    aumid: str | None = None
    publisher: str | None = None
    confidence: float = 1.0
    extra_aliases: tuple[str, ...] = field(default_factory=tuple)
