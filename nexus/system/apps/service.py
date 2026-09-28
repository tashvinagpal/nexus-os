"""AppService: the one object the rest of NEXUS uses for apps, games and folders."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from nexus.system.apps.catalog import AppCatalog, Resolution
from nexus.system.apps.events import AppEvent, EventHub
from nexus.system.apps.launcher import Launcher, LaunchResult, _terminate
from nexus.system.apps.names import normalize, strip_launch_words
from nexus.system.apps.processes import ProcessGuard, ProcessProbe, snapshot_processes
from nexus.system.apps.records import AppKind, AppRecord
from nexus.system.apps.sources import AppSource, default_sources
from nexus.system.games.epic import EpicProvider
from nexus.system.games.providers import GameProvider
from nexus.system.games.rockstar import RockstarProvider
from nexus.system.games.service import GameModeHooks, GameService, GameWatcher, run_in_background
from nexus.system.games.steam import SteamProvider
from nexus.system.platform import Platform
from nexus.system.platform.base import LaunchMethod

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppSettings:
    app_launch_timeout_s: float = 20.0
    game_launch_timeout_s: float = 180.0
    app_close_timeout_s: float = 10.0
    game_close_timeout_s: float = 30.0
    auto_game_mode: bool = True
    watch_games: bool = True
    store_apps: bool = True


def default_providers(platform: Platform) -> list[GameProvider]:
    return [SteamProvider(platform), EpicProvider(platform), RockstarProvider(platform)]


# --- user-defined aliases ("Chrome is my browser") ----------------------------------------------

class AliasStore(Protocol):
    """Persistent, user-visible alias table (the local memory database)."""

    def get_alias(self, alias: str) -> tuple[str, str] | None: ...                 # (app_id, app_name)
    def set_alias(self, alias: str, app_id: str, app_name: str, *,
                  origin_device: str | None = None) -> str | None: ...              # previous app name
    def remove_alias(self, alias: str) -> bool: ...
    def aliases(self) -> list[tuple[str, str]]: ...                                  # (alias, app_name)


_ALIAS_OWNER = {"my", "mera", "meri", "mere", "the", "apna", "apni", "apne"}
_ALIAS_SHAPE = re.compile(r"^[^\W_]+(?: [^\W_]+){0,3}$")
# Names that must keep their meaning: shells/admin words, NEXUS itself, folders, "everything".
RESERVED_ALIASES = {
    "cmd", "command prompt", "terminal", "shell", "powershell", "bash", "console", "admin", "administrator",
    "system", "root", "sudo", "windows", "nexus", "computer", "pc", "laptop", "registry", "everything", "all",
    "anything", "app", "apps", "game", "games", "program", "programs", "folder", "downloads", "documents",
    "desktop", "pictures", "music", "videos", "home", "this pc", "file explorer", "explorer",
}


def normalize_alias(text: str) -> str:
    words = normalize(strip_launch_words(text)).split()
    while words and words[0] in _ALIAS_OWNER:
        words = words[1:]
    return " ".join(words)


@dataclass(frozen=True)
class AliasOutcome:
    ok: bool
    message: str
    refused: bool = False
    options: tuple[str, ...] = ()


class AppService:
    def __init__(self, platform: Platform, *, sources: Iterable[AppSource] | None = None,
                 providers: Iterable[GameProvider] | None = None, processes: ProcessProbe = snapshot_processes,
                 game_mode: GameModeHooks | None = None, settings: AppSettings = AppSettings(),
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 aliases: AliasStore | None = None, guard: ProcessGuard | None = None,
                 terminate: Callable[[int], None] = _terminate,
                 background: Callable[[Callable[[], None]], None] = run_in_background) -> None:
        self.platform = platform
        self.settings = settings
        self.aliases = aliases
        self.events = EventHub()
        self._sources = list(sources) if sources is not None else None
        self.launcher = Launcher(platform, processes=processes, clock=clock, sleep=sleep, guard=guard,
                                 terminate=terminate)
        self.games = GameService(providers if providers is not None else default_providers(platform), self.launcher,
                                 game_mode=game_mode, auto_game_mode=settings.auto_game_mode,
                                 timeout_s=settings.game_launch_timeout_s, events=self.events,
                                 background=background)
        if game_mode is not None:
            self.games.watcher = GameWatcher(lambda: self.catalog().games(), game_mode, processes)
        self._catalog: AppCatalog | None = None
        self._lock = threading.Lock()
        self.discovery_seconds: float | None = None

    def subscribe(self, listener: Callable[[AppEvent], None]) -> None:
        """Background results, e.g. "GTA V is running." after an asynchronous game launch."""
        self.events.subscribe(listener)

    # --- discovery ---------------------------------------------------------------------

    def catalog(self) -> AppCatalog:
        with self._lock:
            if self._catalog is None:
                self._catalog = self._discover()
            return self._catalog

    def refresh(self) -> AppCatalog:
        with self._lock:
            self._catalog = self._discover()
            return self._catalog

    def warm_up(self) -> None:
        """Discover in the background so the first "open X" doesn't wait for it."""
        threading.Thread(target=self.catalog, name="nexus-app-discovery", daemon=True).start()

    def _discover(self) -> AppCatalog:
        started = time.perf_counter()
        sources = self._sources if self._sources is not None else default_sources(
            self.platform, store_apps=self.settings.store_apps)
        entries = []
        for source in sources:
            try:
                entries += list(source.entries())
            except Exception:  # a broken source must not hide the others
                log.exception("app discovery failed for %s", getattr(source, "name", source))
        catalog = AppCatalog.build(entries, self.games.discover())
        self.discovery_seconds = time.perf_counter() - started
        log.info("discovered %d apps/games in %.2fs", len(catalog.records), self.discovery_seconds)
        return catalog

    # --- queries -----------------------------------------------------------------------

    def resolve(self, name: str, *, kinds=None) -> Resolution:
        """User aliases first (exact, deterministic), then the catalog's own names."""
        res = self.catalog().resolve(name, kinds=kinds)
        aliased = self._alias_target(name)
        if aliased is None or (kinds and aliased.kind not in kinds):
            return res
        if res.match is not None and res.match.id != aliased.id and res.score >= 0.99:
            # An installed app now has exactly this name too: show both, don't guess.
            return Resolution(name, candidates=(aliased, res.match), score=1.0)
        return Resolution(name, match=aliased, candidates=(aliased,), score=1.0)

    def _alias_target(self, name: str) -> AppRecord | None:
        if self.aliases is None:
            return None
        entry = self.aliases.get_alias(normalize_alias(name))
        if entry is None:
            return None
        app_id, app_name = entry
        catalog = self.catalog()
        record = catalog.get(app_id)
        if record is None:  # reinstalled/moved: find it again by name, only if unambiguous
            res = catalog.resolve(app_name)
            record = res.match if res.confident else None
        return record

    def find_game(self, name: str) -> Resolution:
        return self.resolve(name, kinds=[AppKind.GAME])

    def is_running(self, record: AppRecord) -> bool:
        return bool(self.launcher.running(record))

    def untrusted_location(self, record: AppRecord) -> bool:
        """Programs sitting in Downloads or temp folders are downloads, not installed apps."""
        if record.launch_method is not LaunchMethod.EXECUTABLE:
            return False
        folders = self.platform.known_folders()
        exe = os.path.normcase(os.path.abspath(record.target))
        return any(exe.startswith(os.path.normcase(str(folders[k])) + os.sep)
                   for k in ("downloads", "temp") if folders.get(k))

    def needs_elevation(self, record: AppRecord) -> bool:
        if record.launch_method is not LaunchMethod.EXECUTABLE:
            return False
        try:
            return bool(self.platform.requires_elevation(Path(record.target)))
        except OSError:
            return False

    # --- actions -----------------------------------------------------------------------

    def launch(self, record: AppRecord, *, origin: str | None = None) -> LaunchResult:
        if record.unsupported_reason:
            return LaunchResult(False, None, record.unsupported_reason, unsupported=True)
        if record.kind is AppKind.GAME:
            return self.games.launch_game(record, origin=origin)  # returns at once; verified in the background
        if not record.mediated:
            pids = self.launcher.running(record)
            if pids:
                return self.launcher.focus_existing(record, pids)
        return self.launcher.start(record, record.spec, timeout_s=self.settings.app_launch_timeout_s)

    def close(self, record: AppRecord, *, force: bool = False) -> LaunchResult:
        game = record.kind is AppKind.GAME
        timeout = self.settings.game_close_timeout_s if game else self.settings.app_close_timeout_s
        result = self.launcher.close(record, timeout_s=timeout, force=force)
        if result.ok and game and self.games.watcher is not None:
            self.games.watcher.check()  # the game is gone: leave game mode now (AI reloads only when needed)
        return result

    def start_game_watcher(self, interval_s: float = 5.0) -> None:
        if self.games.watcher is not None and self.settings.watch_games:
            self.games.watcher.start(interval_s)

    # --- aliases -----------------------------------------------------------------------

    def define_alias(self, alias_text: str, app_query: str, *, origin: str | None = None) -> AliasOutcome:
        if self.aliases is None:
            return AliasOutcome(False, "I can't save app names on this device.")
        alias = normalize_alias(alias_text)
        # Check the raw text: symbols are refused, not silently cleaned ("a/b", "x && del").
        raw_ok = re.fullmatch(r"[^\W_]+(?:[ '\-][^\W_]+)*", alias_text.strip()) is not None
        if not raw_ok or not alias or len(alias) > 40 or not _ALIAS_SHAPE.match(alias) or alias.isdigit():
            return AliasOutcome(False, "I can only use a few ordinary words as a name for an app.", refused=True)
        if alias in RESERVED_ALIASES:
            return AliasOutcome(False, f"I can't use \"{alias}\" as a name for an app - it has to keep its own "
                                       "meaning.", refused=True)
        catalog = self.catalog()
        target = catalog.resolve(app_query)
        if target.ambiguous:
            names = [c.display_name for c in target.candidates]
            listed = ", ".join(names[:-1]) + f" and {names[-1]}"
            return AliasOutcome(False, f"I found {listed}. Which one should be your {alias}?", options=tuple(names))
        record = target.match
        if record is None or not target.confident:
            return AliasOutcome(False, f"I couldn't find an installed app called \"{app_query}\".")
        if record.kind in (AppKind.SHELL, AppKind.ADMIN) or self.untrusted_location(record):
            return AliasOutcome(False, f"I can't give {record.display_name} a name - NEXUS doesn't launch it.",
                                refused=True)
        existing = catalog.resolve(alias)
        if existing.match is not None and existing.score >= 0.99 and existing.match.id != record.id:
            return AliasOutcome(False, f"\"{alias}\" is already the name of {existing.match.display_name}, so I "
                                       f"won't point it at {record.display_name}.", refused=True)
        previous = self.aliases.set_alias(alias, record.id, record.display_name, origin_device=origin)
        note = f" (it used to open {previous})" if previous and previous != record.display_name else ""
        return AliasOutcome(True, f"Got it - \"your {alias}\" now opens {record.display_name}{note}.")

    def remove_alias(self, alias_text: str) -> AliasOutcome:
        alias = normalize_alias(alias_text)
        if self.aliases is None or not self.aliases.remove_alias(alias):
            return AliasOutcome(False, f"You don't have an app called \"{alias}\".")
        return AliasOutcome(True, f"Removed \"{alias}\" - it no longer opens anything.")
