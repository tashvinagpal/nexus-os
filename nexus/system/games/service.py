"""GameService: find_game(), launch_game(), is_game_running() - and GAME_MODE around games."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable, Protocol

from nexus.system.apps.catalog import AppCatalog, Resolution
from nexus.system.apps.events import AppEvent, EventHub
from nexus.system.apps.launcher import Launcher, LaunchResult
from nexus.system.apps.processes import ProcessProbe, running_pids, snapshot_processes
from nexus.system.apps.records import AppKind, AppRecord, GameRecord
from nexus.system.games.providers import GameProvider

log = logging.getLogger(__name__)


class GameModeHooks(Protocol):
    """Implemented by the ResourceManager: frees the GPU/VRAM for games."""

    @property
    def game_mode(self) -> bool: ...
    def enter_game_mode(self) -> list[str]: ...
    def exit_game_mode(self) -> None: ...


class GameWatcher:
    """Turns game mode on while any known game runs and off when it exits - but only game mode it
    turned on itself (a manual /gamemode on is left alone). Never reloads AI models: they come
    back lazily on the next request that needs them."""

    def __init__(self, games: Callable[[], Iterable[AppRecord]], hooks: GameModeHooks,
                 processes: ProcessProbe = snapshot_processes) -> None:
        self.games = games
        self.hooks = hooks
        self.processes = processes
        self._owned = False
        self._thread: threading.Thread | None = None

    def claim(self) -> None:
        self._owned = True

    def release(self) -> None:
        self._owned = False

    def check(self) -> list[str]:
        procs = self.processes()
        running = [g.display_name for g in self.games() if g.verifiable and running_pids(g, procs)]
        if running and not self.hooks.game_mode:
            log.info("game running (%s): entering game mode", running)
            self.hooks.enter_game_mode()
            self._owned = True
        elif not running and self._owned:
            if self.hooks.game_mode:
                log.info("no game running: leaving game mode")
                self.hooks.exit_game_mode()
            self._owned = False
        return running

    def start(self, interval_s: float = 5.0) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while True:
                time.sleep(interval_s)
                try:
                    self.check()
                except Exception:
                    log.exception("game watcher failed")

        self._thread = threading.Thread(target=loop, name="nexus-game-watcher", daemon=True)
        self._thread.start()


def run_in_background(job: Callable[[], None]) -> None:
    threading.Thread(target=job, name="nexus-game-launch", daemon=True).start()


class GameService:
    def __init__(self, providers: Iterable[GameProvider], launcher: Launcher, *,
                 game_mode: GameModeHooks | None = None, auto_game_mode: bool = True,
                 timeout_s: float = 180.0, events: EventHub | None = None,
                 background: Callable[[Callable[[], None]], None] = run_in_background) -> None:
        self.providers = {p.name: p for p in providers}
        self.launcher = launcher
        self.game_mode = game_mode
        self.auto_game_mode = auto_game_mode
        self.timeout_s = timeout_s
        self.events = events or EventHub()
        self.background = background
        self.watcher: GameWatcher | None = None
        self._starting: set[str] = set()
        self._lock = threading.Lock()

    def discover(self) -> list[GameRecord]:
        games: list[GameRecord] = []
        for provider in self.providers.values():
            try:
                games += provider.discover()
            except Exception:  # one broken launcher install must not hide the others
                log.exception("game discovery failed for %s", provider.name)
        return games

    def find_game(self, name: str, catalog: AppCatalog) -> Resolution:
        return catalog.resolve(name, kinds=[AppKind.GAME])

    def is_game_running(self, game: AppRecord) -> bool:
        return bool(self.launcher.running(game))

    def launch_game(self, game: AppRecord, *, origin: str | None = None) -> LaunchResult:
        """Start a game and return at once; verification continues in the background.

        Returns "Starting X..." (pending). Later an AppEvent says "X is running." - only after
        the game's own process was seen - or "I couldn't verify that X started...".
        """
        if game.unsupported_reason:
            return LaunchResult(False, None, game.unsupported_reason, unsupported=True)
        pids = self.launcher.running(game)
        if pids:
            return self.launcher.focus_existing(game, pids)
        with self._lock:
            if game.id in self._starting:
                return LaunchResult(True, None, f"{game.display_name} is already starting - I'll tell you once "
                                                "it's running.", pending=True)
            self._starting.add(game.id)
        entered = False
        # Only games NEXUS can see running get game mode: otherwise nothing could ever end it.
        if game.verifiable and self.game_mode is not None and self.auto_game_mode and not self.game_mode.game_mode:
            self.game_mode.enter_game_mode()  # free VRAM *before* the game starts loading
            entered = True
            if self.watcher is not None:
                self.watcher.claim()
        provider = self.providers.get(game.launcher or "")
        spec = provider.launch_spec(game) if provider is not None and isinstance(game, GameRecord) else game.spec
        before, failed = self.launcher.initiate(game, spec)
        if failed is not None or not game.verifiable:
            with self._lock:
                self._starting.discard(game.id)
            if failed is not None:
                self._give_back(entered)
                return failed
            return self.launcher.await_start(game, before, timeout_s=0)  # honest "can't confirm"

        def monitor() -> None:
            try:
                result = self.launcher.await_start(game, before, timeout_s=self.timeout_s)
            finally:
                with self._lock:
                    self._starting.discard(game.id)
            if not result.ok:
                self._give_back(entered)
            self.events.emit(AppEvent("launch", game.display_name, result.ok, result.verified, result.message,
                                      origin))

        self.background(monitor)
        return LaunchResult(True, None, f"Starting {game.display_name} - I'll tell you once it's running.",
                            pending=True)

    def _give_back(self, entered: bool) -> None:
        if entered and self.game_mode is not None:
            self.game_mode.exit_game_mode()  # nothing is running: give the resources back
            if self.watcher is not None:
                self.watcher.release()
