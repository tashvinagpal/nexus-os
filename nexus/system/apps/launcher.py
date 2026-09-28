"""Start and stop discovered apps, and verify it really happened - from the process table, not from hope.

Starting: initiate() hands the discovered launch spec to the OS; await_start() waits for a new process
that belongs to the app. Apps do both inline (seconds); games do await_start() in the background.

Stopping: only processes the catalog record matches - never a PID from anywhere else - and never a
protected process. Graceful first (ask its windows to close, like clicking X); a forced end is a
separate, explicitly requested operation. "Closed" is only said once the processes are gone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import psutil

from nexus.system.apps.processes import ProcessGuard, ProcessProbe, SystemProcessGuard, running_pids, snapshot_processes
from nexus.system.apps.records import AppKind, AppRecord
from nexus.system.platform import Platform
from nexus.system.platform.base import LaunchError, LaunchSpec

WINDOW_WAIT_S = 5.0


@dataclass(frozen=True)
class LaunchResult:
    ok: bool
    verified: bool | None     # True: checked and true. False: checked, not true. None: can't be / not yet checked.
    message: str
    pids: tuple[int, ...] = ()
    already_running: bool = False
    pending: bool = False     # a game launch still being verified in the background
    refused: bool = False     # NEXUS won't do it (security) - not a failure
    unsupported: bool = False


def _terminate(pid: int) -> None:
    psutil.Process(pid).terminate()


class Launcher:
    def __init__(self, platform: Platform, *, processes: ProcessProbe = snapshot_processes,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 poll_s: float = 0.5, guard: ProcessGuard | None = None,
                 terminate: Callable[[int], None] = _terminate) -> None:
        self.platform = platform
        self.processes = processes
        self.clock = clock
        self.sleep = sleep
        self.poll_s = poll_s
        self.guard = guard or SystemProcessGuard()
        self._terminate = terminate

    def running(self, record: AppRecord) -> list[int]:
        return running_pids(record, self.processes()) if record.verifiable else []

    def focus_existing(self, record: AppRecord, pids: list[int]) -> LaunchResult:
        """Already running: bring it forward instead of starting a duplicate."""
        try:
            focused = self.platform.focus_window(pids)
        except OSError:
            focused = False
        text = (f"{record.display_name} was already running - I switched to it." if focused
                else f"{record.display_name} is already running.")
        return LaunchResult(True, True, text, tuple(pids), already_running=True)

    # --- starting ----------------------------------------------------------------------

    def initiate(self, record: AppRecord, spec: LaunchSpec) -> tuple[set[int], LaunchResult | None]:
        """Ask the OS to start it. Returns (pids that already existed, a failure or None)."""
        before = set(self.running(record))
        try:
            self.platform.launch(spec)
        except LaunchError as exc:
            return before, LaunchResult(False, False, f"I couldn't open {record.display_name}: {exc}.")
        return before, None

    def await_start(self, record: AppRecord, before: set[int], *, timeout_s: float) -> LaunchResult:
        name = record.display_name
        if not record.verifiable:
            via = f"{record.launcher.title()} " if record.launcher else "Windows "
            return LaunchResult(True, None, f"I asked {via}to open {name}, but I can't confirm it actually started.")
        deadline = self.clock() + timeout_s
        new: list[int] = []
        while self.clock() < deadline:
            self.sleep(self.poll_s)
            new = [pid for pid in self.running(record) if pid not in before]
            if new:
                break
        if not new:
            if record.package_family:  # some Store/system apps run outside WindowsApps: can't be sure either way
                return LaunchResult(True, None, f"I asked Windows to open {name}, but I couldn't confirm it started.")
            expected = ", ".join(sorted(record.process_names)) or "its program"
            return LaunchResult(False, False, f"I couldn't verify that {name} started: {expected} didn't appear "
                                              f"within {timeout_s:.0f} seconds.")
        if record.kind is AppKind.GAME:
            return LaunchResult(True, True, f"{name} is running.", tuple(new))
        window_deadline = self.clock() + WINDOW_WAIT_S
        while True:
            pids = set(new) | set(self.running(record))
            try:
                has_window = bool(self.platform.window_pids(pids))
            except OSError:
                has_window = False
            if has_window or self.clock() >= window_deadline:
                break
            self.sleep(self.poll_s)
        text = f"{name} is open." if has_window else f"{name} started (its window hasn't appeared yet)."
        return LaunchResult(True, True, text, tuple(new))

    def start(self, record: AppRecord, spec: LaunchSpec, *, timeout_s: float) -> LaunchResult:
        before, failed = self.initiate(record, spec)
        return failed or self.await_start(record, before, timeout_s=timeout_s)

    # --- stopping ----------------------------------------------------------------------

    def close(self, record: AppRecord, *, timeout_s: float, force: bool = False) -> LaunchResult:
        name = record.display_name
        if not record.verifiable:
            return LaunchResult(False, None, f"I can't tell which program belongs to {name}, so I can't close it "
                                             "safely yet.", unsupported=True)
        procs = [p for p in self.processes() if p.pid in set(self.running(record))]
        if not procs:
            return LaunchResult(True, True, f"{name} isn't running.")
        for proc in procs:
            reason = self.guard.reason(proc)
            if reason:
                return LaunchResult(False, None, f"I can't close {name} through NEXUS - {reason}.", refused=True)
        pids = [p.pid for p in procs]
        if force:
            for pid in pids:  # only pids the catalog record matched, each checked by the guard above
                try:
                    self._terminate(pid)
                except (psutil.Error, OSError):
                    pass
        else:
            try:
                asked = self.platform.close_windows(pids)
            except OSError:
                asked = 0
            if not asked:
                return LaunchResult(False, False, f"I couldn't close {name} gracefully: it has no window I can ask to "
                                                  f"close. Say \"force close {name}\" if you want me to end it.")
        deadline = self.clock() + timeout_s
        while self.clock() < deadline:
            self.sleep(self.poll_s)
            if not self.running(record):
                return LaunchResult(True, True, f"{name} is closed.", tuple(pids))
        if force:
            return LaunchResult(False, False, f"I tried to force-close {name}, but it's still running.")
        return LaunchResult(False, False, f"I asked {name} to close, but it's still running - it may be waiting for "
                                          f"you (for example, to save your work). Say \"force close {name}\" if you "
                                          "want me to end it.")
