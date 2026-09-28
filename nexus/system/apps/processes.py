"""Which processes are running, and which of them belong to an app or game (cross-platform, psutil)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable

import psutil

from nexus.system.apps.records import AppRecord


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    name: str
    exe: str | None  # None when the OS won't tell us (other users' / protected processes)


ProcessProbe = Callable[[], list[ProcInfo]]


def snapshot_processes() -> list[ProcInfo]:
    procs = []
    for p in psutil.process_iter(["pid", "name", "exe"], ad_value=None):
        info = p.info
        procs.append(ProcInfo(info["pid"], info["name"] or "", info["exe"]))
    return procs


def _package_family(exe: str) -> str | None:
    """'...\\WindowsApps\\Microsoft.WindowsCalculator_11.2405.2.0_x64__8wekyb3d8bbwe\\x.exe' -> family name."""
    marker = "\\windowsapps\\"
    low = exe.lower()
    if marker not in low:
        return None
    folder = low.split(marker, 1)[1].split("\\", 1)[0]
    name, publisher = folder.split("_", 1)[0], folder.rsplit("_", 1)[-1]
    return f"{name}_{publisher}"


def belongs_to(record: AppRecord, proc: ProcInfo) -> bool:
    exe = os.path.normcase(proc.exe) if proc.exe else ""
    if exe and exe in record.process_paths:
        return True
    if proc.name.lower() in record.process_names:
        # A game's process must live in the game's folder when we can see where it lives.
        if record.install_dir and exe:
            return exe.startswith(os.path.normcase(record.install_dir))
        return True
    if record.package_family and exe:
        return _package_family(exe) == record.package_family.lower()
    return False


def running_pids(record: AppRecord, procs: Iterable[ProcInfo]) -> list[int]:
    return [p.pid for p in procs if belongs_to(record, p)]


# Processes NEXUS must never close or end, whatever the request.
PROTECTED_PROCESSES = {
    "system", "registry", "secure system", "memory compression", "smss.exe", "csrss.exe", "wininit.exe",
    "winlogon.exe", "services.exe", "lsass.exe", "lsaiso.exe", "svchost.exe", "dwm.exe", "explorer.exe",
    "fontdrvhost.exe", "sihost.exe", "ctfmon.exe", "taskhostw.exe", "runtimebroker.exe", "searchhost.exe",
    "startmenuexperiencehost.exe", "shellexperiencehost.exe", "textinputhost.exe", "securityhealthservice.exe",
    "securityhealthsystray.exe", "msmpeng.exe", "nissrv.exe", "audiodg.exe", "spoolsv.exe", "wudfhost.exe",
    "conhost.exe", "lsm.exe", "applicationframehost.exe", "dllhost.exe", "userinit.exe", "logonui.exe",
}


class ProcessGuard:
    def reason(self, proc: ProcInfo) -> str | None:
        """Why this process must not be closed, or None."""
        raise NotImplementedError


class SystemProcessGuard(ProcessGuard):
    """Protects Windows' own processes, other users' / system-owned processes, and NEXUS itself."""

    def __init__(self) -> None:
        me = psutil.Process()
        self.own = {me.pid} | {p.pid for p in me.parents()}  # NEXUS, its console and terminal
        try:
            self.user = me.username().lower()
        except psutil.Error:
            self.user = None

    def reason(self, proc: ProcInfo) -> str | None:
        if proc.pid in self.own:
            return "it's part of NEXUS itself"
        if proc.name.lower() in PROTECTED_PROCESSES:
            return "it's a protected Windows process"
        try:
            owner = psutil.Process(proc.pid).username().lower()
        except psutil.Error:
            return "it isn't one of your own programs"
        if self.user is None or owner != self.user:
            return "it belongs to the system or another user"
        return None
