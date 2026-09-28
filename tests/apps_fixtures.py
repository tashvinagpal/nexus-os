"""Test doubles for app discovery and launching: a fake OS, a fake process table, real .lnk bytes."""

from __future__ import annotations

import itertools
import struct
import uuid
from pathlib import Path

from nexus.system.apps.processes import ProcInfo
from nexus.system.platform.base import LaunchError, LaunchSpec
from tests.conftest import FakePlatform

_CLSID = uuid.UUID("00021401-0000-0000-c000-000000000046").bytes_le


def build_lnk(target: str | None = None, *, args: str = "", workdir: str | None = None,
              env_target: str | None = None) -> bytes:
    """A minimal but valid [MS-SHLLINK] file, like Windows writes."""
    flags = 0x80  # IsUnicode
    link_info = b""
    if target:
        flags |= 0x2
        volume = struct.pack("<IIII", 0x11, 3, 0x1234, 0x10) + b"\x00"
        base = target.encode("latin-1") + b"\x00"
        header = 0x1C
        vol_off = header
        base_off = vol_off + len(volume)
        suffix_off = base_off + len(base)
        body = volume + base + b"\x00"
        link_info = struct.pack("<7I", header + len(body), header, 1, vol_off, base_off, 0, suffix_off) + body
    strings = b""
    for bit, value in ((0x10, workdir), (0x20, args or None)):
        if value:
            flags |= bit
            strings += struct.pack("<H", len(value)) + value.encode("utf-16-le")
    extra = b""
    if env_target:
        flags |= 0x200
        ansi = env_target.encode("latin-1").ljust(260, b"\x00")
        wide = env_target.encode("utf-16-le").ljust(520, b"\x00")
        extra += struct.pack("<II", 0x314, 0xA0000001) + ansi + wide
    head = struct.pack("<I16sII8s8s8sIiIHHII", 0x4C, _CLSID, flags, 0x20, b"\0" * 8, b"\0" * 8, b"\0" * 8,
                       0, 0, 1, 0, 0, 0, 0)
    return head + link_info + strings + extra + b"\x00\x00\x00\x00"


class FakeProcs:
    """A process table. Programs 'start' a few polls after launch, like real ones."""

    def __init__(self) -> None:
        self.procs: list[ProcInfo] = []
        self._pending: list[list] = []
        self._pids = itertools.count(1000)

    def add(self, name: str, exe: str | None) -> int:
        pid = next(self._pids)
        self.procs.append(ProcInfo(pid, name, exe))
        return pid

    def spawn(self, name: str, exe: str | None, *, after_polls: int = 2) -> None:
        self._pending.append([after_polls, name, exe])

    def remove(self, name: str) -> None:
        self.procs = [p for p in self.procs if p.name.lower() != name.lower()]

    def remove_pids(self, pids) -> None:
        self.procs = [p for p in self.procs if p.pid not in set(pids)]

    def __call__(self) -> list[ProcInfo]:
        for item in self._pending:
            item[0] -= 1
        for item in [i for i in self._pending if i[0] < 0]:
            self._pending.remove(item)
            self.add(item[1], item[2])
        return list(self.procs)


class NameGuard:
    """Test ProcessGuard: protects by name and a set of 'NEXUS' pids (fake pids have no real owner)."""

    def __init__(self, own=()) -> None:
        from nexus.system.apps.processes import PROTECTED_PROCESSES

        self.protected = PROTECTED_PROCESSES
        self.own = set(own)

    def reason(self, proc):
        if proc.pid in self.own:
            return "it's part of NEXUS itself"
        if proc.name.lower() in self.protected:
            return "it's a protected Windows process"
        return None


class Background:
    """Stands in for the background thread of an asynchronous game launch.

    inline (default): the verification job runs immediately, so tests are deterministic.
    defer=True: jobs are held until run_pending() - proves the caller returned first.
    """

    def __init__(self) -> None:
        self.defer = False
        self.pending: list = []

    def __call__(self, job) -> None:
        if self.defer:
            self.pending.append(job)
        else:
            job()

    def run_pending(self) -> None:
        jobs, self.pending = self.pending, []
        for job in jobs:
            job()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class AppsPlatform(FakePlatform):
    """FakePlatform plus the app/game OS surface."""

    name = "fake"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.folders = {role: root / role for role in ("home", "desktop", "documents", "downloads", "pictures",
                                                         "music", "videos", "public_desktop", "start_menu",
                                                         "common_start_menu", "program_data", "local_app_data",
                                                         "temp")}
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        self.registry: dict[tuple[str, str], str] = {}
        self.store: list[tuple[str, str]] = []
        self.launched: list[LaunchSpec] = []
        self.opened_folders: list[str] = []
        self.focused: list[list[int]] = []
        self.focus_ok = True
        self.with_windows: set[int] | None = None  # None = every pid has a window
        self.elevated_exes: set[str] = set()
        self.launch_error: str | None = None
        self.on_launch = None
        self.close_requests: list[list[int]] = []
        self.on_close = None            # e.g. remove the processes, as a well-behaved app would
        self.windows_to_close = True    # False = the app has no window to ask

    def known_folders(self):
        return dict(self.folders)

    def read_registry(self, key, value=""):
        return self.registry.get((key, value))

    def registry_subkeys(self, key):
        prefix = key + "\\"
        return sorted({k[len(prefix):].split("\\")[0] for k, _ in self.registry if k.startswith(prefix)})

    def list_store_apps(self):
        return list(self.store)

    def launch(self, spec: LaunchSpec) -> None:
        if self.launch_error:
            raise LaunchError(self.launch_error)
        self.launched.append(spec)
        if self.on_launch:
            self.on_launch(spec)

    def open_folder(self, path: str) -> None:
        self.opened_folders.append(path)

    def window_pids(self, pids):
        pids = set(pids)
        return pids if self.with_windows is None else pids & self.with_windows

    def focus_window(self, pids) -> bool:
        self.focused.append(sorted(pids))
        return self.focus_ok

    def close_windows(self, pids) -> int:
        pids = sorted(pids)
        self.close_requests.append(pids)
        if self.on_close:
            self.on_close(pids)
        return len(pids) if self.windows_to_close else 0

    def requires_elevation(self, exe: Path):
        return str(exe) in self.elevated_exes


def make_exe(path: Path, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ" + b"\0" * size)
    return path


def make_lnk(folder: Path, name: str, target: str | None = None, **kw) -> Path:
    path = folder / f"{name}.lnk"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_lnk(target, **kw))
    return path


def make_url(folder: Path, name: str, url: str) -> Path:
    path = folder / f"{name}.url"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[InternetShortcut]\nURL={url}\n", encoding="utf-8")
    return path
