"""Discovery sources: where installed apps are found, and what each entry is.

Sources return RawEntry values; classify() decides what kind of thing an entry
is (app, game, shell, admin tool) or drops it (uninstallers, docs, websites).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Protocol

from nexus.system.apps.records import AppKind, RawEntry
from nexus.system.apps.shortcuts import read_lnk, read_url
from nexus.system.platform import Platform

log = logging.getLogger(__name__)

# Command shells, terminals and script hosts: typing into them runs arbitrary commands.
SHELL_EXES = {"cmd.exe", "powershell.exe", "pwsh.exe", "powershell_ise.exe", "wt.exe", "windowsterminal.exe",
              "bash.exe", "sh.exe", "wsl.exe", "git-bash.exe", "git-cmd.exe", "mintty.exe", "conhost.exe",
              "wscript.exe", "cscript.exe", "mshta.exe", "rundll32.exe", "regsvr32.exe", "sqlplus.exe"}
# Language runtimes: always shells, with or without arguments (IDLE is a Python shell too).
INTERPRETERS = {"python.exe", "pythonw.exe", "py.exe", "pyw.exe", "node.exe", "java.exe"}
SCRIPT_EXTS = {".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".hta", ".scr",
               ".com", ".py", ".pyw", ".jar", ".reg"}
ADMIN_EXES = {"regedit.exe", "regedt32.exe", "mmc.exe", "msconfig.exe", "gpedit.exe", "secpol.exe",
              "odbcad32.exe", "iscsicpl.exe", "recoverydrive.exe", "mdsched.exe", "dfrgui.exe", "comexp.exe",
              "lusrmgr.exe", "bcdedit.exe", "diskpart.exe", "compmgmtlauncher.exe", "wf.exe"}
ADMIN_NAMES = re.compile(r"^(run|registry editor|command prompt|services|task scheduler|computer management|"
                         r"event viewer|disk management|device manager|group policy|local security policy|"
                         r"system configuration|component services|windows tools|administrative tools)$", re.I)
JUNK_NAMES = re.compile(r"\b(uninstall|uninstaller|readme|read me|documentation|docs|manual|help|license|licence|"
                        r"release notes|changelog|website|web site|visit|faq|faqs|forum|support center|register|"
                        r"check for updates|what'?s new|about java|module docs|manuals)\b", re.I)
JUNK_EXES = re.compile(r"^(unins\d*|uninstall.*|setup|install(er)?|vc_?redist.*|dxsetup|.*crash.*report.*)\.exe$", re.I)
DOC_EXTS = {".chm", ".txt", ".pdf", ".htm", ".html", ".rtf", ".md", ".ini", ".log", ".url", ".xml", ".msi"}
# Game-launcher URL shortcuts NEXUS recognises but cannot start yet (it only starts steam:// and Epic).
UNSUPPORTED_LAUNCHERS = {"uplay://": "Ubisoft Connect", "ubisoft://": "Ubisoft Connect", "battlenet://": "Battle.net",
                         "origin://": "the EA app", "origin2://": "the EA app", "eadm://": "the EA app",
                         "link2ea://": "the EA app", "goggalaxy://": "GOG Galaxy", "roblox-player:": "Roblox"}
# Shortcuts that ask another launcher to open something: the launcher's process isn't the app itself.
MEDIATED_ARGS = re.compile(r"--launch-product|--cmd\s+launchapp|--package\b|-applaunch|--app-id|--app=|"
                           r"--profile-directory", re.I)


class AppSource(Protocol):
    name: str

    def entries(self) -> Iterable[RawEntry]: ...


def classify(entry: RawEntry) -> tuple[AppKind, str | None] | None:
    """(kind, launcher) or None to drop the entry."""
    name, target, args = entry.name, entry.target, entry.arguments
    if JUNK_NAMES.search(name):
        return None
    if entry.url:
        url = entry.url.lower()
        if url.startswith("steam://rungameid/"):
            return AppKind.GAME, "steam"
        if url.startswith("com.epicgames.launcher://apps/"):
            return AppKind.GAME, "epic"
        for scheme, launcher in UNSUPPORTED_LAUNCHERS.items():
            if url.startswith(scheme):
                return AppKind.GAME, launcher  # known game launcher NEXUS can't drive yet
        return None  # websites and other URL shortcuts are not apps
    if entry.aumid:
        return AppKind.APP, None
    if ADMIN_NAMES.match(name.strip()):
        return (AppKind.SHELL if name.strip().lower() in ("run", "command prompt") else AppKind.ADMIN), None
    if target:
        path = Path(target)
        exe, ext = path.name.lower(), path.suffix.lower()
        if ext in DOC_EXTS or JUNK_EXES.match(exe):
            return None
        if ext == ".msc" or ".msc" in args.lower() or exe in ADMIN_EXES:
            return AppKind.ADMIN, None
        if ext in SCRIPT_EXTS or exe in SHELL_EXES:
            return AppKind.SHELL, None
        if exe in INTERPRETERS:
            return AppKind.SHELL, None  # interpreters run arbitrary code
        if ext != ".exe":
            return None
    if "--launch-product" in args.lower():
        return AppKind.GAME, "riot"
    return AppKind.APP, None


def shortcut_entry(path: Path, source: str, confidence: float) -> RawEntry | None:
    """A RawEntry for one .lnk/.url file (read as data), or None."""
    suffix = path.suffix.lower()
    if suffix == ".lnk":
        link = read_lnk(path)
        if link is None:
            return None
        return RawEntry(name=path.stem, source=source, shortcut=str(path), target=link.target,
                        arguments=link.arguments, working_dir=link.working_dir, confidence=confidence)
    if suffix == ".url":
        url = read_url(path)
        return RawEntry(name=path.stem, source=source, shortcut=str(path), url=url, confidence=confidence) if url else None
    return None


class StartMenuSource:
    name = "start_menu"

    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    def entries(self) -> Iterable[RawEntry]:
        folders = self.platform.known_folders()
        for role in ("common_start_menu", "start_menu"):
            folder = folders.get(role)
            if folder and folder.is_dir():
                for path in folder.rglob("*"):
                    entry = shortcut_entry(path, self.name, 1.0)
                    if entry is not None:
                        yield entry


class DesktopSource:
    name = "desktop"

    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    def entries(self) -> Iterable[RawEntry]:
        folders = self.platform.known_folders()
        for role in ("desktop", "public_desktop"):
            folder = folders.get(role)
            if folder and folder.is_dir():
                # Only shortcuts, never loose .exe files sitting on the Desktop.
                for path in folder.iterdir():
                    entry = shortcut_entry(path, self.name, 0.9)
                    if entry is not None:
                        yield entry


class AppPathsSource:
    """Windows 'App Paths' registrations (e.g. chrome.exe, msedge.exe)."""

    name = "app_paths"
    _KEYS = (r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
             r"HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths",
             r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths")

    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    def entries(self) -> Iterable[RawEntry]:
        windir = os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows"))
        for key in self._KEYS:
            for sub in self.platform.registry_subkeys(key):
                path = (self.platform.read_registry(f"{key}\\{sub}") or "").strip().strip('"')
                if not path.lower().endswith(".exe") or os.path.normcase(path).startswith(windir):
                    continue
                if Path(path).is_file():
                    yield RawEntry(name=Path(sub).stem, source=self.name, target=path, confidence=0.6)


class StoreAppsSource:
    name = "store"

    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    def entries(self) -> Iterable[RawEntry]:
        for display, aumid in self.platform.list_store_apps():
            yield RawEntry(name=display, source=self.name, aumid=aumid, confidence=0.9)


def default_sources(platform: Platform, *, store_apps: bool = True) -> list[AppSource]:
    sources: list[AppSource] = [StartMenuSource(platform), DesktopSource(platform), AppPathsSource(platform)]
    if store_apps:
        sources.append(StoreAppsSource(platform))
    return sources
