"""App, game and folder tools. They only ever start or stop things the discovery catalog found.

Result semantics: SUCCESS "Chrome is open." / FAILURE "I couldn't open Chrome..." /
REFUSAL "I can't launch PowerShell through NEXUS..." (refused=True) / CLARIFICATION (options) /
UNSUPPORTED (unsupported=True). A security refusal is never reported as a failed attempt.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from nexus.system.apps.launcher import LaunchResult
from nexus.system.apps.names import normalize, strip_launch_words
from nexus.system.apps.records import AppKind, AppRecord
from nexus.system.apps.service import AppService
from nexus.tools.base import Param, Risk, Tool, ToolResult, is_network_or_device_path

_EXECUTABLE_LIKE = re.compile(r"\.(exe|bat|cmd|ps1|psm1|vbs|vbe|js|jse|wsf|wsh|hta|scr|com|msi|msix|appx|jar|py|"
                              r"pyw|reg|lnk|dll|sh)\b", re.I)
_PATH_LIKE = re.compile(r"^[a-z]:|[\\/]", re.I)
_PID_LIKE = re.compile(r"^\s*(?:pid|process|proc)?\s*#?\s*\d+\s*$", re.I)
_REFUSE_FILES = ("I can't run files, paths or commands through NEXUS - I only start apps and games that are "
                 "installed on this PC. Running other programs needs a separate, controlled install workflow.")
_REFUSE_PIDS = "I can't close processes by number or path through NEXUS - tell me the app's name instead."

# Folder words in English/Hinglish -> known-folder role.
FOLDERS = {
    "downloads": "downloads", "download": "downloads",
    "documents": "documents", "document": "documents", "docs": "documents",
    "desktop": "desktop", "pictures": "pictures", "photos": "pictures", "images": "pictures",
    "music": "music", "songs": "music", "videos": "videos", "video": "videos",
    "home": "home", "user": "home",
    "this pc": "shell:MyComputerFolder", "my computer": "shell:MyComputerFolder", "computer": "shell:MyComputerFolder",
    "file explorer": "shell:MyComputerFolder", "explorer": "shell:MyComputerFolder",
}
_FOLDER_FILLER = {"folder", "folders", "directory", "my", "mera", "meri", "mere", "wala", "wali", "the", "open",
                  "khol", "kholo", "please", "show", "me", "go", "to"}


def folder_role(text: str) -> str | None:
    words = [w for w in normalize(strip_launch_words(text)).split() if w not in _FOLDER_FILLER]
    return FOLDERS.get(" ".join(words))


def _refusal(record: AppRecord, verb: str = "launch") -> str:
    if record.kind is AppKind.SHELL:
        return (f"I can't {verb} {record.display_name} through NEXUS - it's a command shell or script runner, and "
                "NEXUS never opens shells or runs commands. You can open it yourself from the Start menu.")
    return (f"I can't {verb} {record.display_name} through NEXUS - it's a system administration tool. "
            "You can open it yourself from the Start menu.")


def _as_tool_result(record: AppRecord, result: LaunchResult) -> ToolResult:
    return ToolResult(result.ok, result.message,
                      {"app": record.display_name, "id": record.id, "kind": record.kind.value,
                       "pids": list(result.pids), "already_running": result.already_running,
                       "pending": result.pending},
                      verified=result.verified, refused=result.refused, unsupported=result.unsupported)


def _clarify(apps_listed: list[str], question: str, param: str) -> ToolResult:
    listed = ", ".join(apps_listed[:-1]) + f" and {apps_listed[-1]}"
    return ToolResult(False, f"I found {listed}. {question}", {"candidates": apps_listed, "option_param": param},
                      options=tuple(apps_listed))


def make_app_tools(apps: AppService) -> list[Tool]:
    def _resolve_for(name: str, verb: str) -> AppRecord | ToolResult:
        res = apps.resolve(name)
        if res.ambiguous:
            if all(c.kind in (AppKind.SHELL, AppKind.ADMIN) for c in res.candidates):
                return ToolResult(False, _refusal(res.candidates[0], verb), refused=True)
            return _clarify([c.display_name for c in res.candidates], "Which one do you want?", "name")
        if res.match is None:
            return ToolResult(False, f"I couldn't find an installed app or game called \"{name}\".")
        if res.match.kind in (AppKind.SHELL, AppKind.ADMIN):
            return ToolResult(False, _refusal(res.match, verb), refused=True)
        return res.match

    # --- launch -----------------------------------------------------------------------------

    def launch_app(name: str, origin: str | None) -> ToolResult:
        if _EXECUTABLE_LIKE.search(name) or _PATH_LIKE.search(name) or len(name) > 120:
            return ToolResult(False, _REFUSE_FILES, refused=True)
        found = _resolve_for(name, "launch")
        if isinstance(found, ToolResult):
            return found
        record = found
        if apps.untrusted_location(record):
            return ToolResult(False, f"I can't launch {record.display_name} through NEXUS - it runs from your "
                                     "Downloads or temp folder, and NEXUS doesn't start downloaded programs.",
                              refused=True)
        if apps.needs_elevation(record):
            return ToolResult(False, f"I can't launch {record.display_name} through NEXUS - it needs administrator "
                                     "rights, and NEXUS never starts programs as administrator.", refused=True)
        return _as_tool_result(record, apps.launch(record, origin=origin))

    def progress(args) -> str:
        res = apps.resolve(args["name"])
        record = res.match
        if record is None or record.kind is not AppKind.APP or record.unsupported_reason:
            return ""  # games answer at once ("Starting X..."); refusals/unknowns start nothing
        return "" if apps.is_running(record) else f"opening {record.display_name}"

    # --- close --------------------------------------------------------------------------------

    def _close(name: str, force: bool) -> ToolResult:
        if _PID_LIKE.match(name) or _EXECUTABLE_LIKE.search(name) or _PATH_LIKE.search(name):
            return ToolResult(False, _REFUSE_PIDS, refused=True)
        found = _resolve_for(name, "close")
        if isinstance(found, ToolResult):
            return found
        return _as_tool_result(found, apps.close(found, force=force))

    def close_app(name: str) -> ToolResult:
        return _close(name, force=False)

    def force_close_app(name: str) -> ToolResult:
        return _close(name, force=True)

    # --- aliases ------------------------------------------------------------------------------

    def set_app_alias(app: str, alias: str, origin: str | None) -> ToolResult:
        if _EXECUTABLE_LIKE.search(app) or _PATH_LIKE.search(app):
            return ToolResult(False, _REFUSE_FILES, refused=True)
        out = apps.define_alias(alias, app, origin=origin)
        if out.options:
            return ToolResult(False, out.message, {"candidates": list(out.options), "option_param": "app"},
                              options=out.options)
        return ToolResult(out.ok, out.message, verified=True if out.ok else None, refused=out.refused)

    def remove_app_alias(alias: str) -> ToolResult:
        out = apps.remove_alias(alias)
        return ToolResult(out.ok, out.message, verified=True if out.ok else None)

    # --- folders and listing ----------------------------------------------------------------

    def open_folder(target: str) -> ToolResult:
        role = folder_role(target)
        if role is not None and role.startswith("shell:"):
            path, label = role, "This PC"
        elif role is not None:
            folder = apps.platform.known_folders().get(role)
            if folder is None or not folder.is_dir():
                return ToolResult(False, f"I couldn't find your {role} folder.")
            path, label = str(folder), f"your {role.title()} folder ({folder})"
        else:
            if is_network_or_device_path(target):
                return ToolResult(False, "I can't open network or device paths through NEXUS.", refused=True)
            candidate = Path(os.path.expanduser(target))
            if not candidate.is_absolute() or not candidate.is_dir():
                return ToolResult(False, f"I couldn't find a folder called \"{target}\". Try Downloads, Documents, "
                                         "Desktop, Pictures, Music or Videos, or a full folder path.")
            path, label = str(candidate.resolve()), str(candidate.resolve())
        try:
            apps.platform.open_folder(path)
        except OSError as exc:
            return ToolResult(False, f"I couldn't open {label}: {exc}")
        return ToolResult(True, f"Asked Windows to show {label}.", {"path": path}, verified=None)

    def list_apps(kind: str, query: str | None) -> ToolResult:
        catalog = apps.catalog()
        if kind == "games":
            records = catalog.games()
        else:
            records = [r for r in catalog.records if r.kind is AppKind.APP or (kind == "all" and r.kind is AppKind.GAME)]
        if query:
            q = normalize(query)
            records = [r for r in records if any(q in a for a in r.aliases) or q in normalize(r.display_name)]
        if not records:
            return ToolResult(True, "I didn't find any." if query else f"I didn't find any installed {kind}.",
                              {"count": 0}, verified=True)
        label = {"games": "Installed games", "apps": "Installed apps"}.get(kind, "Installed apps and games")
        shown = [f"{r.display_name} ({r.launcher})" if r.launcher else r.display_name for r in records[:40]]
        more = f" and {len(records) - 40} more" if len(records) > 40 else ""
        return ToolResult(True, f"{label}: " + ", ".join(shown) + more + ".", {"count": len(records)}, verified=True)

    name_param = (Param("name", "str", "the app or game name, e.g. 'Chrome' or 'GTA V'"),)
    return [
        Tool(name="launch_app",
             description="Start an installed application or game by name (e.g. Chrome, Steam, GTA V). "
                         "Only for installed programs - use open_url for websites.",
             risk=Risk.SAFE, params=name_param, run=launch_app, progress=progress, receives_origin=True,
             describe=lambda a: f"Start {a['name']}"),
        Tool(name="close_app",
             description="Close a running app or game gracefully (like clicking its X).",
             risk=Risk.SAFE, params=name_param, run=close_app, describe=lambda a: f"Close {a['name']}"),
        Tool(name="force_close_app",
             description="Force an app or game to end when it won't close normally. Unsaved work is lost.",
             risk=Risk.CONFIRM, params=name_param, run=force_close_app,
             describe=lambda a: f"Force-close {a['name']} (unsaved work in it will be lost)"),
        Tool(name="set_app_alias",
             description="Give an installed app a personal name, e.g. 'Chrome is my browser' -> app=Chrome, "
                         "alias=browser. Then 'open my browser' opens it.",
             risk=Risk.SAFE, receives_origin=True,
             params=(Param("app", "str", "the installed app, e.g. 'Chrome'"),
                     Param("alias", "str", "the personal name, e.g. 'browser'")),
             run=set_app_alias, describe=lambda a: f"Call {a['app']} your {a['alias']}"),
        Tool(name="remove_app_alias",
             description="Remove a personal app name the user set earlier.",
             risk=Risk.SAFE, params=(Param("alias", "str", "the personal name, e.g. 'browser'"),),
             run=remove_app_alias),
        Tool(name="open_folder",
             description="Show a folder in File Explorer: Downloads, Documents, Desktop, Pictures, Music, Videos, "
                         "This PC, or a full folder path.",
             risk=Risk.SAFE, params=(Param("target", "str", "folder name like 'Downloads', or a full path"),),
             run=open_folder),
        Tool(name="list_apps",
             description="List installed games or apps (e.g. 'what games do I have?').",
             risk=Risk.SAFE,
             params=(Param("kind", "str", "games, apps or all", required=False, default="games",
                           choices=("games", "apps", "all")),
                     Param("query", "str", "optional name filter", required=False)),
             run=list_apps),
    ]
