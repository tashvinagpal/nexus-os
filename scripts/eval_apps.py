"""Measure app & game discovery on this PC. Read-only: nothing is launched.

    .venv\\Scripts\\python.exe scripts\\eval_apps.py

1. Shortcut parsing vs Windows' own reader (WScript.Shell, used here only as ground truth)
2. Name resolution for common phrasings (only for apps that are actually installed)
3. Game discovery vs the launchers' own manifests
4. Fast-path routing of launch requests (no model)
5. Discovery time and memory
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus.ai.router import FastPathRouter  # noqa: E402
from nexus.system.apps.service import AppService  # noqa: E402
from nexus.system.apps.shortcuts import read_lnk  # noqa: E402
from nexus.system.games.steam import SteamProvider, parse_vdf  # noqa: E402
from nexus.system.platform import current  # noqa: E402

# phrase -> expected display name (checked only if that app is installed here)
RESOLUTION = {
    "chrome": "Google Chrome", "open chrome": "Google Chrome", "launch google chrome": "Google Chrome",
    "fire up chrome": "Google Chrome", "steam": "Steam", "start steam": "Steam", "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code", "code": "Visual Studio Code", "word": "Word", "excel": "Excel",
    "powerpoint": "PowerPoint", "obs": "OBS Studio", "edge": "Microsoft Edge", "microsoft edge": "Microsoft Edge",
    "epic games": "Epic Games Launcher", "epic launcher": "Epic Games Launcher",
    "rockstar launcher": "Rockstar Games Launcher", "zoom": "Zoom Workplace", "krita": "Krita (x64)",
    "ollama": "Ollama", "opera": "Opera Browser", "qbittorrent": "qBittorrent", "task manager": "Task Manager",
    "calculator": "Calculator", "notepad": "Notepad", "settings": "Settings", "paint": "Paint",
    "whatsapp": "WhatsApp", "microsoft store": "Microsoft Store", "file explorer": "File Explorer",
    "onenote": "OneNote", "outlook": "Outlook",
    "gta": "Grand Theft Auto V Legacy", "gta v": "Grand Theft Auto V Legacy", "gta 5": "Grand Theft Auto V Legacy",
    "gtav": "Grand Theft Auto V Legacy", "grand theft auto": "Grand Theft Auto V Legacy",
    "grand theft auto v": "Grand Theft Auto V Legacy", "once human": "Once Human", "rocket league": "Rocket League",
    "valorant": "VALORANT", "valo": "VALORANT",
}
# phrase -> expected kind (these must be refused, never launched)
REFUSED = {"cmd": "shell", "command prompt": "shell", "git bash": "shell", "python": "shell",
           "registry editor": "admin", "regedit": "admin", "services": "admin", "event viewer": "admin"}
FAST_PATH = {  # text -> (tool, expected display name or target)
    "open chrome": ("launch_app", "Google Chrome"), "launch steam": ("launch_app", "Steam"),
    "Nexus bhai GTA chala de": ("launch_app", "Grand Theft Auto V Legacy"),
    "fire up grand theft auto": ("launch_app", "Grand Theft Auto V Legacy"),
    "chrome khol do": ("launch_app", "Google Chrome"), "vs code open karo": ("launch_app", "Visual Studio Code"),
    "open settings": ("launch_app", "Settings"), "open calculator": ("launch_app", "Calculator"),
    "open downloads": ("open_folder", None), "downloads folder kholo": ("open_folder", None),
    "go to youtube": ("open_url", "https://www.youtube.com"), "go to github.com": ("open_url", "https://github.com"),
    "open youtube": ("open_url", "https://www.youtube.com"),
    "run C:\\Users\\me\\Downloads\\setup.exe": ("launch_app", None),
}
NOT_FAST = ["open spotify", "Nexus, I'm bored. Open GTA.", "yaar kuch khelte hain", "open chrome and word",
            "start a new document in word", "play some music"]


def windows_truth(paths: list[Path]) -> dict[str, tuple[str, str]]:
    script = ("[Console]::InputEncoding=[Text.Encoding]::UTF8; [Console]::OutputEncoding=[Text.Encoding]::UTF8; "
              "$sh = New-Object -ComObject WScript.Shell; $paths = [Console]::In.ReadToEnd() | ConvertFrom-Json; "
              "$paths | ForEach-Object { $l = $sh.CreateShortcut($_); [pscustomobject]@{p=$_; t=$l.TargetPath; "
              "a=$l.Arguments} } | ConvertTo-Json -Compress")
    proc = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                          input=json.dumps([str(p) for p in paths]), capture_output=True, text=True,
                          encoding="utf-8", timeout=120)
    rows = json.loads(proc.stdout or "[]")
    rows = [rows] if isinstance(rows, dict) else rows
    return {r["p"]: (r["t"] or "", (r["a"] or "").strip()) for r in rows}


def main() -> None:
    plat = current()
    rss_before = psutil.Process().memory_info().rss
    started = time.perf_counter()
    apps = AppService(plat)
    catalog = apps.catalog()
    seconds = time.perf_counter() - started
    rss_after = psutil.Process().memory_info().rss
    names = {r.display_name for r in catalog.records}
    print(f"discovery: {len(catalog.records)} records in {seconds:.2f}s, "
          f"+{(rss_after - rss_before) / 2**20:.1f} MB RAM (incl. imports)")

    # 1. shortcut parsing
    folders = plat.known_folders()
    links = [p for role in ("common_start_menu", "start_menu") for p in folders[role].rglob("*.lnk")]
    links += [p for role in ("desktop", "public_desktop") for p in folders[role].glob("*.lnk")]
    truth = windows_truth(links)
    agree = sum(1 for p in links if (s := read_lnk(p)) is not None
                and os.path.normcase(s.target or "") == os.path.normcase(truth.get(str(p), ("", ""))[0])
                and s.arguments == truth.get(str(p), ("", ""))[1])
    print(f"1. shortcut parsing: {agree}/{len(links)} identical to Windows")

    # 2. resolution
    checked = ok = 0
    for phrase, expected in RESOLUTION.items():
        if expected not in names:
            continue
        checked += 1
        match = catalog.resolve(phrase).match
        if match is not None and match.display_name == expected:
            ok += 1
        else:
            got = match.display_name if match else [c.display_name for c in catalog.resolve(phrase).candidates]
            print(f"   MISS {phrase!r} -> {got} (expected {expected})")
    refused_ok = sum(1 for phrase, kind in REFUSED.items()
                     if (m := catalog.resolve(phrase).match) is None or m.kind.value == kind
                     or catalog.resolve(phrase).ambiguous)
    print(f"2. name resolution: {ok}/{checked} correct; shells/admin tools classified for refusal: "
          f"{refused_ok}/{len(REFUSED)}")

    # 3. games vs launcher manifests
    steam = SteamProvider(plat)
    manifest_games = []
    for lib in steam.libraries():
        for m in (lib / "steamapps").glob("appmanifest_*.acf"):
            state = parse_vdf(m.read_text(encoding="utf-8", errors="replace")).get("AppState", {})
            if int(state.get("StateFlags", "0")) & 4:
                manifest_games.append(state.get("name"))
    epic_dir = folders["program_data"] / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
    epic_games = [json.loads(p.read_text(encoding="utf-8-sig")).get("DisplayName") for p in epic_dir.glob("*.item")] \
        if epic_dir.is_dir() else []
    games = catalog.games()
    with_process = [g for g in games if g.install_dir and any(
        (Path(g.install_dir) / p).exists() or any(Path(g.install_dir).rglob(p)) for p in g.process_names)]
    print(f"3. games: {len(games)} found ({', '.join(g.display_name for g in games)}); launcher manifests list "
          f"{len(manifest_games)} Steam ({', '.join(manifest_games)}) + {len(epic_games)} Epic; "
          f"{len(with_process)}/{sum(1 for g in games if g.install_dir)} with the game's own exe located on disk")

    # 4. fast path
    router = FastPathRouter(apps=apps)
    fast_ok = 0
    for text, (tool, expected) in FAST_PATH.items():
        plan = router.route(text)
        step = plan.actions[0] if plan else None
        good = step is not None and step.tool == tool
        if good and tool == "launch_app" and expected:
            good = catalog.resolve(step.args["name"]).match is not None and \
                catalog.resolve(step.args["name"]).match.display_name == expected
        if good and tool == "open_url":
            good = step.args["url"] == expected
        fast_ok += good
        if not good:
            print(f"   FAST-PATH MISS {text!r} -> {step}")
    wrongly_fast = [t for t in NOT_FAST if router.route(t) is not None]
    for t in wrongly_fast:
        print(f"   SHOULD USE PLANNER {t!r} -> {router.route(t).actions[0]}")
    print(f"4. fast path: {fast_ok}/{len(FAST_PATH)} launch/folder/site requests routed correctly without a model; "
          f"{len(NOT_FAST) - len(wrongly_fast)}/{len(NOT_FAST)} uncertain requests correctly left to the planner")


if __name__ == "__main__":
    main()
