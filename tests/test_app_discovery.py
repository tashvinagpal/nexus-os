"""Discovery: shortcut parsing, classification, aliases, merging and name resolution."""

import os
from pathlib import Path

import pytest

from nexus.system.apps.catalog import AppCatalog
from nexus.system.apps.names import aliases_for, strip_launch_words
from nexus.system.apps.records import AppKind, RawEntry
from nexus.system.apps.shortcuts import parse_lnk, read_lnk, read_url
from nexus.system.apps.sources import DesktopSource, StartMenuSource, classify
from nexus.system.platform.base import LaunchMethod
from tests.apps_fixtures import AppsPlatform, build_lnk, make_exe, make_lnk, make_url


# --- .lnk / .url ----------------------------------------------------------------------------

def test_lnk_round_trip():
    s = parse_lnk(build_lnk(r"C:\Apps\Tool\tool.exe", args="--fast", workdir=r"C:\Apps\Tool"))
    assert s.target == r"C:\Apps\Tool\tool.exe" and s.arguments == "--fast" and s.working_dir == r"C:\Apps\Tool"


def test_lnk_environment_variable_target(monkeypatch):
    monkeypatch.setenv("NEXUS_TEST_ROOT", r"D:\Games")
    s = parse_lnk(build_lnk(env_target=r"%NEXUS_TEST_ROOT%\game.exe"))
    assert s.target == r"D:\Games\game.exe"


def test_lnk_without_target_is_a_shell_item():
    assert parse_lnk(build_lnk(args="--launch-product=valorant")).target is None


@pytest.mark.parametrize("junk", [b"", b"not a shortcut", b"\x4c\x00\x00\x00" + b"\x00" * 80])
def test_garbage_is_rejected(junk):
    with pytest.raises(ValueError):
        parse_lnk(junk)


def test_url_shortcut(tmp_path):
    assert read_url(make_url(tmp_path, "GTA", "steam://rungameid/271590")) == "steam://rungameid/271590"


@pytest.mark.skipif(os.name != "nt", reason="reads this PC's real Start menu")
def test_real_windows_shortcut_is_read():
    folder = Path(os.environ["ProgramData"]) / r"Microsoft\Windows\Start Menu\Programs"
    links = [read_lnk(p) for p in list(folder.rglob("*.lnk"))[:40]]
    readable = [link for link in links if link and link.target]
    assert readable and all(Path(link.target).is_absolute() for link in readable)


# --- classification ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry,expected", [
    (RawEntry("Google Chrome", "s", target=r"C:\P\chrome.exe"), (AppKind.APP, None)),
    (RawEntry("Command Prompt", "s", target=r"C:\Windows\system32\cmd.exe"), (AppKind.SHELL, None)),
    (RawEntry("Windows PowerShell", "s", target=r"C:\Windows\powershell.exe"), (AppKind.SHELL, None)),
    (RawEntry("Git Bash", "s", target=r"C:\Git\git-bash.exe"), (AppKind.SHELL, None)),
    (RawEntry("Python 3.10", "s", target=r"C:\Python\python.exe"), (AppKind.SHELL, None)),
    (RawEntry("IDLE", "s", target=r"C:\Python\pythonw.exe", arguments="-m idlelib"), (AppKind.SHELL, None)),
    (RawEntry("Start DB", "s", target=r"C:\db\start.bat"), (AppKind.SHELL, None)),
    (RawEntry("Registry Editor", "s", target=r"C:\Windows\regedit.exe"), (AppKind.ADMIN, None)),
    (RawEntry("services", "s", target=r"C:\Windows\system32\services.msc"), (AppKind.ADMIN, None)),
    (RawEntry("Event Viewer", "s", target=r"C:\Windows\system32\mmc.exe", arguments="eventvwr.msc"),
     (AppKind.ADMIN, None)),
    (RawEntry("Run", "s", shortcut="Run.lnk"), (AppKind.SHELL, None)),
    (RawEntry("GTA V", "s", url="steam://rungameid/271590"), (AppKind.GAME, "steam")),
    (RawEntry("Fortnite", "s", url="com.epicgames.launcher://apps/fn%3Aa%3Ab?action=launch"), (AppKind.GAME, "epic")),
    (RawEntry("VALORANT", "s", shortcut="V.lnk", arguments="--launch-product=valorant"), (AppKind.GAME, "riot")),
    (RawEntry("Calculator", "store", aumid="Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"), (AppKind.APP, None)),
])
def test_classify(entry, expected):
    assert classify(entry) == expected


@pytest.mark.parametrize("entry", [
    RawEntry("Uninstall Zoom", "s", target=r"C:\Zoom\uninstall.exe"),
    RawEntry("Node.js website", "s", url="https://nodejs.org"),
    RawEntry("Python Manuals", "s", target=r"C:\Python\python.chm"),
    RawEntry("Git Release Notes", "s", target=r"C:\Git\ReleaseNotes.html"),
    RawEntry("Setup", "s", target=r"C:\X\setup.exe"),
])
def test_junk_is_dropped(entry):
    assert classify(entry) is None


# --- aliases -----------------------------------------------------------------------------------

def test_game_aliases():
    a = aliases_for("Grand Theft Auto V Legacy")
    assert {"gta", "gta v", "gta 5", "gtav", "gta5", "grand theft auto v", "grand theft auto 5"} <= a


def test_app_aliases():
    assert "chrome" in aliases_for("Google Chrome")
    assert {"vs code", "vscode", "code"} <= aliases_for("Visual Studio Code")
    assert "chrome" in aliases_for("Some Shortcut", exe_stem="chrome")
    assert "launcher" not in aliases_for("X", exe_stem="Launcher")  # generic exe names never become aliases


@pytest.mark.parametrize("text,name", [("open Chrome", "Chrome"), ("fire up Grand Theft Auto", "Grand Theft Auto"),
                                       ("GTA chala de", "GTA"), ("steam start karo", "steam"),
                                       ("launch Google Chrome", "Google Chrome"), ("chrome khol do na", "chrome"),
                                       ("GTA V chala dijiye", "GTA V"), ("steam start kar de", "steam"),
                                       ("GTA khelte hain", "GTA")])
def test_launch_words_are_stripped(text, name):
    assert strip_launch_words(text) == name


# --- catalog: merging and resolution ------------------------------------------------------------

def entries(tmp_path):
    chrome = str(make_exe(tmp_path / "Chrome" / "chrome.exe"))
    return [
        RawEntry("Google Chrome", "start_menu", target=chrome),
        RawEntry("Google Chrome", "desktop", target=chrome.upper(), confidence=0.9),       # same app, other case
        RawEntry("chrome", "app_paths", target=chrome, confidence=0.6),                   # alias enrichment only
        RawEntry("Work - Chrome", "desktop", target=chrome, arguments='--profile-directory="Profile 1"'),
        RawEntry("Grand Theft Auto V Legacy", "start_menu", url="steam://rungameid/271590"),
        RawEntry("Grand Theft Auto IV", "start_menu", url="steam://rungameid/12210"),
        RawEntry("Command Prompt", "start_menu", target=r"C:\Windows\system32\cmd.exe"),
        RawEntry("helperthing", "app_paths", target=str(make_exe(tmp_path / "h" / "helperthing.exe")), confidence=0.6),
    ]


def test_duplicates_merge_and_profiles_stay_separate(tmp_path):
    cat = AppCatalog.build(entries(tmp_path))
    names = [r.display_name for r in cat.records]
    assert names.count("Google Chrome") == 1
    chrome = cat.resolve("chrome").match
    assert chrome.source == "start_menu+desktop+app_paths"
    assert "Work - Chrome" in names
    assert "helperthing" not in names  # App Paths alone never creates an app


def test_open_chrome_resolves_to_the_installed_app(tmp_path):
    res = AppCatalog.build(entries(tmp_path)).resolve("open Chrome")
    assert res.confident and res.match.display_name == "Google Chrome"
    assert res.match.launch_method is LaunchMethod.EXECUTABLE and res.match.target.endswith("chrome.exe")
    assert not res.match.target.startswith("http")


@pytest.mark.parametrize("query", ["gta v", "GTA 5", "gtav", "Grand Theft Auto V", "fire up grand theft auto v"])
def test_aliases_resolve(tmp_path, query):
    res = AppCatalog.build(entries(tmp_path)).resolve(query)
    assert res.match is not None and res.match.display_name == "Grand Theft Auto V Legacy"


def test_ambiguous_names_are_not_guessed(tmp_path):
    res = AppCatalog.build(entries(tmp_path)).resolve("GTA")
    assert res.ambiguous and res.match is None
    assert {c.display_name for c in res.candidates} == {"Grand Theft Auto V Legacy", "Grand Theft Auto IV"}


def test_unknown_names_resolve_to_nothing(tmp_path):
    cat = AppCatalog.build(entries(tmp_path))
    for q in ("spotify", "del /s /q c:\\", "rm -rf", "format c:"):
        assert cat.resolve(q).match is None


def test_typos_still_match(tmp_path):
    assert AppCatalog.build(entries(tmp_path)).resolve("chorme").match.display_name == "Google Chrome"


# --- sources ------------------------------------------------------------------------------------

def test_start_menu_and_desktop_sources(tmp_path):
    plat = AppsPlatform(tmp_path)
    exe = make_exe(tmp_path / "apps" / "obs64.exe")
    make_lnk(plat.folders["common_start_menu"] / "OBS Studio", "OBS Studio", str(exe))
    make_url(plat.folders["start_menu"], "Once Human", "steam://rungameid/2139460")
    make_lnk(plat.folders["desktop"], "OBS Studio", str(exe))
    make_exe(plat.folders["desktop"] / "random.exe")  # loose executables on the Desktop are ignored
    found = list(StartMenuSource(plat).entries()) + list(DesktopSource(plat).entries())
    assert sorted(e.name for e in found) == ["OBS Studio", "OBS Studio", "Once Human"]
    cat = AppCatalog.build(found)
    assert [r.display_name for r in cat.records] == ["OBS Studio", "Once Human"]
    assert cat.resolve("obs").match.kind is AppKind.APP
