"""Steam / Epic / Rockstar providers (fixture installs) and GameService (mock providers)."""

import json

import pytest

from nexus.core.resources import ResourceManager, ResourceState
from nexus.system.apps.catalog import AppCatalog
from nexus.system.apps.launcher import Launcher
from nexus.system.apps.names import aliases_for
from nexus.system.apps.records import AppKind, GameRecord
from nexus.system.games.epic import EpicProvider
from nexus.system.games.providers import game_process_names
from nexus.system.games.rockstar import ROOT as ROCKSTAR, RockstarProvider
from nexus.system.games.service import GameService, GameWatcher
from nexus.system.games.steam import SteamProvider, parse_vdf
from nexus.system.platform.base import LaunchMethod, LaunchSpec
from tests.apps_fixtures import AppsPlatform, Background, FakeClock, FakeProcs, make_exe
from tests.conftest import FakeBackend, FakeProbe


# --- Steam -------------------------------------------------------------------------------------

def manifest(appid, name, installdir, flags=4):
    return f'"AppState"\n{{\n\t"appid"\t\t"{appid}"\n\t"name"\t\t"{name}"\n\t"StateFlags"\t\t"{flags}"\n' \
           f'\t"installdir"\t\t"{installdir}"\n}}\n'


@pytest.fixture
def steam(tmp_path):
    main, second = tmp_path / "Steam", tmp_path / "D" / "SteamLibrary"
    (main / "steamapps").mkdir(parents=True)
    (second / "steamapps").mkdir(parents=True)
    lib = str(second).replace("\\", "\\\\")
    (main / "steamapps" / "libraryfolders.vdf").write_text(
        f'"libraryfolders"\n{{\n\t"0"\n\t{{\n\t\t"path"\t\t"{str(main).replace(chr(92), chr(92) * 2)}"\n\t}}\n'
        f'\t"1"\n\t{{\n\t\t"path"\t\t"{lib}"\n\t}}\n}}\n', encoding="utf-8")
    (main / "steamapps" / "appmanifest_271590.acf").write_text(manifest(271590, "Grand Theft Auto V Legacy",
                                                                        "Grand Theft Auto V"), encoding="utf-8")
    (main / "steamapps" / "appmanifest_228980.acf").write_text(
        manifest(228980, "Steamworks Common Redistributables", "Steamworks Shared"), encoding="utf-8")
    (main / "steamapps" / "appmanifest_999.acf").write_text(manifest(999, "Half Downloaded", "HD", flags=1026 & ~4),
                                                            encoding="utf-8")
    (second / "steamapps" / "appmanifest_2139460.acf").write_text(manifest(2139460, "Once Human", "Once Human"),
                                                                  encoding="utf-8")
    gta = main / "steamapps" / "common" / "Grand Theft Auto V"
    make_exe(gta / "GTA5.exe", 4000)
    make_exe(gta / "PlayGTAV.exe", 100)
    make_exe(gta / "Redistributables" / "Social-Club-Setup.exe", 9000)  # bigger than the game
    once = second / "steamapps" / "common" / "Once Human"
    make_exe(once / "ffmpeg.exe", 9000)                                  # bigger than the game
    make_exe(once / "ONCE_HUMAN.exe", 8000)
    make_exe(once / "UniCrashReporter.exe", 100)
    return main


def test_vdf_parser():
    data = parse_vdf('"a"\n{\n\t"b"\t"1"\n\t"c"\n\t{\n\t\t"d"\t"x\\\\y"\n\t}\n}\n')
    assert data == {"a": {"b": "1", "c": {"d": "x\\y"}}}


def test_steam_discovers_installed_games_across_libraries(tmp_path, steam):
    games = {g.display_name: g for g in SteamProvider(AppsPlatform(tmp_path), steam_dir=steam).discover()}
    assert set(games) == {"Grand Theft Auto V Legacy", "Once Human"}  # no redistributables, no half-installs
    gta = games["Grand Theft Auto V Legacy"]
    assert gta.kind is AppKind.GAME and gta.launcher == "steam" and gta.game_id == "271590"
    assert gta.launch_method is LaunchMethod.URI and gta.target == "steam://rungameid/271590"
    assert gta.process_names == {"gta5.exe", "gta5_enhanced.exe"}
    # unknown title: the process is chosen by name, not by size (ffmpeg.exe is bigger)
    assert games["Once Human"].process_names == {"once_human.exe"}


def test_steam_found_via_registry(tmp_path, steam):
    plat = AppsPlatform(tmp_path)
    plat.registry[(r"HKCU\Software\Valve\Steam", "SteamPath")] = str(steam).replace("\\", "/")
    assert len(SteamProvider(plat).discover()) == 2


def test_no_steam_means_no_games(tmp_path):
    assert SteamProvider(AppsPlatform(tmp_path)).discover() == []


def test_game_process_heuristic_skips_helpers(tmp_path):
    folder = tmp_path / "My Game"
    make_exe(folder / "MyGame_BE.exe", 9000)
    make_exe(folder / "MyGameLauncher.exe", 9000)
    make_exe(folder / "Binaries" / "Win64" / "MyGame-Win64-Shipping.exe", 5000)
    assert game_process_names("My Game", folder) == {"mygame-win64-shipping.exe"}


# --- Epic --------------------------------------------------------------------------------------

def test_epic_manifests(tmp_path):
    plat = AppsPlatform(tmp_path)
    folder = plat.folders["program_data"] / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
    folder.mkdir(parents=True)
    base = {"InstallLocation": str(tmp_path / "rl"), "LaunchExecutable": "Binaries/Win64/RocketLeague.exe",
            "CatalogNamespace": "ns", "CatalogItemId": "item", "AppCategories": ["public", "games"]}
    (folder / "a.item").write_text(json.dumps({**base, "DisplayName": "Rocket League®", "AppName": "Sugar"}),
                                   encoding="utf-8")
    (folder / "b.item").write_text(json.dumps({**base, "DisplayName": "Unreal Engine", "AppName": "UE",
                                                "AppCategories": ["engines"]}), encoding="utf-8")
    (folder / "c.item").write_text(json.dumps({**base, "DisplayName": "Half", "AppName": "H",
                                                "bIsIncompleteInstall": True}), encoding="utf-8")
    [rl] = EpicProvider(plat).discover()
    assert rl.display_name == "Rocket League" and rl.process_names == {"rocketleague.exe"}
    assert rl.target == "com.epicgames.launcher://apps/ns%3Aitem%3ASugar?action=launch&silent=true"


# --- Rockstar ----------------------------------------------------------------------------------

def test_rockstar_launcher_titles_only(tmp_path):
    plat = AppsPlatform(tmp_path)
    rdr2 = tmp_path / "Rockstar" / "Red Dead Redemption 2"
    make_exe(rdr2 / "RDR2.exe")
    plat.registry[(f"{ROCKSTAR}\\Red Dead Redemption 2", "InstallFolder")] = str(rdr2)
    plat.registry[(f"{ROCKSTAR}\\Grand Theft Auto V", "InstallFolderSteam")] = str(tmp_path / "steam" / "gta")
    plat.registry[(f"{ROCKSTAR}\\Launcher", "InstallFolder")] = str(tmp_path)
    [game] = RockstarProvider(plat).discover()
    assert game.display_name == "Red Dead Redemption 2" and game.target.endswith("RDR2.exe")
    assert game.process_names == {"rdr2.exe"}


def test_same_game_from_two_launchers_is_one_game(tmp_path, steam):
    plat = AppsPlatform(tmp_path)
    gta_dir = steam / "steamapps" / "common" / "Grand Theft Auto V"
    rockstar_copy = GameRecord(id="rockstar:gta", display_name="Grand Theft Auto V", aliases=aliases_for("GTA V"),
                               kind=AppKind.GAME, source="rockstar", launch_method=LaunchMethod.EXECUTABLE,
                               target=str(gta_dir / "PlayGTAV.exe"), install_dir=str(gta_dir), launcher="rockstar")
    games = SteamProvider(plat, steam_dir=steam).discover() + [rockstar_copy]
    cat = AppCatalog.build([], games)
    gtas = [g for g in cat.games() if "grand theft auto" in g.display_name.lower()]
    assert len(gtas) == 1 and gtas[0].launcher == "steam"  # Steam owns it: launch through Steam


# --- GameService with mock providers -------------------------------------------------------------

class MockProvider:
    def __init__(self, name, games):
        self.name = name
        self.games = games
        self.specs = []

    def discover(self):
        return list(self.games)

    def launch_spec(self, game):
        spec = LaunchSpec(LaunchMethod.URI, f"{self.name}://run/{game.game_id}")
        self.specs.append(spec)
        return spec


def mock_game(name, launcher, process, game_id="1"):
    return GameRecord(id=f"{launcher}:{game_id}", display_name=name, aliases=aliases_for(name), kind=AppKind.GAME,
                      source=launcher, launch_method=LaunchMethod.URI, target=f"{launcher}://{game_id}",
                      process_names=frozenset({process}), launcher=launcher, game_id=game_id)


@pytest.fixture
def gs(tmp_path):
    plat = AppsPlatform(tmp_path)
    procs, clock = FakeProcs(), FakeClock()
    launcher = Launcher(plat, processes=procs, clock=clock, sleep=clock.sleep, poll_s=0.5)
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    big = FakeBackend("llm:8b@gpu", vram_mb=5100, probe=probe)
    rm.register(big)
    providers = [MockProvider("steam", [mock_game("Grand Theft Auto V Legacy", "steam", "gta5.exe", "271590")]),
                 MockProvider("epic", [mock_game("Rocket League", "epic", "rocketleague.exe", "Sugar")]),
                 MockProvider("rockstar", [])]
    bg, events = Background(), []
    service = GameService(providers, launcher, game_mode=rm, timeout_s=60, background=bg)
    service.events.subscribe(events.append)
    catalog = AppCatalog.build([], service.discover())
    service.watcher = GameWatcher(catalog.games, rm, procs)
    return type("GS", (), dict(service=service, catalog=catalog, plat=plat, procs=procs, clock=clock, rm=rm,
                               big=big, providers=providers, bg=bg, events=events))


def test_find_game_through_the_interface(gs):
    assert gs.service.find_game("gta 5", gs.catalog).match.display_name == "Grand Theft Auto V Legacy"
    assert gs.service.find_game("rocket league", gs.catalog).match.launcher == "epic"
    assert gs.service.find_game("chrome", gs.catalog).match is None


def test_launch_game_uses_its_provider_and_verifies(gs):
    gta = gs.service.find_game("gta", gs.catalog).match
    gs.plat.on_launch = lambda spec: gs.procs.spawn("GTA5.exe", None, after_polls=3)
    result = gs.service.launch_game(gta)
    assert result.ok and result.pending and result.verified is None  # returned before verification
    assert gs.plat.launched == [LaunchSpec(LaunchMethod.URI, "steam://run/271590")]
    [event] = gs.events                                                # verified in the background
    assert event.ok and event.verified is True and event.message == "Grand Theft Auto V Legacy is running."
    assert gs.service.is_game_running(gta)


def test_launch_game_enters_game_mode_first(gs):
    with gs.rm.use("llm:8b@gpu"):
        pass
    assert gs.big.loaded
    seen = []
    gs.plat.on_launch = lambda spec: (seen.append((gs.rm.state, gs.big.loaded)),
                                      gs.procs.spawn("GTA5.exe", None, after_polls=1))
    gs.service.launch_game(gs.catalog.resolve("gta").match)
    assert seen == [(ResourceState.GAME_MODE, False)]  # VRAM freed *before* the game started


def test_failed_game_launch_is_not_reported_as_running_and_frees_game_mode(gs):
    gta = gs.catalog.resolve("gta").match
    gs.bg.defer = True
    result = gs.service.launch_game(gta)  # no process ever appears
    assert result.pending and gs.rm.game_mode  # game mode held while it starts
    gs.bg.run_pending()
    [event] = gs.events
    assert not event.ok and event.verified is False
    assert "couldn't verify" in event.message and "is running" not in event.message
    assert not gs.rm.game_mode  # nothing started: resources given back


def test_already_running_game_is_not_launched_twice(gs):
    gs.procs.add("GTA5.exe", None)
    result = gs.service.launch_game(gs.catalog.resolve("gta").match)
    assert result.already_running and gs.plat.launched == [] and gs.plat.focused


def test_watcher_leaves_game_mode_when_the_game_exits_without_reloading_ai(gs):
    gs.plat.on_launch = lambda spec: gs.procs.spawn("GTA5.exe", None, after_polls=1)
    gs.service.launch_game(gs.catalog.resolve("gta").match)
    assert gs.rm.game_mode and gs.service.watcher.check() == ["Grand Theft Auto V Legacy"]
    gs.procs.remove("GTA5.exe")
    assert gs.service.watcher.check() == []
    assert not gs.rm.game_mode and not gs.big.loaded  # AI comes back lazily, not now


def test_watcher_detects_games_started_outside_nexus(gs):
    gs.procs.add("RocketLeague.exe", None)
    gs.service.watcher.check()
    assert gs.rm.game_mode


def test_watcher_never_undoes_a_manual_game_mode(gs):
    gs.rm.enter_game_mode()  # the user typed /gamemode on
    gs.service.watcher.check()
    assert gs.rm.game_mode


def test_a_broken_provider_does_not_hide_the_others(gs):
    class Broken:
        name = "broken"

        def discover(self):
            raise RuntimeError("corrupt manifest")

    service = GameService([Broken(), *gs.providers], gs.service.launcher)
    assert {g.display_name for g in service.discover()} == {"Grand Theft Auto V Legacy", "Rocket League"}
