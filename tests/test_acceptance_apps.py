"""Subsystem 4 acceptance criteria (App & Game Discovery and Launching), full stack on simulated hardware.

Numbers match the criteria list; #14/#15 are the whole default and live suites.
"""

from types import SimpleNamespace

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.responder import claims_success
from nexus.ai.conversation.types import ActionReport, Say, SayKind, UserInput
from nexus.ai.planner import LLMPlanner
from nexus.ai.providers import HybridChat, ProviderOption, SelectionContext
from nexus.ai.router import FastPathRouter
from nexus.core.kernel import ActionKernel, Status
from nexus.core.resources import Placement, ResourceManager, ResourceState
from nexus.mesh.client import MeshClient
from nexus.mesh.gateway import MeshGateway
from nexus.mesh.registry import DeviceRecord, DeviceRegistry
from nexus.mesh.transport import LoopbackHub
from nexus.security.identity import DeviceKey
from nexus.system.apps.records import RawEntry
from nexus.system.apps.service import AppService, AppSettings
from nexus.system.device import DeviceRole
from nexus.system.games.steam import SteamProvider
from nexus.system.platform.base import LaunchMethod
from nexus.tools import ToolRegistry, default_tools
from nexus.ai.memory.sqlite_store import SQLiteMemoryStore
from tests.apps_fixtures import AppsPlatform, Background, FakeClock, FakeProcs, NameGuard, make_exe
from tests.conftest import FakeChat, FakeProbe
from tests.test_acceptance import assert_honest
from tests.test_game_providers import manifest

CHAT = '{"language": "en", "actions": [], "say": "Want to play something?"}'


class ListSource:
    name = "test"

    def __init__(self, entries):
        self._entries = entries

    def entries(self):
        return list(self._entries)


def build(tmp_path, policy, audit, *, gta_iv=False):
    plat = AppsPlatform(tmp_path)
    procs, clock = FakeProcs(), FakeClock()
    apps_dir = tmp_path / "home" / "apps"
    chrome = make_exe(apps_dir / "Google" / "Chrome" / "chrome.exe")
    steam_exe = make_exe(apps_dir / "Steam" / "steam.exe")
    # A real-format Steam library with GTA V installed
    lib = tmp_path / "SteamLib"
    (lib / "steamapps").mkdir(parents=True)
    (lib / "steamapps" / "appmanifest_271590.acf").write_text(
        manifest(271590, "Grand Theft Auto V Legacy", "Grand Theft Auto V"), encoding="utf-8")
    make_exe(lib / "steamapps" / "common" / "Grand Theft Auto V" / "GTA5.exe", 500)
    if gta_iv:
        (lib / "steamapps" / "appmanifest_12210.acf").write_text(
            manifest(12210, "Grand Theft Auto IV", "GTA IV"), encoding="utf-8")
        make_exe(lib / "steamapps" / "common" / "GTA IV" / "GTAIV.exe", 500)
    entries = [RawEntry("Google Chrome", "start_menu", target=str(chrome)),
               RawEntry("Steam", "start_menu", target=str(steam_exe)),
               RawEntry("Command Prompt", "start_menu", target=r"C:\Windows\system32\cmd.exe")]

    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    big = FakeChat("llm:8b@gpu", vram_mb=5100, ram_mb=500, probe=probe, reply=CHAT)
    cpu = FakeChat("llm:3b@cpu", placement=Placement.CPU, ram_mb=2500, probe=probe, reply=CHAT)
    hybrid = HybridChat([ProviderOption(big, big.spec, 2), ProviderOption(cpu, cpu.spec, 1)], rm, SelectionContext)
    memory = SQLiteMemoryStore(tmp_path / "memory.db")
    bg, app_events = Background(), []
    apps = AppService(plat, sources=[ListSource(entries)], providers=[SteamProvider(plat, steam_dir=lib)],
                      processes=procs, game_mode=rm, clock=clock, sleep=clock.sleep, aliases=memory,
                      background=bg, guard=NameGuard(),
                      settings=AppSettings(app_launch_timeout_s=10, game_launch_timeout_s=60))
    apps.subscribe(app_events.append)
    registry = ToolRegistry(default_tools(plat, memory, apps))
    kernel = ActionKernel(registry, policy, audit)
    engine = ConversationEngine(kernel, LLMPlanner(hybrid, list(registry)), resources=rm,
                                router=FastPathRouter(apps=apps))

    def talk(text, device=None):
        events = []
        engine.handle(UserInput(text, device=device), events.append)
        return events

    def game_starts():
        plat.on_launch = lambda spec: procs.spawn("GTA5.exe", str(lib / "steamapps" / "common" /
                                                                  "Grand Theft Auto V" / "GTA5.exe"), after_polls=2)

    def chrome_starts():
        plat.on_launch = lambda spec: procs.spawn("chrome.exe", str(chrome), after_polls=2)

    return SimpleNamespace(plat=plat, procs=procs, clock=clock, rm=rm, big=big, cpu=cpu, apps=apps, kernel=kernel,
                           engine=engine, talk=talk, chrome=chrome, steam=steam_exe, game_starts=game_starts,
                           chrome_starts=chrome_starts, bg=bg, app_events=app_events, memory=memory)


STARTING = "Starting Grand Theft Auto V Legacy - I'll tell you once it's running."


def later(s):
    """What NEXUS said afterwards, from background verification."""
    return [e.message for e in s.app_events]


@pytest.fixture
def s(tmp_path, policy, audit):
    return build(tmp_path, policy, audit)


def reports(events):
    return [e for e in events if isinstance(e, ActionReport)]


def results(events):
    return [e.text for e in events if isinstance(e, Say) and e.kind is SayKind.RESULT]


def test_S01_open_chrome_is_the_installed_app_not_the_website(s):
    s.chrome_starts()
    events = s.talk("Open Chrome")
    assert [r.tool for r in reports(events)] == ["launch_app"]
    [spec] = s.plat.launched
    assert spec.method is LaunchMethod.EXECUTABLE and spec.target == str(s.chrome)
    assert not s.plat.opened  # no browser URL was opened


@pytest.mark.parametrize("text", ["open Chrome", "start Chrome", "launch Google Chrome", "fire up Chrome",
                                  "chrome khol do", "Nexus bhai chrome chala de"])
def test_S02_app_aliases(s, text):
    s.chrome_starts()
    assert reports(s.talk(text))[0].status is Status.DONE
    assert s.plat.launched[0].target == str(s.chrome)


@pytest.mark.parametrize("text", ["open GTA", "start GTA", "launch GTA V", "fire up Grand Theft Auto",
                                  "Nexus bhai GTA chala de", "GTA 5 start karo"])
def test_S02_game_aliases(s, text):
    s.game_starts()
    events = s.talk(text)
    assert results(events) == [STARTING]
    assert later(s) == ["Grand Theft Auto V Legacy is running."]


@pytest.mark.parametrize("name", ["rm -rf /", "del /s /q c:\\", "format c:", "shutdown /s /t 0", "notepad & calc",
                                  "powershell -enc AAAA"])
def test_S03_unknown_names_are_never_run_as_commands(s, name):
    s.big.reply = f'{{"language": "en", "actions": [{{"tool": "launch_app", "args": {{"name": "{name.replace(chr(92), chr(92) * 2)}"}}}}], "say": ""}}'
    events = s.talk(f"please run {name} right now and also more")
    assert s.plat.launched == []
    assert all(r.status in (Status.FAILED, Status.INVALID, Status.BLOCKED) for r in reports(events))


@pytest.mark.parametrize("text", [r"run C:\Users\me\Downloads\setup.exe", "open random.exe", "run hack.bat",
                                  "start payload.ps1", "launch \\\\server\\share\\tool.exe"])
def test_S04_arbitrary_executables_are_rejected(s, text):
    events = s.talk(text)
    assert s.plat.launched == []
    [refusal] = [e.text for e in events if isinstance(e, Say) and e.kind is SayKind.REFUSAL]
    assert refusal.startswith("I can't run files")  # a refusal, never "that didn't work"
    assert s.big.calls == []  # refused on the fast path, no model needed


def test_S05_already_running_is_detected(s):
    s.procs.add("chrome.exe", str(s.chrome))
    events = s.talk("open chrome")
    assert s.plat.launched == [] and s.plat.focused
    assert "already running" in results(events)[0]


def test_S06_duplicate_matches_ask(tmp_path, policy, audit):
    t = build(tmp_path, policy, audit, gta_iv=True)
    events = t.talk("open GTA")
    [question] = [e.text for e in events if isinstance(e, Say) and e.kind is SayKind.CLARIFY]
    assert "Grand Theft Auto V Legacy" in question and "Grand Theft Auto IV" in question
    assert t.plat.launched == []


def test_S07_success_requires_process_verification(s):
    s.chrome_starts()
    events = s.talk("open chrome")
    [report] = reports(events)
    assert report.status is Status.DONE and report.verified is True
    assert results(events) == ["Google Chrome is open."]


@pytest.mark.parametrize("scenario", ["no process", "os refuses", "game never starts"])
def test_S08_failed_launch_never_claims_success(s, scenario):
    if scenario == "os refuses":
        s.plat.launch_error = "access denied"
    events = s.talk("launch GTA V" if scenario == "game never starts" else "open chrome")
    background = [Say(SayKind.RESULT if e.ok else SayKind.ERROR, e.message) for e in s.app_events]
    assert_honest(events + background)
    assert not any(claims_success(e.text) for e in events + background if isinstance(e, Say))
    if scenario == "game never starts":
        assert results(events) == [STARTING]  # "starting" is not a claim that it runs
        assert later(s)[0].startswith("I couldn't verify that Grand Theft Auto V Legacy started")
    else:
        assert results(events) == []
        assert [e.text for e in events if isinstance(e, Say) and e.kind is SayKind.ERROR][0].startswith(
            "I couldn't")  # a failure, worded as one


def test_S09_gta_v_is_discovered(s):
    game = s.apps.find_game("GTA V").match
    assert game.display_name == "Grand Theft Auto V Legacy" and game.launcher == "steam"
    assert game.target == "steam://rungameid/271590" and "gta5.exe" in game.process_names


def test_S10_provider_interfaces(s):
    """find_game / launch_game / is_game_running work through the provider abstraction."""
    game = s.apps.find_game("gta").match
    assert not s.apps.games.is_game_running(game)
    s.game_starts()
    assert s.apps.games.launch_game(game).pending
    assert later(s) == ["Grand Theft Auto V Legacy is running."]
    assert s.apps.games.is_game_running(game)


def test_S11_game_launch_uses_game_mode_and_ai_comes_back_lazily(s):
    s.talk("yaar I'm bored")                       # the 8B model answers and stays loaded (idle)
    assert s.big.loaded
    vram_before = s.rm.snapshot(fresh=True).vram_used_mb
    states = []
    s.plat.on_launch = lambda spec: (states.append((s.rm.state, s.big.loaded)),
                                     s.procs.spawn("GTA5.exe", None, after_polls=1))
    events = s.talk("launch GTA V")
    assert states == [(ResourceState.GAME_MODE, False)]  # VRAM freed before the game started
    assert vram_before - s.rm.snapshot(fresh=True).vram_used_mb >= 5000
    assert results(events) == [STARTING] and later(s) == ["Grand Theft Auto V Legacy is running."]
    s.procs.remove("GTA5.exe")                     # the game exits
    s.apps.games.watcher.check()
    assert not s.rm.game_mode and not s.big.loaded  # back to normal, no automatic 8B reload


@pytest.mark.parametrize("text", ["open chrome", "launch steam", "GTA chala de", "open downloads",
                                  "go to youtube", "go to github.com"])
def test_S12_high_confidence_requests_use_the_fast_path(s, text):
    s.chrome_starts()
    s.talk(text)
    assert s.big.calls == [] and s.cpu.calls == []


def test_S13_launching_does_not_load_the_8b_model(s):
    s.chrome_starts()
    for text in ("open chrome", "launch steam", "open downloads", "GTA V chala de"):
        s.talk(text)
    assert not s.big.loaded and not s.cpu.loaded


def test_S13_unknown_app_falls_back_to_the_planner(s):
    s.talk("open spotify")  # not installed: not confident -> the planner decides
    assert len(s.big.calls) == 1


def test_model_choosing_a_website_for_an_installed_app_launches_the_app(s):
    """Seen in evaluation: "Nexus, Chrome khol de re" -> the model chose open_url google.com/chrome."""
    s.big.reply = '{"language": "hinglish", "actions": [{"tool": "open_url", "args": {"url": "https://www.google.com/chrome/"}}], "say": ""}'
    s.chrome_starts()
    events = s.talk("Nexus, Chrome ko ek baar khol ke dikhao re")  # not a fast-path phrase: the model plans it
    assert len(s.big.calls) == 1
    assert s.plat.opened == [] and s.plat.launched[0].target == str(s.chrome)
    assert results(events) == ["Google Chrome is open."]


def test_a_named_website_stays_a_website(s):
    s.big.reply = '{"language": "en", "actions": [{"tool": "open_url", "args": {"url": "https://www.google.com/chrome/"}}], "say": ""}'
    s.talk("show me the chrome download page on the website please")
    assert s.plat.opened == ["https://www.google.com/chrome/"] and s.plat.launched == []


@pytest.mark.parametrize("text", ["Nexus, Chrome khol de re", "chrome khol do ji"])
def test_haryanvi_and_polite_endings_use_the_fast_path(s, text):
    s.chrome_starts()
    s.talk(text)
    assert s.big.calls == [] and s.plat.launched[0].target == str(s.chrome)


def test_model_cannot_launch_an_app_the_user_never_mentioned(s):
    """Seen in evaluation: "wo browser wali baat bhool ja" -> the model proposed launch_app Steam."""
    s.big.reply = '{"language": "hinglish", "actions": [{"tool": "launch_app", "args": {"name": "Steam"}}], "say": ""}'
    events = s.talk("wo browser wali baat bhool ja")
    assert s.plat.launched == []
    assert [e.text for e in events if isinstance(e, Say) and e.kind is SayKind.CLARIFY] == ["Should I open Steam? (yes/no)"]
    s.plat.on_launch = lambda spec: s.procs.spawn("steam.exe", spec.target, after_polls=1)
    events = s.talk("haan")
    assert len(s.plat.launched) == 1 and results(events) == ["Steam is open."]


def test_model_launch_of_a_mentioned_app_runs_directly(s):
    s.big.reply = '{"language": "en", "actions": [{"tool": "launch_app", "args": {"name": "Grand Theft Auto V"}}], "say": ""}'
    s.game_starts()
    events = s.talk("Nexus, I'm bored. Open GTA.")  # not a fast-path phrase: the model plans it
    assert len(s.big.calls) == 1
    assert results(events) == [STARTING] and later(s) == ["Grand Theft Auto V Legacy is running."]


def test_phone_starts_gta_on_the_pc(tmp_path, policy, audit):
    """PHONE: 'Nexus, start GTA on my PC.' -> authenticated -> security check -> launch -> verify -> 'is running'."""
    t = build(tmp_path, policy, audit)
    pc, phone = DeviceKey.generate(name="pc", role=DeviceRole.DESKTOP), DeviceKey.generate(name="p", role=DeviceRole.MOBILE)
    registry = DeviceRegistry(None)
    registry.add(DeviceRecord(phone.device_id, phone.identity.public_key, "p", DeviceRole.MOBILE, "t"))
    gateway = MeshGateway(pc, "nx-t", registry, t.kernel, engine=t.engine, resources=t.rm)
    hub = LoopbackHub()
    gateway.attach(hub.connect(pc.device_id, gateway.receive))
    gateway.watch_apps(t.apps)
    client = MeshClient(phone, nexus_id="nx-t", core_id=pc.device_id, core_public_key=pc.identity.public_key)
    client.connect(hub)
    events = client.say("Nexus, start GTA")
    assert events[0]["status"] == "needs_confirmation" and t.plat.launched == []  # the PC's policy decides
    t.bg.defer = True
    t.game_starts()
    done = client.confirm(events[-1]["confirmation_id"], True)
    assert [e["text"] for e in done if e["type"] == "say"][-1] == STARTING  # the phone gets an answer at once
    t.bg.run_pending()                                                        # the PC verifies in the background
    later_on_phone = [m.body["event"]["text"] for m in client.inbox if "in_reply_to" not in m.body]
    assert later_on_phone == ["Grand Theft Auto V Legacy is running."]  # signed by the PC, sent to the phone
