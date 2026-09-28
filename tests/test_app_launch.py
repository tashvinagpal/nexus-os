"""Launching through the Action Kernel: verification, safety refusals, already-running, clarification."""

import pytest

from nexus.ai.conversation.engine import ConversationEngine, pick_option
from nexus.ai.conversation.types import ActionReport, Say, SayKind, UserInput
from nexus.ai.conversation.responder import claims_success
from nexus.ai.router import FastPathRouter
from nexus.core.kernel import Status, ToolRequest
from nexus.system.apps.records import RawEntry
from nexus.system.apps.service import AppService, AppSettings
from nexus.system.platform.base import LaunchMethod
from nexus.tools import ToolRegistry, default_tools
from nexus.core.kernel import ActionKernel
from tests.apps_fixtures import AppsPlatform, Background, FakeClock, FakeProcs, NameGuard, make_exe
from tests.conftest import ScriptedPlanner, plan


class ListSource:
    name = "test"

    def __init__(self, entries):
        self._entries = entries

    def entries(self):
        return list(self._entries)


@pytest.fixture
def env(tmp_path, policy, audit):
    plat = AppsPlatform(tmp_path)
    procs, clock = FakeProcs(), FakeClock()
    apps_dir = tmp_path / "home" / "apps"  # a normal install location
    chrome = make_exe(apps_dir / "Chrome" / "chrome.exe")
    steam = make_exe(apps_dir / "Steam" / "steam.exe")
    downloaded = make_exe(plat.folders["downloads"] / "cracked" / "game.exe")
    admin_tool = make_exe(apps_dir / "Tools" / "needsadmin.exe")
    plat.elevated_exes.add(str(admin_tool))
    entries = [
        RawEntry("Google Chrome", "start_menu", target=str(chrome)),
        RawEntry("Steam", "start_menu", target=str(steam)),
        RawEntry("Grand Theft Auto V Legacy", "start_menu", url="steam://rungameid/271590"),
        RawEntry("Grand Theft Auto IV", "start_menu", url="steam://rungameid/12210"),
        RawEntry("Command Prompt", "start_menu", target=r"C:\Windows\system32\cmd.exe"),
        RawEntry("Registry Editor", "start_menu", target=r"C:\Windows\regedit.exe"),
        RawEntry("Cracked Game", "desktop", target=str(downloaded)),
        RawEntry("Admin Tool", "start_menu", target=str(admin_tool)),
        RawEntry("Among Us", "desktop", target=str(make_exe(apps_dir / "BS" / "HD-Player.exe")),
                 arguments='--cmd launchApp --package "com.innersloth"'),
        RawEntry("Calculator", "store", aumid="Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
    ]
    bg, terminated, app_events = Background(), [], []
    apps = AppService(plat, sources=[ListSource(entries)], providers=[], processes=procs,
                      settings=AppSettings(app_launch_timeout_s=10, game_launch_timeout_s=60),
                      clock=clock, sleep=clock.sleep, background=bg, guard=NameGuard(),
                      terminate=lambda pid: (terminated.append(pid), procs.remove_pids([pid])))
    apps.subscribe(app_events.append)
    kernel = ActionKernel(ToolRegistry(default_tools(plat, None, apps)), policy, audit)

    def run(name, tool="launch_app"):
        return kernel.submit(ToolRequest(tool, {"name": name}))

    return type("Env", (), dict(plat=plat, procs=procs, clock=clock, apps=apps, kernel=kernel, run=run,
                                chrome=chrome, steam=steam, audit=audit, bg=bg, terminated=terminated,
                                events=app_events))


# --- verification ------------------------------------------------------------------------------

def test_launch_success_requires_the_process(env):
    env.plat.on_launch = lambda spec: env.procs.spawn("chrome.exe", str(env.chrome), after_polls=2)
    out = env.run("open Chrome")
    assert out.status is Status.DONE and out.result.verified is True
    assert out.message == "Google Chrome is open."
    assert env.plat.launched[0].method is LaunchMethod.EXECUTABLE
    assert env.plat.launched[0].target == str(env.chrome)  # the discovered exe, not chrome.google.com


def test_process_without_window_is_reported_honestly(env):
    env.plat.with_windows = set()
    env.plat.on_launch = lambda spec: env.procs.spawn("chrome.exe", str(env.chrome), after_polls=1)
    out = env.run("chrome")
    assert out.status is Status.DONE and "window hasn't appeared" in out.message


def test_failed_launch_never_claims_success(env):
    out = env.run("Chrome")  # the process never appears
    assert out.status is Status.FAILED and out.result.verified is False
    assert "couldn't verify" in out.message and not claims_success(out.message)


def test_os_refusal_is_reported_with_the_reason(env):
    env.plat.launch_error = "the file is damaged"
    out = env.run("Chrome")
    assert out.status is Status.FAILED and out.message == "I couldn't open Google Chrome: the file is damaged."


def test_game_failure_is_reported_when_background_verification_times_out(env):
    out = env.run("GTA V")  # no gta5.exe ever appears
    assert out.status is Status.DONE and out.result.verified is None and out.result.data["pending"]
    assert out.message.startswith("Starting Grand Theft Auto V Legacy")
    [event] = env.events
    assert not event.ok and event.verified is False
    assert event.message.startswith("I couldn't verify that Grand Theft Auto V Legacy started")


def test_game_launch_returns_before_verification(env):
    env.bg.defer = True
    env.plat.on_launch = lambda spec: env.procs.spawn("GTA5.exe", None, after_polls=3)
    out = env.run("GTA V")
    assert out.message.startswith("Starting") and env.events == []   # NEXUS is free again
    env.bg.run_pending()                                              # ...verification finishes later
    assert [e.message for e in env.events] == ["Grand Theft Auto V Legacy is running."]


def test_a_second_request_while_starting_does_not_launch_twice(env):
    env.bg.defer = True
    env.run("GTA V")
    again = env.run("GTA V")
    assert "already starting" in again.message and len(env.plat.launched) == 1


def test_unsupported_launcher_is_not_a_failure(tmp_path, policy, audit):
    plat = AppsPlatform(tmp_path)
    apps = AppService(plat, sources=[ListSource([RawEntry("Assassin's Creed", "desktop", url="uplay://launch/720")])],
                      providers=[], processes=FakeProcs())
    kernel = ActionKernel(ToolRegistry(default_tools(plat, None, apps)), policy, audit)
    out = kernel.submit(ToolRequest("launch_app", {"name": "assassins creed"}))
    assert out.status is Status.UNSUPPORTED and plat.launched == []
    assert out.message == "I don't support launching games through Ubisoft Connect yet."


# --- already running --------------------------------------------------------------------------------

def test_already_running_app_is_focused_not_relaunched(env):
    env.procs.add("chrome.exe", str(env.chrome))
    out = env.run("chrome")
    assert out.status is Status.DONE and out.result.data["already_running"]
    assert env.plat.launched == [] and env.plat.focused
    assert "already running" in out.message


def test_already_running_but_window_cannot_be_focused(env):
    env.procs.add("chrome.exe", str(env.chrome))
    env.plat.focus_ok = False
    assert env.run("chrome").message == "Google Chrome is already running."


# --- safety -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["spotify", "definitely not an app", "del /s /q C:\\*", "format c:",
                                  "rm -rf /", "shutdown /s"])
def test_unknown_names_are_never_executed(env, name):
    out = env.run(name)
    assert out.status in (Status.FAILED, Status.BLOCKED) and env.plat.launched == []


@pytest.mark.parametrize("name", [r"C:\Users\me\Downloads\random.exe", "random.exe", "setup.msi", "script.ps1",
                                  "evil.bat", "payload.vbs", r"..\..\x.exe", "\\\\server\\share\\x.exe", "tool.cmd"])
def test_arbitrary_files_and_paths_are_rejected(env, name):
    out = env.run(name)
    assert out.status is Status.BLOCKED and "can't run files" in out.message and env.plat.launched == []


@pytest.mark.parametrize("name,word", [("cmd", "shell"), ("command prompt", "shell"),
                                       ("registry editor", "administration")])
def test_shells_and_admin_tools_are_refused(env, name, word):
    out = env.run(name)
    assert out.status is Status.BLOCKED and word in out.message and env.plat.launched == []
    assert out.message.startswith("I can't launch")


def test_downloaded_programs_are_refused(env):
    out = env.run("Cracked Game")
    assert out.status is Status.BLOCKED and "Downloads" in out.message and env.plat.launched == []


def test_programs_needing_admin_are_refused(env):
    out = env.run("Admin Tool")
    assert out.status is Status.BLOCKED and "administrator" in out.message and env.plat.launched == []


# --- unverifiable launches ---------------------------------------------------------------------------

def test_launcher_mediated_app_is_not_claimed_as_started(env):
    out = env.run("Among Us")
    assert out.status is Status.DONE and out.result.verified is None
    assert "can't confirm" in out.message and not claims_success(out.message)


def test_store_app_verified_by_its_package(env):
    exe = r"C:\Program Files\WindowsApps\Microsoft.WindowsCalculator_11.2405.2.0_x64__8wekyb3d8bbwe\CalculatorApp.exe"
    env.plat.on_launch = lambda spec: env.procs.spawn("CalculatorApp.exe", exe, after_polls=1)
    out = env.run("calculator")
    assert out.status is Status.DONE and out.result.verified is True
    assert env.plat.launched[0].method is LaunchMethod.PACKAGED


def test_store_app_unconfirmed_is_honest(env):
    out = env.run("calculator")
    assert out.status is Status.DONE and out.result.verified is None and "couldn't confirm" in out.message


# --- clarification ---------------------------------------------------------------------------------

def test_duplicate_matches_ask_instead_of_guessing(env):
    out = env.run("GTA")
    assert out.status is Status.NEEDS_CLARIFICATION and env.plat.launched == []
    assert set(out.result.options) == {"Grand Theft Auto V Legacy", "Grand Theft Auto IV"}
    assert out.message.startswith("I found") and out.message.endswith("Which one do you want?")


@pytest.mark.parametrize("answer,picked", [("GTA V", "Grand Theft Auto V Legacy"), ("the 4 one", "Grand Theft Auto IV"),
                                           ("first", "A"), ("dusra wala", "B"), ("IV", "Grand Theft Auto IV")])
def test_pick_option(answer, picked):
    options = ("A", "B") if picked in ("A", "B") else ("Grand Theft Auto V Legacy", "Grand Theft Auto IV")
    assert pick_option(answer, options) == picked


def test_conversation_asks_then_launches_the_choice(env):
    engine = ConversationEngine(env.kernel, ScriptedPlanner(), router=FastPathRouter(apps=env.apps))
    events = []
    engine.handle(UserInput("open GTA"), events.append)
    [question] = [e for e in events if isinstance(e, Say) and e.kind is SayKind.CLARIFY]
    assert "Which one" in question.text and env.plat.launched == []
    env.plat.on_launch = lambda spec: env.procs.spawn("GTA5.exe", None, after_polls=1)
    events = []
    engine.handle(UserInput("GTA V"), events.append)
    assert [s.target for s in env.plat.launched] == ["steam://rungameid/271590"]
    assert [e.text for e in events if isinstance(e, Say) and e.kind is SayKind.RESULT] == [
        "Starting Grand Theft Auto V Legacy - I'll tell you once it's running."]
    assert [e.message for e in env.events] == ["Grand Theft Auto V Legacy is running."]


def test_answering_no_to_a_choice_does_nothing(env):
    engine = ConversationEngine(env.kernel, ScriptedPlanner(), router=FastPathRouter(apps=env.apps))
    engine.handle(UserInput("open GTA"), lambda e: None)
    events = []
    engine.handle(UserInput("nahi rehne de"), events.append)
    assert env.plat.launched == [] and "won't start anything" in events[-1].text


# --- folders and listing ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["Downloads", "my downloads folder", "downloads wala folder"])
def test_open_known_folder(env, target):
    out = env.kernel.submit(ToolRequest("open_folder", {"target": target}))
    assert out.status is Status.DONE and out.result.verified is None
    assert env.plat.opened_folders == [str(env.plat.folders["downloads"])]


def test_open_folder_by_path_and_refusals(env, tmp_path):
    folder = tmp_path / "home" / "projects"
    folder.mkdir()
    assert env.kernel.submit(ToolRequest("open_folder", {"target": str(folder)})).status is Status.DONE
    assert env.kernel.submit(ToolRequest("open_folder", {"target": "\\\\evil\\share"})).status is Status.BLOCKED
    for bad in (str(env.chrome), "not a folder at all"):
        assert env.kernel.submit(ToolRequest("open_folder", {"target": bad})).status is Status.FAILED
    assert env.plat.opened_folders == [str(folder.resolve())]


def test_list_games(env):
    out = env.kernel.submit(ToolRequest("list_apps", {"kind": "games"}))
    assert "Grand Theft Auto V Legacy" in out.message and "Chrome" not in out.message
