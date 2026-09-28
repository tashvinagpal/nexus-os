"""SUCCESS / FAILURE / REFUSAL / CLARIFICATION / UNSUPPORTED are worded differently, and the console stays
responsive while a game starts in the background."""

import threading
import time

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.types import Say, SayKind, UserInput
from nexus.ai.router import FastPathRouter
from nexus.console import Console
from nexus.core.kernel import ActionKernel
from nexus.core.resources import ResourceManager
from nexus.core.runtime import Runtime
from nexus.system.apps.records import RawEntry
from nexus.system.apps.service import AppService, AppSettings
from nexus.tools import ToolRegistry, default_tools
from tests.apps_fixtures import AppsPlatform, Background, FakeClock, FakeProcs, NameGuard, make_exe
from tests.conftest import FakeProbe, ScriptedPlanner


class ListSource:
    name = "test"

    def __init__(self, entries):
        self._entries = entries

    def entries(self):
        return list(self._entries)


def entries(tmp_path):
    return [RawEntry("Google Chrome", "start_menu", target=str(make_exe(tmp_path / "apps" / "chrome.exe"))),
            RawEntry("Windows PowerShell", "start_menu", target=r"C:\Windows\powershell.exe"),
            RawEntry("Outlook (classic)", "start_menu", target=str(make_exe(tmp_path / "o" / "OUTLOOK.EXE"))),
            RawEntry("Outlook", "store", aumid="Microsoft.OutlookForWindows_8wekyb3d8bbwe!Microsoft.OutlookforWindows"),
            RawEntry("Assassin's Creed", "desktop", url="uplay://launch/720"),
            RawEntry("Grand Theft Auto V Legacy", "start_menu", url="steam://rungameid/271590")]


@pytest.fixture
def talk(tmp_path, policy, audit):
    plat = AppsPlatform(tmp_path)
    procs, clock = FakeProcs(), FakeClock()
    apps = AppService(plat, sources=[ListSource(entries(tmp_path))], providers=[], processes=procs, clock=clock,
                      sleep=clock.sleep, background=Background(), guard=NameGuard())
    engine = ConversationEngine(ActionKernel(ToolRegistry(default_tools(plat, None, apps)), policy, audit),
                                ScriptedPlanner(), router=FastPathRouter(apps=apps))

    def say(text, *, error=None):
        plat.launch_error = error
        if "chrome" in text and not error:
            plat.on_launch = lambda spec: procs.spawn("chrome.exe", spec.target, after_polls=1)
        events = []
        engine.handle(UserInput(text), events.append)
        return [e for e in events if isinstance(e, Say)][-1]

    return say


def test_success(talk):
    s = talk("open chrome")
    assert (s.kind, s.text) == (SayKind.RESULT, "Google Chrome is open.")


def test_failure_is_worded_as_a_failure(talk):
    s = talk("open chrome", error="access denied")
    assert (s.kind, s.text) == (SayKind.ERROR, "I couldn't open Google Chrome: access denied.")


def test_refusal_is_not_described_as_a_failure(talk):
    s = talk("open powershell")
    assert s.kind is SayKind.REFUSAL
    assert s.text.startswith("I can't launch Windows PowerShell through NEXUS")
    assert "didn't work" not in s.text and "fail" not in s.text.lower() and "couldn't" not in s.text


def test_clarification(talk):
    s = talk("open outlook")
    assert s.kind is SayKind.CLARIFY
    assert s.text in ("I found Outlook (classic) and Outlook. Which one do you want?",
                      "I found Outlook and Outlook (classic). Which one do you want?")


def test_unsupported(talk):
    s = talk("open assassins creed")
    assert (s.kind, s.text) == (SayKind.UNSUPPORTED, "I don't support launching games through Ubisoft Connect yet.")


def test_policy_refusals_use_refusal_language_too(talk, sandbox):
    from nexus.ai.conversation.responder import ResponseComposer
    from nexus.core.kernel import Outcome, Status

    s = ResponseComposer().result(Outcome("r", Status.BLOCKED, "Refused: C:\\Windows\\x is inside a protected "
                                                                 "location (C:\\Windows)."), "en")
    assert s.kind is SayKind.REFUSAL and s.text.startswith("I can't do that: C:\\Windows\\x is inside")


# --- the console is not blocked while a game starts -----------------------------------------------

def test_console_stays_responsive_while_a_game_starts(tmp_path, policy, audit):
    """Real threads and real time: the game's process appears ~1 s after launch."""
    plat = AppsPlatform(tmp_path)
    procs = FakeProcs()
    probe = FakeProbe()
    rm = ResourceManager(probe, snapshot_ttl=0)
    apps = AppService(plat, sources=[ListSource(entries(tmp_path))], providers=[], processes=procs, game_mode=rm,
                      guard=NameGuard(), settings=AppSettings(game_launch_timeout_s=10))  # real clock, real thread
    plat.on_launch = lambda spec: procs.spawn("GTA5.exe", None, after_polls=2)  # polls every 0.5 s
    kernel = ActionKernel(ToolRegistry(default_tools(plat, None, apps)), policy, audit)
    engine = ConversationEngine(kernel, ScriptedPlanner(), router=FastPathRouter(apps=apps))
    rt = Runtime(settings=None, platform=plat, kernel=kernel, registry=kernel.registry, policy=policy, audit=audit,
                 resources=rm, apps=apps)
    assistant = type("A", (), {"engine": engine, "chat": type("C", (), {"last_choice": None})()})()
    out, stamps, lock = [], {}, threading.Lock()

    def output(line):
        with lock:
            out.append((time.perf_counter(), line))

    def input_fn(prompt):
        if "sent" not in stamps:
            stamps["sent"] = time.perf_counter()
            return "launch GTA V"
        stamps["back"] = time.perf_counter()  # the console is asking for the next command
        deadline = time.perf_counter() + 8
        while time.perf_counter() < deadline and not any("is running" in line for _, line in out):
            time.sleep(0.02)
        return "/quit"

    Console(rt, assistant, input_fn=input_fn, output=output).run()
    lines = [line for _, line in out]
    starting = next(t for t, line in out if "Starting Grand Theft Auto V Legacy" in line)
    running = next(t for t, line in out if line == "nexus> Grand Theft Auto V Legacy is running.")
    print(f"\nturn returned after {stamps['back'] - stamps['sent']:.3f}s; 'is running' printed {running - stamps['back']:.3f}s after that")
    assert stamps["back"] - stamps["sent"] < 0.5, "the console waited for the game"
    assert running > stamps["back"] > starting  # "is running" arrived later, while NEXUS was free
    assert rm.game_mode  # game mode is on while the game runs
    assert lines.index(next(l for l in lines if "Starting" in l)) < lines.index(
        "nexus> Grand Theft Auto V Legacy is running.")
