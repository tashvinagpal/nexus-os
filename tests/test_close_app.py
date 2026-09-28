"""close_app / force_close_app: catalog-only targets, graceful first, verified, never protected, never a raw PID."""

import os
import subprocess

import psutil
import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.types import Say, SayKind, UserInput
from nexus.ai.router import FastPathRouter
from nexus.core.kernel import ActionKernel, Status, ToolRequest
from nexus.core.resources import ResourceManager
from nexus.system.apps.processes import ProcInfo, SystemProcessGuard
from nexus.system.apps.records import RawEntry
from nexus.system.apps.service import AppService, AppSettings
from nexus.system.games.steam import SteamProvider
from nexus.tools import ToolRegistry, default_tools
from tests.apps_fixtures import AppsPlatform, Background, FakeClock, FakeProcs, NameGuard, make_exe
from tests.conftest import FakeBackend, FakeProbe, ScriptedPlanner, plan
from tests.test_game_providers import manifest


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
    apps_dir = tmp_path / "home" / "apps"
    chrome = make_exe(apps_dir / "Chrome" / "chrome.exe")
    steam = make_exe(apps_dir / "Steam" / "steam.exe")
    explorer = make_exe(tmp_path / "win" / "explorer.exe")
    lib = tmp_path / "SteamLib"
    (lib / "steamapps").mkdir(parents=True)
    (lib / "steamapps" / "appmanifest_271590.acf").write_text(
        manifest(271590, "Grand Theft Auto V Legacy", "Grand Theft Auto V"), encoding="utf-8")
    gta_exe = make_exe(lib / "steamapps" / "common" / "Grand Theft Auto V" / "GTA5.exe", 500)
    nexus_like = make_exe(apps_dir / "Nexus" / "nexushost.exe")
    entries = [RawEntry("Google Chrome", "start_menu", target=str(chrome)),
               RawEntry("Steam", "start_menu", target=str(steam)),
               RawEntry("Windows Shell", "start_menu", target=str(explorer)),
               RawEntry("Nexus Host", "start_menu", target=str(nexus_like)),
               RawEntry("Among Us", "desktop", target=str(make_exe(apps_dir / "BS" / "HD-Player.exe")),
                        arguments='--cmd launchApp --package "x"')]
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    big = FakeBackend("llm:8b@gpu", vram_mb=5100, probe=probe)
    rm.register(big)
    terminated = []
    own_pid = procs.add("nexushost.exe", str(nexus_like))  # stands in for NEXUS's own process
    apps = AppService(plat, sources=[ListSource(entries)], providers=[SteamProvider(plat, steam_dir=lib)],
                      processes=procs, game_mode=rm, clock=clock, sleep=clock.sleep, background=Background(),
                      guard=NameGuard(own={own_pid}), settings=AppSettings(app_close_timeout_s=10),
                      terminate=lambda pid: (terminated.append(pid), procs.remove_pids([pid])))
    kernel = ActionKernel(ToolRegistry(default_tools(plat, None, apps)), policy, audit)

    def close(name, tool="close_app"):
        return kernel.submit(ToolRequest(tool, {"name": name}))

    def app_closes_when_asked():
        plat.on_close = lambda pids: procs.remove_pids(pids)

    return type("Env", (), dict(plat=plat, procs=procs, apps=apps, kernel=kernel, close=close, rm=rm, big=big,
                                chrome=chrome, steam=steam, explorer=explorer, gta_exe=gta_exe,
                                terminated=terminated, own_pid=own_pid, app_closes_when_asked=app_closes_when_asked))


def test_successful_close_is_verified(env):
    pid = env.procs.add("chrome.exe", str(env.chrome))
    env.app_closes_when_asked()
    out = env.close("chrome")
    assert out.status is Status.DONE and out.result.verified is True and out.message == "Google Chrome is closed."
    assert env.plat.close_requests == [[pid]] and env.terminated == []  # graceful, not killed


def test_already_closed(env):
    out = env.close("chrome")
    assert out.status is Status.DONE and out.message == "Google Chrome isn't running."
    assert env.plat.close_requests == []


def test_app_that_refuses_to_close_is_reported_honestly(env):
    env.procs.add("chrome.exe", str(env.chrome))  # asked, but it keeps running (e.g. "save changes?")
    out = env.close("chrome")
    assert out.status is Status.FAILED and out.result.verified is False
    assert "still running" in out.message and "closed" not in out.message.replace("close", "")
    assert env.terminated == []  # never escalates on its own


def test_no_window_to_ask(env):
    env.procs.add("chrome.exe", str(env.chrome))
    env.plat.windows_to_close = False
    out = env.close("chrome")
    assert out.status is Status.FAILED and "no window" in out.message and env.terminated == []


@pytest.mark.parametrize("name", ["windows shell", "nexus host"])
def test_protected_processes_are_refused(env, name):
    env.procs.add("explorer.exe", str(env.explorer))
    out = env.close(name)
    assert out.status is Status.BLOCKED and out.message.startswith("I can't close")
    assert env.plat.close_requests == [] and env.terminated == []


def test_protected_processes_are_refused_even_by_force(env):
    env.procs.add("explorer.exe", str(env.explorer))
    out = env.close("windows shell", tool="force_close_app")
    assert out.status is Status.NEEDS_CONFIRMATION
    done = env.kernel.resolve(out.confirmation_id, True)
    assert done.status is Status.BLOCKED and env.terminated == []


@pytest.mark.parametrize("name", ["spotify", "definitely not running", "the thing"])
def test_unknown_apps_are_not_closed(env, name):
    env.procs.add("chrome.exe", str(env.chrome))
    out = env.close(name)
    assert out.status is Status.FAILED and env.plat.close_requests == [] and env.terminated == []


@pytest.mark.parametrize("name", ["1234", "pid 1234", "PID #4", "process 88", r"C:\Windows\explorer.exe",
                                  "chrome.exe", "taskkill /f /im chrome.exe"])
def test_no_arbitrary_pid_or_path_termination(env, name):
    env.procs.add("chrome.exe", str(env.chrome))
    for tool in ("close_app", "force_close_app"):
        out = env.close(name, tool=tool)
        assert out.status in (Status.BLOCKED, Status.NEEDS_CONFIRMATION, Status.FAILED)
        if out.status is Status.NEEDS_CONFIRMATION:
            assert env.kernel.resolve(out.confirmation_id, True).status in (Status.BLOCKED, Status.FAILED)
    assert env.plat.close_requests == [] and env.terminated == []


def test_a_pid_argument_is_rejected_by_the_kernel(env):
    out = env.kernel.submit(ToolRequest("close_app", {"name": "chrome", "pid": 4}))
    assert out.status is Status.INVALID


def test_close_never_runs_a_shell(env, monkeypatch):
    def no_shell(*a, **k):
        raise AssertionError("close must not start any process")

    for target in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, target, no_shell)
    monkeypatch.setattr(os, "system", no_shell)
    env.procs.add("chrome.exe", str(env.chrome))
    env.app_closes_when_asked()
    assert env.close("chrome").status is Status.DONE


def test_force_close_needs_confirmation_and_only_ends_matched_processes(env):
    pid = env.procs.add("chrome.exe", str(env.chrome))
    other = env.procs.add("steam.exe", str(env.steam))
    out = env.close("chrome", tool="force_close_app")
    assert out.status is Status.NEEDS_CONFIRMATION and "unsaved work" in out.message and env.terminated == []
    done = env.kernel.resolve(out.confirmation_id, True)
    assert done.status is Status.DONE and done.message == "Google Chrome is closed."
    assert env.terminated == [pid] and other not in env.terminated


def test_unverifiable_app_close_is_unsupported(env):
    out = env.close("among us")
    assert out.status is Status.UNSUPPORTED and env.plat.close_requests == []


def test_closing_a_game_leaves_game_mode_without_reloading_ai(env):
    env.procs.add("GTA5.exe", str(env.gta_exe))
    env.apps.games.watcher.check()
    assert env.rm.game_mode
    env.app_closes_when_asked()
    out = env.close("GTA")
    assert out.status is Status.DONE and out.message == "Grand Theft Auto V Legacy is closed."
    assert not env.rm.game_mode and not env.big.loaded


# --- conversation / fast path ---------------------------------------------------------------------

@pytest.mark.parametrize("text,tool", [("close Chrome", "close_app"), ("shut down Steam", "close_app"),
                                       ("GTA band kar", "close_app"), ("chrome band kar do", "close_app"),
                                       ("quit steam", "close_app"), ("force close chrome", "force_close_app"),
                                       ("kill chrome", "force_close_app")])
def test_close_requests_use_the_fast_path(env, text, tool):
    step = FastPathRouter(apps=env.apps).route(text).actions[0]
    assert step.tool == tool


@pytest.mark.parametrize("text", ["shut down my pc", "shutdown computer", "close the door", "stop the music",
                                  "close chrome and steam"])
def test_non_app_close_requests_go_to_the_planner(env, text):
    assert FastPathRouter(apps=env.apps).route(text) is None


def test_model_cannot_close_an_app_the_user_never_mentioned(env):
    env.procs.add("steam.exe", str(env.steam))
    engine = ConversationEngine(env.kernel, ScriptedPlanner(plan("", ("close_app", {"name": "Steam"}))),
                                router=FastPathRouter(apps=env.apps))
    events = []
    engine.handle(UserInput("I'm done for today"), events.append)
    assert [e.text for e in events if isinstance(e, Say) and e.kind is SayKind.CLARIFY] == \
        ["Should I close Steam? (yes/no)"]
    assert env.plat.close_requests == []


# --- the real guard -----------------------------------------------------------------------------------

def test_real_guard_protects_nexus_windows_and_other_users():
    guard = SystemProcessGuard()
    me = psutil.Process()
    assert guard.reason(ProcInfo(me.pid, me.name(), me.exe())) == "it's part of NEXUS itself"
    assert "protected Windows process" in guard.reason(ProcInfo(999999, "explorer.exe", None))
    assert guard.reason(ProcInfo(4, "System", None))  # system-owned (pid 4 on Windows)
