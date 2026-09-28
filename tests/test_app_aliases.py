"""User-defined app names: "Chrome is my browser" -> "open my browser"."""

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.types import Say, SayKind, UserInput
from nexus.ai.memory.sqlite_store import SQLiteMemoryStore
from nexus.ai.router import FastPathRouter
from nexus.core.kernel import ActionKernel, Status, ToolRequest
from nexus.system.apps.records import RawEntry
from nexus.system.apps.service import AppService
from nexus.tools import ToolRegistry, default_tools
from tests.apps_fixtures import AppsPlatform, Background, FakeClock, FakeProcs, NameGuard, make_exe
from tests.conftest import ScriptedPlanner


class ListSource:
    name = "test"

    def __init__(self, entries):
        self.entries_list = entries

    def entries(self):
        return list(self.entries_list)


def build(tmp_path, policy, audit, extra=()):
    plat = AppsPlatform(tmp_path)
    procs, clock = FakeProcs(), FakeClock()
    apps_dir = tmp_path / "home" / "apps"
    exes = {n: make_exe(apps_dir / n / f"{n}.exe") for n in ("chrome", "firefox", "photoshop", "steam", "obs64")}
    entries = [RawEntry("Google Chrome", "start_menu", target=str(exes["chrome"])),
               RawEntry("Mozilla Firefox", "start_menu", target=str(exes["firefox"])),
               RawEntry("Adobe Photoshop 2025", "start_menu", target=str(exes["photoshop"])),
               RawEntry("Steam", "start_menu", target=str(exes["steam"])),
               RawEntry("OBS Studio", "start_menu", target=str(exes["obs64"])),
               RawEntry("Windows PowerShell", "start_menu", target=r"C:\Windows\powershell.exe"),
               *extra]
    memory = SQLiteMemoryStore(tmp_path / "memory.db")
    apps = AppService(plat, sources=[ListSource(entries)], providers=[], processes=procs, clock=clock,
                      sleep=clock.sleep, aliases=memory, background=Background(), guard=NameGuard())
    kernel = ActionKernel(ToolRegistry(default_tools(plat, memory, apps)), policy, audit)
    engine = ConversationEngine(kernel, ScriptedPlanner(), router=FastPathRouter(apps=apps))

    def talk(text, device=None):
        events = []
        engine.handle(UserInput(text, device=device), events.append)
        return events

    def starts(name, exe):
        plat.on_launch = lambda spec: procs.spawn(name, str(exe), after_polls=1)

    return type("E", (), dict(plat=plat, procs=procs, apps=apps, kernel=kernel, talk=talk, memory=memory,
                              exes=exes, starts=starts))


@pytest.fixture
def e(tmp_path, policy, audit):
    return build(tmp_path, policy, audit)


def says(events, kind):
    return [x.text for x in events if isinstance(x, Say) and x.kind is kind]


@pytest.mark.parametrize("define,use,exe", [
    ("Chrome is my browser", "Open my browser", "chrome"),
    ("Call Photoshop my editor", "Launch my editor", "photoshop"),
    ("Remember that Steam is my game launcher", "Open my game launcher", "steam"),
    ("OBS mera recorder hai", "recorder khol do", "obs64"),
])
def test_create_and_use(e, define, use, exe):
    events = e.talk(define)
    assert says(events, SayKind.RESULT)[0].startswith("Got it")
    e.starts(f"{exe}.exe", e.exes[exe])
    events = e.talk(use)
    assert e.plat.launched[-1].target == str(e.exes[exe])
    assert says(events, SayKind.RESULT)[-1].endswith("is open.")  # names the real app, e.g. "Google Chrome is open."


def test_alias_resolution_is_exact_and_deterministic(e):
    e.talk("Chrome is my browser")
    for _ in range(3):
        assert e.apps.resolve("my browser").match.display_name == "Google Chrome"
        assert e.apps.resolve("browser").match.display_name == "Google Chrome"
    assert e.apps.resolve("my browsr").match is None or e.apps.resolve("my browsr").match.display_name != "Google Chrome"


def test_replacing_an_alias_says_what_changed(e):
    e.talk("Chrome is my browser")
    events = e.talk("Firefox is my browser")
    assert "used to open Google Chrome" in says(events, SayKind.RESULT)[0]
    assert e.apps.resolve("my browser").match.display_name == "Mozilla Firefox"


def test_delete_alias(e):
    e.talk("Chrome is my browser")
    events = e.talk("remove my browser alias")
    assert says(events, SayKind.RESULT)[0].startswith("Removed")
    assert e.memory.aliases() == []
    assert e.apps.resolve("my browser").match is None


def test_aliases_survive_a_restart(tmp_path, policy, audit):
    first = build(tmp_path, policy, audit)
    first.talk("Chrome is my browser")
    first.memory.close()
    second = build(tmp_path, policy, audit)  # a fresh NEXUS on the same memory database
    assert second.apps.resolve("my browser").match.display_name == "Google Chrome"


def test_ambiguity_is_shown_not_guessed(tmp_path, policy, audit):
    e = build(tmp_path, policy, audit)
    e.talk("Chrome is my browser")
    e.memory.close()
    # later an app literally called "Browser" is installed: both are offered
    later = build(tmp_path, policy, audit, extra=[RawEntry("Browser", "start_menu",
                                                           target=str(make_exe(tmp_path / "b" / "browser.exe")))])
    res = later.apps.resolve("browser")
    assert res.ambiguous and {c.display_name for c in res.candidates} == {"Google Chrome", "Browser"}
    events = later.talk("open my browser")
    assert "Google Chrome" in says(events, SayKind.CLARIFY)[0] and later.plat.launched == []


@pytest.mark.parametrize("define,word", [
    ("PowerShell is my browser", "I can't give Windows PowerShell a name"),  # shells can't be aliased
    ("Chrome is my cmd", "keep its own meaning"),                            # reserved words
    ("Chrome is my powershell", "keep its own meaning"),
    ("Chrome is my steam", "already the name of Steam"),                     # can't hijack another app's name
    ("Chrome is my downloads", "keep its own meaning"),
])
def test_malicious_or_conflicting_aliases_are_refused(e, define, word):
    events = e.talk(define)
    refusals = says(events, SayKind.REFUSAL)
    assert refusals and word in refusals[0]
    assert e.memory.aliases() == []


@pytest.mark.parametrize("app,alias", [(r"C:\evil\payload.exe", "browser"), ("rm -rf /", "cleaner"),
                                       ("chrome", "br;owser && del"), ("chrome", "a/b"), ("chrome", "123"),
                                       ("chrome", "one two three four five"), ("spotify", "music player")])
def test_invalid_aliases_via_the_tool(e, app, alias):
    out = e.kernel.submit(ToolRequest("set_app_alias", {"app": app, "alias": alias}))
    assert out.status in (Status.BLOCKED, Status.FAILED)
    assert e.memory.aliases() == []


def test_ambiguous_target_asks_which(tmp_path, policy, audit):
    e = build(tmp_path, policy, audit, extra=[RawEntry("Chrome Beta", "start_menu",
                                                       target=str(make_exe(tmp_path / "cb" / "chrome.exe")))])
    out = e.kernel.submit(ToolRequest("set_app_alias", {"app": "chrome", "alias": "browser"}))
    assert out.status is Status.NEEDS_CLARIFICATION and "Which one should be your browser" in out.message
    assert out.result.data["option_param"] == "app"


def test_aliases_are_device_local_and_visible(e):
    out = e.kernel.submit(ToolRequest("set_app_alias", {"app": "chrome", "alias": "browser"}, device="nxd-phone"))
    e.kernel.resolve(out.confirmation_id, True, device="nxd-phone")  # from another device: confirmed first
    row = e.memory._db.execute("SELECT scope, origin_device FROM app_aliases").fetchone()
    assert tuple(row) == ("device", "nxd-phone")
    listing = e.kernel.submit(ToolRequest("list_memories")).message
    assert "you call Google Chrome your browser" in listing
    assert e.memory.changes_since(None) == ([], None)  # nothing offered for sync


def test_alias_that_is_not_an_app_stays_a_memory(e):
    events = e.talk("Remember that Priya is my sister")  # Priya isn't an installed app
    assert e.memory.aliases() == [] and e.memory.all()[0].text == "Priya is my sister"
    assert says(events, SayKind.RESULT)[0].startswith("Got it, I'll remember")
