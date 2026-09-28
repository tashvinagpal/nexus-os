"""Acceptance criteria on the real hardware: RTX 3060 + Ollama + llama3.1:8b.

Loads and unloads real models on the GPU, so it is opt-in (not while gaming):

    .venv\\Scripts\\python.exe -m pytest -m live
"""

import time
from types import SimpleNamespace

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.types import ActionReport, Say, SayKind, UserInput
from nexus.ai.memory.sqlite_store import SQLiteMemoryStore
from nexus.ai.planner import LLMPlanner
from nexus.ai.providers import ChatMessage, HybridChat, ProviderOption, SelectionContext
from nexus.ai.providers.ollama import OllamaChat
from nexus.ai.router import FastPathRouter
from nexus.core.config import PROJECT_ROOT, load_ai_settings, load_settings
from nexus.core.kernel import ActionKernel, Status
from nexus.core.resources import Placement, ResourceManager
from nexus.security.audit import AuditLog
from nexus.security.permissions import Policy
from nexus.system import info
from nexus.system.platform import current
from nexus.tools import ToolRegistry, default_tools


def _ready() -> bool:
    big = OllamaChat("llama3.1:8b", placement=Placement.GPU, ram_mb=500, vram_mb=5100)
    return bool(info.gpus()) and big.available()


pytestmark = [pytest.mark.live, pytest.mark.skipif(not _ready(), reason="needs an NVIDIA GPU and Ollama with llama3.1:8b")]


def vram_used_mb() -> float:
    time.sleep(0.5)  # let the driver settle
    return info.gpus()[0]["vram_used_mb"]


def loaded_models() -> set[str]:
    return {m["name"] for m in OllamaChat("x", placement=Placement.GPU, ram_mb=0)._get("/api/ps")["models"]}


@pytest.fixture
def big():
    llm = OllamaChat("llama3.1:8b", placement=Placement.GPU, ram_mb=500, vram_mb=5100, idle_unload_s=3)
    yield llm
    llm.unload()


@pytest.fixture
def cpu():
    llm = OllamaChat("llama3.2", placement=Placement.CPU, ram_mb=2500, idle_unload_s=30)
    yield llm
    llm.unload()


def test_L04_nexus_unloads_the_real_8b_when_idle():
    """NEXUS's own idle cleanup (Ollama's timer is set long so it can't be what unloads it)."""
    llm = OllamaChat("llama3.1:8b", placement=Placement.GPU, ram_mb=500, vram_mb=5100, idle_unload_s=120)
    now = [0.0]
    rm = ResourceManager(clock=lambda: now[0])
    rm.register(llm)
    try:
        before = vram_used_mb()
        with rm.use(llm.spec.name):
            llm.chat([ChatMessage("user", "Say hi.")])
        loaded = vram_used_mb()
        assert llm.is_loaded() and loaded - before >= 3500, f"8B used only {loaded - before} MB?"
        now[0] += 121  # two idle minutes pass
        assert rm.release_idle() == [llm.spec.name]
        after = vram_used_mb()
        assert not llm.is_loaded() and loaded - after >= 3500, f"unloading freed only {loaded - after} MB"
        print(f"\n8B loaded: +{loaded - before:.0f} MB VRAM; NEXUS idle unload freed {loaded - after:.0f} MB")
    finally:
        llm.unload()


def test_L04_model_unloads_even_if_nexus_is_not_running(big):
    """Safety net: Ollama's keep-alive mirrors NEXUS's timeout, so a crash can't leave 5 GB pinned."""
    big.chat([ChatMessage("user", "Say hi.")])
    assert big.is_loaded()
    deadline = time.monotonic() + 15
    while big.is_loaded() and time.monotonic() < deadline:  # nobody calls release_idle here
        time.sleep(0.5)
    assert not big.is_loaded()


def test_L05_real_game_mode_frees_vram_and_uses_cpu(big, cpu):
    rm = ResourceManager()
    hybrid = HybridChat([ProviderOption(big, big.spec, 2), ProviderOption(cpu, cpu.spec, 1)], rm, SelectionContext)
    hybrid.chat([ChatMessage("user", "Say hi.")])
    assert hybrid.last_choice == big.spec.name
    loaded = vram_used_mb()
    assert rm.enter_game_mode() == [big.spec.name]
    freed_to = vram_used_mb()
    assert loaded - freed_to >= 4000, f"game mode freed only {loaded - freed_to} MB"
    reply = hybrid.chat([ChatMessage("user", "Say hi.")])
    assert reply.text and hybrid.last_choice == cpu.spec.name
    assert vram_used_mb() - freed_to < 500, "the CPU fallback must not use the GPU"
    print(f"\ngame mode: VRAM {loaded:.0f} -> {freed_to:.0f} MB; AI answered on CPU")


def test_L06_real_stack_answers_simple_requests_without_loading_a_model(tmp_path):
    from nexus.core.assistant import create_assistant
    from nexus.core.runtime import create_runtime

    rt = create_runtime(load_settings({"NEXUS_DATA_DIR": str(tmp_path / "var")}), current())
    a = create_assistant(rt, load_ai_settings(PROJECT_ROOT / "config" / "nexus.toml"))
    models_before, vram_before = loaded_models(), vram_used_mb()
    for text in ["GPU ka temperature kya hai?", "what time is it?", "how much RAM am I using",
                 "Remember that Chrome is my preferred browser", "what do you remember about me?"]:
        events = []
        a.engine.handle(UserInput(text), events.append)
        assert [e.status for e in events if isinstance(e, ActionReport)] == [Status.DONE], text
    assert a.chat.last_choice is None  # no language model was even selected
    assert loaded_models() == models_before
    assert vram_used_mb() - vram_before < 300
    rt.memory.close()


# --- Subsystem 4: real discovery and a real, verified launch -----------------------------------------
# Launches Character Map (a tiny built-in Windows tool) and closes only the process NEXUS started.
# Games and browsers are never launched by the tests.

@pytest.fixture(scope="module")
def real_apps():
    from nexus.system.apps.service import AppService

    service = AppService(current())
    service.catalog()
    return service


def _close_started(pids):
    """Test cleanup: end only the processes this test started, and wait until they are really gone."""
    import psutil

    procs = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            procs.append(proc)
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=5)


def test_L07_real_discovery_finds_this_pcs_apps_and_games(real_apps):
    cat = real_apps.catalog()
    print(f"\n{len(cat.records)} apps/games discovered in {real_apps.discovery_seconds:.2f}s")
    chrome = cat.resolve("open chrome").match
    assert chrome is not None and chrome.target.lower().endswith("chrome.exe")
    steam_gta = [g for g in cat.games() if g.launcher == "steam" and "grand theft auto v" in g.display_name.lower()]
    if steam_gta:  # installed on the dev PC
        assert cat.resolve("GTA V").match == steam_gta[0]
        assert "gta5.exe" in steam_gta[0].process_names
    assert cat.resolve("cmd").match.kind.value == "shell"


def test_L08_real_launch_is_verified_then_detected_as_running(real_apps, tmp_path):
    from nexus.core.kernel import ActionKernel, ToolRequest
    from nexus.tools import ToolRegistry, default_tools

    record = real_apps.resolve("character map").match
    if record is None:
        pytest.skip("Character Map is not in this PC's Start menu")
    if real_apps.is_running(record):
        pytest.skip("Character Map is already open")
    kernel = ActionKernel(ToolRegistry(default_tools(current(), None, real_apps)),
                          Policy(protected=[], safe_roots=[tmp_path]), AuditLog(tmp_path / "audit.jsonl"))
    first = kernel.submit(ToolRequest("launch_app", {"name": "character map"}))
    try:
        assert first.status is Status.DONE and first.result.verified is True, first.message
        assert first.result.data["pids"]
        second = kernel.submit(ToolRequest("launch_app", {"name": "charmap"}))
        assert second.result.data["already_running"] and "already running" in second.message or "switched" in second.message
        print(f"\n{first.message} / {second.message}")
    finally:
        _close_started(first.result.data["pids"] if first.result else [])


def test_L09_real_stack_launches_on_the_fast_path_without_a_model(tmp_path):
    from nexus.core.assistant import create_assistant
    from nexus.core.runtime import create_runtime

    rt = create_runtime(load_settings({"NEXUS_DATA_DIR": str(tmp_path / "var")}), current())
    a = create_assistant(rt, load_ai_settings(PROJECT_ROOT / "config" / "nexus.toml"))
    record = rt.apps.resolve("character map").match
    if record is None or rt.apps.is_running(record):
        pytest.skip("Character Map unavailable or already open")
    models_before = loaded_models()
    events = []
    a.engine.handle(UserInput("open character map"), events.append)
    reports = [e for e in events if isinstance(e, ActionReport)]
    try:
        assert [(r.tool, r.status, r.verified) for r in reports] == [("launch_app", Status.DONE, True)]
        assert a.chat.last_choice is None and loaded_models() == models_before
    finally:
        _close_started(rt.apps.launcher.running(record))
        rt.memory.close()


def test_L10_real_close_app_is_graceful_and_verified(real_apps, tmp_path):
    """close_app on a real window: WM_CLOSE (no kill, no shell), then verified from the process table."""
    import psutil
    from nexus.core.kernel import ToolRequest

    record = real_apps.resolve("character map").match
    if record is None or real_apps.is_running(record):
        pytest.skip("Character Map unavailable or already open")
    kernel = ActionKernel(ToolRegistry(default_tools(current(), None, real_apps)),
                          Policy(protected=[], safe_roots=[tmp_path]), AuditLog(tmp_path / "audit.jsonl"))
    opened = kernel.submit(ToolRequest("launch_app", {"name": "character map"}))
    pids = opened.result.data["pids"] if opened.result else []
    try:
        assert opened.status is Status.DONE and opened.result.verified is True, opened.message
        closed = kernel.submit(ToolRequest("close_app", {"name": "character map"}))
        assert closed.status is Status.DONE and closed.result.verified is True, closed.message
        assert closed.message == "Character Map is closed."
        assert not any(psutil.pid_exists(p) and psutil.Process(p).name().lower() == "charmap.exe" for p in pids)
        again = kernel.submit(ToolRequest("close_app", {"name": "character map"}))
        assert again.message == "Character Map isn't running."
        refused = kernel.submit(ToolRequest("close_app", {"name": "1234"}))
        assert refused.status is Status.BLOCKED
        print(f"\n{opened.message} -> {closed.message} -> {again.message}")
    finally:
        _close_started(pids)


def test_L11_real_game_mode_lifecycle_async_launch_and_close(big):
    """Character Map stands in for a game (no real game is launched): the 8B model is loaded; the async launch
    returns at once and enters game mode (VRAM freed); verification arrives in the background; closing it
    leaves game mode; the 8B model is NOT reloaded."""
    import os
    from pathlib import Path

    from nexus.system.apps.launcher import Launcher
    from nexus.system.apps.names import aliases_for
    from nexus.system.apps.processes import running_pids, snapshot_processes
    from nexus.system.apps.records import AppKind, GameRecord
    from nexus.system.games.service import GameService, GameWatcher
    from nexus.system.platform.base import LaunchMethod

    exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "charmap.exe"
    game = GameRecord(id="test:charmap", display_name="Test Game", aliases=aliases_for("Test Game"),
                      kind=AppKind.GAME, source="test", launch_method=LaunchMethod.EXECUTABLE, target=str(exe),
                      process_names=frozenset({"charmap.exe"}), launcher=None)
    if not exe.is_file() or running_pids(game, snapshot_processes()):
        pytest.skip("charmap.exe unavailable or already running")
    rm = ResourceManager()
    rm.register(big)
    with rm.use(big.spec.name):
        big.chat([ChatMessage("user", "Say hi.")])
    loaded = vram_used_mb()
    games = GameService([], Launcher(current()), game_mode=rm, timeout_s=20)
    games.watcher = GameWatcher(lambda: [game], rm)
    events = []
    games.events.subscribe(events.append)
    try:
        started = time.perf_counter()
        result = games.launch_game(game)
        returned = time.perf_counter() - started
        assert result.pending and returned < 1.0, f"launch_game blocked for {returned:.2f}s"
        assert rm.game_mode and not big.is_loaded()
        in_game = vram_used_mb()
        assert loaded - in_game >= 3500, f"game mode freed only {loaded - in_game} MB"
        deadline = time.monotonic() + 20
        while not events and time.monotonic() < deadline:
            time.sleep(0.1)
        assert [(e.ok, e.verified, e.message) for e in events] == [(True, True, "Test Game is running.")]
        closed = games.launcher.close(game, timeout_s=10)
        assert closed.ok and closed.verified is True, closed.message
        games.watcher.check()
        assert not rm.game_mode
        time.sleep(2)
        assert not big.is_loaded() and loaded_models() == set(), "the 8B model came back on its own"
        print(f"\nlaunch_game returned in {returned * 1000:.0f} ms; VRAM {loaded:.0f} -> {in_game:.0f} MB in game "
              f"mode; verified in background; closed; game mode off; 8B not reloaded")
    finally:
        _close_started(running_pids(game, snapshot_processes()))


def test_L02_real_model_converses_acts_and_remembers(tmp_path, big):
    memory = SQLiteMemoryStore(tmp_path / "memory.db")
    registry = ToolRegistry(default_tools(current(), memory))
    kernel = ActionKernel(registry, Policy(protected=[], safe_roots=[tmp_path]), AuditLog(tmp_path / "audit.jsonl"))
    rm = ResourceManager()
    hybrid = HybridChat([ProviderOption(big, big.spec, 1)], rm, SelectionContext)
    engine = ConversationEngine(kernel, LLMPlanner(hybrid, list(registry)), resources=rm, memory=memory,
                                router=FastPathRouter())

    def talk(text):
        events = []
        engine.handle(UserInput(text), events.append)
        return SimpleNamespace(reports=[e for e in events if isinstance(e, ActionReport)],
                               replies=[e.text for e in events if isinstance(e, Say) and e.kind is SayKind.REPLY])

    chat = talk("kaise ho bhai?")
    assert chat.reports == [] and chat.replies

    target = tmp_path / "nexus-live-check"
    act = talk(f"Create a folder at {target}")
    assert [(r.tool, r.status, r.verified) for r in act.reports] == [("create_folder", Status.DONE, True)]
    assert target.is_dir()

    assert talk("Remember that Chrome is my preferred browser").reports[0].status is Status.DONE
    answer = talk("Which browser do I prefer?")
    assert answer.reports == [] and "chrome" in " ".join(answer.replies).lower()
    memory.close()
