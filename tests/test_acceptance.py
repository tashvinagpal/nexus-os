"""NEXUS acceptance criteria (locked 2026-09-28).

Each test is named after the criterion it proves and runs in the default suite
on simulated hardware. tests/test_acceptance_live.py proves the resource
criteria again on the real RTX 3060 + Ollama:  pytest -m live
"""

import ast
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.responder import claims_success
from nexus.ai.conversation.types import ActionReport, Say, SayKind, UserInput
from nexus.ai.memory import MemoryScope
from nexus.ai.planner import LLMPlanner
from nexus.ai.providers import HybridChat, Locality, ProviderInfo, ProviderOption, SelectionContext, select_provider
from nexus.ai.router import FastPathRouter
from nexus.ai.voice.speaker import Speaker
from nexus.core.config import PROJECT_ROOT, load_ai_settings
from nexus.core.kernel import Status
from nexus.core.resources import Placement, ResourceManager, ResourceState, probe_system
from nexus.mesh.client import MeshClient
from nexus.mesh.gateway import MeshGateway
from nexus.mesh.messages import MessageKind, seal
from nexus.mesh.registry import DeviceRecord, DeviceRegistry
from nexus.mesh.transport import LoopbackHub
from nexus.security.identity import DeviceKey, NexusIdentity
from nexus.security.permissions import DeviceRules, Policy
from nexus.system.device import DeviceCapabilities, DeviceRole, detect_capabilities, supported_models
from nexus.tools.base import Risk, ToolResult
from tests.conftest import FakeBackend, FakeChat, FakeProbe, Recorder, ScriptedPlanner, make_tool, plan

NEXUS_SRC = PROJECT_ROOT / "nexus"
CHAT = '{"language": "hinglish", "actions": [], "say": "Want to play something? GTA is installed."}'


def reply_with(tool, **args):
    import json
    return json.dumps({"language": "en", "actions": [{"tool": tool, "args": args}], "say": ""})


@pytest.fixture
def stack(make_kernel, memory_store, clock):
    """Full NEXUS stack on simulated hardware: 12 GB GPU, 16 GB RAM, big GPU model + CPU fallback."""
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    big = FakeChat("llm:8b@gpu", vram_mb=5100, ram_mb=500, probe=probe, reply=CHAT)
    cpu = FakeChat("llm:3b@cpu", placement=Placement.CPU, ram_mb=2500, probe=probe, reply=CHAT)
    hybrid = HybridChat([ProviderOption(big, big.spec, 2), ProviderOption(cpu, cpu.spec, 1)], rm, SelectionContext)
    kernel = make_kernel(memory=memory_store)
    engine = ConversationEngine(kernel, LLMPlanner(hybrid, list(kernel.registry)), resources=rm,
                                memory=memory_store, router=FastPathRouter())

    def talk(text, device=None):
        events = []
        engine.handle(UserInput(text, device=device), events.append)
        return events

    return SimpleNamespace(probe=probe, rm=rm, big=big, cpu=cpu, hybrid=hybrid, kernel=kernel, engine=engine,
                           memory=memory_store, clock=clock, talk=talk)


def says(events, kind=None):
    return [e.text for e in events if isinstance(e, Say) and (kind is None or e.kind is kind)]


def reports(events):
    return [e for e in events if isinstance(e, ActionReport)]


def assert_honest(events):
    """Any sentence that claims success must come right after a DONE, verified action."""
    last = None
    for e in events:
        if isinstance(e, ActionReport):
            last = e
        elif isinstance(e, Say) and claims_success(e.text):
            assert e.kind is SayKind.RESULT, f"claim outside a result: {e.text!r}"
            assert last is not None and last.status is Status.DONE and last.verified is True, e.text


def python_files():
    return [p for p in NEXUS_SRC.rglob("*.py")]


# === 1. NEXUS can converse without executing a tool =====================================

def test_A01_converses_without_executing_a_tool(stack, audit):
    events = stack.talk("Nexus yaar I'm bored.")
    assert says(events, SayKind.REPLY) == ["Want to play something? GTA is installed."]
    assert reports(events) == []
    assert not any(e["event"] == "request" for e in audit.tail(100))


# === 2. NEXUS can execute an action and verify it =======================================

def test_A02_executes_an_action_and_verifies_it(stack, sandbox):
    target = sandbox / "home" / "acceptance"
    stack.big.reply = reply_with("create_folder", path=str(target))
    events = stack.talk("make a folder called acceptance")
    [report] = reports(events)
    assert report.status is Status.DONE and report.verified is True
    assert target.is_dir()  # checked independently of NEXUS
    assert says(events, SayKind.RESULT) == [f"Created the folder {target}."]


# === 3. NEXUS never claims an unverified action succeeded ===============================

def _scenarios(sandbox):
    ok_unverifiable = make_tool("ping", run=Recorder(ToolResult(True, "Asked the system to do it.", verified=None)))
    return {
        "tool fails": (plan("GTA V is ready!", ("t", {})), make_tool("t", run=Recorder(ToolResult(False, "Crashed.")))),
        "tool crashes": (plan("Done!", ("t", {})), make_tool("t", run=Recorder(exc=RuntimeError("boom")))),
        "verification fails": (plan("", ("t", {})),
                               make_tool("t", run=Recorder(ToolResult(True, "GTA V is ready.", verified=False)))),
        "unverifiable": (plan("I've opened it!", ("ping", {})), ok_unverifiable),
        "blocked": (plan("Formatting C drive...", ("create_folder", {"path": str(sandbox / "system" / "x")})), None),
        "confirmation declined": (plan("Deleted!", ("t", {})), make_tool("t", risk=Risk.CONFIRM, run=Recorder())),
        "model lies in chat": (plan("I've launched GTA V for you, it is running."), None),
        "unknown tool": (plan("Windows Defender disabled.", ("disable_defender", {})), None),
    }


@pytest.mark.parametrize("name", ["tool fails", "tool crashes", "verification fails", "unverifiable", "blocked",
                                  "confirmation declined", "model lies in chat", "unknown tool"])
def test_A03_never_claims_unverified_success(make_kernel, sandbox, name):
    the_plan, tool = _scenarios(sandbox)[name]
    engine = ConversationEngine(make_kernel(*([tool] if tool else [])), ScriptedPlanner(the_plan))
    events = []
    engine.handle(UserInput("do it"), events.append)
    if name == "confirmation declined":
        engine.handle(UserInput("nahi"), events.append)
    assert_honest(events)
    if name == "model lies in chat":
        assert "haven't done anything" in says(events)[0]


def test_A03_verified_success_is_reported(make_kernel):
    tool = make_tool("launch", run=Recorder(ToolResult(True, "GTA V is ready.", verified=True)))
    engine = ConversationEngine(make_kernel(tool), ScriptedPlanner(plan("", ("launch", {}))))
    events = []
    engine.handle(UserInput("GTA chala de"), events.append)
    assert_honest(events)
    assert says(events, SayKind.RESULT) == ["GTA V is ready."]


# === 4. Large models unload when no longer needed ========================================

def test_A04_large_model_unloads_after_idle(stack):
    stack.talk("yaar I'm bored")
    assert stack.big.loaded
    vram_loaded = stack.rm.snapshot(fresh=True).vram_used_mb
    stack.clock.now += 301
    assert stack.rm.release_idle() == ["llm:8b@gpu"]
    assert not stack.big.loaded
    assert vram_loaded - stack.rm.snapshot(fresh=True).vram_used_mb == 5100


def test_A04_housekeeping_unloads_without_being_asked(stack):
    stack.talk("yaar I'm bored")
    stack.clock.now += 301
    stack.rm.start_housekeeping(interval_s=0.01)
    deadline = time.monotonic() + 2
    while stack.big.loaded and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not stack.big.loaded


def test_A04_kept_loaded_only_by_explicit_choice(clock):
    ai = load_ai_settings(PROJECT_ROOT / "config" / "nexus.toml")
    assert ai.keep_llm_loaded is False and ai.llm_idle_unload_seconds <= 600  # shipped default
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    kept = FakeBackend("kept", idle_unload_s=float("inf"), probe=probe)
    rm.register(kept)
    with rm.use("kept"):
        pass
    clock.now += 10 ** 9
    assert rm.release_idle() == [] and kept.loaded
    assert rm.status()["models"][0]["kept_loaded"] is True
    rm.enter_game_mode()
    assert not kept.loaded  # the game still wins


# === 5. Game mode frees significant GPU/VRAM ============================================

def test_A05_game_mode_frees_vram_and_falls_back_to_cpu(stack):
    stack.talk("yaar I'm bored")
    before = stack.rm.snapshot(fresh=True).vram_used_mb
    stack.rm.enter_game_mode()
    after = stack.rm.snapshot(fresh=True).vram_used_mb
    assert before - after >= 5000
    stack.talk("tell me a joke")
    assert stack.hybrid.last_choice == "llm:3b@cpu"
    assert not stack.big.loaded and stack.rm.snapshot(fresh=True).vram_used_mb == after


# === 6. Simple requests don't load the 8B model ==========================================

@pytest.mark.parametrize("text", ["GPU ka temperature kya hai?", "what time is it?", "how much RAM am I using",
                                  "kitni storage bachi hai?", "Remember that Chrome is my preferred browser",
                                  "what do you remember about me?"])
def test_A06_simple_requests_load_no_model(stack, text):
    events = stack.talk(text)
    assert [r.status for r in reports(events)] == [Status.DONE]
    assert stack.big.calls == [] and stack.cpu.calls == []
    assert not stack.big.loaded and not stack.cpu.loaded


def test_A06_conversation_still_uses_the_model(stack):
    stack.talk("Nexus yaar I'm bored.")
    assert len(stack.big.calls) == 1


# === 7. A paired phone cannot bypass PC security =========================================

@pytest.fixture
def mesh(make_kernel, memory_store, fake_platform, audit, clock):
    pc = DeviceKey.generate(name="pc", role=DeviceRole.DESKTOP)
    phone = DeviceKey.generate(name="phone", role=DeviceRole.MOBILE)
    laptop = DeviceKey.generate(name="laptop", role=DeviceRole.LAPTOP)
    nexus = NexusIdentity("nx-acceptance")
    registry = DeviceRegistry(None)
    for key in (phone, laptop):
        registry.add(DeviceRecord(key.device_id, key.identity.public_key, key.identity.name, key.identity.role, "t"))
    policy = Policy(protected=fake_platform.protected_paths(), safe_roots=fake_platform.safe_roots(),
                    remote_devices={laptop.device_id: DeviceRules(blocked=frozenset({"create_folder"}))})
    kernel = make_kernel(memory=memory_store, policy_override=policy)
    probe = FakeProbe()
    caps = DeviceCapabilities(role=DeviceRole.DESKTOP, cpu_threads=12, ram_mb=16000, gpu_name="RTX 3060",
                              vram_mb=12288, has_battery=False, supported_models=("llm:llama3.1:8b@gpu",))
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0, capabilities=caps)
    engine = ConversationEngine(kernel, ScriptedPlanner(plan("PC is fine, bhai.")), resources=rm,
                                memory=memory_store, router=FastPathRouter())
    gateway = MeshGateway(pc, nexus.nexus_id, registry, kernel, engine=engine, resources=rm)
    hub = LoopbackHub()
    gateway.attach(hub.connect(pc.device_id, gateway.receive))

    def client(key):
        c = MeshClient(key, nexus_id=nexus.nexus_id, core_id=pc.device_id, core_public_key=pc.identity.public_key)
        c.connect(hub)
        return c

    return SimpleNamespace(pc=pc, phone=phone, laptop=laptop, nexus=nexus, hub=hub, client=client,
                           memory=memory_store, audit=audit)


def test_A07_phone_cannot_bypass_pc_security(mesh, sandbox):
    phone, laptop = mesh.client(mesh.phone), mesh.client(mesh.laptop)
    # protected location: refused outright
    assert phone.request("create_folder", {"path": str(sandbox / "system" / "x")})[0]["status"] == "blocked"
    # even a safe folder needs the PC's confirmation rules
    target = sandbox / "home" / "from-phone"
    events = phone.request("create_folder", {"path": str(target)})
    assert events[0]["status"] == "needs_confirmation" and not target.exists()
    # another paired device can't approve the phone's request
    assert laptop.confirm(events[1]["confirmation_id"], True)[0]["status"] == "invalid"
    assert not target.exists()
    # per-device permissions: the laptop is not allowed this tool at all
    assert laptop.request("create_folder", {"path": str(target)})[0]["status"] == "blocked"
    # private memories never reach the phone
    mesh.memory.remember("My bank is ABC", scope=MemoryScope.PRIVATE)
    events = phone.request("list_memories")
    done = phone.confirm(events[1]["confirmation_id"], True)
    assert "ABC" not in done[0]["message"]


def test_A07_strangers_and_forgeries_get_nothing(mesh):
    stranger = DeviceKey.generate(name="evil", role=DeviceRole.MOBILE)
    evil = MeshClient(stranger, nexus_id=mesh.nexus.nexus_id, core_id=mesh.pc.device_id,
                      core_public_key=mesh.pc.identity.public_key)
    evil.connect(mesh.hub)
    assert evil.say("delete everything") == []
    assert mesh.audit.tail(1)[0]["event"] == "mesh_rejected"


# === 8. Device identity is separate from user identity ===================================

def test_A08_device_identity_is_separate_from_user_identity(tmp_path):
    nexus = NexusIdentity.load_or_create(tmp_path / "nexus.json")
    pc = DeviceKey.load_or_create(tmp_path / "pc.json", name="pc", role=DeviceRole.DESKTOP)
    phone = DeviceKey.load_or_create(tmp_path / "phone.json", name="phone", role=DeviceRole.MOBILE)
    assert nexus.nexus_id.startswith("nx-") and pc.device_id.startswith("nxd-")
    assert pc.device_id != phone.device_id != nexus.nexus_id
    assert pc.identity.public_key != phone.identity.public_key
    assert "private_key" not in (tmp_path / "nexus.json").read_text()  # the user identity holds no device key
    assert nexus.nexus_id not in (tmp_path / "pc.json").read_text()      # the device key is not tied to the id
    # permissions are per device, not per user
    policy = Policy(protected=[], safe_roots=[tmp_path],
                    remote_devices={phone.device_id: DeviceRules(allow_without_confirmation=frozenset({"x"}))})
    tool = make_tool("x")
    assert policy.evaluate(tool, {}, device=phone.device_id).verdict.value == "allow"
    assert policy.evaluate(tool, {}, device=pc.device_id).verdict.value == "confirm"


# === 9. Memory can remain local ==========================================================

def test_A09_memory_stays_local(tmp_path, memory_store):
    rec = memory_store.remember("Chrome is my preferred browser")
    assert rec.scope is MemoryScope.DEVICE  # local unless the user chooses to share
    memory_store.remember("bank is ABC", scope=MemoryScope.PRIVATE)
    memory_store.remember("wfh today", scope=MemoryScope.TEMPORARY)
    assert memory_store.changes_since(None) == ([], None)  # nothing is ever offered for sync
    network = {"socket", "urllib", "http", "requests", "ssl", "ftplib", "smtplib", "websocket", "httpx"}
    for path in (NEXUS_SRC / "ai" / "memory").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(n.split(".")[0] in network for n in names), f"{path.name} imports {names}"


def test_A09_memory_database_lives_in_nexus_data_folder(tmp_path):
    from nexus.core.config import load_settings
    from nexus.core.runtime import create_runtime
    from tests.conftest import FakePlatform

    (tmp_path / "home").mkdir()
    rt = create_runtime(load_settings({"NEXUS_DATA_DIR": str(tmp_path / "var")}), FakePlatform(tmp_path))
    rt.memory.remember("x is y")
    assert (tmp_path / "var" / "memory" / "memory.db").exists()
    rt.memory.close()


# === 10. The architecture can eventually run on Linux ====================================

WINDOWS_ONLY = re.compile(r"\bwinreg\b|\bwindll\b|os\.startfile|\b_winapi\b|\bmsvcrt\b|\bwin32api\b|\bpywintypes\b")


def test_A10_os_specific_code_stays_behind_the_platform_layer():
    offenders = [str(p.relative_to(PROJECT_ROOT)) for p in python_files()
                 if WINDOWS_ONLY.search(p.read_text(encoding="utf-8")) and p.name != "windows.py"]
    assert offenders == []


def test_A10_platform_is_chosen_at_runtime(monkeypatch):
    from nexus.system import platform as plat

    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(NotImplementedError, match="v0.9"):
        plat.current()


def test_A10_core_runs_on_a_non_windows_platform(make_kernel, fake_platform, sandbox):
    """The whole suite runs the kernel, policy, tools and engine on FakePlatform, not Windows."""
    assert fake_platform.name == "fake"
    engine = ConversationEngine(make_kernel(), ScriptedPlanner(plan("", ("create_folder",
                                                                        {"path": str(sandbox / "home" / "l")}))))
    events = []
    engine.handle(UserInput("go"), events.append)
    assert reports(events)[0].status is Status.DONE


# === 11. The architecture can support a mobile client ====================================

def test_A11_mobile_client(mesh):
    phone_caps = DeviceCapabilities(role=DeviceRole.MOBILE, cpu_threads=8, ram_mb=6000, gpu_name=None,
                                    vram_mb=None, has_battery=True)
    assert not phone_caps.can_execute_desktop_tools
    pc_models = [FakeChat("llm:llama3.1:8b@gpu", vram_mb=5100, ram_mb=500).spec]
    assert supported_models(pc_models, ram_mb=phone_caps.ram_mb, vram_mb=phone_caps.vram_mb) == ()
    phone = mesh.client(mesh.phone)
    assert [e["text"] for e in phone.say("kaise ho bhai?")] == ["PC is fine, bhai."]  # conversation
    reading = phone.say("PC ka kya haal hai?")  # fast path: a real reading, allowed remotely without asking
    assert reading[0]["tool"] == "get_system_info" and reading[0]["status"] == "done"
    status = phone.status()
    assert status["device"]["supported_models"] == ["llm:llama3.1:8b@gpu"]  # "the PC can run it, I can't"
    assert status["state"] == "idle"
    # the phone only accepts replies signed by its paired PC
    impostor = DeviceKey.generate(name="fake-pc", role=DeviceRole.DESKTOP)
    mesh.hub.deliver(seal(impostor, nexus_id=mesh.nexus.nexus_id, recipient=mesh.phone.device_id,
                          kind=MessageKind.ERROR, body={"error": "trust me"}))
    assert phone.dropped == 1


# === 12. The system remains usable with keyboard and mouse ===============================

INPUT_TAKEOVER = re.compile(r"SetWindowsHookEx|BlockInput|pynput|pyautogui|keyboard\.hook|mouse\.hook|"
                            r"CurrentVersion\\\\Run|schtasks|CreateService|SetValueEx")


def test_A12_nexus_never_takes_over_input_or_autostarts():
    offenders = [str(p.relative_to(PROJECT_ROOT)) for p in python_files()
                 if INPUT_TAKEOVER.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def run_nexus(tmp_path, lines, *extra):
    env = {**os.environ, "NEXUS_DATA_DIR": str(tmp_path / "var")}
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-m", "nexus", *extra], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, timeout=60, env=env, cwd=str(PROJECT_ROOT))
    return proc, time.monotonic() - started


def test_A12_works_without_the_ai_layer_and_exits_cleanly(tmp_path):
    proc, seconds = run_nexus(tmp_path, ["/tools", "/run get_datetime", "/quit"], "--no-ai")
    assert proc.returncode == 0, proc.stderr
    assert "get_system_info" in proc.stdout and "It's " in proc.stdout
    assert seconds < 20


# === Resource requirements ================================================================

def test_R1_resource_manager_knows_the_machine():
    snap = probe_system()
    caps = detect_capabilities()
    rm = ResourceManager(capabilities=caps)
    st = rm.status()
    for key in ("ram_total_mb", "ram_available_mb", "vram_total_mb", "vram_used_mb", "cpu_percent", "gpu_percent",
                "on_battery", "battery_percent", "network_up", "nexus_rss_mb"):
        assert key in st["resources"]
    for key in ("role", "cpu_name", "ram_mb", "gpu_name", "vram_mb", "has_battery", "network_up",
                "supported_models"):
        assert key in st["device"]
    assert "models" in st and st["state"] == "idle"
    assert snap.ram_available_mb <= snap.ram_total_mb


def test_R2_a_model_loads_only_if_it_fits_the_budget(clock):
    probe = FakeProbe(vram_used=8000)  # e.g. a game is using 8 GB of the 12 GB
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    big = FakeBackend("8b", vram_mb=5100, probe=probe)
    rm.register(big)
    admission = rm.admit(big.spec)
    assert not admission.ok and "VRAM" in admission.reason and not big.loaded


def test_R3_idle_nexus_is_light(tmp_path):
    proc, _ = run_nexus(tmp_path, ["/status", "/quit"])
    assert proc.returncode == 0, proc.stderr
    rss = int(re.search(r"NEXUS itself (\d+) MB", proc.stdout).group(1))
    assert rss < 150, f"idle NEXUS uses {rss} MB"
    model_lines = [line for line in proc.stdout.splitlines() if "llm:" in line]
    assert model_lines and all("not loaded" in line or "loaded" not in line.split()[1] for line in model_lines)


class FakeTTS(FakeBackend):
    def __init__(self, name, rm=None, **kw):
        super().__init__(name, **kw)
        self.info = ProviderInfo(name=name, locality=Locality.LOCAL)
        self.rm = rm
        self.spoken = []

    def speak(self, text, *, language=None):
        self.spoken.append((text, self.loaded, self.rm.state if self.rm else None))

    def stop(self):
        pass


def test_R4_voice_output_loads_only_while_speaking(clock):
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    tts = FakeTTS("tts:voice", rm=rm, placement=Placement.GPU, vram_mb=500, ram_mb=300, idle_unload_s=60, probe=probe)
    speaker = Speaker(tts, rm)
    assert not tts.loaded
    assert speaker.say("GTA V is ready.")
    assert tts.spoken == [("GTA V is ready.", True, ResourceState.SPEAKING)]
    clock.now += 61
    rm.release_idle()
    assert not tts.loaded
    rm.enter_game_mode()  # a GPU voice must not compete with the game: text only
    assert speaker.say("hello") is False and not tts.loaded


def test_R5_laptop_adapts_to_battery_and_hardware(clock):
    probe = FakeProbe(on_battery=True)
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    big = FakeChat("8b@gpu", vram_mb=5100, ram_mb=500, probe=probe)
    small = FakeChat("3b@gpu", vram_mb=2500, ram_mb=800, probe=probe)
    cpu = FakeChat("3b@cpu", placement=Placement.CPU, ram_mb=2500, probe=probe)
    options = [ProviderOption(big, big.spec, 3), ProviderOption(small, small.spec, 2), ProviderOption(cpu, cpu.spec, 1)]
    on_battery = SelectionContext(power_saving=bool(rm.snapshot().on_battery))
    assert select_provider(options, rm, on_battery)[0].provider is small
    # a weaker laptop: 8 GB RAM, no GPU -> only the CPU model is supported
    weak = supported_models([o.spec for o in options], ram_mb=8000, vram_mb=None)
    assert weak == ("3b@cpu",)
    assert select_provider(options, rm, SelectionContext(supported=frozenset(weak)))[0].provider is cpu


def test_R6_this_pc_reports_its_capabilities():
    ai = load_ai_settings(PROJECT_ROOT / "config" / "nexus.toml")
    specs = [FakeChat(f"llm:{o.model}@{o.placement}", placement=Placement(o.placement), ram_mb=o.ram_mb,
                      vram_mb=o.vram_mb).spec for o in ai.llm]
    caps = detect_capabilities(model_specs=specs)
    d = caps.to_dict()
    for key in ("cpu_name", "cpu_threads", "ram_mb", "gpu_name", "vram_mb", "has_battery", "on_battery",
                "network_up", "role", "supported_models"):
        assert key in d
    if (caps.vram_mb or 0) >= 12000:  # this dev PC: RTX 3060 12 GB runs every configured model
        assert set(caps.supported_models) == {s.name for s in specs}


def test_R7_local_first_nothing_requires_the_cloud(clock):
    ai = load_ai_settings(PROJECT_ROOT / "config" / "nexus.toml")
    assert ai.allow_cloud is False
    assert re.match(r"https?://(127\.0\.0\.1|localhost)[:/]", ai.ollama_url)
    probe = FakeProbe()
    rm = ResourceManager(probe, clock=clock, snapshot_ttl=0)
    local = FakeChat("local", probe=probe)
    cloud = FakeChat("cloud", locality=Locality.CLOUD)
    lan = FakeChat("lan", locality=Locality.LAN)
    options = [ProviderOption(cloud, None, 9), ProviderOption(lan, None, 8), ProviderOption(local, local.spec, 1)]
    offline = SelectionContext(allow_cloud=False, network_available=False)
    assert select_provider(options, rm, offline)[0].provider is local
