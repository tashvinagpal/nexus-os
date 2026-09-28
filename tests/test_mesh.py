"""A paired phone talks to the PC over the (in-process) mesh. The PC's security
engine stays authoritative: pairing grants nothing the policy doesn't allow."""

from dataclasses import replace

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.mesh.gateway import MeshGateway
from nexus.mesh.messages import Envelope, MessageError, MessageKind, seal, signature_valid
from nexus.mesh.registry import DeviceRecord, DeviceRegistry
from nexus.mesh.transport import LoopbackHub
from nexus.security.identity import DeviceKey, NexusIdentity, b64d, device_id_for
from nexus.system.device import DeviceRole
from tests.conftest import ScriptedPlanner, plan

NEXUS = "nx-test"


def pair(registry: DeviceRegistry, key: DeviceKey) -> None:
    ident = key.identity
    registry.add(DeviceRecord(ident.device_id, ident.public_key, ident.name, ident.role, "2026-09-28T00:00:00Z"))


class Mesh:
    def __init__(self, make_kernel, resources, planner=None, tmp_path=None):
        self.pc = DeviceKey.generate(name="pc", role=DeviceRole.DESKTOP)
        self.phone = DeviceKey.generate(name="phone", role=DeviceRole.MOBILE)
        self.laptop = DeviceKey.generate(name="laptop", role=DeviceRole.LAPTOP)
        self.registry = DeviceRegistry(tmp_path / "devices.json" if tmp_path else None)
        pair(self.registry, self.phone)
        pair(self.registry, self.laptop)
        self.kernel = make_kernel()
        engine = ConversationEngine(self.kernel, planner, resources=resources) if planner else None
        self.gateway = MeshGateway(self.pc, NEXUS, self.registry, self.kernel, engine=engine, resources=resources)

    def send(self, key, kind, body, **overrides):
        env = seal(key, nexus_id=NEXUS, recipient=self.pc.device_id, kind=kind, body=body)
        return self.gateway.handle(replace(env, **overrides) if overrides else env)

    def events(self, replies):
        assert all(signature_valid(r, b64d(self.pc.identity.public_key)) for r in replies)
        return [r.body["event"] for r in replies if r.kind is MessageKind.TURN_EVENT]


@pytest.fixture
def mesh(make_kernel, resources, tmp_path):
    def build(*plans):
        return Mesh(make_kernel, resources, ScriptedPlanner(*plans) if plans else None, tmp_path)
    return build


# --- identity --------------------------------------------------------------------------

def test_device_id_is_derived_from_key():
    key = DeviceKey.generate(name="x", role=DeviceRole.DESKTOP)
    assert key.device_id == device_id_for(b64d(key.identity.public_key))
    assert key.device_id.startswith("nxd-")


def test_identity_persists(tmp_path):
    a = DeviceKey.load_or_create(tmp_path / "device.json", name="pc", role=DeviceRole.DESKTOP)
    b = DeviceKey.load_or_create(tmp_path / "device.json", name="other", role=DeviceRole.LAPTOP)
    assert a.device_id == b.device_id and b.identity.name == "pc"
    n1 = NexusIdentity.load_or_create(tmp_path / "nexus.json")
    assert NexusIdentity.load_or_create(tmp_path / "nexus.json") == n1


def test_registry_rejects_id_that_does_not_match_key(tmp_path):
    key = DeviceKey.generate(name="x", role=DeviceRole.MOBILE)
    reg = DeviceRegistry(tmp_path / "d.json")
    with pytest.raises(ValueError):
        reg.add(DeviceRecord("nxd-someoneelse", key.identity.public_key, "x", DeviceRole.MOBILE, "t"))


def test_registry_persists_and_revokes(tmp_path):
    key = DeviceKey.generate(name="phone", role=DeviceRole.MOBILE)
    reg = DeviceRegistry(tmp_path / "d.json")
    pair(reg, key)
    reg.revoke(key.device_id)
    again = DeviceRegistry(tmp_path / "d.json")
    assert again.get(key.device_id).revoked and again.get(key.device_id).role is DeviceRole.MOBILE


# --- envelopes -------------------------------------------------------------------------

def test_signature_covers_the_body():
    key = DeviceKey.generate(name="x", role=DeviceRole.MOBILE)
    env = seal(key, nexus_id=NEXUS, recipient="nxd-pc", kind=MessageKind.USER_INPUT, body={"text": "hi"})
    pub = b64d(key.identity.public_key)
    assert signature_valid(env, pub)
    assert not signature_valid(replace(env, body={"text": "format C:"}), pub)
    assert Envelope.from_json(env.to_json()) == env


@pytest.mark.parametrize("body", [{}, {"text": 5}, {"text": "hi", "extra": 1}])
def test_malformed_bodies_rejected(body):
    key = DeviceKey.generate(name="x", role=DeviceRole.MOBILE)
    with pytest.raises(MessageError):
        seal(key, nexus_id=NEXUS, recipient="nxd-pc", kind=MessageKind.USER_INPUT, body=body)


def test_oversized_or_garbage_envelopes_rejected():
    with pytest.raises(MessageError):
        Envelope.from_json("x" * 70000)
    with pytest.raises(MessageError):
        Envelope.from_json('{"v": 1}')


# --- gateway: authentication -----------------------------------------------------------

def test_phone_conversation_reaches_the_pc(mesh):
    m = mesh(plan("Haan bhai, sab theek hai!", language="hinglish"))
    events = m.events(m.send(m.phone, MessageKind.USER_INPUT, {"text": "PC ka haal?"}))
    assert events == [{"type": "say", "kind": "reply", "text": "Haan bhai, sab theek hai!",
                       "language": "hinglish"}]


def test_unpaired_device_gets_nothing(mesh, audit):
    m = mesh(plan("should not be reached"))
    stranger = DeviceKey.generate(name="evil", role=DeviceRole.MOBILE)
    assert m.send(stranger, MessageKind.USER_INPUT, {"text": "hi"}) == []
    assert audit.tail(1)[0]["event"] == "mesh_rejected"
    assert "not paired" in audit.tail(1)[0]["reason"]


def test_forged_sender_id_fails_signature(mesh, audit):
    m = mesh(plan("nope"))
    attacker = DeviceKey.generate(name="evil", role=DeviceRole.MOBILE)
    env = seal(attacker, nexus_id=NEXUS, recipient=m.pc.device_id, kind=MessageKind.USER_INPUT,
               body={"text": "hi"})
    assert m.gateway.handle(replace(env, sender=m.phone.device_id)) == []
    assert audit.tail(1)[0]["reason"] == "invalid signature"


def test_revoked_device_rejected(mesh):
    m = mesh(plan("nope"))
    m.registry.revoke(m.phone.device_id)
    assert m.send(m.phone, MessageKind.USER_INPUT, {"text": "hi"}) == []


def test_replay_rejected(mesh):
    m = mesh(plan("first"), plan("second"))
    env = seal(m.phone, nexus_id=NEXUS, recipient=m.pc.device_id, kind=MessageKind.USER_INPUT, body={"text": "hi"})
    assert m.gateway.handle(env)
    assert m.gateway.handle(env) == []


def test_stale_message_rejected(mesh):
    m = mesh(plan("nope"))
    env = seal(m.phone, nexus_id=NEXUS, recipient=m.pc.device_id, kind=MessageKind.USER_INPUT,
               body={"text": "hi"}, now_ms=0)
    assert m.gateway.handle(env) == []


def test_wrong_nexus_or_recipient_rejected(mesh):
    m = mesh(plan("nope"))
    env = seal(m.phone, nexus_id="nx-other", recipient=m.pc.device_id, kind=MessageKind.USER_INPUT,
               body={"text": "hi"})
    assert m.gateway.handle(env) == []
    env = seal(m.phone, nexus_id=NEXUS, recipient=m.laptop.device_id, kind=MessageKind.USER_INPUT,
               body={"text": "hi"})
    assert m.gateway.handle(env) == []


# --- gateway: authorization (the kernel stays in charge) -------------------------------

def test_phone_action_needs_confirmation_on_pc(mesh, sandbox):
    """'Nexus, make a folder on my PC' from the phone: the PC asks first, even inside safe folders."""
    m = mesh()
    target = sandbox / "home" / "from-phone"
    events = m.events(m.send(m.phone, MessageKind.ACTION_REQUEST,
                             {"tool": "create_folder", "args": {"path": str(target)}}))
    assert events[0]["status"] == "needs_confirmation" and not target.exists()
    cid = events[1]["confirmation_id"]

    # Another paired device can't approve the phone's request.
    events = m.events(m.send(m.laptop, MessageKind.CONFIRM, {"confirmation_id": cid, "approved": True}))
    assert events[0]["status"] == "invalid" and not target.exists()

    events = m.events(m.send(m.phone, MessageKind.CONFIRM, {"confirmation_id": cid, "approved": True}))
    assert events[0]["status"] == "done" and target.is_dir()


def test_phone_can_read_status_without_confirmation(mesh):
    m = mesh()
    events = m.events(m.send(m.phone, MessageKind.ACTION_REQUEST,
                             {"tool": "get_system_info", "args": {"section": "memory"}}))
    assert events[0]["status"] == "done"


def test_phone_cannot_touch_protected_paths(mesh, sandbox):
    m = mesh()
    events = m.events(m.send(m.phone, MessageKind.ACTION_REQUEST,
                             {"tool": "create_folder", "args": {"path": str(sandbox / "system" / "x")}}))
    assert events[0]["status"] == "blocked"


def test_phone_voice_request_goes_through_the_same_engine(mesh, sandbox):
    target = sandbox / "home" / "mods"
    m = mesh(plan("Sure.", ("create_folder", {"path": str(target)})))
    events = m.events(m.send(m.phone, MessageKind.USER_INPUT, {"text": "make a mods folder"}))
    assert [e["type"] for e in events] == ["say", "action", "say", "confirmation"]
    assert events[1]["status"] == "needs_confirmation" and not target.exists()
    events = m.events(m.send(m.phone, MessageKind.CONFIRM,
                             {"confirmation_id": events[3]["confirmation_id"], "approved": True}))
    assert events[0]["status"] == "done" and target.is_dir()


def test_status_request(mesh):
    m = mesh()
    [reply] = m.send(m.phone, MessageKind.STATUS_REQUEST, {})
    assert reply.kind is MessageKind.STATUS and reply.body["status"]["state"] == "idle"
    assert reply.body["in_reply_to"]


def test_end_to_end_over_loopback_transport(mesh):
    m = mesh(plan("PC is fine."))
    hub = LoopbackHub()
    inbox = []
    m.gateway.attach(hub.connect(m.pc.device_id, m.gateway.receive))
    phone_link = hub.connect(m.phone.device_id, inbox.append)
    phone_link.send(seal(m.phone, nexus_id=NEXUS, recipient=m.pc.device_id, kind=MessageKind.USER_INPUT,
                         body={"text": "status?"}))
    assert [e.body["event"]["text"] for e in inbox] == ["PC is fine."]
    assert all(e.sender == m.pc.device_id for e in inbox)
