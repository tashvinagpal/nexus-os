from datetime import datetime, timedelta, timezone

import pytest

from nexus.ai.memory import MemoryScope
from nexus.ai.memory.sqlite_store import SQLiteMemoryStore, tokens
from nexus.core.kernel import Status, ToolRequest
from nexus.tools.memory import looks_like_secret


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def tick(self, **kw):
        self.now += timedelta(**kw)


# --- store -----------------------------------------------------------------------------

def test_remember_and_recall(memory_store):
    memory_store.remember("Chrome is my preferred browser")
    memory_store.remember("My favourite game is GTA V")
    [hit] = memory_store.recall("What browser do I prefer?", limit=1)
    assert hit.text == "Chrome is my preferred browser"
    assert memory_store.recall("mera favourite game kaunsa hai?")[0].text == "My favourite game is GTA V"


def test_recall_works_for_devanagari(memory_store):
    memory_store.remember("मेरा पसंदीदा ब्राउज़र क्रोम है")
    assert memory_store.recall("पसंदीदा ब्राउज़र")
    assert "ब्राउज़र" in tokens("मेरा पसंदीदा ब्राउज़र")  # vowel signs stay inside words


def test_same_fact_is_not_duplicated(memory_store):
    a = memory_store.remember("Chrome is my preferred browser")
    b = memory_store.remember("chrome is my preferred browser")
    assert a.id == b.id and len(memory_store.all()) == 1


def test_default_scope_is_device_local(memory_store):
    assert memory_store.remember("x is y").scope is MemoryScope.DEVICE


def test_private_memories_are_not_recalled_into_prompts(memory_store):
    memory_store.remember("My bank is ABC Bank", scope=MemoryScope.PRIVATE)
    assert memory_store.recall("which bank") == []
    assert memory_store.recall("which bank", include_private=True)[0].text == "My bank is ABC Bank"
    assert memory_store.all(include_private=False) == []


def test_temporary_memories_expire(tmp_path):
    clock = Clock()
    store = SQLiteMemoryStore(tmp_path / "m.db", clock=clock)
    store.remember("I'm working from home today", scope=MemoryScope.TEMPORARY, ttl_s=3600)
    assert len(store.all()) == 1
    clock.tick(hours=2)
    assert store.all() == []


def test_forget_erases_text_from_disk(tmp_path):
    path = tmp_path / "m.db"
    store = SQLiteMemoryStore(path)
    rec = store.remember("my-very-unique-secret-hobby is knitting")
    assert store.forget(rec.id)
    assert store.get(rec.id) is None and store.all() == []
    store.close()
    assert b"my-very-unique-secret-hobby" not in path.read_bytes()


def test_forgetting_a_synced_memory_leaves_an_empty_tombstone(memory_store):
    rec = memory_store.remember("I like biryani", scope=MemoryScope.SYNCED)
    memory_store.forget(rec.id)
    tomb = memory_store.get(rec.id)
    assert tomb.deleted and tomb.text == ""
    assert memory_store.forget(rec.id) is False


def test_memories_persist_across_restarts(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "m.db")
    store.remember("Chrome is my preferred browser")
    store.close()
    again = SQLiteMemoryStore(tmp_path / "m.db")
    assert again.all()[0].text == "Chrome is my preferred browser"


# --- sync (local-first: only 'synced' ever leaves) ------------------------------------

def test_only_synced_memories_are_offered_for_sync(memory_store):
    memory_store.remember("device fact")
    memory_store.remember("private fact", scope=MemoryScope.PRIVATE)
    memory_store.remember("temporary fact", scope=MemoryScope.TEMPORARY)
    shared = memory_store.remember("shared fact", scope=MemoryScope.SYNCED)
    changes, cursor = memory_store.changes_since(None)
    assert [c.id for c in changes] == [shared.id]
    assert memory_store.changes_since(cursor)[0] == []


def test_two_devices_merge_synced_memories(tmp_path):
    clock = Clock()
    pc = SQLiteMemoryStore(tmp_path / "pc.db", clock=clock)
    laptop = SQLiteMemoryStore(tmp_path / "laptop.db", clock=clock)
    rec = pc.remember("I like biryani", scope=MemoryScope.SYNCED, origin_device="nxd-pc")
    pc.remember("PC-only fact")
    assert laptop.apply_changes(pc.changes_since(None)[0]) == 1
    assert [r.text for r in laptop.all()] == ["I like biryani"]
    # forgotten on the laptop -> the tombstone removes it on the PC too
    clock.tick(seconds=5)
    laptop.forget(rec.id)
    pc.apply_changes(laptop.changes_since(None)[0])
    assert [r.text for r in pc.all()] == ["PC-only fact"]


def test_incoming_non_synced_records_are_ignored(tmp_path):
    a, b = SQLiteMemoryStore(tmp_path / "a.db"), SQLiteMemoryStore(tmp_path / "b.db")
    private = a.remember("secret stuff", scope=MemoryScope.PRIVATE)
    assert b.apply_changes([private]) == 0 and b.all() == []


def test_older_remote_change_does_not_overwrite_newer_local(tmp_path):
    clock = Clock()
    a = SQLiteMemoryStore(tmp_path / "a.db", clock=clock)
    rec = a.remember("fact v1", scope=MemoryScope.SYNCED)
    old = a.get(rec.id)
    clock.tick(seconds=10)
    a.forget(rec.id)
    assert a.apply_changes([old]) == 0 and a.get(rec.id).deleted


# --- tools, through the Action Kernel --------------------------------------------------

def test_remember_tool_is_verified_and_audited(make_kernel, memory_store, audit):
    k = make_kernel(memory=memory_store)
    out = k.submit(ToolRequest("remember", {"text": "Chrome is my preferred browser"}))
    assert out.status is Status.DONE and out.result.verified is True
    assert memory_store.all()[0].text == "Chrome is my preferred browser"
    assert any(e["event"] == "executed" and e["tool"] == "remember" for e in audit.tail(10))


@pytest.mark.parametrize("text", ["my bank password is hunter2", "my ATM PIN is 4321", "OTP 993211",
                                  "card 4111 1111 1111 1111", "my api key is sk-abc"])
def test_secrets_are_refused(make_kernel, memory_store, text):
    assert looks_like_secret(text)
    out = make_kernel(memory=memory_store).submit(ToolRequest("remember", {"text": text}))
    assert out.status is Status.BLOCKED and "password manager" in out.message  # a refusal, not a failure
    assert memory_store.all() == []


def test_list_memories(make_kernel, memory_store):
    k = make_kernel(memory=memory_store)
    assert "don't remember anything" in k.submit(ToolRequest("list_memories")).message
    memory_store.remember("Chrome is my preferred browser")
    memory_store.remember("My bank is ABC", scope=MemoryScope.PRIVATE)
    text = k.submit(ToolRequest("list_memories")).message
    assert "Chrome" in text and "ABC [private]" in text


def test_private_memories_never_go_to_another_device(make_kernel, memory_store):
    memory_store.remember("My bank is ABC", scope=MemoryScope.PRIVATE)
    memory_store.remember("Chrome is my preferred browser")
    k = make_kernel(memory=memory_store)
    out = k.submit(ToolRequest("list_memories", device="nxd-phone"))
    cid = out.confirmation_id
    done = k.resolve(cid, True, device="nxd-phone")
    assert "Chrome" in done.message and "ABC" not in done.message


def test_remote_memory_records_origin_device(make_kernel, memory_store):
    k = make_kernel(memory=memory_store)
    out = k.submit(ToolRequest("remember", {"text": "I'm travelling next week"}, device="nxd-phone"))
    k.resolve(out.confirmation_id, True, device="nxd-phone")
    assert memory_store.all()[0].origin_device == "nxd-phone"


def test_origin_cannot_be_supplied_as_an_argument(make_kernel, memory_store):
    out = make_kernel(memory=memory_store).submit(
        ToolRequest("remember", {"text": "x is y", "origin": "nxd-fake"}))
    assert out.status is Status.INVALID


def test_forget_asks_first_and_shows_exactly_what(make_kernel, memory_store):
    memory_store.remember("Chrome is my preferred browser")
    memory_store.remember("My favourite game is GTA V")
    k = make_kernel(memory=memory_store)
    out = k.submit(ToolRequest("forget_memory", {"query": "browser"}))
    assert out.status is Status.NEEDS_CONFIRMATION
    assert 'Forget "Chrome is my preferred browser"' in out.message
    assert len(memory_store.all()) == 2
    done = k.resolve(out.confirmation_id, True)
    assert done.status is Status.DONE and [m.text for m in memory_store.all()] == ["My favourite game is GTA V"]


def test_forget_with_no_match_changes_nothing(make_kernel, memory_store):
    memory_store.remember("My favourite game is GTA V")
    k = make_kernel(memory=memory_store)
    out = k.submit(ToolRequest("forget_memory", {"query": "biryani"}))
    assert "none matches" in out.message
    assert k.resolve(out.confirmation_id, True).status is Status.FAILED
    assert len(memory_store.all()) == 1
