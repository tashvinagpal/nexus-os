import json

import pytest

from nexus.security.audit import AuditLog


def test_chain_verifies(audit):
    for i in range(5):
        audit.record("event", n=i, path=audit.path)  # Paths are stored as text
    ok, detail = audit.verify()
    assert ok, detail
    assert "5 entries" in detail


def test_chain_continues_across_restarts(audit):
    audit.record("first")
    reopened = AuditLog(audit.path)
    reopened.record("second")
    assert reopened.verify()[0]
    entries = reopened.tail(2)
    assert entries[1]["prev"] == entries[0]["hash"]


def test_edited_entry_detected(audit):
    for i in range(3):
        audit.record("event", n=i)
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["n"] = 99
    lines[1] = json.dumps(entry)
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, detail = audit.verify()
    assert not ok and "line 2" in detail


def test_deleted_entry_detected(audit):
    for i in range(3):
        audit.record("event", n=i)
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, detail = audit.verify()
    assert not ok and "chain broken" in detail


def test_damaged_tail_refuses_to_start_new_chain(audit):
    audit.record("event")
    with audit.path.open("a", encoding="utf-8") as f:
        f.write("garbage\n")
    with pytest.raises(RuntimeError, match="damaged"):
        AuditLog(audit.path)


def test_unicode_is_preserved(audit):
    audit.record("request", text="Nexus bhai GTA chala de - नमस्ते")
    assert audit.verify()[0]
    assert audit.tail(1)[0]["text"].endswith("नमस्ते")
