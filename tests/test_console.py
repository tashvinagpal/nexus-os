import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.console import Console, parse_args
from nexus.core.runtime import Runtime
from tests.conftest import ScriptedPlanner, plan


def test_parse_args():
    assert parse_args('query="gta v mods" engine=youtube') == {"query": "gta v mods", "engine": "youtube"}
    assert parse_args(r"path=C:\Users\me\x url=https://a.com/?q=1") == {
        "path": r"C:\Users\me\x", "url": "https://a.com/?q=1"}
    assert parse_args("") == {}
    with pytest.raises(ValueError):
        parse_args(r"path=C:\Program Files\x")  # unquoted space


class FakeAssistant:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.chat = type("Chat", (), {"last_choice": None})()


def session(make_kernel, audit, fake_platform, lines, planner=None):
    kernel = make_kernel()
    rt = Runtime(settings=None, platform=fake_platform, kernel=kernel,
                 registry=kernel.registry, policy=kernel.policy, audit=audit)
    assistant = FakeAssistant(ConversationEngine(kernel, planner)) if planner else None
    feed, out = iter(lines), []
    Console(rt, assistant, input_fn=lambda _: next(feed), output=out.append).run()
    return "\n".join(out)


def test_console_commands(make_kernel, audit, fake_platform, sandbox):
    outside = sandbox / "outside" / "x"
    text = session(make_kernel, audit, fake_platform, [
        "/tools",
        "/run get_system_info section=memory",
        f"/run create_folder path={outside}",
        "n",
        "/run create_folder path=relative",
        "/bogus",
        "/audit verify",
        "/quit",
    ])
    assert "create_folder" in text and "get_system_info" in text
    assert "Memory" in text
    assert "Asking because" in text and "[cancelled]" in text
    assert not outside.exists()
    assert "full path" in text
    assert "Unknown command '/bogus'" in text
    assert "OK:" in text


def test_chat_without_ai_layer_is_honest(make_kernel, audit, fake_platform):
    text = session(make_kernel, audit, fake_platform, ["nexus bhai gta chala de", "/quit"])
    assert "AI layer isn't running" in text


def test_chat_goes_through_engine(make_kernel, audit, fake_platform, sandbox):
    planner = ScriptedPlanner(
        plan("Kaafi badhiya bhai!", language="hinglish"),
        plan("Sure.", ("create_folder", {"path": str(sandbox / "outside" / "games")})),
    )
    text = session(make_kernel, audit, fake_platform,
                   ["kaise ho bhai", "make a games folder there", "haan", "/quit"], planner)
    assert "nexus> Kaafi badhiya bhai!" in text
    assert "[needs your OK] create_folder" in text
    assert "[done] create_folder" in text
    assert (sandbox / "outside" / "games").is_dir()


def test_console_ignores_byte_order_mark(make_kernel, audit, fake_platform):
    text = session(make_kernel, audit, fake_platform, ["\ufeff/tools", "/quit"])
    assert "get_system_info" in text and "Unknown command" not in text


def test_console_confirm_yes(make_kernel, audit, fake_platform, sandbox):
    outside = sandbox / "outside" / "y"
    text = session(make_kernel, audit, fake_platform, [f"/run create_folder path={outside}", "y", "/quit"])
    assert "[done]" in text and outside.is_dir()
