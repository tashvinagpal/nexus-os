import json

import pytest

from nexus.ai.memory import MemoryRecord
from nexus.ai.planner import LLMPlanner, PlanError, parse_plan
from nexus.ai.planner.llm_planner import describe_tools, plan_schema
from nexus.ai.providers.base import ChatMessage
from nexus.tools import default_tools
from tests.conftest import FakeChat


@pytest.fixture
def tools(fake_platform):
    return default_tools(fake_platform)


def reply(**data):
    return json.dumps({"language": "en", "actions": [], "say": "", **data})


def test_chat_only(tools):
    p = parse_plan(reply(say="Main badhiya hoon!", language="hinglish"), tools)
    assert p.say == "Main badhiya hoon!" and p.actions == () and p.language == "hinglish"


def test_action_plan(tools):
    p = parse_plan(reply(say="Sure.", actions=[{"tool": "get_system_info", "args": {"section": "gpu"}}]), tools)
    assert p.actions[0].tool == "get_system_info" and p.actions[0].args == {"section": "gpu"}


def test_unknown_args_and_empty_optionals_dropped(tools):
    p = parse_plan(reply(actions=[{"tool": "search_web",
                                   "args": {"query": "gta mods", "engine": "", "reason": "fun"}}]), tools)
    assert p.actions[0].args == {"query": "gta mods"}


def test_unknown_tools_pass_through_for_the_kernel_to_refuse(tools):
    p = parse_plan(reply(actions=[{"tool": "run_shell", "args": {"cmd": "del /s"}}]), tools)
    assert p.actions[0].tool == "run_shell"


def test_invalid_language_defaults_to_english(tools):
    assert parse_plan(reply(say="hi", language="klingon"), tools).language == "en"


def test_action_count_capped(tools):
    acts = [{"tool": "get_system_info", "args": {}}] * 6
    assert len(parse_plan(reply(actions=acts), tools, max_actions=3).actions) == 3


@pytest.mark.parametrize("bad", ["not json", "[]", reply(actions="nope"), reply(say="", actions=[])])
def test_bad_output_raises(tools, bad):
    with pytest.raises(PlanError):
        parse_plan(bad, tools)


def test_schema_restricts_tools_to_registered_names(tools):
    schema = plan_schema([t.name for t in tools], 3)
    items = schema["properties"]["actions"]["items"]["properties"]["tool"]
    assert set(items["enum"]) == {"get_system_info", "get_datetime", "open_url", "search_web", "create_folder"}
    assert list(schema["properties"]) == ["language", "actions", "say"]


def test_tool_descriptions_for_prompt(tools):
    text = describe_tools(tools)
    assert "get_system_info(section?: all|cpu|memory|gpu|storage)" in text
    assert "create_folder(path: path)" in text


def test_planner_sends_system_history_and_user(tools):
    chat = FakeChat("llm", reply=reply(say="hello"))
    planner = LLMPlanner(chat, tools)
    history = [ChatMessage("user", "earlier"), ChatMessage("assistant", "reply")]
    memory = MemoryRecord("id1", "Chrome is the preferred browser", "t", "t", None)
    plan = planner.plan("hi", history=history, memories=[memory])
    messages, schema = chat.calls[0]
    assert messages[0].role == "system" and "open_url" in messages[0].content
    assert "Chrome is the preferred browser" in messages[0].content
    assert [m.content for m in messages[1:]] == ["earlier", "reply", "hi"]
    assert schema is planner.schema and plan.say == "hello"
