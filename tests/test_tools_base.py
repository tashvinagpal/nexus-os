from pathlib import Path

import pytest

from nexus.tools.base import Param, Risk, Tool, ToolRegistry, ValidationError
from tests.conftest import make_tool


def tool_with(*params: Param) -> Tool:
    return make_tool("t", params=params)


def test_missing_required_argument():
    with pytest.raises(ValidationError, match="needs 'q'"):
        tool_with(Param("q", "str", "query")).validate({})


def test_unknown_argument_rejected():
    with pytest.raises(ValidationError, match="does not take: rm"):
        tool_with(Param("q", "str", "query")).validate({"q": "x", "rm": "-rf"})


def test_optional_default_and_choices():
    t = tool_with(Param("mode", "str", "m", required=False, default="a", choices=("a", "b")))
    assert t.validate({}) == {"mode": "a"}
    assert t.validate({"mode": "b"}) == {"mode": "b"}
    with pytest.raises(ValidationError, match="one of"):
        t.validate({"mode": "c"})


@pytest.mark.parametrize("raw,expected", [("yes", True), ("0", False), (True, True), ("off", False)])
def test_bool_coercion(raw, expected):
    assert tool_with(Param("b", "bool", "flag")).validate({"b": raw}) == {"b": expected}


def test_number_coercion():
    t = tool_with(Param("n", "int", "n"))
    assert t.validate({"n": "42"}) == {"n": 42}
    with pytest.raises(ValidationError):
        t.validate({"n": "lots"})
    with pytest.raises(ValidationError):
        t.validate({"n": True})


def test_empty_and_oversized_text_rejected():
    t = tool_with(Param("q", "str", "q"))
    with pytest.raises(ValidationError):
        t.validate({"q": "   "})
    with pytest.raises(ValidationError):
        t.validate({"q": "x" * 5000})
    with pytest.raises(ValidationError):
        t.validate({"q": 123})


@pytest.mark.parametrize("url", ["file:///C:/Windows/System32/cmd.exe", "javascript:alert(1)",
                                 "ms-settings:display", "github.com", "https://"])
def test_non_web_urls_rejected(url):
    with pytest.raises(ValidationError, match="http"):
        tool_with(Param("u", "url", "u")).validate({"u": url})


def test_relative_paths_rejected():
    with pytest.raises(ValidationError, match="full path"):
        tool_with(Param("p", "path", "p")).validate({"p": "Desktop\\stuff"})


@pytest.mark.parametrize("raw", [r"\\evil-server\share\x", "//evil-server/share/x", r"\\?\C:\Windows"])
def test_network_and_device_paths_rejected_before_resolving(raw):
    with pytest.raises(ValidationError, match="network"):
        tool_with(Param("p", "path", "p")).validate({"p": raw})


def test_paths_are_expanded_and_resolved(tmp_path):
    t = tool_with(Param("p", "path", "p"))
    assert t.validate({"p": str(tmp_path / "a" / ".." / "b")}) == {"p": (tmp_path / "b").resolve()}
    assert t.validate({"p": "~"})["p"] == Path.home().resolve()


def test_registry_rejects_duplicates_and_lists_sorted():
    reg = ToolRegistry([make_tool("zeta"), make_tool("alpha")])
    assert [t.name for t in reg] == ["alpha", "zeta"]
    with pytest.raises(ValueError, match="already registered"):
        reg.register(make_tool("alpha"))


def test_invalid_tool_name():
    with pytest.raises(ValueError):
        make_tool("Open App!")


def test_json_schema_for_ai_function_calling():
    t = make_tool("search", risk=Risk.SAFE, params=(
        Param("query", "str", "what to search"),
        Param("engine", "str", "engine", required=False, default="google", choices=("google", "bing")),
    ))
    fn = t.json_schema()["function"]
    assert fn["name"] == "search"
    assert fn["parameters"]["required"] == ["query"]
    assert fn["parameters"]["properties"]["engine"] == {
        "type": "string", "description": "engine", "enum": ["google", "bing"]}


def test_summary_masks_sensitive_values():
    t = make_tool("login", params=(Param("user", "str", "u"), Param("pin", "str", "p", sensitive=True)))
    assert "1234" not in t.summary({"user": "me", "pin": "1234"})
