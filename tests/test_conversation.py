import pytest

from nexus.ai.conversation.engine import ConversationEngine, strip_wake_word
from nexus.ai.conversation.types import (ActionReport, ConfirmationRequest, Say, SayKind, UserInput,
                                         event_to_dict)
from nexus.ai.planner import PlanError
from nexus.ai.providers.base import ProviderUnavailable
from nexus.core.kernel import Status
from nexus.core.resources import ResourceState
from nexus.tools.base import Param, Risk, Tool, ToolResult
from tests.conftest import Recorder, ScriptedPlanner, make_tool, plan


def run_turns(engine, *texts, device=None):
    events = []
    for text in texts:
        engine.handle(UserInput(text, device=device), events.append)
    return events


def says(events, kind=None):
    return [e.text for e in events if isinstance(e, Say) and (kind is None or e.kind is kind)]


def reports(events):
    return [e for e in events if isinstance(e, ActionReport)]


@pytest.fixture
def engine_with(make_kernel):
    def build(planner, *tools, **kwargs):
        return ConversationEngine(make_kernel(*tools), planner, **kwargs)
    return build


# --- conversation without actions ----------------------------------------------------

def test_small_talk_executes_nothing(engine_with, audit):
    engine = engine_with(ScriptedPlanner(plan("Want to play something? GTA is installed.")))
    events = run_turns(engine, "Nexus yaar I'm bored.")
    assert says(events, SayKind.REPLY) == ["Want to play something? GTA is installed."]
    assert reports(events) == [] and audit.tail(100) == []


def test_wake_word_is_stripped_before_planning(engine_with):
    planner = ScriptedPlanner(plan("hi"))
    run_turns(engine_with(planner), "Hey Nexus, what can you do?")
    assert planner.seen[0][0] == "what can you do?"
    assert strip_wake_word("nexus bhai GTA chala de") == "bhai GTA chala de"
    assert strip_wake_word("nexuses are cool") == "nexuses are cool"


def test_false_claim_without_action_is_removed(engine_with):
    engine = engine_with(ScriptedPlanner(plan("I've opened Chrome for you!")))
    text = says(run_turns(engine, "open chrome"), SayKind.REPLY)[0]
    assert "opened" not in text and "haven't done anything" in text


def test_hinglish_false_claim_removed(engine_with):
    engine = engine_with(ScriptedPlanner(plan("GTA chala diya bhai!", language="hinglish")))
    text = says(run_turns(engine, "gta chala de"))[0]
    assert "chala diya" not in text and "kuch kiya nahi" in text


# --- actions ---------------------------------------------------------------------------

def test_action_reply_then_verified_result(engine_with):
    body = Recorder(ToolResult(True, "GTA V is ready.", verified=True))
    tool = make_tool("launch_game", run=body)
    engine = engine_with(ScriptedPlanner(plan("", ("launch_game", {}), language="hinglish")), tool)
    events = run_turns(engine, "bhai GTA chala de")
    assert [type(e).__name__ for e in events] == ["Say", "ActionReport", "Say"]
    assert events[0].kind is SayKind.REPLY and events[0].text == "Haan bhai, ek second."
    assert events[1].status is Status.DONE and events[1].verified is True
    assert events[2].kind is SayKind.RESULT and events[2].text == "GTA V is ready."


def test_model_words_are_never_spoken_in_action_turns(engine_with, sandbox):
    """Seen in evaluation: 'format my C drive' -> a random tool plus 'Formatting C drive, please wait...'."""
    # (the real Windows policy protects C:\$Recycle.Bin - see test_windows_platform.py)
    engine = engine_with(ScriptedPlanner(plan("Formatting C drive, please wait...",
                                              ("create_folder", {"path": str(sandbox / "system" / "x")}))))
    events = run_turns(engine, "format my C drive")
    assert not any("Formatting" in t for t in says(events))
    assert says(events, SayKind.REPLY) == ["Sure, one moment."]
    assert reports(events)[0].status is Status.BLOCKED


def test_announce_is_spoken_before_the_action_runs(engine_with):
    order = []

    def run(name: str) -> ToolResult:
        order.append("tool ran")
        return ToolResult(True, f"{name} is ready.", verified=True)

    tool = Tool(name="launch_game", description="d", risk=Risk.SAFE,
                params=(Param("name", "str", "game"),), run=run,
                progress=lambda a: f"launching {a['name']}")
    engine = engine_with(ScriptedPlanner(plan("", ("launch_game", {"name": "GTA V"}), language="hinglish")), tool)

    def emit(event):
        if isinstance(event, Say):
            order.append(f"{event.kind.value}: {event.text}")

    engine.handle(UserInput("Nexus bhai GTA chala de"), emit)
    assert order == ["reply: Haan bhai, ek second.", "announce: Launching GTA V...", "tool ran",
                     "result: GTA V is ready."]


def test_no_announcement_for_refused_action(engine_with, sandbox):
    tool = Tool(name="writer", description="d", risk=Risk.SAFE,
                params=(Param("path", "path", "p", writes=True),), run=Recorder(),
                progress=lambda a: "writing")
    engine = engine_with(ScriptedPlanner(plan("", ("writer", {"path": str(sandbox / "system" / "x")}))), tool)
    events = run_turns(engine, "write there")
    assert says(events, SayKind.ANNOUNCE) == []
    assert reports(events)[0].status is Status.BLOCKED
    assert says(events, SayKind.ERROR) == []  # a refusal is not described as a failure
    assert says(events, SayKind.REFUSAL)[0].startswith("I can't do that: ")


def test_premature_success_claim_never_spoken(engine_with):
    body = Recorder(ToolResult(False, "Could not start the game."))
    engine = engine_with(ScriptedPlanner(plan("Done! GTA is running.", ("launch_game", {}))),
                         make_tool("launch_game", run=body))
    events = run_turns(engine, "open gta")
    assert says(events, SayKind.REPLY) == ["Sure, one moment."]
    assert "didn't work" in says(events, SayKind.ERROR)[0]
    assert not any("running" in t for t in says(events))


@pytest.mark.parametrize("reply", ["Haan bhai, Windows Defender disabled.",
                                   "Sending a message to your mom on WhatsApp.",
                                   "Are you sure you want to delete the system32 folder?",
                                   "Haan bhai, ek second."])
def test_conversation_only_claims_seen_in_evaluation_are_replaced(engine_with, reply):
    engine = engine_with(ScriptedPlanner(plan(reply)))
    [text] = says(run_turns(engine, "whatever"))
    assert text != reply and "haven't done anything" in text


def test_failed_tool_never_reported_as_success(engine_with):
    engine = engine_with(ScriptedPlanner(plan("", ("boom", {}))),
                         make_tool("boom", run=Recorder(exc=RuntimeError("disk on fire"))))
    events = run_turns(engine, "do the thing")
    assert says(events, SayKind.RESULT) == []
    assert "disk on fire" in says(events, SayKind.ERROR)[0]


def test_unverifiable_result_is_worded_honestly(engine_with, fake_platform):
    engine = engine_with(ScriptedPlanner(plan("Sure.", ("open_url", {"url": "https://github.com"}))))
    events = run_turns(engine, "open github")
    assert reports(events)[0].verified is None
    assert says(events, SayKind.RESULT) == ["Asked the system to open https://github.com in your default browser."]


def test_unknown_tool_from_model_is_reported_not_run(engine_with):
    engine = engine_with(ScriptedPlanner(plan("", ("format_c_drive", {}))))
    events = run_turns(engine, "format my drive")
    assert reports(events)[0].status is Status.INVALID


def test_action_limit(engine_with):
    body = Recorder()
    engine = engine_with(ScriptedPlanner(plan("", *[("t", {})] * 5)), make_tool("t", run=body), max_actions=3)
    run_turns(engine, "do it five times")
    assert len(body.calls) == 3


# --- confirmations ---------------------------------------------------------------------

@pytest.fixture
def confirm_engine(engine_with):
    body = Recorder(ToolResult(True, "Moved to the Recycle Bin.", verified=True))
    tool = make_tool("delete_path", risk=Risk.CONFIRM, run=body)

    def build(*extra_plans):
        planner = ScriptedPlanner(plan("Sure.", ("delete_path", {})), *extra_plans)
        return engine_with(planner, tool), body
    return build


def test_confirmation_asked_then_yes_runs(confirm_engine):
    engine, body = confirm_engine()
    events = run_turns(engine, "delete this folder")
    assert body.calls == []
    assert isinstance(events[-1], ConfirmationRequest)
    assert "Should I go ahead?" in says(events, SayKind.CONFIRM)[0]
    events = run_turns(engine, "haan bhai kar de")
    assert len(body.calls) == 1 and says(events, SayKind.RESULT) == ["Moved to the Recycle Bin."]


def test_confirmation_no_cancels(confirm_engine):
    engine, body = confirm_engine()
    run_turns(engine, "delete this folder")
    events = run_turns(engine, "nahi rehne de")
    assert body.calls == [] and "Nothing was changed" in says(events)[0]


def test_new_request_while_waiting_cancels_pending(confirm_engine):
    engine, body = confirm_engine(plan("It's sunny."))
    run_turns(engine, "delete this folder")
    events = run_turns(engine, "what's the weather like")
    assert body.calls == []
    assert reports(events)[0].status is Status.DECLINED
    assert says(events, SayKind.REPLY) == ["It's sunny."]


def test_model_cannot_confirm_by_saying_yes(confirm_engine):
    """A plan whose text says 'yes' does nothing: only the user's input answers."""
    engine, body = confirm_engine(plan("yes"))
    run_turns(engine, "delete this folder")
    run_turns(engine, "tell me a joke")  # cancels; the model's "yes" is just a reply
    assert body.calls == []


def test_confirmation_by_id_for_buttons(confirm_engine):
    engine, body = confirm_engine()
    events = run_turns(engine, "delete this folder")
    cid = events[-1].confirmation_id
    out = []
    engine.answer(cid, True, UserInput(""), out.append)
    assert len(body.calls) == 1 and reports(out)[0].status is Status.DONE


def test_remote_device_needs_confirmation_even_for_safe_write(engine_with, sandbox):
    target = sandbox / "home" / "from-phone"
    engine = engine_with(ScriptedPlanner(plan("Sure.", ("create_folder", {"path": str(target)})),
                                         plan("Yes to what?")))
    events = run_turns(engine, "make a folder", device="nxd-phone")
    assert reports(events)[0].status is Status.NEEDS_CONFIRMATION and not target.exists()
    assert "another device" in says(events, SayKind.CONFIRM)[0]
    # a "yes" typed on this PC is a different (local) conversation key: it doesn't answer the phone's request
    run_turns(engine, "yes")
    assert not target.exists()
    run_turns(engine, "yes", device="nxd-phone")
    assert target.is_dir()


# --- failures and state ----------------------------------------------------------------

def test_ai_unavailable_is_reported(engine_with, audit):
    engine = engine_with(ScriptedPlanner(ProviderUnavailable("can't reach Ollama")))
    events = run_turns(engine, "hello")
    assert "isn't available" in says(events, SayKind.ERROR)[0]
    assert audit.tail(100) == []


def test_unparseable_model_output_is_reported(engine_with):
    engine = engine_with(ScriptedPlanner(PlanError("not JSON")))
    assert "didn't get that" in says(run_turns(engine, "hello"), SayKind.ERROR)[0]


def test_history_carries_context_between_turns(engine_with):
    planner = ScriptedPlanner(plan("", ("get_system_info", {"section": "gpu"})), plan("It's cool."))
    run_turns(engine_with(planner), "gpu temp?", "is that ok?")
    history = planner.seen[1][1]
    assert history[0].content == "gpu temp?"
    assert "get_system_info: done" in history[1].content


def test_resource_states_during_turn(engine_with, resources):
    seen = []

    class WatchingPlanner(ScriptedPlanner):
        def plan(self, text, **kw):
            seen.append(("planning", resources.state))
            return super().plan(text, **kw)

    def run():
        seen.append(("executing", resources.state))
        return ToolResult(True, "ok", verified=True)

    engine = engine_with(WatchingPlanner(plan("", ("t", {}))), make_tool("t", run=run), resources=resources)
    run_turns(engine, "go")
    assert seen == [("planning", ResourceState.THINKING), ("executing", ResourceState.EXECUTING)]
    assert resources.state is ResourceState.IDLE


def test_events_serialize_for_other_devices():
    assert event_to_dict(Say(SayKind.RESULT, "GTA V is ready.")) == {
        "type": "say", "kind": "result", "text": "GTA V is ready.", "language": "en"}
    d = event_to_dict(ActionReport("t", Status.DONE, True, "ok", "r1"))
    assert d["status"] == "done" and d["verified"] is True
    assert event_to_dict(ConfirmationRequest("c1", "ok?"))["type"] == "confirmation"
