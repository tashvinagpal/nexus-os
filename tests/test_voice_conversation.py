"""Voice is just another way to type: the same engine, kernel, policy and confirmations.

A spoken request becomes UserInput(interface="voice"); nothing about voice
can approve an action by itself, and a "yes" that wasn't heard clearly never does.
"""

from __future__ import annotations

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.conversation.types import ConfirmationRequest, Say, SayKind, UserInput
from nexus.ai.router import FastPathRouter
from nexus.ai.voice.text import romanize
from nexus.console import Console
from nexus.core.kernel import Status
from nexus.core.runtime import Runtime
from nexus.tools.base import Param, Risk, ToolResult
from tests.conftest import Recorder, ScriptedPlanner, make_tool, plan


def voice(text: str, confidence: float | None = 0.9, device=None) -> UserInput:
    return UserInput(text, interface="voice", confidence=confidence, device=device)


def run(engine, *inputs):
    events = []
    for item in inputs:
        engine.handle(item if isinstance(item, UserInput) else UserInput(item), events.append)
    return events


@pytest.fixture
def confirm_engine(make_kernel):
    body = Recorder(ToolResult(True, "Moved to the Recycle Bin.", verified=True))
    tool = make_tool("delete_path", risk=Risk.CONFIRM, run=body)
    planner = ScriptedPlanner(plan("Sure.", ("delete_path", {})), plan("Sure.", ("delete_path", {})))
    return ConversationEngine(make_kernel(tool), planner), body


def test_voice_request_goes_through_the_kernel_like_typed_text(confirm_engine):
    engine, body = confirm_engine
    events = run(engine, voice("delete this folder"))
    assert body.calls == []  # a CONFIRM tool still asks - voice changes nothing
    assert isinstance(events[-1], ConfirmationRequest)


def test_clear_spoken_yes_approves(confirm_engine):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    run(engine, voice("haan kar do", confidence=0.85))
    assert len(body.calls) == 1


def test_unclear_spoken_yes_never_approves_and_asks_again(confirm_engine):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    events = run(engine, voice("yes", confidence=0.45))
    assert body.calls == []
    assert [e.text for e in events if isinstance(e, Say)] == [
        "I didn't hear that clearly. Say yes or no again - or type it."]
    # still pending: a clear answer (spoken or typed) decides
    run(engine, UserInput("yes"))
    assert len(body.calls) == 1


def test_spoken_yes_without_a_confidence_is_treated_as_unclear(confirm_engine):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    run(engine, voice("yes", confidence=None))
    assert body.calls == []


def test_unclear_spoken_no_still_cancels(confirm_engine):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    events = run(engine, voice("no", confidence=0.3))  # saying no is always safe
    assert body.calls == [] and "Nothing was changed" in [e.text for e in events if isinstance(e, Say)][0]


def test_romanized_hindi_yes_approves(confirm_engine):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    run(engine, voice(romanize("हाँ कर दो"), confidence=0.8))
    assert len(body.calls) == 1


def test_threshold_is_configurable(make_kernel):
    body = Recorder(ToolResult(True, "done", verified=True))
    tool = make_tool("delete_path", risk=Risk.CONFIRM, run=body)
    engine = ConversationEngine(make_kernel(tool), ScriptedPlanner(plan("Sure.", ("delete_path", {}))),
                                voice_confirm_min_confidence=0.95)
    run(engine, voice("delete this folder"), voice("yes", confidence=0.9))
    assert body.calls == []


def test_a_phone_cannot_answer_the_pcs_question_by_voice(confirm_engine):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    run(engine, voice("yes", confidence=0.99, device="nxd-phone"))
    assert body.calls == []


def test_blocked_actions_stay_blocked_by_voice(make_kernel, audit):
    from nexus.ai.conversation.types import ActionReport

    body = Recorder(ToolResult(True, "x", verified=True))
    engine = ConversationEngine(make_kernel(make_tool("wipe", risk=Risk.HIGH, run=body)),
                                ScriptedPlanner(plan("Sure.", ("wipe", {})), plan("Okay.")))
    events = run(engine, voice("wipe everything", confidence=0.99), voice("yes", confidence=0.99))
    assert body.calls == []
    assert [e.status for e in events if isinstance(e, ActionReport)] == [Status.BLOCKED]


@pytest.mark.parametrize("spoken, expected", [
    ("समय क्या हुआ है", "समय क्या हुआ है"),   # the fast path knows this Devanagari phrase: kept as spoken
    ("कितने बजे हैं", "कितने बजे हैं"),
    ("जीटीए चला दे", "jitie chala de"),      # not understood as written: romanized for router/planner
])
def test_original_script_is_kept_when_the_fast_path_understands_it(spoken, expected):
    from nexus.ai.voice.base import Transcript
    from tests.test_voice_input import SPEECH, build
    from tests.voice_fixtures import FakeCapture, FakeDevices, FakeStream, mic

    router = FastPathRouter()
    vi, _, _ = build(FakeDevices(mic("Microphone (USB)")), FakeCapture({"Microphone (USB)": FakeStream(SPEECH)}),
                     Transcript(spoken, "hi", 0.8), fast_path=lambda text: router.route(text) is not None)
    assert vi.listen().text == expected


# --- spoken answers to a pending confirmation (real retest phrases) --------------------------------

YES_ANSWERS = ["haan ji", "haanji", "anji", "अंजी", "haan, kar do", "haan, kar de", "yes"]
NO_ANSWERS = ["nahi", "nahin rahne do", "nahi chahiye"]


@pytest.mark.parametrize("answer", YES_ANSWERS)
def test_clear_spoken_yes_forms_approve(confirm_engine, answer):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    run(engine, voice(answer, confidence=0.64))  # "अंजी" was heard at 0.64 in the retest
    assert len(body.calls) == 1


@pytest.mark.parametrize("answer", YES_ANSWERS)
def test_unclear_spoken_yes_forms_never_approve(confirm_engine, answer):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    events = run(engine, voice(answer, confidence=0.57))  # ... and at 0.57
    assert body.calls == []
    assert [e.text for e in events if isinstance(e, Say)] == [
        "I didn't hear that clearly. Say yes or no again - or type it."]
    run(engine, voice("haan ji", confidence=0.9))  # the question is still open for a clear answer
    assert len(body.calls) == 1


@pytest.mark.parametrize("answer", NO_ANSWERS)
@pytest.mark.parametrize("confidence", [0.3, 0.9])
def test_spoken_no_forms_always_cancel(confirm_engine, answer, confidence):
    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    events = run(engine, voice(answer, confidence=confidence))
    assert body.calls == [] and "Nothing was changed" in [e.text for e in events if isinstance(e, Say)][0]


@pytest.mark.parametrize("confidence, approved", [(0.64, True), (0.57, False)])
def test_haan_ji_heard_as_anji_end_to_end(confirm_engine, confidence, approved):
    """The retest transcript "अंजी" through the real listener normalization into the engine."""
    from nexus.ai.voice.base import Transcript
    from tests.test_voice_input import SPEECH, build
    from tests.voice_fixtures import FakeCapture, FakeDevices, FakeStream, mic

    engine, body = confirm_engine
    run(engine, voice("delete this folder"))
    vi, _, _ = build(FakeDevices(mic("Microphone (USB)")), FakeCapture({"Microphone (USB)": FakeStream(SPEECH)}),
                     Transcript("अंजी", "hi", confidence), fast_path=lambda text: False)
    heard = vi.listen()
    assert not heard.guessed
    run(engine, UserInput(heard.text, interface="voice", confidence=heard.confidence, inferred=heard.guessed))
    assert len(body.calls) == (1 if approved else 0)


# --- uncertain speech never acts without asking -------------------------------------------------

@pytest.fixture
def launch_engine(make_kernel):
    from nexus.ai.conversation.types import ActionStep, Plan

    body = Recorder(ToolResult(True, "Google Chrome is open.", verified=True))
    launch = make_tool("launch_app", run=body, params=(Param("name", "str", "app"),))

    class Router:
        apps = None

        def route(self, text):
            if "chrome" in text.lower():
                return Plan("", (ActionStep("launch_app", {"name": "Chrome"}),), acknowledge=False)
            if "time" in text.lower() or "baje" in text.lower() or "vaje" in text.lower():
                return Plan("", (ActionStep("get_datetime", {}),), acknowledge=False)
            return None

    kernel = make_kernel(launch)  # get_datetime is one of the real default tools
    return ConversationEngine(kernel, ScriptedPlanner(plan("Sure.", ("launch_app", {"name": "Chrome"}))),
                              router=Router()), body, None


def read_time(events) -> bool:
    from nexus.ai.conversation.types import ActionReport

    return any(isinstance(e, ActionReport) and e.tool == "get_datetime" and e.status is Status.DONE for e in events)


def says_text(events):
    return [e.text for e in events if isinstance(e, Say)]


@pytest.mark.parametrize("item", [
    voice("Chrome khol do", confidence=0.45),  # like the real "download folder kholo" at 0.45
    UserInput("Chrome khol do", interface="voice", confidence=0.9, inferred=True),  # "Coldo" -> "khol do"
])
def test_unclear_or_guessed_voice_asks_before_acting(launch_engine, item):
    engine, body, _ = launch_engine
    events = run(engine, item)
    assert body.calls == []
    assert "Should I open Chrome? (yes/no)" in says_text(events)[0]
    assert "not sure I heard you right" in says_text(events)[0]


def test_clear_voice_or_typed_request_acts_directly(launch_engine):
    engine, body, _ = launch_engine
    run(engine, voice("Chrome khol do", confidence=0.85))
    run(engine, UserInput("Chrome khol do"))
    assert len(body.calls) == 2


def test_reading_the_time_never_asks(launch_engine):
    engine, _, _ = launch_engine
    assert read_time(run(engine, voice("kinne vaje ne", confidence=0.55)))


def test_answering_the_check(launch_engine):
    engine, body, _ = launch_engine
    run(engine, voice("Chrome khol do", confidence=0.45))
    events = run(engine, voice("haan", confidence=0.5))  # an unclear yes doesn't count
    assert body.calls == [] and "didn't hear that clearly" in says_text(events)[0]
    run(engine, voice("haan kar do", confidence=0.8))
    assert len(body.calls) == 1


def test_a_guessed_yes_does_not_count(launch_engine):
    engine, body, _ = launch_engine
    run(engine, voice("Chrome khol do", confidence=0.45))
    run(engine, UserInput("haan kar do", interface="voice", confidence=0.9, inferred=True))  # "Han, Cardu"
    assert body.calls == []
    run(engine, UserInput("yes"))  # typed
    assert len(body.calls) == 1


@pytest.mark.parametrize("answer", ["nahi, rehne do", "nahin rahne do", "no", "chhod do"])
def test_no_to_the_check_cancels_even_when_unclear(launch_engine, answer):
    engine, body, _ = launch_engine
    run(engine, voice("Chrome khol do", confidence=0.45))
    events = run(engine, voice(answer, confidence=0.3))
    assert body.calls == [] and says_text(events) == ["Okay, I won't do that."]
    assert engine._voice_checks == {}  # nothing is pending any more: a later "yes" can't revive it


def test_something_else_drops_the_unconfirmed_request(launch_engine):
    engine, body, _ = launch_engine
    run(engine, voice("Chrome khol do", confidence=0.45))
    events = run(engine, voice("kitne baje hain", confidence=0.9))
    assert body.calls == [] and read_time(events)


def test_planner_actions_from_unclear_speech_ask_too(launch_engine):
    engine, body, _ = launch_engine
    events = run(engine, voice("wo browser wala khol", confidence=0.5))  # not the fast path: the planner
    assert body.calls == [] and "Should I open Chrome?" in says_text(events)[0]


def test_unclear_yes_to_a_which_one_question_is_asked_again(make_kernel):
    from nexus.ai.conversation.engine import _PendingChoice
    from nexus.ai.conversation.types import ActionStep

    body = Recorder(ToolResult(True, "Steam is open.", verified=True))
    launch = make_tool("launch_app", run=body, params=(Param("name", "str", "app"),))
    engine = ConversationEngine(make_kernel(launch), ScriptedPlanner())
    engine._choices[("default", None)] = _PendingChoice(ActionStep("launch_app", {"name": "Steam"}), "name",
                                                         ("Steam",), "en")
    events = run(engine, voice("haan", confidence=0.4))
    assert body.calls == [] and "didn't hear that clearly" in says_text(events)[0]
    run(engine, voice("haan", confidence=0.9))
    assert len(body.calls) == 1


# --- fast path: spoken Hindi reaches the same no-model routes ------------------------------

@pytest.mark.parametrize("spoken", ["कितने बजे हैं", "समय क्या हुआ है", "What time is it?", "ਕਿੰਨੇ ਵਜੇ ਨੇ"])
def test_spoken_time_questions_use_the_fast_path(spoken):
    router = FastPathRouter()
    routed = router.route(spoken) or router.route(romanize(spoken))  # as spoken, or romanized
    assert routed is not None and routed.actions[0].tool == "get_datetime"


# --- console -------------------------------------------------------------------------------------

class FakeVoice:
    def __init__(self, results) -> None:
        from nexus.system.audio.devices import MicrophoneManager
        from tests.voice_fixtures import FakeDevices, mic

        self.results = list(results)
        self.mics = MicrophoneManager(FakeDevices(mic("Microphone (USB)")))
        self.device_poll_s = 0
        self.last_timings = {"transcribing_s": 0.4}
        self.calls = 0

    def listen(self, *, cancel=None, on_status=lambda _: None):
        self.calls += 1
        on_status("listening with Microphone (USB)")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeAssistant:
    def __init__(self, engine, voice) -> None:
        self.engine = engine
        self.chat = type("Chat", (), {"last_choice": None})()
        self.voice = voice
        self.voice_note = ""


def heard(text, original=None, confidence=0.9, language="en"):
    from nexus.ai.voice.base import Transcript
    from nexus.ai.voice.listener import Heard

    return Heard(text, original or text, Transcript(original or text, language, confidence), "Microphone (USB)", 1.0)


def console_session(make_kernel, audit, fake_platform, lines, voice_results, planner=None, tools=()):
    kernel = make_kernel(*tools)
    rt = Runtime(settings=None, platform=fake_platform, kernel=kernel, registry=kernel.registry,
                 policy=kernel.policy, audit=audit)
    engine = ConversationEngine(kernel, planner or ScriptedPlanner(plan("Hello!")), router=FastPathRouter())
    fake_voice = FakeVoice(voice_results)
    feed, out = iter(lines), []
    Console(rt, FakeAssistant(engine, fake_voice), input_fn=lambda _: next(feed), output=out.append).run()
    return "\n".join(out), fake_voice


def test_console_listen_runs_the_spoken_request(make_kernel, audit, fake_platform):
    text, fv = console_session(make_kernel, audit, fake_platform, ["/listen", "/quit"],
                               [heard("kitne baje hain", "कितने बजे हैं", language="hi")])
    assert 'heard: "कितने बजे हैं" -> "kitne baje hain"' in text
    assert "(hi, confidence 0.90" in text
    assert "[done] get_datetime" in text  # fast path, through the kernel


def test_enter_on_an_empty_line_is_push_to_talk(make_kernel, audit, fake_platform):
    text, fv = console_session(make_kernel, audit, fake_platform, ["", "/quit"], [heard("What time is it?")])
    assert fv.calls == 1 and "[done] get_datetime" in text


def test_voice_errors_are_shown_and_the_console_keeps_running(make_kernel, audit, fake_platform):
    from nexus.ai.voice.base import ListeningCancelled, NoSpeech, STTUnavailable
    from nexus.system.audio.errors import NoMicrophone

    text, fv = console_session(make_kernel, audit, fake_platform, ["/l", "/l", "/l", "", "/tools", "/quit"], [
        NoMicrophone("No microphone is connected, so voice input is off for now (typing still works)."),
        NoSpeech("I didn't hear anything in 6 seconds."), ListeningCancelled("Cancelled."),
        STTUnavailable("I can't transcribe speech right now: model not downloaded")])
    assert "(voice) No microphone is connected" in text
    assert "(voice) I didn't hear anything" in text and "(cancelled)" in text
    assert "(voice) I can't transcribe speech right now" in text
    assert "get_system_info" in text  # still working afterwards


def test_unexpected_voice_failure_does_not_end_the_console(make_kernel, audit, fake_platform):
    text, fv = console_session(make_kernel, audit, fake_platform, ["/listen", "/tools", "/quit"],
                               [OSError("PortAudio: unanticipated host error")])
    assert "Voice input failed unexpectedly (OSError" in text and "Typing still works" in text
    assert "get_system_info" in text


def test_console_without_voice_explains(make_kernel, audit, fake_platform):
    kernel = make_kernel()
    rt = Runtime(settings=None, platform=fake_platform, kernel=kernel, registry=kernel.registry,
                 policy=kernel.policy, audit=audit)
    feed, out = iter(["/listen", "", "/mic", "/quit"]), []
    Console(rt, None, input_fn=lambda _: next(feed), output=out.append).run()
    text = "\n".join(out)
    assert text.count("Voice input isn't available") == 2  # empty Enter does nothing without voice


def test_mic_command(make_kernel, audit, fake_platform):
    text, _ = console_session(make_kernel, audit, fake_platform, ["/mic", "/mic use nothing", "/mic auto", "/quit"],
                              [])
    assert "* Microphone (USB)" in text and "No audio input matches 'nothing'" in text
    assert "Choosing the microphone automatically - now Microphone (USB)." in text
