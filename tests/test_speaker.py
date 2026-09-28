"""The Speaker (TTS manager): lifecycle, chunking, stop, queueing, languages, failures, resources."""

from __future__ import annotations

import threading
import time

import pytest

from nexus.ai.conversation.types import Say, SayKind
from nexus.ai.voice.base import TTSError
from nexus.ai.voice.speaker import Speaker, SpeechSettings
from nexus.core.resources import ResourceManager, ResourceState
from nexus.system.audio.playback import AudioOutputError, NoAudioOutput
from tests.conftest import FakeClock, FakeProbe
from tests.tts_fixtures import FakeSink, FakeVoice


def build(*, sink=None, settings=None, voice=None, clock=None, **kw):
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0, **({"clock": clock} if clock else {}))
    voice = voice or FakeVoice(**kw)
    sink = sink or FakeSink()
    return Speaker(voice, rm, sink=sink, settings=settings), voice, sink, rm


def spoken(voice) -> list[str]:
    return [text for text, _, _ in voice.calls]


# --- lifecycle --------------------------------------------------------------------------------------

def test_model_loads_on_first_speech_and_is_reused():
    speaker, voice, sink, rm = build()
    assert not voice.loaded  # registered, not loaded
    assert speaker.say("Opening Google Chrome.")
    assert voice.loads == 1 and voice.loaded
    assert "load_s" in speaker.last  # the cold start is measured
    assert speaker.say("It's 3:42 PM.")
    assert voice.loads == 1 and "load_s" not in speaker.last  # warm


def test_idle_model_unloads_and_reloads_when_needed():
    clock = FakeClock()
    speaker, voice, sink, rm = build(clock=clock)
    speaker.say("Hello.")
    clock.now += 121  # idle_unload_s = 120
    assert rm.release_idle() == ["tts:kokoro@cpu"] and not voice.loaded
    assert speaker.say("Hello again.")
    assert voice.loads == 2


def test_load_failure_is_reported_and_recovered_from():
    speaker, voice, sink, rm = build(fail_load=True)
    assert speaker.say("Hello.") is False
    assert "damaged" in speaker.problem and sink.played == []
    voice.fail_load = False  # e.g. the model was downloaded again
    assert speaker.say("Hello.")
    assert speaker.problem == ""


def test_prepare_loads_in_the_background():
    speaker, voice, sink, rm = build()
    speaker.prepare()
    deadline = time.time() + 2
    while not voice.loaded and time.time() < deadline:
        time.sleep(0.01)
    assert voice.loaded and sink.played == []  # loaded, nothing said


def test_speaking_state_and_idle_after():
    states = []
    speaker, voice, sink, rm = build()
    original = voice.synthesize

    def watch(text, **kw):
        states.append(rm.state)
        return original(text, **kw)

    voice.synthesize = watch
    speaker.say("One. Two.")
    assert set(states) == {ResourceState.SPEAKING} and rm.state is ResourceState.IDLE and speaker.state == "idle"


# --- sentences ---------------------------------------------------------------------------------------

def test_long_reply_is_spoken_sentence_by_sentence_in_order():
    speaker, voice, sink, rm = build()
    speaker.say("Opening Google Chrome. It should be ready in a second. Tell me if it isn't.")
    assert spoken(voice)[0] == "Opening Google Chrome."
    assert len(sink.played) == len(voice.calls) >= 2


def test_the_next_sentence_is_synthesized_while_the_first_plays():
    speaker, voice, sink, rm = build(sink=FakeSink(instant=False, play_seconds=0.3))
    order = []
    original_play, original_synth = sink.play, voice.synthesize
    sink.play = lambda clip: (order.append("play"), original_play(clip))[1]
    voice.synthesize = lambda text, **kw: (order.append("synth"), original_synth(text, **kw))[1]
    speaker.say("First sentence here. Second sentence here.")
    assert order == ["synth", "play", "synth", "play"]


def test_markdown_and_links_are_not_read_out():
    speaker, voice, sink, rm = build()
    speaker.say("Sure! **Opening Chrome now.**\nhttps://example.com/some/long/path")
    assert " ".join(spoken(voice)) == "Sure! Opening Chrome now. example.com."


@pytest.mark.parametrize("text", ["", "   ", "***"])
def test_nothing_to_say_is_not_an_error(text):
    speaker, voice, sink, rm = build()
    assert speaker.say(text) is False and voice.calls == [] and not voice.loaded


def test_very_long_replies_are_capped():
    speaker, voice, sink, rm = build(settings=SpeechSettings(max_chars=200))
    speaker.say(" ".join(f"Sentence {i} of a very long reply." for i in range(100)))
    assert spoken(voice)[-1].endswith("The rest is on the screen.")
    assert sum(len(t) for t in spoken(voice)) < 300


# --- stop / interrupt -----------------------------------------------------------------------------------

def test_stop_silences_at_once_and_skips_the_rest():
    speaker, voice, sink, rm = build(sink=FakeSink(instant=False, play_seconds=5))
    speaker.say_async("First sentence. Second sentence. Third sentence.")
    deadline = time.time() + 2
    while not sink.playbacks and time.time() < deadline:
        time.sleep(0.01)
    started = time.perf_counter()
    assert speaker.stop() is True
    assert time.perf_counter() - started < 0.2
    assert speaker.wait(2)
    assert sink.playbacks[0].stopped and len(sink.played) == 1  # the rest was never played
    assert speaker.last["stopped"] is True


def test_speak_stop_speak_again_many_times():
    speaker, voice, sink, rm = build(sink=FakeSink(instant=False, play_seconds=2))
    for i in range(6):
        speaker.say_async(f"Reply number {i}. It is long enough to interrupt.")
        deadline = time.time() + 2
        while len(sink.playbacks) <= i and time.time() < deadline:
            time.sleep(0.005)
        assert speaker.stop()
        assert speaker.wait(2)
    assert len(sink.played) == 6 and all(p.stopped for p in sink.playbacks)
    assert voice.loads == 1  # stopping never kills the voice engine
    speaker.sink.instant = True
    assert speaker.say("Finally a whole reply.")


def test_stop_when_silent_and_stop_drops_the_queue():
    speaker, voice, sink, rm = build(sink=FakeSink(instant=False, play_seconds=1))
    assert speaker.stop() is False
    speaker.say_async("One.")
    speaker.say_async("Two.")
    speaker.say_async("Three.")
    speaker.stop()
    assert speaker.wait(2)
    assert "Three." not in spoken(voice) and "Two." not in spoken(voice)


def test_say_async_returns_immediately_and_replies_play_in_order():
    speaker, voice, sink, rm = build(sink=FakeSink(instant=False, play_seconds=0.05))
    started = time.perf_counter()
    for text in ("Launching GTA V.", "Starting it now.", "GTA V is running."):
        speaker(Say(SayKind.RESULT, text))
    assert time.perf_counter() - started < 0.1  # the console is never blocked
    assert speaker.wait(3)
    assert spoken(voice) == ["Launching GTA V.", "Starting it now.", "GTA V is running."]


# --- languages ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, language, provider_language, said", [
    ("Opening Google Chrome.", "en", "en", "Opening Google Chrome."),
    ("क्रोम खुल गया है।", "hi", "hi", "क्रोम खुल गया है।"),
    ("Haan bhai, ek second.", "hinglish", "hi", "हां भाई, एक second."),
    ("ਕਰ ਦਿਆਂ?", "pa", "hi", "कर दिआं?"),                   # Punjabi read with the Hindi voice (approximate)
    ("क्रोम खुल गया है।", "en", "hi", "क्रोम खुल गया है।"),  # the script decides, not the tag
])
def test_voice_follows_the_reply_language(text, language, provider_language, said):
    speaker, voice, sink, rm = build()
    speaker.say(text, language=language)
    assert voice.calls[0][0] == said and voice.calls[0][1] == provider_language


def test_romanized_hinglish_can_use_the_english_voice():
    speaker, voice, sink, rm = build(settings=SpeechSettings(romanized="english"))
    speaker.say("Haan bhai, ek second.", language="hinglish")
    assert voice.calls[0][:2] == ("Haan bhai, ek second.", "en")


def test_installed_app_names_stay_english_in_hinglish():
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0)
    voice, sink = FakeVoice(), FakeSink()
    speaker = Speaker(voice, rm, sink=sink, names=lambda: ["valorant"])
    speaker.say("valorant chala raha hoon", language="hinglish")
    assert voice.calls[0][0] == "valorant चला रहा हूं."


# --- resources and the user -------------------------------------------------------------------------------------

def test_quiet_in_game_mode_and_the_model_is_freed():
    speaker, voice, sink, rm = build()
    speaker.say("Hello.")
    rm.enter_game_mode()
    assert speaker.say("GTA V is running.") is False
    assert not voice.loaded and sink.played[-1:] and len(sink.played) == 1


def test_game_mode_can_keep_talking_if_configured():
    speaker, voice, sink, rm = build(settings=SpeechSettings(game_mode="speak"))
    rm.enter_game_mode()
    assert speaker.say("GTA V is running.")


def test_never_talks_over_the_microphone():
    speaker, voice, sink, rm = build()
    done = threading.Event()
    with rm.busy(ResourceState.LISTENING):
        speaker.say_async("GTA V is running.")  # a background result arrives while you speak
        time.sleep(0.3)
        assert sink.played == []
        done.set()
    assert speaker.wait(3) and len(sink.played) == 1


def test_overlapping_activities_on_two_threads_never_leave_a_stale_state():
    speaker, voice, sink, rm = build(sink=FakeSink(instant=False, play_seconds=0.3))
    speaker.say_async("Speaking in the background.")
    time.sleep(0.05)
    with rm.busy(ResourceState.THINKING):  # the user typed something meanwhile
        assert rm.activity is ResourceState.THINKING
        time.sleep(0.4)  # the speech ends inside this block
    assert speaker.wait(2)
    assert rm.activity is ResourceState.IDLE


def test_disabled_speaks_nothing():
    speaker, voice, sink, rm = build(settings=SpeechSettings(enabled=False))
    speaker(Say(SayKind.REPLY, "Hello."))
    assert speaker.say("Hello.") is False and voice.calls == []


# --- failures are never fatal ---------------------------------------------------------------------------------------

def test_no_audio_output():
    speaker, voice, sink, rm = build(sink=FakeSink(device=None))
    assert speaker.say("Hello.") is False
    assert speaker.problem == "no audio output device is connected"
    assert voice.calls == [] and not voice.loaded  # nothing synthesized for nobody


@pytest.mark.parametrize("error", [NoAudioOutput("No audio output device is connected"),
                                   AudioOutputError("Couldn't play on Speakers: device busy")])
def test_output_errors_while_starting_to_play(error):
    speaker, voice, sink, rm = build()
    sink.error_on_play = error
    assert speaker.say("Hello.") is False and str(error) in speaker.problem


def test_device_lost_while_playing():
    speaker, voice, sink, rm = build()
    sink.error_during_play = "the audio output stopped playing (disconnected?)"
    assert speaker.say("First. Second.") is False
    assert "disconnected" in speaker.problem and len(sink.played) == 1


def test_synthesis_failure():
    speaker, voice, sink, rm = build(fail=TTSError("Speech synthesis failed: engine crashed"))
    assert speaker.say("Hello.") is False and "crashed" in speaker.problem


def test_the_speaker_thread_survives_unexpected_errors():
    speaker, voice, sink, rm = build(fail=RuntimeError("bug in a provider"))
    speaker.say_async("Hello.")
    assert speaker.wait(2)
    voice.fail = None
    speaker.say_async("Still here.")
    assert speaker.wait(2) and spoken(voice)[-1] == "Still here."


def test_status():
    speaker, voice, sink, rm = build()
    speaker.say("Hello.")
    st = speaker.status()
    assert st["provider"] == "tts:kokoro@cpu" and st["loaded"] and st["output"] == "Speakers (Fake USB Audio)"
    assert st["last"]["first_audio_s"] is not None and st["state"] == "idle"


# --- the original interface still works (acceptance R4) -------------------------------------------------------------

def test_speak_only_providers_are_called_directly():
    from nexus.ai.voice.base import SilentTTS

    speaker = Speaker(SilentTTS())
    assert speaker.say("Hello.") is True
    speaker(Say(SayKind.REPLY, "Hello."))  # returns at once, no thread


# --- safety ---------------------------------------------------------------------------------------------------------------

def test_speech_can_never_run_anything():
    """The Speaker only receives text that was already decided and shown; it has no path to the kernel."""
    import ast
    import inspect

    import nexus.ai.voice.hinglish as hinglish
    import nexus.ai.voice.kokoro as kokoro
    import nexus.ai.voice.speaker as speaker_module
    import nexus.ai.voice.speech_text as speech_text
    import nexus.system.audio.playback as playback

    for module in (speaker_module, speech_text, hinglish, kokoro, playback):
        tree = ast.parse(inspect.getsource(module))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert not any(name and (name.startswith("nexus.core.kernel") or name.startswith("nexus.tools")
                                 or name == "subprocess" or name.startswith("nexus.ai.planner")) for name in imported), \
            module.__name__
    speaker, voice, sink, rm = build()
    speaker.say("Delete the folder C:\\Users\\me\\Documents now.")
    assert spoken(voice) == ["Delete the folder Documents now."]  # said, never done
