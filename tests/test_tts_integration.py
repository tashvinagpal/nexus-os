"""Voice output inside NEXUS: keyboard/voice -> assistant -> speech, failures, interrupts, commands."""

from __future__ import annotations

import time

import pytest

from nexus.ai.conversation.engine import ConversationEngine
from nexus.ai.router import FastPathRouter
from nexus.ai.voice.base import SilentTTS, TTSError
from nexus.ai.voice.speaker import Speaker
from nexus.console import Console
from nexus.core.resources import ResourceManager
from nexus.core.runtime import Runtime
from tests.conftest import FakeProbe, ScriptedPlanner, plan
from tests.tts_fixtures import FakeSink, FakeVoice


class FakeAssistant:
    def __init__(self, engine, speaker, voice=None, speech_note="") -> None:
        self.engine = engine
        self.chat = type("Chat", (), {"last_choice": None})()
        self.speaker = speaker
        self.voice = voice
        self.voice_note = ""
        self.speech_note = speech_note


def session(make_kernel, audit, fake_platform, lines, *, planner=None, voice=None, sink=None, speaker=None,
            voice_input=None, speech_note=""):
    kernel = make_kernel()
    rt = Runtime(settings=None, platform=fake_platform, kernel=kernel, registry=kernel.registry,
                 policy=kernel.policy, audit=audit)
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0)
    voice = voice or FakeVoice()
    sink = sink or FakeSink()
    speaker = speaker or Speaker(voice, rm, sink=sink)
    engine = ConversationEngine(kernel, planner or ScriptedPlanner(plan("Hello! How can I help?")),
                                router=FastPathRouter(), resources=rm)
    out = []
    feed = iter(lines)

    def next_line(_):
        speaker.wait(3)  # let the previous reply finish, like a person listening
        return next(feed)

    Console(rt, FakeAssistant(engine, speaker, voice_input, speech_note), input_fn=next_line,
            output=out.append).run()
    return "\n".join(out), voice, sink, speaker


def spoken(voice):
    return [text for text, _, _ in voice.calls]


def test_keyboard_reply_is_spoken(make_kernel, audit, fake_platform):
    text, voice, sink, _ = session(make_kernel, audit, fake_platform, ["what time is it?", "/quit"])
    assert "nexus> It's" in text  # shown ...
    said = " ".join(spoken(voice))
    assert said.startswith("It's") and len(sink.played) >= 1  # ... and spoken


def test_conversation_reply_is_spoken_in_its_language(make_kernel, audit, fake_platform):
    planner = ScriptedPlanner(plan("Ekdum badhiya bhai! Tu bata?", language="hinglish"))
    text, voice, _, _ = session(make_kernel, audit, fake_platform, ["kaisa hai bhai", "/quit"], planner=planner)
    assert "nexus> Ekdum badhiya bhai! Tu bata?" in text        # the screen keeps the original
    assert voice.calls[0] == ("एकदम बढ़िया भाई!", "hi", None)    # speech gets the Hindi voice


def test_voice_input_reply_is_spoken(make_kernel, audit, fake_platform):
    from tests.test_voice_conversation import FakeVoice as FakeVoiceInput, heard

    fv = FakeVoiceInput([heard("What time is it?")])
    text, voice, _, _ = session(make_kernel, audit, fake_platform, ["/listen", "/quit"], voice_input=fv)
    assert 'heard: "What time is it?"' in text and spoken(voice)[0].startswith("It's")


def test_speech_failure_never_breaks_the_assistant(make_kernel, audit, fake_platform):
    voice = FakeVoice(fail=TTSError("Speech synthesis failed: engine crashed"))
    text, voice, _, speaker = session(make_kernel, audit, fake_platform, ["what time is it?", "hello", "/quit"],
                                      voice=voice)
    assert text.count("nexus>") == 2  # both replies shown
    assert "crashed" in speaker.problem


def test_a_broken_speaker_never_breaks_the_console(make_kernel, audit, fake_platform):
    class Broken(Speaker):
        def __call__(self, event):
            raise RuntimeError("audio driver exploded")

    broken = Broken(SilentTTS())
    text, *_ = session(make_kernel, audit, fake_platform, ["hello", "/tools", "/quit"], speaker=broken)
    assert "nexus> Hello!" in text and "get_system_info" in text


def test_no_output_device_keeps_text_working(make_kernel, audit, fake_platform):
    text, voice, _, speaker = session(make_kernel, audit, fake_platform, ["hello", "/tts", "/quit"],
                                      sink=FakeSink(device=None))
    assert "nexus> Hello!" in text and voice.calls == []
    assert "output: none - no audio output device is connected" in text


def test_typing_interrupts_speech(make_kernel, audit, fake_platform):
    sink = FakeSink(instant=False, play_seconds=5)
    kernel = make_kernel()
    rt = Runtime(settings=None, platform=fake_platform, kernel=kernel, registry=kernel.registry,
                 policy=kernel.policy, audit=audit)
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0)
    voice = FakeVoice()
    speaker = Speaker(voice, rm, sink=sink)
    engine = ConversationEngine(kernel, ScriptedPlanner(plan("A long answer. With two sentences.")), resources=rm)
    lines = iter(["tell me something", "/tools", "/quit"])
    stamps = []

    def next_line(_):
        time.sleep(0.2)  # the user types while NEXUS is still talking
        stamps.append(speaker.speaking)
        return next(lines)

    out = []
    Console(rt, FakeAssistant(engine, speaker), input_fn=next_line, output=out.append).run()
    assert stamps[1] is True                     # it was speaking when the next line came in
    assert sink.playbacks[0].stopped             # ... and that line stopped it
    assert len(sink.played) == 1                 # the second sentence was never played


def test_tts_commands(make_kernel, audit, fake_platform):
    text, voice, sink, speaker = session(
        make_kernel, audit, fake_platform,
        ["/tts", "/tts off", "hello", "/tts on", "/tts test", "/tts test Namaste bhai", "/stop", "/tts bogus", "/quit"])
    assert "voice output: on   provider: tts:kokoro@cpu" in text
    assert "Voice output off (text only)." in text and "Voice output on." in text
    assert "(spoke" in text and "first sound after" in text
    assert spoken(voice)[0] == "Hello, I'm Nexus." and "नमस्ते" not in text  # the screen isn't converted
    assert "(NEXUS wasn't speaking)" in text and "usage: /tts" in text


def test_off_means_nothing_is_spoken(make_kernel, audit, fake_platform):
    text, voice, *_ = session(make_kernel, audit, fake_platform, ["/tts off", "hello", "/quit"])
    assert "nexus> Hello!" in text and voice.calls == []


def test_voice_loads_while_nexus_thinks(make_kernel, audit, fake_platform):
    text, voice, *_ = session(make_kernel, audit, fake_platform, ["hello", "/quit"])
    assert voice.loads == 1  # preloaded (or loaded) once, used for the reply


# --- wiring ------------------------------------------------------------------------------------------

def test_voice_output_off_in_config(tmp_path):
    from nexus.core.assistant import _tts_provider
    from nexus.core.config import TTSSettings

    provider, note = _tts_provider(None, TTSSettings(enabled=False))
    assert provider is None and "turned off" in note


def test_missing_model_means_text_only(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from nexus.ai.voice import kokoro
    from nexus.core.assistant import _tts_provider
    from nexus.core.config import TTSSettings

    def unavailable(self):
        self.problem = "model not downloaded - run: scripts\\get_tts_model.py fp32"
        return False

    monkeypatch.setattr(kokoro.KokoroTTS, "available", unavailable)
    runtime = SimpleNamespace(settings=SimpleNamespace(data_dir=tmp_path))
    provider, note = _tts_provider(runtime, TTSSettings())
    assert provider is None and "get_tts_model.py" in note


def test_status_shows_voice_output(make_kernel, audit, fake_platform):
    text, *_ = session(make_kernel, audit, fake_platform, ["/tts", "/quit"], speech_note="")
    assert "output: Speakers (Fake USB Audio)" in text
