"""VoiceInput: microphone -> utterance -> text, with every failure handled (no hardware needed)."""

from __future__ import annotations

import threading

import pytest

from nexus.ai.providers.selection import SelectionContext
from nexus.ai.voice.base import (ListeningCancelled, NoSpeech, STTFailed, STTTimeout, STTUnavailable, Transcript,
                                 Unintelligible)
from nexus.ai.voice.listener import VoiceInput
from nexus.ai.voice.recorder import VADConfig
from nexus.ai.voice.stt import HybridSTT, STTOption
from nexus.core.resources import ResourceManager, ResourceState
from nexus.system.audio.devices import MicrophoneManager
from nexus.system.audio.errors import MicrophoneDenied, MicrophoneLost, MicrophoneUnavailable, NoMicrophone
from tests.conftest import FakeProbe
from tests.voice_fixtures import FakeCapture, FakeDevices, FakeStream, FakeSTT, TickClock, mic, silence, voice

SPEECH = silence(500) + voice(1200) + silence(1200)


def build(devices: FakeDevices, capture: FakeCapture, *results, permission=None, stt=None, **kw):
    rm = ResourceManager(FakeProbe(), snapshot_ttl=0)
    provider = stt or FakeSTT("base", results=list(results))
    hybrid = HybridSTT([STTOption(provider, provider.spec, 1, game_mode_ok=True)], rm, lambda: SelectionContext())
    vi = VoiceInput(MicrophoneManager(devices), capture, hybrid, rm, permission=permission, **kw)
    return vi, provider, rm


def test_heard_text_with_details():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    vi, stt, _ = build(devices, capture, Transcript("What time is it?", "en", 0.93))
    statuses = []
    heard = vi.listen(on_status=statuses.append)
    assert heard.text == "What time is it?" and heard.original == heard.text
    assert heard.confidence == 0.93 and heard.microphone == "Microphone (USB)"
    assert "listening with Microphone (USB)" in statuses and "transcribing" in statuses
    assert stt.calls[0][0].seconds > 1.0  # the recorded clip was transcribed
    assert set(vi.last_timings) == {"recording_s", "transcribing_s", "audio_s"}


def test_state_is_listening_while_recording_and_idle_after():
    states = []
    devices = FakeDevices(mic("Microphone (USB)"))

    class Watch(FakeStream):
        def read(self, timeout):
            states.append(rm.state)
            return super().read(timeout)

    capture = FakeCapture({"Microphone (USB)": Watch(SPEECH)})
    vi, stt, rm = build(devices, capture)
    vi.listen()
    assert set(states) == {ResourceState.LISTENING} and rm.state is ResourceState.IDLE


# --- microphones -----------------------------------------------------------------------

def test_no_microphone_is_reported_and_nothing_is_opened():
    devices = FakeDevices(mic("Stereo Mix", loopback=True), mic("Microphone (e2eSoft iVCam)", bus="virtual"))
    capture = FakeCapture()
    vi, stt, _ = build(devices, capture)
    with pytest.raises(NoMicrophone) as err:
        vi.listen()
    message = str(err.value)
    assert "No microphone is connected" in message and "typing still works" in message
    assert "iVCam" in message and "/mic use" in message and "no restart needed" in message
    assert capture.opened == [] and stt.loads == 0


def test_permission_denied_by_privacy_settings():
    devices = FakeDevices(mic("Microphone (USB)"))
    vi, _, _ = build(devices, FakeCapture(), permission=lambda: False)
    with pytest.raises(MicrophoneDenied) as err:
        vi.listen()
    assert "Privacy & security > Microphone" in str(err.value)


def test_permission_denied_when_opening():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture()
    capture.deny("Microphone (USB)")
    vi, _, _ = build(devices, capture)
    with pytest.raises(MicrophoneDenied):
        vi.listen()


def test_microphone_that_wont_open_falls_back_to_the_next():
    devices = FakeDevices(mic("Microphone (USB)", comms=True), mic("Headset (BT)", form_factor="headset",
                                                                   bus="bluetooth"))
    capture = FakeCapture({"Headset (BT)": FakeStream(SPEECH)})
    capture.refused["Microphone (USB)"] = MicrophoneUnavailable("Couldn't open Microphone (USB): device busy")
    vi, _, _ = build(devices, capture)
    statuses = []
    heard = vi.listen(on_status=statuses.append)
    assert heard.microphone == "Headset (BT)"
    assert capture.opened == ["Microphone (USB)", "Headset (BT)"]
    assert any("couldn't open Microphone (USB)" in s for s in statuses)
    # remembered for a while: the next turn goes straight to the working one
    capture.streams["Headset (BT)"] = FakeStream(SPEECH)
    vi.listen()
    assert capture.opened[-1] == "Headset (BT)" and capture.opened.count("Microphone (USB)") == 1


def test_no_microphone_can_be_opened():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture()  # refuses everything
    vi, _, _ = build(devices, capture)
    with pytest.raises(MicrophoneUnavailable) as err:
        vi.listen()
    assert "No microphone could be opened" in str(err.value)


def test_disconnect_before_speaking_switches_microphone_and_keeps_listening():
    devices = FakeDevices(mic("Microphone (USB)", comms=True), mic("Microphone (Realtek)", bus="builtin"))
    dying = FakeStream(silence(300))  # unplugged 0.3 s in
    capture = FakeCapture({"Microphone (USB)": dying, "Microphone (Realtek)": FakeStream(SPEECH)})
    vi, _, _ = build(devices, capture)
    vi.recorder._clock = TickClock(0.5)

    def unplug_on_loss(timeout, _read=dying.read):
        block = _read(timeout)
        if block is None:
            devices.unplug("Microphone (USB)")
        return block

    dying.read = unplug_on_loss
    statuses = []
    heard = vi.listen(on_status=statuses.append)
    assert heard.microphone == "Microphone (Realtek)"
    assert any("disconnected - listening with Microphone (Realtek)" in s for s in statuses)
    assert dying.closed


def test_disconnect_mid_sentence_asks_to_repeat_with_the_next_microphone():
    devices = FakeDevices(mic("Microphone (USB)", comms=True), mic("Microphone (Realtek)", bus="builtin"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(silence(400) + voice(800))})
    vi, stt, _ = build(devices, capture)
    vi.recorder._clock = TickClock(0.5)
    with pytest.raises(MicrophoneLost) as err:
        vi.listen()
    assert "stopped while you were speaking" in str(err.value)
    assert "Switched to Microphone (Realtek)" in str(err.value)
    assert stt.calls == []  # half a sentence is never acted on


def test_disconnect_with_no_other_microphone():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(silence(300))})
    vi, _, _ = build(devices, capture)
    vi.recorder._clock = TickClock(0.5)
    with pytest.raises(NoMicrophone):
        vi.listen()


def test_microphone_plugged_in_after_start_is_used_without_restart():
    devices = FakeDevices()
    capture = FakeCapture({"Headset (AirPods)": FakeStream(SPEECH)})
    vi, _, _ = build(devices, capture)
    with pytest.raises(NoMicrophone):
        vi.listen()
    devices.plug(mic("Headset (AirPods)", form_factor="headset", bus="bluetooth"))
    assert vi.listen().microphone == "Headset (AirPods)"


# --- speech ---------------------------------------------------------------------------------

def test_silence_is_no_speech_and_nothing_is_transcribed():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(silence(8000))})
    vi, stt, _ = build(devices, capture, vad=VADConfig(no_speech_timeout_s=2))
    with pytest.raises(NoSpeech):
        vi.listen()
    assert stt.calls == []


@pytest.mark.parametrize("transcript, error", [
    (Transcript("", "en", None), NoSpeech),
    (Transcript("[Music]", "en", 0.9), NoSpeech),
    (Transcript("Thanks for watching!", "en", 0.5, no_speech_probability=0.6), NoSpeech),
    (Transcript("uh open the the", "en", 0.31), Unintelligible),
])
def test_empty_noise_or_unclear_speech_is_not_acted_on(transcript, error):
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    vi, _, _ = build(devices, capture, transcript)
    with pytest.raises(error) as err:
        vi.listen()
    if error is Unintelligible:
        assert "uh open the the" in str(err.value)  # shows what it heard, doesn't act on it


def test_a_confidently_heard_thank_you_is_kept():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    vi, _, _ = build(devices, capture, Transcript("Thank you.", "en", 0.92, no_speech_probability=0.02))
    assert vi.listen().text == "Thank you."


@pytest.mark.parametrize("error", [STTTimeout("too slow"), STTFailed("crashed")])
def test_transcription_failures_surface_as_voice_errors(error):
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    vi, _, rm = build(devices, capture, error)
    with pytest.raises(type(error)):
        vi.listen()
    assert rm.state is ResourceState.IDLE


def test_no_stt_model_is_reported_before_recording():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    stt = FakeSTT("small", is_available=False)
    stt.problem = "model not downloaded"
    vi, _, _ = build(devices, capture, stt=stt)
    with pytest.raises(STTUnavailable) as err:
        vi.listen()
    assert "model not downloaded" in str(err.value)
    assert capture.opened == []  # you weren't asked to speak for nothing


def test_insufficient_memory_is_reported():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    rm = ResourceManager(FakeProbe(ram_free=2100), snapshot_ttl=0)
    stt = FakeSTT("small", ram_mb=560)
    hybrid = HybridSTT([STTOption(stt, stt.spec, 1)], rm, lambda: SelectionContext())
    vi = VoiceInput(MicrophoneManager(devices), capture, hybrid, rm)
    with pytest.raises(STTUnavailable) as err:
        vi.listen()
    assert "RAM" in str(err.value)


def test_cancel_while_listening():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(b"", tail=silence(600), realtime=True)})
    vi, stt, rm = build(devices, capture)
    cancel = threading.Event()
    threading.Timer(0.15, cancel.set).start()
    with pytest.raises(ListeningCancelled):
        vi.listen(cancel=cancel)
    assert stt.calls == [] and rm.state is ResourceState.IDLE
    assert capture.streams["Microphone (USB)"].closed


def test_ctrl_c_while_listening_becomes_a_cancel():
    devices = FakeDevices(mic("Microphone (USB)"))

    class Interrupted(FakeStream):
        def read(self, timeout):
            raise KeyboardInterrupt

    capture = FakeCapture({"Microphone (USB)": Interrupted(b"")})
    vi, _, rm = build(devices, capture)
    with pytest.raises(ListeningCancelled):
        vi.listen()
    assert capture.streams["Microphone (USB)"].closed and rm.state is ResourceState.IDLE


# --- languages -------------------------------------------------------------------------------

@pytest.mark.parametrize("original, language, acted", [
    ("कितने बजे हैं", "hi", "kitne baje hain"),
    ("नेक्सस, जीटीए चला दे", "hi", "jitie chala de"),
    ("ਕਿੰਨੇ ਵਜੇ ਨੇ", "pa", "kitne baje ne"),     # Punjabi forms -> the Hinglish the planner knows
    ("GTA chala de bhai", "en", "GTA chala de bhai"),
    ("Chrome खोल दो", "hi", "Chrome khol do"),
])
def test_hindi_and_punjabi_scripts_are_romanized_for_nexus(original, language, acted):
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    vi, _, _ = build(devices, capture, Transcript(original, language, 0.8))
    heard = vi.listen()
    assert heard.text == acted and heard.original == original


def test_romanizing_can_be_turned_off():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    vi, _, _ = build(devices, capture, Transcript("कितने बजे हैं", "hi", 0.8), romanize=False)
    assert vi.listen().text == "कितने बजे हैं"


def test_configured_language_is_passed_as_a_hint():
    devices = FakeDevices(mic("Microphone (USB)"))
    capture = FakeCapture({"Microphone (USB)": FakeStream(SPEECH)})
    vi, stt, _ = build(devices, capture, language="hi")
    vi.listen()
    assert stt.calls[0][1] == "hi"
