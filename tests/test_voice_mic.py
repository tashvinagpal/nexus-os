"""The real microphone path on this PC (pytest -m mic).

Uses whatever physical microphone is connected at the time - none is hard-coded.
With no usable microphone the tests are skipped as PENDING (not passed).

M01 needs no speaking: it opens the chosen microphone and checks audio arrives.
M02 asks you to speak: set NEXUS_MIC_SPEAK=1 and say "What time is it?" when prompted.
For Hindi/Hinglish/Punjabi/Haryanvi, run scripts/voice_check.py.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from nexus.system.audio.capture import PortAudioCapture
from nexus.system.audio.devices import MicrophoneManager

pytestmark = [pytest.mark.mic, pytest.mark.skipif(sys.platform != "win32", reason="Windows audio devices")]


@pytest.fixture(scope="module")
def microphones():
    from nexus.system.platform.windows import WindowsPlatform

    plat = WindowsPlatform()
    if plat.microphone_allowed() is False:
        pytest.fail("microphone access is blocked in Windows privacy settings")
    manager = MicrophoneManager(plat.audio_inputs)
    scan = manager.scan()
    if scan.chosen is None:
        ignored = "; ".join(f"{d.name}: {why}" for d, why in scan.ignored) or "no audio inputs at all"
        pytest.skip(f"PENDING - no usable physical microphone is connected ({ignored})")
    return manager, scan.chosen


def test_M01_chosen_microphone_opens_and_delivers_audio(microphones):
    manager, mic = microphones
    stream = PortAudioCapture().open(mic)
    blocks = []
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            block = stream.read(0.2)
            if block:
                blocks.append(block)
    finally:
        stream.close()
    seconds = len(b"".join(blocks)) / 2 / stream.sample_rate
    print(f"\n{mic.name}: {seconds:.2f} s of audio in 2 s at {stream.sample_rate} Hz")
    assert seconds >= 1.5, f"{mic.name} delivered only {seconds:.2f} s of audio in 2 s"


@pytest.mark.skipif(os.environ.get("NEXUS_MIC_SPEAK") != "1", reason="set NEXUS_MIC_SPEAK=1 to speak into the mic")
def test_M02_spoken_request_through_the_whole_pipeline(microphones, make_kernel):
    from nexus.ai.conversation.engine import ConversationEngine
    from nexus.ai.conversation.types import ActionReport, UserInput
    from nexus.ai.providers.selection import SelectionContext
    from nexus.ai.router import FastPathRouter
    from nexus.ai.voice.listener import VoiceInput
    from nexus.ai.voice.stt import HybridSTT, STTOption
    from nexus.ai.voice.whisper import WhisperSTT
    from nexus.core.kernel import Status
    from nexus.core.resources import ResourceManager

    manager, mic = microphones
    rm = ResourceManager()
    small = WhisperSTT("small", ram_mb=560, threads=6)
    try:
        router = FastPathRouter()
        vi = VoiceInput(manager, PortAudioCapture(), HybridSTT([STTOption(small, small.spec, 1)], rm,
                                                               lambda: SelectionContext()), rm,
                        fast_path=lambda text: router.route(text) is not None)
        print(f'\n>>> Using {mic.name}. Say "What time is it?" now.')
        heard = vi.listen(on_status=lambda s: print(f"    ({s})"))
        print(f'    heard: "{heard.original}" ({heard.transcript.language}, {heard.confidence:.2f}), '
              f"transcribed in {vi.last_timings['transcribing_s']:.2f} s")
        class NoModel:
            def plan(self, *a, **k):
                raise AssertionError(f"not understood by the fast path: {heard.text!r}")

        events = []
        ConversationEngine(make_kernel(), NoModel(), router=router).handle(
            UserInput(heard.text, interface="voice", confidence=heard.confidence), events.append)
        report = [e for e in events if isinstance(e, ActionReport)][0]
        assert report.tool == "get_datetime" and report.status is Status.DONE
    finally:
        small.unload()
