"""Voice output on the real audio output of this PC (pytest -m speaker). You will hear two short sentences.

Uses whatever the Windows default output is at the time - nothing is hard-coded.
With no output device the tests are skipped as PENDING.
"""

from __future__ import annotations

import sys
import time

import pytest

from nexus.ai.voice.kokoro import KOKORO_FILES, MODELS_DIR, KokoroTTS
from nexus.ai.voice.speaker import Speaker
from nexus.core.resources import ResourceManager
from nexus.system.audio.playback import PortAudioSink

pytestmark = [pytest.mark.speaker, pytest.mark.skipif(sys.platform != "win32", reason="Windows audio devices"),
              pytest.mark.skipif(not all((MODELS_DIR / KOKORO_FILES[k][0]).is_file() for k in ("fp32", "voices")),
                                 reason="run scripts/get_tts_model.py fp32 first")]


@pytest.fixture(scope="module")
def speaker():
    from nexus.system.platform.windows import WindowsPlatform

    plat = WindowsPlatform()
    sink = PortAudioSink(plat.audio_outputs, prepare_thread=plat.prepare_audio_thread)
    ready, device = sink.ready()
    if not ready:
        pytest.skip(f"PENDING - {device}")
    tts = KokoroTTS("fp32", threads=4)
    sp = Speaker(tts, ResourceManager(), sink=sink)
    sp.device = device
    yield sp
    sp.stop()
    tts.unload()


def test_H01_speaks_through_the_default_output(speaker):
    assert speaker.say("This is Nexus, testing the speakers.")
    last = speaker.last
    print(f"\n{speaker.device}: first sound after {last['first_audio_s']} s "
          f"(load {last.get('load_s', 0)} s), done in {last['total_s']} s")
    assert last["first_audio_s"] is not None and not last["stopped"]


def test_H02_stops_at_once(speaker):
    speaker.say_async("I will now read a long answer. " * 4)
    deadline = time.time() + 10
    while speaker.state != "playing" and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.5)
    started = time.perf_counter()
    assert speaker.stop()
    stop_s = time.perf_counter() - started
    assert speaker.wait(5)
    print(f"\nsilent {stop_s * 1000:.0f} ms after stop()")
    assert stop_s < 0.3


def test_H03_plays_from_background_threads_after_the_main_thread_used_audio():
    """The microphone code starts PortAudio on the main thread; speech plays on its own thread.
    Without COM on that thread, WASAPI refused to start the stream (measured). Silent audio."""
    import threading

    from nexus.ai.voice.base import AudioClip
    from nexus.system.platform.windows import WindowsPlatform

    plat = WindowsPlatform()
    sink = PortAudioSink(plat.audio_outputs, prepare_thread=plat.prepare_audio_thread)
    if not sink.ready()[0]:
        pytest.skip("PENDING - no audio output")
    silence = AudioClip(bytes(2) * 4800, 24000)
    results = []

    def play():
        try:
            sink.play(silence).wait(3)
            results.append("ok")
        except Exception as exc:
            results.append(str(exc))

    play()  # main thread first
    for _ in range(3):
        t = threading.Thread(target=play)
        t.start()
        t.join()
    assert results == ["ok"] * 4, results
