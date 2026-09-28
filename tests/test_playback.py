"""Playing speech on the default output: device choice, fallbacks, stop, device changes, no output."""

from __future__ import annotations

import sys
import threading
import time
from array import array

import pytest

from nexus.ai.voice.base import AudioClip
from nexus.system.audio.playback import AudioOutputError, NoAudioOutput, PortAudioSink, resample_pcm16
from nexus.system.platform.base import AudioOutput

CLIP = AudioClip(b"\x10\x00" * 2400, 24000)  # 0.1 s


class FakeOutStream:
    def __init__(self, sd, **kw):
        self.sd, self.kw = sd, kw
        self.started = self.aborted = self.closed = False
        sd.streams.append(self)
        reason = sd.refuse(kw)
        if reason:
            raise sd.PortAudioError(reason)

    def start(self):
        self.started = True
        if self.sd.run_callbacks:
            threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        buf = bytearray(2 * 240)
        while not self.aborted:
            try:
                self.kw["callback"](memoryview(buf), 240, None, None)
                self.sd.heard.extend(buf)
            except self.sd.CallbackStop:
                self.kw["finished_callback"]()
                return
            time.sleep(0.001)

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


class FakeSD:
    class PortAudioError(Exception):
        pass

    class CallbackStop(Exception):
        pass

    def __init__(self, devices, refuse=lambda kw: None, run_callbacks=True, default_output=None):
        self.devices = devices  # [(host api, name, output channels, rate)]
        self.refuse = refuse
        self.run_callbacks = run_callbacks
        self.streams: list[FakeOutStream] = []
        self.heard = bytearray()
        self.reinit = 0
        idx = default_output if default_output is not None else (0 if devices else -1)
        self.default = type("D", (), {"device": [None, idx]})()

    def query_hostapis(self):
        apis = {}
        for i, (api, *_r) in enumerate(self.devices):
            apis.setdefault(api, []).append(i)
        return [{"name": a, "devices": idx} for a, idx in apis.items()]

    def query_devices(self, index):
        api, name, channels, rate = self.devices[index]
        return {"name": name, "max_output_channels": channels, "default_samplerate": rate}

    def RawOutputStream(self, **kw):
        return FakeOutStream(self, **kw)

    def WasapiSettings(self, auto_convert=False):
        return ("wasapi", auto_convert)

    def _terminate(self):
        self.reinit += 1

    def _initialize(self):
        pass


SPEAKERS = AudioOutput("{usb}", "Speakers (AB13X USB Audio)", form_factor="speakers", bus="usb", default=True)
MONITOR = AudioOutput("{hdmi}", "VG270 M3 (NVIDIA High Definition Audio)", form_factor="display", bus="builtin")


def sink_with(sd, outputs, **kw) -> PortAudioSink:
    sink = PortAudioSink(lambda: list(outputs), **kw)
    sink._sd = sd
    return sink


def test_plays_on_the_windows_default_output_through_wasapi():
    sd = FakeSD([("MME", "Speakers (AB13X USB Audio)", 2, 44100), ("Windows WASAPI", "VG270 M3 (NVIDIA High Definition Audio)", 2, 48000),
                 ("Windows WASAPI", "Speakers (AB13X USB Audio)", 2, 48000)])
    sink = sink_with(sd, [MONITOR, SPEAKERS])
    playback = sink.play(CLIP)
    assert playback.wait(2)
    kw = sd.streams[-1].kw
    assert kw["device"] == 2 and kw["samplerate"] == 24000 and kw["channels"] == 1
    assert kw["extra_settings"] == ("wasapi", True)
    assert bytes(sd.heard[:len(CLIP.pcm16)]) == CLIP.pcm16  # exactly the clip, then silence
    assert playback.started_at is not None and sink.ready() == (True, "Speakers (AB13X USB Audio)")


def test_no_default_uses_any_active_output_and_ignores_unplugged_ones():
    unplugged = AudioOutput("{bt}", "Headphones (AirPods Pro)", state="unplugged", default=True)
    sd = FakeSD([("Windows WASAPI", MONITOR.name, 2, 48000)])
    sink = sink_with(sd, [unplugged, MONITOR])
    assert sink.current() == MONITOR
    assert sink.play(CLIP).wait(2)


def test_no_output_device_at_all():
    sink = sink_with(FakeSD([]), [AudioOutput("{bt}", "Headphones", state="unplugged")])
    assert sink.ready() == (False, "no audio output device is connected")
    with pytest.raises(NoAudioOutput) as err:
        sink.play(CLIP)
    assert "text still works" in str(err.value)


def test_falls_back_to_the_devices_own_rate():
    refused = lambda kw: "Invalid sample rate" if kw["samplerate"] == 24000 else None  # noqa: E731
    sd = FakeSD([("Windows WASAPI", SPEAKERS.name, 2, 48000)], refuse=refused)
    playback = sink_with(sd, [SPEAKERS]).play(CLIP)
    assert playback.wait(2) and sd.streams[-1].kw["samplerate"] == 48000


def test_every_format_refused_is_an_error_not_a_crash():
    sd = FakeSD([("Windows WASAPI", SPEAKERS.name, 2, 48000)], refuse=lambda kw: "Device unavailable [-9985]")
    with pytest.raises(AudioOutputError) as err:
        sink_with(sd, [SPEAKERS]).play(CLIP)
    assert "Device unavailable" in str(err.value)


def test_stop_is_immediate_and_a_new_play_stops_the_old_one():
    sd = FakeSD([("Windows WASAPI", SPEAKERS.name, 2, 48000)])
    sink = sink_with(sd, [SPEAKERS])
    long = AudioClip(b"\x10\x00" * 24000 * 20, 24000)  # 20 s
    first = sink.play(long)
    time.sleep(0.02)
    started = time.perf_counter()
    sink.stop()
    assert time.perf_counter() - started < 0.1 and first.wait(1) and first.stopped
    assert sd.streams[0].aborted and sd.streams[0].closed
    second = sink.play(long)
    sink.play(CLIP)  # replaces the one still playing
    assert second.stopped


def test_default_output_change_rereads_devices_but_never_while_recording():
    outputs = [SPEAKERS]
    recording = {"now": False}
    sd = FakeSD([("Windows WASAPI", SPEAKERS.name, 2, 48000), ("Windows WASAPI", MONITOR.name, 2, 48000)])
    sink = sink_with(sd, outputs, can_refresh=lambda: not recording["now"])
    sink.play(CLIP).wait(2)
    outputs[:] = [AudioOutput(MONITOR.id, MONITOR.name, default=True)]  # user switched output in Windows
    recording["now"] = True
    sink.play(CLIP).wait(2)
    assert sd.reinit == 0  # not while a microphone records
    recording["now"] = False
    outputs[:] = [SPEAKERS]
    sink.play(CLIP).wait(2)
    assert sd.reinit == 1 and sd.streams[-1].kw["device"] == 0


def test_output_connected_after_start_is_found_by_rereading():
    sd = FakeSD([])
    sink = sink_with(sd, [SPEAKERS])
    original = sd._initialize

    def plug():
        sd.devices = [("Windows WASAPI", SPEAKERS.name, 2, 48000)]
        original()

    sd._initialize = plug
    assert sink.play(CLIP).wait(2) and sd.reinit == 1


def test_device_that_stops_playing_is_reported():
    sd = FakeSD([("Windows WASAPI", SPEAKERS.name, 2, 48000)], run_callbacks=False)  # unplugged: no callbacks
    sink = sink_with(sd, [SPEAKERS])
    clock = {"t": 0.0}

    def tick():
        clock["t"] += 0.5
        return clock["t"]

    sink._clock = tick
    playback = sink.play(CLIP)
    assert playback.wait(10)
    assert "disconnected" in playback.error


def test_every_playing_thread_is_prepared():
    # Measured on Windows: WASAPI playback from a thread without COM fails to start, so the
    # platform's prepare_audio_thread runs on each thread that plays (it is idempotent per thread).
    prepared = []
    sd = FakeSD([("Windows WASAPI", SPEAKERS.name, 2, 48000)])
    sink = sink_with(sd, [SPEAKERS], prepare_thread=lambda: prepared.append(threading.current_thread().name))
    sink.play(CLIP).wait(2)
    t = threading.Thread(target=lambda: sink.play(CLIP).wait(2), name="nexus-speaker")
    t.start()
    t.join()
    assert prepared == ["MainThread", "nexus-speaker"]


def test_resampling():
    samples = array("h", [0, 1000, 2000, 3000])
    up = array("h")
    up.frombytes(resample_pcm16(samples.tobytes(), 24000, 48000))
    assert len(up) == 8 and up[0] == 0 and up[2] == 1000 and up[1] == 500
    assert resample_pcm16(b"", 24000, 48000) == b""
    assert resample_pcm16(samples.tobytes(), 24000, 24000) == samples.tobytes()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Core Audio")
def test_windows_lists_only_playback_devices():
    from nexus.system.platform.windows import WindowsPlatform

    plat = WindowsPlatform()
    outputs = plat.audio_outputs()
    inputs = plat.audio_inputs()
    assert all(o.state in ("active", "disabled", "unplugged") for o in outputs)
    assert sum(o.default for o in outputs) <= 1
    assert not {o.id for o in outputs} & {i.id for i in inputs}  # a microphone is never an output
    assert all("Microphone" not in o.name and "Stereo Mix" not in o.name for o in outputs)
