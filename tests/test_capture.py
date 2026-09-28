"""PortAudio capture wrapper: device matching, format fallbacks, downmix, errors, re-scan.

A fake `sounddevice` module stands in for the hardware; the last test uses the
real PortAudio but never opens a device.
"""

from __future__ import annotations

from array import array

import pytest

from nexus.system.audio.capture import PortAudioCapture
from nexus.system.audio.errors import MicrophoneDenied, MicrophoneUnavailable
from tests.voice_fixtures import mic


class FakeRawStream:
    def __init__(self, sd, **kw) -> None:
        self.kw = kw
        self.started = self.stopped = self.closed = False
        sd.streams.append(self)
        if sd.refuse(kw):
            raise sd.PortAudioError(sd.refuse(kw))

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    @property
    def active(self):
        return self.started and not self.stopped


class FakeSD:
    class PortAudioError(Exception):
        pass

    def __init__(self, devices, refuse=lambda kw: None) -> None:
        # devices: [(host api, name, max input channels, default rate)]
        self.devices = devices
        self.refuse = refuse
        self.streams: list[FakeRawStream] = []
        self.reinit = 0

    def query_hostapis(self):
        apis = {}
        for i, (api, *_rest) in enumerate(self.devices):
            apis.setdefault(api, []).append(i)
        return [{"name": api, "devices": idx, "default_input_device": idx[0]} for api, idx in apis.items()]

    def query_devices(self, index):
        api, name, channels, rate = self.devices[index]
        return {"name": name, "max_input_channels": channels, "default_samplerate": rate}

    def RawInputStream(self, **kw):
        return FakeRawStream(self, **kw)

    def WasapiSettings(self, auto_convert=False):
        return ("wasapi", auto_convert)

    def _terminate(self):
        self.reinit += 1

    def _initialize(self):
        pass


def capture_with(sd) -> PortAudioCapture:
    cap = PortAudioCapture()
    cap._sd = sd
    return cap


USB = "Microphone (USB Audio Device)"


def test_opens_through_wasapi_at_16k_mono_with_windows_conversion():
    sd = FakeSD([("MME", USB[:31], 2, 44100), ("Windows WASAPI", USB, 2, 48000)])
    stream = capture_with(sd).open(mic(USB))
    kw = sd.streams[-1].kw
    assert kw["device"] == 1 and kw["samplerate"] == 16000 and kw["channels"] == 1 and kw["dtype"] == "int16"
    assert kw["extra_settings"] == ("wasapi", True) and kw["blocksize"] == 480
    assert stream.sample_rate == 16000 and stream.active
    stream.close()
    assert sd.streams[-1].closed


def test_falls_back_to_the_native_rate_and_then_stereo():
    refused = lambda kw: "Invalid sample rate" if kw["samplerate"] == 16000 else (  # noqa: E731
        "Invalid number of channels" if kw["channels"] == 1 else None)
    sd = FakeSD([("Windows WASAPI", USB, 2, 48000)], refuse=refused)
    stream = capture_with(sd).open(mic(USB))
    assert (stream.sample_rate, sd.streams[-1].kw["channels"]) == (48000, 2)
    # Stereo from the device is delivered as mono (first channel).
    callback = sd.streams[-1].kw["callback"]
    callback(array("h", [100, -1, 200, -2, 300, -3]).tobytes(), 3, None, None)
    assert array("h", stream.read(0.1)).tolist() == [100, 200, 300]


def test_blocks_arrive_through_the_callback_and_timeouts_return_none():
    sd = FakeSD([("Windows WASAPI", USB, 1, 16000)])
    stream = capture_with(sd).open(mic(USB))
    assert stream.read(0.01) is None
    sd.streams[-1].kw["callback"](b"\x01\x00" * 480, 480, None, None)
    assert stream.read(0.1) == b"\x01\x00" * 480


def test_mme_names_are_cut_to_31_characters_and_still_match():
    long_name = "Microphone (Logitech USB Headset H390)"
    sd = FakeSD([("MME", long_name[:31], 1, 44100)])
    capture_with(sd).open(mic(long_name))
    assert sd.streams[-1].kw["device"] == 0


def test_wdm_ks_pins_are_never_used():
    sd = FakeSD([("Windows WDM-KS", "Stereo Mix (Realtek HD Audio Stereo input)", 2, 44100),
                 ("Windows WDM-KS", USB, 2, 44100)])
    with pytest.raises(MicrophoneUnavailable):
        capture_with(sd).open(mic(USB))
    assert capture_with(sd).list_inputs() == []


def test_device_connected_after_start_is_found_by_rescanning():
    sd = FakeSD([])
    cap = capture_with(sd)
    original = sd._initialize

    def plug_in():
        sd.devices = [("Windows WASAPI", USB, 1, 48000)]
        original()

    sd._initialize = plug_in
    cap.open(mic(USB))
    assert sd.reinit == 1


def test_missing_device_is_unavailable():
    with pytest.raises(MicrophoneUnavailable) as err:
        capture_with(FakeSD([])).open(mic(USB))
    assert "isn't available" in str(err.value)


def test_access_denied_becomes_a_privacy_message():
    sd = FakeSD([("Windows WASAPI", USB, 1, 48000)],
                refuse=lambda kw: "Unanticipated host error [PaErrorCode -9999]: 'Access is denied' [0x80070005]")
    with pytest.raises(MicrophoneDenied) as err:
        capture_with(sd).open(mic(USB))
    assert "Privacy & security > Microphone" in str(err.value)


def test_every_format_refused_is_unavailable_not_a_crash():
    sd = FakeSD([("Windows WASAPI", USB, 2, 48000)], refuse=lambda kw: "Device unavailable [-9985]")
    with pytest.raises(MicrophoneUnavailable) as err:
        capture_with(sd).open(mic(USB))
    assert "Device unavailable" in str(err.value)


def test_fallback_device_list_uses_one_host_api():
    sd = FakeSD([("MME", "Microphone (USB Audio Dev", 1, 44100), ("Windows WASAPI", USB, 1, 48000),
                 ("Windows WASAPI", "Headset (BT)", 1, 16000)])
    names = [d.name for d in capture_with(sd).list_inputs()]
    assert names == [USB, "Headset (BT)"]  # WASAPI's full names, not MME's duplicates


def test_real_portaudio_rescan_and_listing_do_not_open_anything():
    pytest.importorskip("sounddevice")
    cap = PortAudioCapture()
    cap.refresh()
    inputs = cap.list_inputs()
    assert all(d.id.startswith("portaudio:") for d in inputs)
    with pytest.raises(MicrophoneUnavailable):
        cap.open(mic("No Such Microphone 12345"))
