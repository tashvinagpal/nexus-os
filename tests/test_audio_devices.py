"""Microphone enumeration, automatic selection, fallback, hotplug and no-microphone handling.

No specific microphone is ever assumed: every test builds its own device list.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from nexus.system.audio.devices import MicrophoneManager, classify
from tests.voice_fixtures import FakeDevices, mic


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


# --- which inputs count as microphones ---------------------------------------------

@pytest.mark.parametrize("dev, usable, automatic", [
    (mic("Microphone (USB Audio Device)"), True, True),
    (mic("Headset (AirPods Pro Hands-Free)", form_factor="headset", bus="bluetooth"), True, True),
    (mic("Microphone (Realtek(R) Audio)", bus="builtin"), True, True),
    (mic("Headset Microphone (Logitech H390)", form_factor="headset"), True, True),
    (mic("Line In (Realtek(R) Audio)", form_factor="line_in", bus="builtin"), True, True),
    (mic("Input (Some Unknown Box)", form_factor="unknown", bus="other"), True, True),
    # never automatic
    (mic("Microphone (e2eSoft iVCam)", bus="virtual"), True, False),
    (mic("CABLE Output (VB-Audio Virtual Cable)", bus="virtual"), True, False),
    # never at all
    (mic("Stereo Mix (Realtek(R) Audio)", form_factor="unknown", bus="builtin", loopback=True), False, False),
    (mic("What U Hear (Sound Blaster)", bus="builtin"), False, False),                 # by name alone
    (mic("Monitor of Built-in Audio Analog Stereo", bus="other"), False, False),       # Linux loopback source
    (mic("Microphone (moto g86 Hands-Free HF Audio)", bus="bluetooth", remote=True), False, False),
    (mic("Microphone (Realtek(R) Audio)", state="unplugged"), False, False),
    (mic("Microphone (Webcam)", state="disabled"), False, False),
])
def test_classification(dev, usable, automatic):
    verdict = classify(dev)
    assert (verdict.usable, verdict.automatic) == (usable, automatic)
    if not automatic:
        assert verdict.note  # /mic explains why


def test_loopback_note_explains_it_records_the_pc():
    assert "what this PC plays" in classify(mic("Stereo Mix", loopback=True)).note


# --- choosing ------------------------------------------------------------------------

def test_default_communications_microphone_is_preferred():
    devices = FakeDevices(mic("Microphone (USB Mic)", default=True),
                          mic("Headset (AirPods Pro Hands-Free)", form_factor="headset", bus="bluetooth", comms=True),
                          mic("Microphone (Realtek(R) Audio)", bus="builtin"))
    scan = MicrophoneManager(devices).scan()
    assert scan.chosen.name == "Headset (AirPods Pro Hands-Free)"
    assert [d.name for d in scan.candidates][1] == "Microphone (USB Mic)"  # then the default


def test_default_microphone_second_then_kind_and_bus():
    devices = FakeDevices(mic("Microphone (Realtek(R) Audio)", bus="builtin"),
                          mic("Line In", form_factor="line_in", bus="usb"),
                          mic("Microphone (BT)", bus="bluetooth"),
                          mic("Headset (USB)", form_factor="headset"),
                          mic("Microphone (USB)"))
    names = [d.name for d in MicrophoneManager(devices).scan().candidates]
    assert names == ["Headset (USB)", "Microphone (USB)", "Microphone (Realtek(R) Audio)", "Microphone (BT)",
                     "Line In"]


def test_default_that_is_virtual_or_loopback_is_not_used():
    devices = FakeDevices(mic("Microphone (e2eSoft iVCam)", bus="virtual", comms=True, default=True),
                          mic("Stereo Mix", loopback=True, default=True),
                          mic("Microphone (USB)"))
    assert MicrophoneManager(devices).scan().chosen.name == "Microphone (USB)"


def test_no_hard_coded_device_any_single_usable_input_is_chosen():
    for dev in (mic("Brand New Mic 3000"), mic("Headset (Some BT Thing)", form_factor="headset", bus="bluetooth"),
                mic("Microfone (Placa de som USB)")):  # any name, any language
        assert MicrophoneManager(FakeDevices(dev)).scan().chosen == dev


def test_playback_only_system_has_no_microphone():
    # The platform only ever lists capture endpoints; speakers never appear. Nothing usable -> None.
    devices = FakeDevices(mic("Stereo Mix", loopback=True), mic("Microphone (e2eSoft iVCam)", bus="virtual"))
    scan = MicrophoneManager(devices).scan()
    assert scan.chosen is None and scan.candidates == ()
    assert {d.name for d, _ in scan.ignored} == {"Stereo Mix", "Microphone (e2eSoft iVCam)"}


def test_no_devices_at_all():
    manager = MicrophoneManager(FakeDevices())
    scan = manager.scan()
    assert scan.chosen is None and scan.error is None
    assert any("No audio inputs" in line for line in manager.describe(scan))


def test_enumeration_failure_is_reported_not_raised():
    devices = FakeDevices(mic("Microphone (USB)"))
    devices.error = OSError("audio service not running")
    scan = MicrophoneManager(devices).scan()
    assert scan.chosen is None and "audio service not running" in scan.error


# --- fallback -----------------------------------------------------------------------------

def test_failed_microphone_falls_back_and_is_retried_later():
    clock = Clock()
    devices = FakeDevices(mic("Microphone (USB)", comms=True), mic("Headset (BT)", form_factor="headset",
                                                                   bus="bluetooth"))
    manager = MicrophoneManager(devices, clock=clock, retry_after_s=30)
    first = manager.scan().chosen
    manager.mark_failed(first, "in use by another app")
    scan = manager.scan()
    assert scan.chosen.name == "Headset (BT)"
    assert any("in use by another app" in why for _, why in scan.ignored)
    clock.now += 31
    assert manager.scan().chosen.name == "Microphone (USB)"  # tried again after a while


def test_mark_working_clears_a_failure():
    devices = FakeDevices(mic("Microphone (USB)"))
    manager = MicrophoneManager(devices)
    dev = manager.scan().chosen
    manager.mark_failed(dev, "x")
    assert manager.scan().chosen is None
    manager.mark_working(dev)
    assert manager.scan().chosen == dev


# --- hotplug: no restart needed ------------------------------------------------------------

def test_microphone_connected_after_start_is_found_and_announced():
    devices = FakeDevices()
    manager = MicrophoneManager(devices)
    changes = []
    manager.subscribe(changes.append)
    assert manager.scan().chosen is None
    devices.plug(mic("Headset (AirPods Pro Hands-Free)", form_factor="headset", bus="bluetooth"))
    assert manager.scan().chosen.name == "Headset (AirPods Pro Hands-Free)"
    assert len(changes) == 1
    text = changes[0].describe()
    assert "Microphone connected: Headset (AirPods Pro Hands-Free)." in text
    assert "NEXUS will listen with Headset (AirPods Pro Hands-Free)." in text


def test_active_microphone_disconnected_switches_to_the_next():
    devices = FakeDevices(mic("Microphone (USB)", comms=True), mic("Microphone (Realtek(R) Audio)", bus="builtin"))
    manager = MicrophoneManager(devices)
    changes = []
    manager.subscribe(changes.append)
    assert manager.scan().chosen.name == "Microphone (USB)"
    devices.unplug("Microphone (USB)")
    assert manager.scan().chosen.name == "Microphone (Realtek(R) Audio)"
    assert changes[-1].describe() == ("Microphone disconnected: Microphone (USB). "
                                      "NEXUS will listen with Microphone (Realtek(R) Audio).")


def test_last_microphone_disconnected_and_reconnected():
    devices = FakeDevices(mic("Microphone (USB)"))
    manager = MicrophoneManager(devices)
    changes = []
    manager.subscribe(changes.append)
    manager.scan()
    devices.unplug("Microphone (USB)")
    assert manager.scan().chosen is None
    assert "No microphone is available now." in changes[-1].describe()
    devices.plug(mic("Microphone (USB)"))
    assert manager.scan().chosen.name == "Microphone (USB)"
    assert "Microphone connected" in changes[-1].describe()


def test_unplugged_state_counts_as_disconnected():
    usb = mic("Microphone (USB)")
    devices = FakeDevices(usb)
    manager = MicrophoneManager(devices)
    manager.scan()
    devices.inputs = [mic("Microphone (USB)", state="unplugged")]  # jack pulled: still listed, not usable
    assert manager.scan().chosen is None


def test_no_event_when_nothing_changed_and_a_failure_is_not_a_disconnect():
    devices = FakeDevices(mic("A"), mic("B"))
    manager = MicrophoneManager(devices)
    changes = []
    manager.subscribe(changes.append)
    manager.scan()
    manager.scan()
    assert changes == []
    manager.mark_failed(manager.last_scan.chosen, "busy")
    manager.scan()
    assert all(not c.disconnected for c in changes)


def test_background_watcher_notices_plugging_in():
    devices = FakeDevices()
    manager = MicrophoneManager(devices)
    seen = threading.Event()
    manager.subscribe(lambda change: seen.set() if change.connected else None)
    manager.scan()
    manager.start_watching(interval_s=0.02)
    try:
        devices.plug(mic("Microphone (USB)"))
        assert seen.wait(2), "the watcher didn't notice the new microphone"
    finally:
        manager.stop_watching()


def test_a_broken_listener_does_not_break_scanning():
    devices = FakeDevices()
    manager = MicrophoneManager(devices)
    manager.subscribe(lambda change: 1 / 0)
    manager.scan()
    devices.plug(mic("Microphone (USB)"))
    assert manager.scan().chosen is not None


# --- choosing one yourself ---------------------------------------------------------------

def test_use_by_part_of_the_name_and_back_to_auto():
    devices = FakeDevices(mic("Microphone (USB Audio Device)", comms=True), mic("Headset (AirPods Pro Hands-Free)",
                                                                                form_factor="headset"))
    manager = MicrophoneManager(devices)
    ok, message = manager.use("airpods")
    assert ok and "AirPods" in message
    assert manager.scan().chosen.name == "Headset (AirPods Pro Hands-Free)"
    ok, _ = manager.use("auto")
    assert ok and manager.scan().chosen.name == "Microphone (USB Audio Device)"


def test_virtual_microphone_only_when_picked():
    devices = FakeDevices(mic("Microphone (e2eSoft iVCam)", bus="virtual"))
    manager = MicrophoneManager(devices)
    assert manager.scan().chosen is None
    ok, _ = manager.use("ivcam")
    assert ok and manager.scan().chosen.name == "Microphone (e2eSoft iVCam)"


def test_picked_microphone_missing_falls_back_to_automatic_with_a_note():
    devices = FakeDevices(mic("Microphone (USB)"))
    manager = MicrophoneManager(devices, preferred="Blue Yeti")
    scan = manager.scan()
    assert scan.chosen.name == "Microphone (USB)" and "Blue Yeti" in scan.note


def test_picking_an_unplugged_microphone_waits_for_it():
    devices = FakeDevices(mic("Microphone (USB)"), mic("Headset (AirPods)", state="unplugged", form_factor="headset"))
    manager = MicrophoneManager(devices)
    ok, message = manager.use("airpods")
    assert ok and "isn't connected now" in message
    assert manager.scan().chosen.name == "Microphone (USB)"
    devices.inputs[1] = mic("Headset (AirPods)", form_factor="headset")
    assert manager.scan().chosen.name == "Headset (AirPods)"


@pytest.mark.parametrize("name, expected", [
    ("stereo", "records what this PC plays"),
    ("moto", "not a microphone here"),
    ("webcam", "disabled"),
    ("nothing-like-this", "No audio input matches"),
])
def test_use_refuses_what_is_not_a_microphone(name, expected):
    devices = FakeDevices(mic("Stereo Mix (Realtek(R) Audio)", loopback=True),
                          mic("Microphone (moto g86 HF Audio)", remote=True, bus="bluetooth"),
                          mic("Microphone (Webcam)", state="disabled"))
    ok, message = MicrophoneManager(devices).use(name)
    assert not ok and expected in message


def test_ambiguous_name_asks_for_more():
    devices = FakeDevices(mic("Microphone (USB A)"), mic("Microphone (USB B)"))
    ok, message = MicrophoneManager(devices).use("usb")
    assert not ok and "Microphone (USB A)" in message and "Microphone (USB B)" in message


def test_describe_marks_the_chosen_one():
    devices = FakeDevices(mic("Microphone (USB)", comms=True), mic("Stereo Mix", loopback=True))
    lines = MicrophoneManager(devices).describe()
    assert "  * Microphone (USB)  (default for calls)" in lines
    assert any(line.startswith("  - Stereo Mix:") for line in lines)
    assert lines[-1].startswith("  Using Microphone (USB)")


# --- the real Windows device list (read-only) ------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows Core Audio")
def test_windows_lists_capture_endpoints_read_only():
    from nexus.system.platform.windows import WindowsPlatform

    plat = WindowsPlatform()
    started = time.perf_counter()
    inputs = plat.audio_inputs()
    took = time.perf_counter() - started
    assert took < 1.0
    assert all(d.state in ("active", "disabled", "unplugged") for d in inputs)
    assert all(d.id and d.name for d in inputs)
    assert sum(d.default for d in inputs) <= 1 and sum(d.default_communications for d in inputs) <= 1
    for d in inputs:
        if "stereo mix" in d.name.lower():
            assert not classify(d).usable  # recognised structurally or by name, never used
    # And from another thread (the hotplug watcher runs on one).
    result = []
    t = threading.Thread(target=lambda: result.append(plat.audio_inputs()))
    t.start()
    t.join()
    assert [d.id for d in result[0]] == [d.id for d in inputs]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows privacy settings")
def test_windows_microphone_privacy_setting_is_readable():
    from nexus.system.platform.windows import WindowsPlatform

    assert WindowsPlatform().microphone_allowed() in (True, False, None)
