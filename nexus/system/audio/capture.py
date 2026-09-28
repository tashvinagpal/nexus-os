"""Capturing audio from a microphone through PortAudio (the `sounddevice` package).

PortAudio is cross-platform: WASAPI (shared mode) on Windows, ALSA/PulseAudio
on Linux. The device to open comes from MicrophoneManager; PortAudio's own
list is only used to find that device (and as a fallback device list on a
platform without its own). The stream delivers mono 16-bit PCM in small
blocks; nothing is written to disk.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from typing import Any, Protocol

from nexus.system.audio.errors import MicrophoneDenied, MicrophoneLost, MicrophoneUnavailable
from nexus.system.platform.base import AudioInput

log = logging.getLogger(__name__)

# Best first. WDM-KS is skipped: it bypasses the system mixer and lists disabled/unplugged pins.
_HOST_APIS = ("Windows WASAPI", "MME", "Windows DirectSound", "ALSA", "PulseAudio", "JACK Audio Connection Kit",
              "Core Audio")
_DENIED = re.compile(r"0x80070005|access\s*(is\s*)?denied|permission", re.IGNORECASE)
_MME_NAME_LIMIT = 31  # MME cuts device names to 31 characters


class CaptureStream(Protocol):
    sample_rate: int

    def read(self, timeout: float) -> bytes | None:
        """Next block of mono PCM16, or None if nothing arrived within `timeout` seconds."""
        ...

    @property
    def active(self) -> bool: ...

    def close(self) -> None: ...


class CaptureBackend(Protocol):
    def open(self, mic: AudioInput, *, sample_rate: int = 16000, block_ms: int = 30) -> CaptureStream: ...

    def refresh(self) -> None:
        """Re-read the backend's device list (after a device was plugged in)."""
        ...


class _PortAudioStream:
    def __init__(self, sd: Any, device: int, sample_rate: int, channels: int, block: int,
                 extra: Any = None) -> None:
        self.sample_rate = sample_rate
        self._channels = channels
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=400)  # ~12 s of 30 ms blocks
        self.overflows = 0

        def callback(indata, frames, time_info, status) -> None:  # PortAudio's thread
            if status:
                self.overflows += 1
            data = bytes(indata)
            if self._channels > 1:  # keep the first channel
                data = memoryview(data).cast("h")[::self._channels].tobytes()
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                pass

        self._stream = sd.RawInputStream(device=device, samplerate=sample_rate, channels=channels, dtype="int16",
                                         blocksize=block, callback=callback, extra_settings=extra)
        self._stream.start()

    def read(self, timeout: float) -> bytes | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def active(self) -> bool:
        try:
            return bool(self._stream.active)
        except Exception:
            return False

    def close(self) -> None:
        for step in (self._stream.stop, self._stream.close):
            try:
                step()
            except Exception:  # a device that vanished may fail to stop cleanly
                pass


class PortAudioCapture:
    def __init__(self) -> None:
        self._sd: Any = None
        self._lock = threading.Lock()

    def _module(self) -> Any:
        if self._sd is None:
            try:
                import sounddevice
            except (ImportError, OSError) as exc:
                raise MicrophoneUnavailable(
                    "Voice input needs the 'sounddevice' package: .venv\\Scripts\\python.exe -m pip install "
                    f"-e \".[voice]\" ({exc})") from None
            self._sd = sounddevice
        return self._sd

    def refresh(self) -> None:
        sd = self._module()
        with self._lock:
            # PortAudio reads its device list once; re-initialising it picks up newly connected devices.
            sd._terminate()
            sd._initialize()

    # --- a device list for platforms without their own (none needed on Windows) ---------

    def list_inputs(self) -> list[AudioInput]:
        sd = self._module()
        apis = sd.query_hostapis()
        seen: dict[str, AudioInput] = {}
        for api_name in _HOST_APIS:
            api = next((a for a in apis if a["name"] == api_name), None)
            if api is None:
                continue
            for index in api["devices"]:
                info = sd.query_devices(index)
                if info["max_input_channels"] > 0 and info["name"] not in seen:
                    seen[info["name"]] = AudioInput(id=f"portaudio:{info['name']}", name=info["name"],
                                                    default=index == api["default_input_device"])
            if seen:
                break  # one host API is enough; the others list the same devices again
        return list(seen.values())

    # --- opening -------------------------------------------------------------------------

    def _find(self, mic: AudioInput) -> tuple[int, str] | None:
        sd = self._module()
        apis = sd.query_hostapis()
        for api_name in _HOST_APIS:
            api = next((a for a in apis if a["name"] == api_name), None)
            if api is None:
                continue
            for index in api["devices"]:
                info = sd.query_devices(index)
                if info["max_input_channels"] <= 0:
                    continue
                name = info["name"]
                if name == mic.name or (api_name == "MME" and len(name) >= _MME_NAME_LIMIT - 1
                                        and mic.name.startswith(name)):
                    return index, api_name
        return None

    def open(self, mic: AudioInput, *, sample_rate: int = 16000, block_ms: int = 30) -> CaptureStream:
        sd = self._module()
        found = self._find(mic)
        if found is None:
            self.refresh()  # connected after PortAudio started?
            found = self._find(mic)
        if found is None:
            raise MicrophoneUnavailable(f"{mic.name} isn't available to open right now.")
        index, api_name = found
        info = sd.query_devices(index)
        native = int(info["default_samplerate"]) or sample_rate
        attempts: list[tuple[int, int, Any]] = []
        if api_name == "Windows WASAPI":  # shared mode; Windows converts rate and channels for us
            attempts.append((sample_rate, 1, sd.WasapiSettings(auto_convert=True)))
        attempts += [(sample_rate, 1, None), (native, 1, None),
                     (native, min(2, int(info["max_input_channels"])), None)]
        last: Exception | None = None
        for rate, channels, extra in attempts:
            try:
                stream = _PortAudioStream(sd, index, rate, channels, max(1, rate * block_ms // 1000), extra)
                log.info("microphone open: %s via %s at %d Hz, %d ch", mic.name, api_name, rate, channels)
                return stream
            except Exception as exc:  # sd.PortAudioError, ValueError for unsupported settings
                last = exc
                if _DENIED.search(str(exc)):
                    raise MicrophoneDenied(
                        "Microphone access is blocked. Allow it in Windows Settings > Privacy & security > "
                        "Microphone (\"Let desktop apps access your microphone\").") from None
        raise MicrophoneUnavailable(f"Couldn't open {mic.name}: {last}")


__all__ = ["CaptureBackend", "CaptureStream", "MicrophoneLost", "PortAudioCapture"]
