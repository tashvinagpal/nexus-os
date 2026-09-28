"""Playing speech on the system's default audio output, stoppable at once.

Output side only - microphones are nexus/system/audio/capture.py. The output
is whatever the OS uses as its default playback device right now (speakers,
USB or Bluetooth headphones, a monitor's speakers); nothing is hard-coded.
If the default changes, PortAudio's device list is re-read before the next
sentence (never while a microphone is recording). Playback runs in PortAudio's
own thread: play() returns at once, stop() silences it immediately.
"""

from __future__ import annotations

import logging
import threading
import time
from array import array
from typing import Any, Callable, Protocol

from nexus.ai.voice.base import AudioClip
from nexus.system.platform.base import AudioOutput

log = logging.getLogger(__name__)

_HOST_APIS = ("Windows WASAPI", "MME", "Windows DirectSound", "ALSA", "PulseAudio", "Core Audio")


class AudioOutputError(Exception):
    """Sound couldn't be played. str(error) is written for the user; never fatal."""


class NoAudioOutput(AudioOutputError):
    """No playback device is connected or enabled."""


class Playback(Protocol):
    started_at: float | None   # when the first audio reached the device (for latency measurements)
    error: str | None

    def wait(self, timeout: float | None = None) -> bool:
        """True once finished (played out or stopped); False if still playing after `timeout`."""
        ...

    def stop(self) -> None: ...


class AudioSink(Protocol):
    def ready(self) -> tuple[bool, str]:
        """(True, device name) if something can play now, else (False, why)."""
        ...

    def play(self, clip: AudioClip) -> Playback:
        """Start playing and return at once. Raises AudioOutputError."""
        ...

    def stop(self) -> None:
        """Silence whatever is playing, immediately."""
        ...


def resample_pcm16(pcm: bytes, src: int, dst: int) -> bytes:
    """Linear resampling of mono PCM16 (a fallback for devices that won't take the voice's rate)."""
    if src == dst or not pcm:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    n_out = int(len(samples) * dst / src)
    step = src / dst
    out = array("h", bytes(2 * n_out))
    last = len(samples) - 1
    for i in range(n_out):
        pos = i * step
        j = int(pos)
        frac = pos - j
        a = samples[j]
        b = samples[j + 1] if j < last else a
        out[i] = int(a + (b - a) * frac)
    return out.tobytes()


class _PortAudioPlayback:
    def __init__(self, sd: Any, device: int, pcm: bytes, rate: int, extra: Any,
                 clock: Callable[[], float]) -> None:
        self._clock = clock
        self._data = pcm
        self._pos = 0
        self._done = threading.Event()
        self._last_progress = clock()
        self.started_at: float | None = None
        self.stopped = False
        self.error: str | None = None
        self.duration = len(pcm) / 2 / rate

        def callback(outdata, frames, time_info, status) -> None:  # PortAudio's thread
            if self.started_at is None:
                self.started_at = self._clock()
            self._last_progress = self._clock()
            n = frames * 2
            chunk = self._data[self._pos:self._pos + n]
            self._pos += n
            if len(chunk) < n:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = b"\x00" * (n - len(chunk))
                raise sd.CallbackStop
            outdata[:] = chunk

        self._stream = sd.RawOutputStream(device=device, samplerate=rate, channels=1, dtype="int16",
                                          callback=callback, finished_callback=self._done.set, extra_settings=extra)
        self._stream.start()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else self._clock() + timeout
        while not self._done.wait(0.05):
            if self._clock() - self._last_progress > 2.0 + (0 if self.started_at else 1.0):
                self.error = "the audio output stopped playing (disconnected?)"
                log.warning("playback stalled: %s", self.error)
                self.stop()
                return True
            if deadline is not None and self._clock() >= deadline:
                return False
        self._close()
        return True

    def stop(self) -> None:
        if not self._done.is_set():
            self.stopped = True
            try:
                self._stream.abort()
            except Exception:
                pass
        self._done.set()
        self._close()

    def _close(self) -> None:
        try:
            self._stream.close()
        except Exception:
            pass


class PortAudioSink:
    def __init__(self, list_outputs: Callable[[], list[AudioOutput]] | None = None, *,
                 can_refresh: Callable[[], bool] = lambda: True, clock: Callable[[], float] = time.monotonic,
                 prepare_thread: Callable[[], None] | None = None) -> None:
        self._list_outputs = list_outputs
        self._prepare_thread = prepare_thread  # e.g. COM for WASAPI on the speech thread (Platform.prepare_audio_thread)
        self._can_refresh = can_refresh  # e.g. not while a microphone is recording
        self._clock = clock
        self._sd: Any = None
        self._current: _PortAudioPlayback | None = None
        self._lock = threading.Lock()
        self._initialised_for: str | None = None  # the default output when PortAudio last read its list
        self.last_device: str | None = None

    def _module(self) -> Any:
        if self._sd is None:
            try:
                import sounddevice
            except (ImportError, OSError) as exc:
                raise AudioOutputError(f"Audio output needs the 'sounddevice' package ({exc}).") from None
            self._sd = sounddevice
        return self._sd

    # --- which device ---------------------------------------------------------------------

    def outputs(self) -> list[AudioOutput]:
        if self._list_outputs is None:
            return []
        try:
            return list(self._list_outputs())
        except Exception as exc:  # a broken audio stack must not take NEXUS down
            log.warning("listing audio outputs failed: %s", exc)
            return []

    def current(self) -> AudioOutput | None:
        """The OS default output if it is active; else any active output; None if there is none."""
        if self._list_outputs is None:  # no platform list (e.g. Linux for now): PortAudio's default
            sd = self._module()
            index = sd.default.device[1]
            if index is None or index < 0:
                return None
            info = sd.query_devices(index)
            return AudioOutput(id=f"portaudio:{info['name']}", name=info["name"], default=True)
        active = [o for o in self.outputs() if o.state == "active"]
        return next((o for o in active if o.default), None) or (active[0] if active else None)

    def ready(self) -> tuple[bool, str]:
        try:
            out = self.current()
        except AudioOutputError as exc:
            return False, str(exc)
        if out is None:
            return False, "no audio output device is connected"
        return True, out.name

    def _find(self, sd: Any, name: str) -> tuple[int, str] | None:
        apis = sd.query_hostapis()
        for api_name in _HOST_APIS:
            api = next((a for a in apis if a["name"] == api_name), None)
            if api is None:
                continue
            for index in api["devices"]:
                info = sd.query_devices(index)
                if info["max_output_channels"] <= 0:
                    continue
                if info["name"] == name or (api_name == "MME" and len(info["name"]) >= 30
                                            and name.startswith(info["name"])):
                    return index, api_name
        return None

    # --- playing -----------------------------------------------------------------------------

    def play(self, clip: AudioClip) -> Playback:
        if self._prepare_thread is not None:
            self._prepare_thread()
        sd = self._module()
        out = self.current()
        if out is None:
            raise NoAudioOutput("No audio output device is connected, so NEXUS can't speak (text still works).")
        with self._lock:
            self._stop_locked()
            if self._initialised_for not in (None, out.id) and self._can_refresh():
                log.info("default audio output changed to %s: re-reading the device list", out.name)
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception as exc:
                    log.warning("re-reading audio devices failed: %s", exc)
            self._initialised_for = out.id
            found = self._find(sd, out.name)
            if found is None and self._can_refresh():
                sd._terminate()
                sd._initialize()
                found = self._find(sd, out.name)
            if found is None:
                default = sd.default.device[1]
                if default is None or default < 0:
                    raise NoAudioOutput(f"{out.name} can't be opened for playback right now.")
                found = (default, "default")
            index, api_name = found
            native = int(sd.query_devices(index)["default_samplerate"]) or clip.sample_rate
            attempts: list[tuple[int, Any]] = []
            if api_name == "Windows WASAPI":  # shared mode; Windows converts the voice's rate for us
                attempts.append((clip.sample_rate, sd.WasapiSettings(auto_convert=True)))
            attempts += [(clip.sample_rate, None), (native, None)]
            last: Exception | None = None
            for rate, extra in attempts:
                pcm = clip.pcm16 if rate == clip.sample_rate else resample_pcm16(clip.pcm16, clip.sample_rate, rate)
                try:
                    playback = _PortAudioPlayback(sd, index, pcm, rate, extra, self._clock)
                except Exception as exc:  # sd.PortAudioError, ValueError for unsupported settings
                    last = exc
                    continue
                self._current = playback
                if self.last_device != out.name:
                    log.info("speaking through %s (%s)", out.name, api_name)
                self.last_device = out.name
                return playback
            raise AudioOutputError(f"Couldn't play on {out.name}: {last}")

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        current, self._current = self._current, None
        if current is not None:
            current.stop()
