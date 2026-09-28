"""Choosing a microphone automatically: any suitable input, never a hard-coded one.

Order (the first one that opens wins, the rest are fallbacks):
  1. the microphone you picked (`/mic use <name>` or `[voice] microphone`), when it is connected
  2. the system's default *communications* microphone (Windows: Sound settings > Communications)
  3. the system's default microphone
  4. any other connected input: headset > microphone > unknown input > line in; USB > built-in > Bluetooth

Never chosen automatically:
  - loopback inputs that record what the PC plays ("Stereo Mix", "What U Hear", "Monitor of ...")
  - audio coming from another device (a phone's call or music over Bluetooth)
  - virtual devices (virtual cameras and cables such as iVCam) - only if you pick one yourself
  - disabled or unplugged inputs; playback devices are never inputs at all

The list is read again on every /listen, every few seconds in the background
and whenever a microphone fails, so a microphone connected after NEXUS started
is used without a restart and a disconnected one is replaced by the next.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

from nexus.system.platform.base import AudioInput

log = logging.getLogger(__name__)

_LOOPBACK_NAME = re.compile(r"stereo\s*mix|what\s*u\s*hear|wave\s*out\s*mix|loopback|^monitor of\b", re.IGNORECASE)
_KIND_RANK = {"headset": 0, "handset": 1, "microphone": 2, "unknown": 3, "other": 3, "line_in": 4}
_BUS_RANK = {"usb": 0, "builtin": 1, "bluetooth": 2, "other": 3, "virtual": 4}


@dataclass(frozen=True)
class Verdict:
    usable: bool     # may be opened at all
    automatic: bool  # may be chosen without the user asking for it
    note: str = ""   # why not (shown by /mic)


def classify(dev: AudioInput) -> Verdict:
    if dev.loopback or _LOOPBACK_NAME.search(dev.name):
        return Verdict(False, False, "records what this PC plays - not a microphone")
    if dev.remote:
        return Verdict(False, False, "sound from another device (a phone call or music) - not a microphone here")
    if dev.state == "disabled":
        return Verdict(False, False, "disabled in the system sound settings")
    if dev.state != "active":
        return Verdict(False, False, "not connected" if dev.state == "unplugged" else f"not available ({dev.state})")
    if dev.bus == "virtual":
        return Verdict(True, False, "virtual device - used only if you pick it with /mic use")
    return Verdict(True, True)


def _rank(dev: AudioInput) -> tuple:
    return (not dev.default_communications, not dev.default, _KIND_RANK.get(dev.form_factor, 3),
            _BUS_RANK.get(dev.bus, 3), dev.name.lower(), dev.id)


@dataclass(frozen=True)
class MicScan:
    inputs: tuple[AudioInput, ...]
    candidates: tuple[AudioInput, ...]              # best first: NEXUS tries them in this order
    ignored: tuple[tuple[AudioInput, str], ...]     # with the reason
    error: str | None = None                        # the device list couldn't be read
    note: str = ""

    @property
    def chosen(self) -> AudioInput | None:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True)
class MicChange:
    connected: tuple[AudioInput, ...]
    disconnected: tuple[AudioInput, ...]
    previous: AudioInput | None
    current: AudioInput | None

    def describe(self) -> str:
        parts = [f"Microphone connected: {d.name}." for d in self.connected]
        parts += [f"Microphone disconnected: {d.name}." for d in self.disconnected]
        if self.current is None:
            if self.previous is not None:
                parts.append("No microphone is available now.")
        elif self.previous is None or self.current.id != self.previous.id:
            parts.append(f"NEXUS will listen with {self.current.name}.")
        return " ".join(parts)


class MicrophoneManager:
    def __init__(self, list_inputs: Callable[[], list[AudioInput]], *, preferred: str = "auto",
                 clock: Callable[[], float] = time.monotonic, retry_after_s: float = 30.0) -> None:
        self._list = list_inputs
        self._preferred = preferred.strip() or "auto"
        self._clock = clock
        self._retry_after_s = retry_after_s
        self._failed: dict[str, tuple[float, str]] = {}
        self._lock = threading.RLock()
        self._last: MicScan | None = None
        self._last_available: tuple[AudioInput, ...] = ()
        self._listeners: list[Callable[[MicChange], None]] = []
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None

    @property
    def preferred(self) -> str:
        return self._preferred

    @property
    def last_scan(self) -> MicScan | None:
        return self._last

    def subscribe(self, listener: Callable[[MicChange], None]) -> None:
        self._listeners.append(listener)

    # --- scanning ------------------------------------------------------------------

    def scan(self) -> MicScan:
        """Read the device list again and rank what can be used right now."""
        error = None
        try:
            inputs = tuple(self._list())
        except Exception as exc:  # a broken audio stack must not take NEXUS down
            log.warning("listing audio inputs failed: %s", exc)
            inputs, error = (), f"couldn't read the list of audio devices ({exc})"
        now = self._clock()
        available: list[AudioInput] = []
        ignored: list[tuple[AudioInput, str]] = []
        for dev in inputs:
            verdict = classify(dev)
            if verdict.automatic or (verdict.usable and self._picked(dev)):
                available.append(dev)
            else:
                ignored.append((dev, verdict.note))
        available.sort(key=_rank)
        note = ""
        if self._preferred != "auto":
            picked = [d for d in available if self._picked(d)]
            if picked:
                available = picked + [d for d in available if d not in picked]
            else:
                note = f"The microphone you picked ('{self._preferred}') isn't connected - choosing automatically."
        candidates = []
        for dev in available:
            failed = self._failed.get(dev.id)
            if failed is not None and failed[0] > now:
                ignored.append((dev, f"couldn't be used a moment ago: {failed[1]}"))
            else:
                candidates.append(dev)
        scan = MicScan(inputs, tuple(candidates), tuple(ignored), error, note)
        with self._lock:
            previous, previous_available = self._last, self._last_available
            self._last, self._last_available = scan, tuple(available)
        if previous is not None:
            change = self._diff(previous, previous_available, scan, tuple(available))
            if change is not None:
                for listener in list(self._listeners):
                    try:
                        listener(change)
                    except Exception:
                        log.exception("microphone change listener failed")
        return scan

    @staticmethod
    def _diff(prev: MicScan, prev_available: tuple[AudioInput, ...], scan: MicScan,
              available: tuple[AudioInput, ...]) -> MicChange | None:
        old = {d.id: d for d in prev_available}
        new = {d.id: d for d in available}
        connected = tuple(d for i, d in new.items() if i not in old)
        disconnected = tuple(d for i, d in old.items() if i not in new)
        before, now = prev.chosen, scan.chosen
        chosen_changed = (before is None) != (now is None) or (before and now and before.id != now.id)
        if not connected and not disconnected and not chosen_changed:
            return None
        return MicChange(connected, disconnected, before, now)

    # --- choices and failures ------------------------------------------------------------

    def _picked(self, dev: AudioInput) -> bool:
        pref = self._preferred.lower()
        return pref != "auto" and (pref in dev.name.lower() or dev.id.lower() == pref)

    def use(self, name: str) -> tuple[bool, str]:
        """Pick a microphone by (part of) its name; "auto" goes back to the automatic choice."""
        name = name.strip()
        if name.lower() in ("", "auto"):
            self._preferred = "auto"
            chosen = self.scan().chosen
            return True, ("Choosing the microphone automatically"
                          + (f" - now {chosen.name}." if chosen else " - none is connected right now."))
        inputs = self.scan().inputs
        matches = [d for d in inputs if name.lower() in d.name.lower() or d.id == name]
        if not matches:
            return False, f"No audio input matches '{name}'. /mic lists them."
        allowed = [d for d in matches if classify(d).usable or (d.state == "unplugged" and not d.loopback
                                                                and not d.remote)]
        if not allowed:
            return False, f"{matches[0].name} can't be used: {classify(matches[0]).note}."
        if len({d.name for d in allowed}) > 1:
            return False, (f"'{name}' matches " + ", ".join(sorted({d.name for d in allowed}))
                           + ". Use more of the name.")
        self._preferred = name
        self._failed.pop(allowed[0].id, None)
        self.scan()
        if classify(allowed[0]).usable:
            return True, f"NEXUS will listen with {allowed[0].name}."
        return True, f"{allowed[0].name} isn't connected now - NEXUS will use it once it is."

    def mark_failed(self, dev: AudioInput, reason: str) -> None:
        """Skip this microphone for a while so the next one is tried (it is retried later)."""
        self._failed[dev.id] = (self._clock() + self._retry_after_s, reason)

    def mark_working(self, dev: AudioInput) -> None:
        self._failed.pop(dev.id, None)

    # --- background watching ---------------------------------------------------------------

    def start_watching(self, interval_s: float = 5.0) -> None:
        """Rescan every few seconds (a few milliseconds each) so plug/unplug is noticed and announced."""
        if self._watcher is not None or interval_s <= 0:
            return

        def loop() -> None:
            while not self._stop.wait(interval_s):
                self.scan()

        self._watcher = threading.Thread(target=loop, name="nexus-microphones", daemon=True)
        self._watcher.start()

    def stop_watching(self) -> None:
        self._stop.set()

    # --- /mic -----------------------------------------------------------------------------

    def describe(self, scan: MicScan | None = None) -> list[str]:
        scan = scan or self.scan()
        lines = []
        if scan.error:
            lines.append(f"  {scan.error}")
        chosen = scan.chosen
        for dev in scan.candidates:
            flags = [f for f, on in (("default for calls", dev.default_communications),
                                     ("default", dev.default)) if on]
            mark = "*" if dev is chosen else "+"
            tail = f"  ({', '.join(flags)})" if flags else ""
            lines.append(f"  {mark} {dev.name}{tail}")
        for dev, why in scan.ignored:
            lines.append(f"  - {dev.name}: {why}")
        if not scan.inputs and not scan.error:
            lines.append("  No audio inputs at all.")
        if scan.note:
            lines.append(f"  {scan.note}")
        lines.append(f"  {'Using ' + chosen.name if chosen else 'No usable microphone right now'} "
                     f"(choice: {self._preferred}).")
        return lines
