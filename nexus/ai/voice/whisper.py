"""faster-whisper speech-to-text: local, loaded only while needed.

`WhisperSTT` is both an STTProvider and a ModelBackend: the ResourceManager
decides when it loads (admission against free RAM/VRAM, never on the GPU in
game mode) and unloads it when idle. "Loaded" means a worker process
(nexus/ai/voice/whisper_worker.py) holds the model; unloading ends that
process, which returns every byte it used - model, libraries and, on a GPU,
all of its VRAM.

Models (CTranslate2 conversions of OpenAI Whisper) live in models/stt/<name>/
and are never downloaded automatically: scripts/get_stt_model.py <name> shows
the size first. Nothing is sent anywhere while transcribing.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from importlib.util import find_spec
from pathlib import Path
from typing import IO, Any, Callable

from nexus.ai.providers.base import Locality, ProviderInfo
from nexus.ai.voice.base import (AudioClip, ListeningCancelled, STTFailed, STTTimeout, STTUnavailable,
                                 Transcript)
from nexus.core.config import PROJECT_ROOT
from nexus.core.resources import ModelSpec, Placement

log = logging.getLogger(__name__)

# name -> (Hugging Face repository, download size in MB). Sizes are the repositories' model files.
WHISPER_MODELS = {
    "tiny": ("Systran/faster-whisper-tiny", 75),
    "base": ("Systran/faster-whisper-base", 145),
    "small": ("Systran/faster-whisper-small", 484),
    "medium": ("Systran/faster-whisper-medium", 1530),
    "large-v3": ("Systran/faster-whisper-large-v3", 3090),
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1620),
}
MODELS_DIR = PROJECT_ROOT / "models" / "stt"
_BROKEN_FOR_S = 300.0  # after a failed load, other options are used for this long


def _worker_python() -> str:
    """pythonw on Windows: no console (and no console host process) for a background worker."""
    windowless = Path(sys.executable).with_name("pythonw.exe")
    return str(windowless) if sys.platform == "win32" and windowless.is_file() else sys.executable


def model_folder(name: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"faster-whisper-{name}"


class WhisperSTT:
    def __init__(self, model: str, *, placement: Placement = Placement.CPU, compute_type: str = "int8",
                 threads: int = 4, ram_mb: int, vram_mb: int = 0, idle_unload_s: float = 60.0,
                 languages: tuple[str, ...] = ("en", "hi", "pa"), beam_size: int = 1, timeout_s: float = 30.0,
                 load_timeout_s: float = 90.0, models_dir: Path = MODELS_DIR, log_file: Path | None = None,
                 command: list[str] | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.model = model
        self.folder = model_folder(model, models_dir)
        self.placement = placement
        self.compute_type = compute_type
        self.threads = threads
        self.languages = languages
        self.beam_size = beam_size
        self.timeout_s = timeout_s
        self.load_timeout_s = load_timeout_s
        self.log_file = log_file
        self._command = command  # tests run a fake worker
        self._clock = clock
        self.info = ProviderInfo(name=f"stt:whisper-{model}@{placement.value}", locality=Locality.LOCAL,
                                 model=model)
        self.spec = ModelSpec(self.info.name, "stt", placement, ram_mb, vram_mb, idle_unload_s)
        self._proc: subprocess.Popen[bytes] | None = None
        self._replies: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._broken_until = 0.0
        self.problem = ""               # why it isn't available (shown by /listen and /status)
        self.last_load_seconds: float | None = None

    # --- availability -----------------------------------------------------------------

    def available(self) -> bool:
        if self._clock() < self._broken_until:
            return False
        if self._command is not None:
            return True
        if find_spec("faster_whisper") is None:
            self.problem = "faster-whisper isn't installed (pip install -e \".[voice]\")"
            return False
        if not (self.folder / "model.bin").is_file():
            self.problem = (f"model not downloaded - run: .venv\\Scripts\\python.exe scripts\\get_stt_model.py "
                            f"{self.model}")
            return False
        self.problem = ""
        return True

    # --- ModelBackend --------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self.is_loaded() else None  # type: ignore[union-attr]

    def memory_mb(self) -> int:
        """RAM held by the worker right now (with a venv, Python runs as a child of a small launcher)."""
        import psutil

        pid = self.pid
        if pid is None:
            return 0
        try:
            root = psutil.Process(pid)
            return sum(p.memory_info().rss for p in [root, *root.children(recursive=True)]) // 2**20
        except psutil.Error:
            return 0

    def load(self) -> None:
        if self.is_loaded():
            return
        if not self.available():
            raise STTUnavailable(f"{self.info.name}: {self.problem or 'not available'}")
        cmd = self._command or [_worker_python(), "-m", "nexus.ai.voice.whisper_worker", "--model", str(self.folder),
                                "--device", "cuda" if self.placement is Placement.GPU else "cpu",
                                "--compute-type", self.compute_type, "--threads", str(self.threads)]
        started = time.perf_counter()
        stderr: IO[Any] | int = subprocess.DEVNULL
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            stderr = self.log_file.open("ab")
        # PASSIVE: CPU threads sleep between requests instead of spinning (measured: ~1 s of 100% CPU
        # after every transcription otherwise, and no difference in speed).
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8", "OMP_WAIT_POLICY": "PASSIVE"}
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                          cwd=str(PROJECT_ROOT), env=env,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self._fail(f"couldn't start the speech engine: {exc}")
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()  # type: ignore[union-attr]  # the child has its own handle
        self._replies = queue.Queue()
        threading.Thread(target=self._read_replies, args=(self._proc,), name="nexus-stt-reader",
                         daemon=True).start()
        try:
            ready = self._replies.get(timeout=self.load_timeout_s)
        except queue.Empty:
            ready = {"ok": False, "error": f"no answer within {self.load_timeout_s:g} s"}
        if not ready or not ready.get("ok"):
            self._kill()
            error = (ready or {}).get("error", "the speech engine exited while loading")
            self._fail(f"couldn't load {self.info.name}: {error}")
        self.last_load_seconds = time.perf_counter() - started
        log.info("%s loaded in %.1f s (pid %s)", self.info.name, self.last_load_seconds, self._proc.pid)

    def unload(self) -> None:
        proc = self._proc
        if proc is None:
            return
        with self._lock:
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(b'{"op": "quit"}\n')
                    proc.stdin.flush()
                proc.wait(timeout=3)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._kill()
        log.info("%s unloaded", self.info.name)

    def _fail(self, message: str) -> None:
        self._broken_until = self._clock() + _BROKEN_FOR_S
        self.problem = message
        log.warning("%s", message)
        raise STTUnavailable(message)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        for pipe in (proc.stdin, proc.stdout):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass

    def _read_replies(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        replies = self._replies
        for line in proc.stdout:
            try:
                replies.put(json.loads(line))
            except ValueError:
                log.warning("stt worker wrote %r", line[:200])
        replies.put(None)  # the worker ended

    # --- STTProvider ---------------------------------------------------------------------

    def transcribe(self, audio: AudioClip, *, language_hint: str | None = None,
                   cancel: threading.Event | None = None, prompt: str | None = None) -> Transcript:
        if not audio.pcm16:
            raise STTFailed("There was no audio to transcribe.")
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise STTFailed("The speech engine isn't running.")
            request = {"op": "transcribe", "bytes": len(audio.pcm16), "rate": audio.sample_rate,
                       "language": language_hint, "languages": list(self.languages), "beam_size": self.beam_size}
            if prompt:
                request["prompt"] = prompt  # e.g. the wake word, so the model spells it the usual way
            try:
                proc.stdin.write(json.dumps(request).encode() + b"\n" + audio.pcm16)
                proc.stdin.flush()
            except OSError:
                self._kill()
                raise STTFailed("The speech engine stopped unexpectedly.") from None
            reply = self._await_reply(cancel)
        if not reply.get("ok"):
            raise STTFailed(f"Speech recognition failed: {reply.get('error')}")
        return Transcript(text=str(reply.get("text") or ""), language=reply.get("language"),
                          confidence=reply.get("confidence"), language_probability=reply.get("language_probability"),
                          no_speech_probability=reply.get("no_speech_probability"),
                          audio_seconds=reply.get("audio_seconds"), seconds=reply.get("seconds"),
                          model=self.info.name, speech_seconds=reply.get("speech_seconds"))

    def _await_reply(self, cancel: threading.Event | None) -> dict[str, Any]:
        deadline = self._clock() + self.timeout_s
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    self._kill()  # a half-finished answer would be read by the next request
                    raise ListeningCancelled("Cancelled.")
                remaining = deadline - self._clock()
                if remaining <= 0:
                    self._kill()
                    raise STTTimeout(f"Transcribing took longer than {self.timeout_s:g} seconds, so I stopped it.")
                try:
                    reply = self._replies.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if reply is None:
                    self._kill()
                    raise STTFailed("The speech engine stopped unexpectedly (it restarts next time).")
                return reply
        except KeyboardInterrupt:  # Ctrl+C in the console
            self._kill()
            raise ListeningCancelled("Cancelled.") from None
