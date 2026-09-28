"""A model in its own process, spoken to over pipes (used by text-to-speech).

Why a separate process: unloading ends it, so all of its memory (model and
libraries) really comes back; a stuck request can be stopped; a crash in
native code can't take NEXUS down. (Speech-to-text does the same in
whisper.py, which predates this helper.)

Protocol: each message is one JSON line, optionally followed by `bytes`
bytes of payload. The worker's first message is {"ok": true, "event": "ready"}
(or {"ok": false, "error": ...}).
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
from pathlib import Path
from typing import IO, Any, Callable

log = logging.getLogger(__name__)


class WorkerError(Exception):
    """The worker couldn't start, died, or didn't answer in time."""


def windowless_python() -> str:
    """pythonw on Windows: no console (and no console host process) for a background worker."""
    windowless = Path(sys.executable).with_name("pythonw.exe")
    return str(windowless) if sys.platform == "win32" and windowless.is_file() else sys.executable


def _read_exact(stream: IO[bytes], n: int) -> bytes:
    chunks, left = [], n
    while left > 0:
        chunk = stream.read(left)
        if not chunk:
            raise EOFError("pipe closed")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


class WorkerProcess:
    def __init__(self, command: list[str], *, cwd: Path, env: dict[str, str] | None = None,
                 log_file: Path | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.command = command
        self.cwd = cwd
        self.env = env or {}
        self.log_file = log_file
        self._clock = clock
        self._proc: subprocess.Popen[bytes] | None = None
        self._replies: queue.Queue[tuple[dict[str, Any], bytes] | None] = queue.Queue()
        self._lock = threading.Lock()
        self.ready: dict[str, Any] = {}

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self.alive else None  # type: ignore[union-attr]

    def memory_mb(self) -> int:
        """RAM held by the worker (with a venv, Python runs as a child of a small launcher)."""
        import psutil

        pid = self.pid
        if pid is None:
            return 0
        try:
            root = psutil.Process(pid)
            return sum(p.memory_info().rss for p in [root, *root.children(recursive=True)]) // 2**20
        except psutil.Error:
            return 0

    def start(self, timeout_s: float) -> dict[str, Any]:
        """Start and wait for "ready". Raises WorkerError (the process is gone again)."""
        stderr: IO[Any] | int = subprocess.DEVNULL
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            stderr = self.log_file.open("ab")
        try:
            self._proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                          cwd=str(self.cwd), env={**os.environ, **self.env},
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            raise WorkerError(f"couldn't start: {exc}") from None
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()  # type: ignore[union-attr]  # the child has its own handle
        self._replies = queue.Queue()
        threading.Thread(target=self._read, args=(self._proc, self._replies), name="nexus-worker-reader",
                         daemon=True).start()
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            reply = ({"ok": False, "error": f"no answer within {timeout_s:g} s"}, b"")
        if reply is None or not reply[0].get("ok"):
            self.kill()
            raise WorkerError((reply or ({"error": "exited while loading"}, b""))[0].get("error", "failed"))
        self.ready = reply[0]
        return self.ready

    def request(self, message: dict[str, Any], *, timeout_s: float,
                cancel: threading.Event | None = None) -> tuple[dict[str, Any], bytes]:
        """One request and its reply. On timeout, cancel or a dead worker it is killed (never reused
        half-way through an answer) and WorkerError is raised."""
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise WorkerError("not running")
            try:
                proc.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
                proc.stdin.flush()
            except OSError:
                self.kill()
                raise WorkerError("stopped unexpectedly") from None
            deadline = self._clock() + timeout_s
            while True:
                if cancel is not None and cancel.is_set():
                    self.kill()
                    raise WorkerError("cancelled")
                remaining = deadline - self._clock()
                if remaining <= 0:
                    self.kill()
                    raise WorkerError(f"no answer within {timeout_s:g} s")
                try:
                    reply = self._replies.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if reply is None:
                    self.kill()
                    raise WorkerError("stopped unexpectedly")
                return reply

    def stop(self) -> None:
        """Ask it to quit, then make sure it is gone."""
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
            self.kill()

    def kill(self) -> None:
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

    @staticmethod
    def _read(proc: subprocess.Popen[bytes], replies: queue.Queue) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                try:
                    header = json.loads(line)
                except ValueError:
                    log.warning("worker wrote %r", line[:200])
                    continue
                payload = _read_exact(proc.stdout, int(header.get("bytes", 0))) if header.get("bytes") else b""
                replies.put((header, payload))
        except (EOFError, OSError, ValueError):
            pass
        replies.put(None)  # the worker ended
