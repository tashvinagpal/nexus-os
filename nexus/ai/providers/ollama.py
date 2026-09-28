"""Local LLMs through Ollama (http://127.0.0.1:11434).

One OllamaChat instance = one model at one placement (GPU or CPU-only). It is
both a ChatProvider and a ModelBackend, so the ResourceManager decides when it
may load and unloads it when idle.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from nexus.ai.providers.base import ChatMessage, ChatReply, Locality, ProviderInfo, ProviderUnavailable
from nexus.core.resources import ModelSpec, Placement

DEFAULT_URL = "http://127.0.0.1:11434"
# Never route local model traffic through a system proxy.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class OllamaChat:
    def __init__(self, model: str, *, placement: Placement, ram_mb: int, vram_mb: int = 0,
                 idle_unload_s: float = 300.0, num_ctx: int = 4096, base_url: str = DEFAULT_URL,
                 timeout_s: float = 120.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.spec = ModelSpec(name=f"llm:{model}@{placement.value}", kind="llm", placement=placement,
                              ram_mb=ram_mb, vram_mb=vram_mb if placement is Placement.GPU else 0,
                              idle_unload_s=idle_unload_s)
        self.info = ProviderInfo(name=self.spec.name, locality=Locality.LOCAL, model=model)
        self._options: dict[str, Any] = {"num_ctx": num_ctx}
        if placement is Placement.CPU:
            self._options["num_gpu"] = 0  # keep every layer off the GPU
        self._available_cache: tuple[float, bool] | None = None

    # --- ChatProvider ------------------------------------------------------------

    def available(self) -> bool:
        now = time.monotonic()
        if self._available_cache and now - self._available_cache[0] < 10:
            return self._available_cache[1]
        try:
            names = {m["name"] for m in self._get("/api/tags").get("models", [])}
            ok = self.model in names or f"{self.model}:latest" in names
        except ProviderUnavailable:
            ok = False
        self._available_cache = (now, ok)
        return ok

    def chat(self, messages: Sequence[ChatMessage], *, schema: dict[str, Any] | None = None,
             temperature: float = 0.2) -> ChatReply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "keep_alive": self._keep_alive(),
            "options": {**self._options, "temperature": temperature},
        }
        if schema is not None:
            body["format"] = schema
        started = time.perf_counter()
        data = self._post("/api/chat", body)
        text = data.get("message", {}).get("content", "")
        stats = {k: data.get(k) for k in ("prompt_eval_count", "eval_count", "load_duration", "total_duration")}
        return ChatReply(text=text, model=self.model, seconds=time.perf_counter() - started, stats=stats)

    # --- ModelBackend ------------------------------------------------------------

    def load(self) -> None:
        # An empty prompt makes Ollama load the model and return.
        self._post("/api/generate", {"model": self.model, "keep_alive": self._keep_alive(),
                                     "options": self._options})

    def unload(self) -> None:
        self._post("/api/generate", {"model": self.model, "keep_alive": 0})
        deadline = time.monotonic() + 10
        while self.is_loaded() and time.monotonic() < deadline:  # wait until memory is really freed
            time.sleep(0.2)

    def is_loaded(self) -> bool:
        try:
            running = self._get("/api/ps").get("models", [])
        except ProviderUnavailable:
            return False
        for m in running:
            if m.get("name") in (self.model, f"{self.model}:latest"):
                on_gpu = (m.get("size_vram") or 0) > 0
                return on_gpu == (self.spec.placement is Placement.GPU)
        return False

    # --- HTTP --------------------------------------------------------------------

    def _keep_alive(self) -> str:
        # Ollama's own timer mirrors ours, so the model unloads even if NEXUS is closed.
        if math.isinf(self.spec.idle_unload_s):
            return "-1"  # the user chose to keep it loaded (game mode still unloads it)
        return f"{int(self.spec.idle_unload_s)}s"

    def _get(self, path: str) -> dict[str, Any]:
        return self._request(urllib.request.Request(self.base_url + path), timeout=5)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        return self._request(req, timeout=self.timeout_s)

    def _request(self, req: urllib.request.Request, *, timeout: float) -> dict[str, Any]:
        try:
            with _opener.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ProviderUnavailable(f"Ollama error {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ProviderUnavailable(f"can't reach Ollama at {self.base_url} ({exc})") from None
