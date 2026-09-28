"""OllamaChat against a fake Ollama HTTP server (always runs) and, if Ollama is
running with llama3.2 installed, against the real thing."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nexus.ai.providers.base import ChatMessage, ProviderUnavailable
from nexus.ai.providers.ollama import OllamaChat
from nexus.core.resources import Placement


class FakeOllama(BaseHTTPRequestHandler):
    loaded: dict = {}
    requests: list = []

    def log_message(self, *args):
        pass

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": "llama3.2:latest"}, {"name": "llama3.1:8b"}]})
        elif self.path == "/api/ps":
            self._json({"models": [{"name": n, "size_vram": v} for n, v in self.loaded.items()]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append((self.path, body))
        name = body["model"] if ":" in body["model"] else body["model"] + ":latest"
        if self.path == "/api/generate":
            if body.get("keep_alive") == 0:
                self.loaded.pop(name, None)
            else:
                self.loaded[name] = 0 if body.get("options", {}).get("num_gpu") == 0 else 2_000_000_000
            self._json({"done": True})
        elif self.path == "/api/chat":
            self._json({"message": {"role": "assistant", "content": '{"say":"hi"}'}, "eval_count": 3})


@pytest.fixture
def server():
    FakeOllama.loaded, FakeOllama.requests = {}, []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_availability_and_chat(server):
    llm = OllamaChat("llama3.2", placement=Placement.GPU, ram_mb=800, vram_mb=2500, base_url=server)
    assert llm.available()
    reply = llm.chat([ChatMessage("user", "hi")], schema={"type": "object"})
    assert reply.text == '{"say":"hi"}'
    path, body = FakeOllama.requests[-1]
    assert path == "/api/chat" and body["format"] == {"type": "object"} and body["stream"] is False
    assert body["keep_alive"] == "300s" and body["options"]["num_ctx"] == 4096


def test_missing_model_is_unavailable(server):
    assert not OllamaChat("mistral", placement=Placement.GPU, ram_mb=1, base_url=server).available()


def test_load_unload_and_placement(server):
    gpu = OllamaChat("llama3.2", placement=Placement.GPU, ram_mb=800, vram_mb=2500, base_url=server)
    cpu = OllamaChat("llama3.2", placement=Placement.CPU, ram_mb=2500, base_url=server)
    assert cpu.spec.vram_mb == 0 and cpu.spec.name == "llm:llama3.2@cpu"
    cpu.load()
    assert FakeOllama.requests[-1][1]["options"]["num_gpu"] == 0
    assert cpu.is_loaded() and not gpu.is_loaded()  # loaded on CPU is not "loaded on GPU"
    cpu.unload()
    assert not cpu.is_loaded()


def test_unreachable_server():
    llm = OllamaChat("llama3.2", placement=Placement.GPU, ram_mb=1, base_url="http://127.0.0.1:9")
    assert not llm.available() and not llm.is_loaded()
    with pytest.raises(ProviderUnavailable, match="can't reach Ollama"):
        llm.chat([ChatMessage("user", "hi")])


def _real_ollama():
    llm = OllamaChat("llama3.2", placement=Placement.GPU, ram_mb=800, vram_mb=2500, idle_unload_s=60)
    return llm if llm.available() else None


@pytest.mark.live
@pytest.mark.skipif(_real_ollama() is None, reason="Ollama with llama3.2 not available")
def test_real_ollama_structured_output():
    llm = _real_ollama()
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
    try:
        reply = llm.chat([ChatMessage("user", "What is 2+3? Reply as JSON.")], schema=schema)
        assert json.loads(reply.text)["answer"] == 5
        assert llm.is_loaded()
    finally:
        llm.unload()
    assert not llm.is_loaded()
