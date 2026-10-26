import ollama
import pytest

from src.errors import OllamaUnavailableError
from src.gemma_client import GemmaClient


class FakeOllama:
    def __init__(self, reject_format=False, error=None):
        self.reject_format = reject_format
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if self.reject_format and "format" in kwargs:
            raise ollama.ResponseError("structured output is unavailable", 501)
        return {"message": {"content": "<think>hidden</think>{\"ok\": true}"}}


def client_with(fake):
    c = GemmaClient()
    c.client = fake
    return c


def test_json_mode_sends_format_when_supported():
    fake = FakeOllama()
    out = client_with(fake).chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert out == '{"ok": true}'
    assert fake.calls[0]["format"] == "json" and fake.calls[0]["think"] is False


def test_falls_back_when_runner_rejects_json_format():
    fake = FakeOllama(reject_format=True)
    client = client_with(fake)
    out = client.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert out == '{"ok": true}'
    assert len(fake.calls) == 2 and "format" not in fake.calls[1]
    client.chat([{"role": "user", "content": "again"}], json_mode=True)
    assert len(fake.calls) == 3 and "format" not in fake.calls[2]     # remembered, no wasted call


def test_missing_model_message():
    fake = FakeOllama(error=ollama.ResponseError("model 'gemma4:e4b' not found", 404))
    with pytest.raises(OllamaUnavailableError, match="ollama pull"):
        client_with(fake).chat([{"role": "user", "content": "hi"}])


def test_connection_error_message():
    fake = FakeOllama(error=ConnectionError("refused"))
    with pytest.raises(OllamaUnavailableError, match="brew services start ollama"):
        client_with(fake).chat([{"role": "user", "content": "hi"}])
