"""Thin wrapper around Ollama's chat API for Gemma 4.

Responsibilities:
* send messages with sensible options (context size, low temperature)
* turn Gemma 4's "thinking" off and strip any reasoning text that slips through
* convert every Ollama failure into one friendly OllamaUnavailableError
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src import config
from src.errors import OllamaUnavailableError

_THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|?>thought.*?<channel\|>", re.DOTALL | re.IGNORECASE),
]


def strip_thinking(text: str) -> str:
    for pattern in _THINK_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


@dataclass
class HealthStatus:
    ollama_running: bool
    chat_model_ready: bool
    embed_model_ready: bool
    message: str

    @property
    def ok(self) -> bool:
        return self.ollama_running and self.chat_model_ready and self.embed_model_ready


class GemmaClient:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        import ollama

        self.model = model or config.CHAT_MODEL
        self.client = ollama.Client(host=host or config.OLLAMA_HOST)
        # Some Ollama runners (e.g. the MLX engine on Apple Silicon) don't support
        # forced JSON output and return HTTP 501. We then rely on the prompt asking
        # for JSON and on our own parser, and stop asking for the option.
        self.json_format_supported = True

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        import ollama

        kwargs = dict(
            model=self.model,
            messages=messages,
            options={
                "num_ctx": config.NUM_CTX,
                "temperature": config.TEMPERATURE if temperature is None else temperature,
            },
            keep_alive=config.KEEP_ALIVE,
        )
        if json_mode and self.json_format_supported:
            kwargs["format"] = "json"
        try:
            try:
                response = self._send(kwargs)
            except ollama.ResponseError as exc:
                if "format" in kwargs and _json_format_unsupported(exc):
                    self.json_format_supported = False
                    kwargs.pop("format")
                    response = self._send(kwargs)
                else:
                    raise
        except ollama.ResponseError as exc:
            if exc.status_code == 404 or "not found" in str(exc).lower():
                raise OllamaUnavailableError(
                    f"The model '{self.model}' is not downloaded. Run: ollama pull {self.model}"
                ) from exc
            raise OllamaUnavailableError(f"Gemma returned an error: {exc}") from exc
        except Exception as exc:
            raise OllamaUnavailableError(
                "Can't reach Ollama. Start it with: brew services start ollama"
            ) from exc
        return strip_thinking(response["message"]["content"] or "")

    def _send(self, kwargs: dict):
        try:
            return self.client.chat(think=False, **kwargs)
        except TypeError:  # older ollama-python without the `think` option
            return self.client.chat(**kwargs)


def _json_format_unsupported(exc) -> bool:
    text = str(exc).lower()
    return getattr(exc, "status_code", None) == 501 or "structured output" in text or "format" in text and "unsupported" in text


def check_health(host: str | None = None) -> HealthStatus:
    """Is Ollama running, and are both models downloaded? Used by the UI sidebar."""
    import ollama

    try:
        listed = ollama.Client(host=host or config.OLLAMA_HOST).list()
    except Exception:
        return HealthStatus(False, False, False, "Ollama is not running. Start it with: brew services start ollama")

    names = set()
    for m in listed.get("models", []):
        name = m.get("model") or m.get("name") or ""
        names.add(name)
        names.add(name.removesuffix(":latest"))

    chat_ok = config.CHAT_MODEL in names
    embed_ok = config.EMBED_MODEL in names
    missing = [m for m, ok in ((config.CHAT_MODEL, chat_ok), (config.EMBED_MODEL, embed_ok)) if not ok]
    message = "Ready" if not missing else "Missing model(s). Run: " + " && ".join(f"ollama pull {m}" for m in missing)
    return HealthStatus(True, chat_ok, embed_ok, message)
