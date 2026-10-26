"""Turn text into vectors with EmbeddingGemma running in Ollama.

EmbeddingGemma works best when documents and questions get different
prefixes, so we add them here instead of in every caller.
"""

from __future__ import annotations

from typing import Protocol

from src import config
from src.errors import OllamaUnavailableError


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        batch_size: int = 16,
    ) -> None:
        import ollama

        self.model = model or config.EMBED_MODEL
        self.client = ollama.Client(host=host or config.OLLAMA_HOST)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prompts = [f"title: none | text: {t}" for t in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prompts), self.batch_size):
            vectors.extend(self._embed(prompts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"task: search result | query: {text}"])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        import ollama

        try:
            response = self.client.embed(model=self.model, input=inputs)
        except ollama.ResponseError as exc:
            if exc.status_code == 404 or "not found" in str(exc).lower():
                raise OllamaUnavailableError(
                    f"The embedding model '{self.model}' is not downloaded. "
                    f"Run: ollama pull {self.model}"
                ) from exc
            raise OllamaUnavailableError(f"Ollama returned an error: {exc}") from exc
        except Exception as exc:  # connection refused, timeout, etc.
            raise OllamaUnavailableError(
                "Can't reach Ollama. Start it with: brew services start ollama"
            ) from exc
        return [list(v) for v in response["embeddings"]]
