"""Shared test helpers. Tests never call Ollama: they use a tiny fake embedder."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pymupdf
import pytest

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "what", "which",
    "how", "does", "do", "by", "for", "on", "with", "it", "its", "this", "that",
    "from", "as", "be", "at", "who", "why", "called", "into",
}


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, offline, good enough to
    test ranking, filtering and the threshold logic."""

    dim = 4096

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in _STOP or len(word) < 3:
                continue
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        v[-1] += 0.01  # never a zero vector
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return (SAMPLE_DIR / "photosynthesis_notes.pdf").read_bytes()


@pytest.fixture
def scanned_pdf_bytes() -> bytes:
    return (SAMPLE_DIR / "photosynthesis_notes_scanned.pdf").read_bytes()


def make_pdf(pages: list[str], password: str | None = None) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=12)
    kwargs = {}
    if password:
        kwargs = dict(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password
        )
    data = doc.tobytes(**kwargs)
    doc.close()
    return data
