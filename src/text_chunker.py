"""Split page text into small, overlapping passages for retrieval.

Rules:
* A chunk never crosses a page boundary, so every chunk has one exact page number.
* Chunks are built from whole sentences where possible (~CHUNK_WORDS words).
* Neighbouring chunks share a few sentences (overlap) so an idea split across
  two chunks can still be found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src import config
from src.document_loader import LoadedDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    chunk_index: int          # position of the chunk within its page
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def split_sentences(text: str) -> list[str]:
    """Split on sentence endings and on line breaks (bullets, headings)."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return sentences


def _split_long(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_page_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    units: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence.split()) > max_words:
            units.extend(_split_long(sentence, max_words))
        else:
            units.append(sentence)

    chunks: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        n = len(unit.split())
        if current and current_words + n > max_words:
            chunks.append(current)
            # carry the last sentences (up to overlap_words) into the next chunk
            carry: list[str] = []
            carry_words = 0
            for prev in reversed(current):
                w = len(prev.split())
                if carry_words + w > overlap_words:
                    break
                carry.insert(0, prev)
                carry_words += w
            current, current_words = carry, carry_words
        current.append(unit)
        current_words += n
    if current:
        chunks.append(current)

    texts = [" ".join(c) for c in chunks]
    # merge a tiny final chunk into the previous one so headings don't stand alone
    if len(texts) >= 2 and len(texts[-1].split()) < 20:
        tail = texts.pop()
        extra = [w for w in tail.split()]
        texts[-1] = texts[-1] + " " + " ".join(extra)
    return texts


def chunk_document(
    doc: LoadedDocument,
    max_words: int | None = None,
    overlap_words: int | None = None,
) -> list[Chunk]:
    max_words = max_words or config.CHUNK_WORDS
    overlap_words = config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    chunks: list[Chunk] = []
    for page in doc.pages:
        if not page.text:
            continue
        for i, text in enumerate(chunk_page_text(page.text, max_words, overlap_words)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-p{page.page_number}-c{i}",
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    page_number=page.page_number,
                    chunk_index=i,
                    text=text,
                )
            )
    return chunks
