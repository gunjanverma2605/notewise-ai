import pytest

from src.document_loader import load_document, load_text
from src.text_chunker import chunk_document, chunk_page_text, split_sentences


def test_chunks_keep_page_numbers_and_unique_ids(sample_pdf_bytes):
    doc = load_document(sample_pdf_bytes, "notes.pdf")
    chunks = chunk_document(doc)
    assert {c.page_number for c in chunks} == set(range(1, 9))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        page_text = doc.pages[c.page_number - 1].text
        # every sentence of a chunk comes from its own page (never crosses pages)
        assert c.text.split(". ")[0][:40] in " ".join(page_text.split())


def test_chunks_respect_size_limit(sample_pdf_bytes):
    doc = load_document(sample_pdf_bytes, "notes.pdf")
    for c in chunk_document(doc, max_words=60, overlap_words=10):
        assert c.word_count <= 60 + 20  # +20 allowance for a merged tiny tail


def test_overlap_repeats_last_sentence():
    text = " ".join(f"Sentence number {i} is about leaves." for i in range(30))
    chunks = chunk_page_text(text, max_words=30, overlap_words=10)
    assert len(chunks) > 1
    last_sentence_of_first = chunks[0].split(". ")[-1].rstrip(".")
    assert last_sentence_of_first in chunks[1]


def test_very_long_sentence_is_split():
    text = " ".join(["word"] * 500)
    chunks = chunk_page_text(text, max_words=100, overlap_words=0)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 500


def test_split_sentences_handles_bullets_and_lines():
    sents = split_sentences("Title\nFirst point. Second point!\n- bullet item")
    assert sents == ["Title", "First point.", "Second point!", "- bullet item"]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_document(load_text("Some text here."), max_words=10, overlap_words=10)
