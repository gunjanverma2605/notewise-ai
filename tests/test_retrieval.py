import pytest

from src.document_loader import load_document, load_text
from src.retrieval import NoteIndex

# Threshold tuned for the fake embedder only. The real value for EmbeddingGemma
# is calibrated with scripts/check_retrieval.py.
FAKE_MIN = 0.2


@pytest.fixture
def index(tmp_path, fake_embedder):
    return NoteIndex(fake_embedder, persist_dir=tmp_path / "chroma")


@pytest.fixture
def notes(sample_pdf_bytes):
    return load_document(sample_pdf_bytes, "photosynthesis_notes.pdf")


def test_indexing_is_idempotent(index, notes):
    first = index.add_document(notes)
    second = index.add_document(notes)  # e.g. Streamlit rerun or re-upload
    assert first > 0 and first == second
    assert index.chunk_count(notes.doc_id) == first
    assert index.has_document(notes.doc_id)


def test_supported_question_finds_right_page(index, notes):
    index.add_document(notes)
    result = index.search("Which enzyme joins carbon dioxide to RuBP in phase 1 of the Calvin cycle?", [notes.doc_id], min_score=FAKE_MIN)
    assert result.supported
    assert result.chunks[0].chunk.page_number == 4
    assert result.chunks[0].chunk.doc_name == "photosynthesis_notes.pdf"


def test_question_spanning_two_pages(index, notes):
    index.add_document(notes)
    result = index.search(
        "photolysis thylakoid oxygen and Calvin cycle stroma RuBP", [notes.doc_id], top_k=6, min_score=FAKE_MIN
    )
    # top-k returns passages from several pages, so Gemma can combine them
    pages = {c.chunk.page_number for c in result.chunks}
    assert {3, 4} <= pages
    assert result.supported


def test_unsupported_question_is_flagged(index, notes):
    index.add_document(notes)
    result = index.search("What is the GDP of India in 2025?", [notes.doc_id], min_score=FAKE_MIN)
    assert not result.supported
    assert result.evidence == []


def test_search_only_uses_selected_documents(index, notes):
    other = load_text("Newton's laws describe force, mass and acceleration.", name="physics.txt")
    index.add_document(notes)
    index.add_document(other)
    result = index.search("force mass acceleration Newton", [notes.doc_id], min_score=FAKE_MIN)
    assert all(c.chunk.doc_id == notes.doc_id for c in result.chunks)
    assert not result.supported


def test_no_documents_means_not_supported(index, notes):
    index.add_document(notes)
    assert not index.search("chlorophyll", [], min_score=FAKE_MIN).supported
    assert not index.search("chlorophyll", ["missing-doc-id"], min_score=FAKE_MIN).supported
    assert not index.search("   ", [notes.doc_id], min_score=FAKE_MIN).supported


def test_index_persists_across_restarts(tmp_path, fake_embedder, notes):
    NoteIndex(fake_embedder, persist_dir=tmp_path / "db").add_document(notes)
    reopened = NoteIndex(fake_embedder, persist_dir=tmp_path / "db")
    assert reopened.has_document(notes.doc_id)


def test_delete_document(index, notes):
    index.add_document(notes)
    index.delete_document(notes.doc_id)
    assert not index.has_document(notes.doc_id)


def test_get_chunks_returns_page_order(index, notes):
    index.add_document(notes)
    chunks = index.get_chunks([notes.doc_id])
    assert len(chunks) == index.chunk_count(notes.doc_id)
    assert [c.page_number for c in chunks] == sorted(c.page_number for c in chunks)
    assert index.get_chunks([]) == []
