import pytest

from src import config
from src.document_loader import compute_doc_id, load_document, load_text
from src.errors import DocumentError
from src.ocr_processor import tesseract_available
from tests.conftest import make_pdf

needs_tesseract = pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")


def test_text_pdf_reads_every_page_with_page_numbers(sample_pdf_bytes):
    doc = load_document(sample_pdf_bytes, "photosynthesis_notes.pdf")
    assert doc.page_count == 8
    assert [p.page_number for p in doc.pages] == list(range(1, 9))
    assert all(p.method == "text" for p in doc.pages)
    assert "Calvin cycle" in doc.pages[3].text
    assert "thylakoid" in doc.pages[1].text


@needs_tesseract
def test_scanned_pdf_uses_ocr(scanned_pdf_bytes):
    doc = load_document(scanned_pdf_bytes, "scanned.pdf")
    assert doc.page_count == 8
    assert doc.ocr_pages == list(range(1, 9))
    assert "chlorophyll" in doc.pages[1].text.lower()
    assert "rubisco" in doc.pages[3].text.lower()


def test_scanned_pdf_without_ocr_is_rejected(scanned_pdf_bytes):
    with pytest.raises(DocumentError, match="No readable text"):
        load_document(scanned_pdf_bytes, "scanned.pdf", use_ocr=False)


def test_blank_pdf_is_rejected():
    with pytest.raises(DocumentError, match="No readable text"):
        load_document(make_pdf(["", ""]), "blank.pdf")


def test_partly_blank_pdf_warns_about_empty_page():
    long_text = "Photosynthesis happens in chloroplasts using light energy."
    doc = load_document(make_pdf([long_text, ""]), "partial.pdf", use_ocr=False)
    assert doc.empty_pages == [2]
    assert any("page(s): 2" in w for w in doc.warnings)


def test_corrupted_pdf_is_rejected():
    with pytest.raises(DocumentError, match="could not be opened"):
        load_document(b"%PDF-1.4 this is not really a pdf", "broken.pdf")


def test_password_protected_pdf_is_rejected():
    data = make_pdf(["Secret notes about photosynthesis and chlorophyll."], password="1234")
    with pytest.raises(DocumentError, match="password"):
        load_document(data, "locked.pdf")


def test_empty_file_is_rejected():
    with pytest.raises(DocumentError, match="empty"):
        load_document(b"", "notes.pdf")


def test_unsupported_extension_is_rejected():
    with pytest.raises(DocumentError, match="not supported"):
        load_document(b"hello", "notes.docx")


def test_too_many_pages_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_PAGES", 2)
    with pytest.raises(DocumentError, match="limit"):
        load_document(make_pdf(["a", "b", "c"]), "long.pdf")


def test_txt_file_and_form_feed_pages():
    doc = load_document("First page text.\fSecond page text.".encode(), "notes.txt")
    assert doc.source_type == "txt"
    assert [p.page_number for p in doc.pages] == [1, 2]


def test_non_utf8_txt_still_loads():
    doc = load_document("Café notes about leaves".encode("latin-1"), "notes.txt")
    assert "leaves" in doc.pages[0].text


def test_pasted_text_and_empty_paste():
    doc = load_text("Chlorophyll absorbs red and blue light.")
    assert doc.source_type == "paste" and doc.page_count == 1
    with pytest.raises(DocumentError):
        load_text("   \n  ")


def test_running_header_and_page_labels_are_removed(sample_pdf_bytes):
    doc = load_document(sample_pdf_bytes, "notes.pdf")
    for page in doc.pages:
        assert "Biology Notes - Photosynthesis" not in page.text
        assert not page.text.rstrip().endswith(f"Page {page.page_number}")
    assert doc.pages[0].text.startswith("Introduction to Photosynthesis")


def test_doc_id_is_stable(sample_pdf_bytes):
    assert compute_doc_id(sample_pdf_bytes) == compute_doc_id(sample_pdf_bytes)
    a = load_document(sample_pdf_bytes, "a.pdf")
    b = load_document(sample_pdf_bytes, "renamed.pdf")
    assert a.doc_id == b.doc_id
