"""Load PDFs, TXT files and pasted notes into page-numbered text.

Pipeline for PDFs:
    validate -> for every page: try normal text extraction
             -> if the page has (almost) no text, run OCR on it
             -> keep the original page number with the text
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src import config
from src.errors import DocumentError
from src.ocr_processor import ocr_page, tesseract_available

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class Page:
    page_number: int          # 1-based, matches the page number the student sees
    text: str
    method: str               # "text", "ocr" or "empty"


@dataclass
class LoadedDocument:
    doc_id: str               # stable hash of the file contents
    name: str
    source_type: str          # "pdf", "txt" or "paste"
    pages: list[Page]
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def ocr_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.method == "ocr"]

    @property
    def empty_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.method == "empty"]


def compute_doc_id(data: bytes) -> str:
    """Same file -> same id, so re-uploading never creates duplicates."""
    return hashlib.sha256(data).hexdigest()[:16]


def clean_text(text: str) -> str:
    """Normalise whitespace but keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(data: bytes, filename: str, use_ocr: bool = True) -> LoadedDocument:
    """Entry point: pick the right loader from the file extension."""
    _validate_size(data)
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentError(
            f"'{filename}' is not supported. Please upload a PDF or a .txt file."
        )
    if ext == ".pdf":
        return load_pdf(data, filename, use_ocr=use_ocr)
    return load_text(_decode(data), filename, source_type="txt", raw=data)


def load_pdf(data: bytes, filename: str, use_ocr: bool = True) -> LoadedDocument:
    _validate_size(data)
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentError(
            f"'{filename}' could not be opened. The file may be damaged or not a real PDF."
        ) from exc

    with pdf:
        if pdf.needs_pass:
            raise DocumentError(
                f"'{filename}' is password-protected. Please remove the password and upload it again."
            )
        if pdf.page_count == 0:
            raise DocumentError(f"'{filename}' has no pages.")
        if pdf.page_count > config.MAX_PAGES:
            raise DocumentError(
                f"'{filename}' has {pdf.page_count} pages. The limit is {config.MAX_PAGES} pages."
            )

        ocr_ready = use_ocr and tesseract_available()
        warnings: list[str] = []
        pages: list[Page] = []

        for index, pdf_page in enumerate(pdf):               # every page, not just the first few
            page_number = index + 1
            try:
                text = clean_text(pdf_page.get_text("text"))
            except Exception:
                text = ""
            method = "text"

            if len(text) < config.MIN_CHARS_FOR_TEXT_PAGE:
                if ocr_ready:
                    try:
                        ocr_text = clean_text(ocr_page(pdf_page))
                    except Exception:
                        ocr_text = ""
                        warnings.append(f"OCR failed on page {page_number}.")
                    if len(ocr_text) > len(text):
                        text, method = ocr_text, "ocr"
                elif use_ocr:
                    warnings.append(
                        f"Page {page_number} looks scanned, but Tesseract OCR is not installed."
                    )

            if not text:
                method = "empty"
            pages.append(Page(page_number=page_number, text=text, method=method))

    pages = remove_repeated_lines(pages)
    doc = LoadedDocument(
        doc_id=compute_doc_id(data),
        name=filename,
        source_type="pdf",
        pages=pages,
        warnings=warnings,
    )
    if doc.char_count == 0:
        raise DocumentError(
            f"No readable text was found in '{filename}'. "
            "If it is a photo or scan, check that the text is clear and printed (not handwritten)."
        )
    if doc.empty_pages:
        doc.warnings.append(
            "No text found on page(s): " + ", ".join(map(str, doc.empty_pages)) + "."
        )
    return doc


def load_text(
    text: str, name: str = "Pasted notes", source_type: str = "paste", raw: bytes | None = None
) -> LoadedDocument:
    """Plain text has no real pages. A form-feed character (\\f) starts a new page;
    otherwise the whole text is page 1."""
    parts = [clean_text(part) for part in text.split("\f")]
    pages = [
        Page(page_number=i + 1, text=part, method="text" if part else "empty")
        for i, part in enumerate(parts)
    ]
    if not any(p.text for p in pages):
        raise DocumentError("The notes are empty. Please paste or upload some text.")
    pages = [p for p in pages if p.text] or pages
    return LoadedDocument(
        doc_id=compute_doc_id(raw if raw is not None else text.encode("utf-8")),
        name=name,
        source_type=source_type,
        pages=pages,
    )


def save_upload(data: bytes, doc: LoadedDocument, upload_dir: str | Path | None = None) -> Path:
    """Keep a copy of the original file as data/uploads/<doc_id>.<ext>, so a saved
    chat can rebuild its notes later. Pasted notes are saved as .txt."""
    folder = Path(upload_dir or config.UPLOAD_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    ext = ".pdf" if doc.source_type == "pdf" else ".txt"
    path = folder / f"{doc.doc_id}{ext}"
    if not path.exists():
        path.write_bytes(data)
    return path


_PAGE_LABEL = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)


def remove_repeated_lines(pages: list[Page], min_pages: int = 3) -> list[Page]:
    """Drop running headers/footers (a line that appears on most pages) and bare
    page labels like 'Page 3'. They add noise to every passage."""
    with_text = [p for p in pages if p.text]
    counts: dict[str, int] = {}
    if len(with_text) >= min_pages:
        for p in with_text:
            for line in {ln.strip().lower() for ln in p.text.split("\n") if ln.strip()}:
                counts[line] = counts.get(line, 0) + 1
    repeated = {line for line, n in counts.items() if n >= max(min_pages, len(with_text) * 0.6)}

    cleaned: list[Page] = []
    for p in pages:
        kept = [
            ln for ln in p.text.split("\n")
            if ln.strip().lower() not in repeated and not _PAGE_LABEL.match(ln.strip())
        ]
        text = clean_text("\n".join(kept))
        method = p.method if text else "empty"
        cleaned.append(Page(page_number=p.page_number, text=text, method=method))
    return cleaned


def _validate_size(data: bytes) -> None:
    if not data:
        raise DocumentError("The file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_FILE_MB:
        raise DocumentError(
            f"The file is {size_mb:.1f} MB. The limit is {config.MAX_FILE_MB} MB."
        )


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")
