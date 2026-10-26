"""Build the demo PDFs from sample_data/photosynthesis_notes.txt.

Creates:
  sample_data/photosynthesis_notes.pdf          normal PDF with selectable text (8 pages)
  sample_data/photosynthesis_notes_scanned.pdf  image-only PDF, to test OCR

Run from the project folder:  python scripts/make_sample_pdfs.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sample_data" / "photosynthesis_notes.txt"
TEXT_PDF = ROOT / "sample_data" / "photosynthesis_notes.pdf"
SCANNED_PDF = ROOT / "sample_data" / "photosynthesis_notes_scanned.pdf"


def split_pages(raw: str) -> list[tuple[str, str]]:
    pages = []
    for block in raw.split("\n# "):
        block = block.lstrip("# ").strip()
        if not block:
            continue
        title, _, body = block.partition("\n")
        pages.append((title.strip(), body.strip()))
    return pages


def build_text_pdf(pages: list[tuple[str, str]], out: Path) -> None:
    doc = pymupdf.open()
    for number, (title, body) in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((50, 60), "Biology Notes - Photosynthesis", fontsize=9, color=(0.4, 0.4, 0.4))
        page.insert_text((50, 95), title, fontsize=17, fontname="hebo")
        box = pymupdf.Rect(50, 115, 545, 800)
        paragraphs = "\n\n".join(body.split("\n"))
        overflow = page.insert_textbox(box, paragraphs, fontsize=11, lineheight=1.35)
        if overflow < 0:
            raise RuntimeError(f"Text did not fit on page {number}: {title}")
        page.insert_text((280, 825), f"Page {number}", fontsize=9, color=(0.4, 0.4, 0.4))
    doc.save(out)
    doc.close()


def build_scanned_pdf(text_pdf: Path, out: Path, dpi: int = 130) -> None:
    """Render each page to a picture, then put only the pictures in a new PDF."""
    src = pymupdf.open(text_pdf)
    dst = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        new_page = dst.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=pix.tobytes("jpeg", jpg_quality=70))
    dst.save(out)
    src.close()
    dst.close()


if __name__ == "__main__":
    pages = split_pages(SRC.read_text(encoding="utf-8"))
    build_text_pdf(pages, TEXT_PDF)
    build_scanned_pdf(TEXT_PDF, SCANNED_PDF)
    print(f"Created {TEXT_PDF.name} ({len(pages)} pages) and {SCANNED_PDF.name}")
