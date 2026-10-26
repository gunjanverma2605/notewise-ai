"""OCR for scanned PDF pages using Tesseract.

OCR only turns a picture of text into real text. It does NOT make the
language model able to read more pages; RAG (retrieval.py) handles that.
"""

from __future__ import annotations

import io
from functools import lru_cache

from src import config


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    """True if both pytesseract and the Tesseract program are installed."""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_page(page, dpi: int | None = None, lang: str | None = None) -> str:
    """Render one PyMuPDF page to an image and read its text with Tesseract."""
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi or config.OCR_DPI)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, lang=lang or config.OCR_LANG)
