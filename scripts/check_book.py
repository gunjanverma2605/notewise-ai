"""Check the Revision Book with REAL Gemma 4 (needs Ollama running).

Run:  python scripts/check_book.py [path/to/notes.pdf]
Saves revision_book_check.html in the project folder (open it, Cmd+P to save as PDF).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.book import generate_book, render_download  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.embeddings import OllamaEmbedder  # noqa: E402
from src.errors import NoteWiseError  # noqa: E402
from src.gemma_client import GemmaClient, check_health  # noqa: E402
from src.mind_map import generate_mind_map  # noqa: E402
from src.retrieval import NoteIndex  # noqa: E402


def main() -> None:
    health = check_health()
    print("Ollama:", health.message)
    if not health.ok:
        sys.exit(1)
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_data" / "photosynthesis_notes.pdf"
    doc = load_document(pdf.read_bytes(), pdf.name)
    index = NoteIndex(OllamaEmbedder(), persist_dir=tempfile.mkdtemp(prefix="notewise_book_"))
    index.add_document(doc)
    chunks = index.get_chunks([doc.doc_id])
    gemma = GemmaClient()
    start = time.time()
    mm = generate_mind_map(chunks, gemma)
    print(f"Mind map: {len(mm.branches)} topics in {time.time() - start:.1f}s")
    bk = generate_book(mm, lambda name, page: [c for c in chunks if c.page_number == page], gemma,
                       on_progress=lambda d, t: print(f"  chapter {d}/{t} done"))
    print(f"Book ready in {time.time() - start:.1f}s")
    for c in bk.chapters:
        print(f"- {c.title} (p. {c.page_number}) grounded={c.grounded}")
    out = ROOT / "revision_book_check.html"
    out.write_text(render_download(bk), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    try:
        main()
    except NoteWiseError as exc:
        print("ERROR:", exc)
        sys.exit(1)
