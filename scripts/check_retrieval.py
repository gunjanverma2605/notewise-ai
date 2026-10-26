"""Phase 1 check with the REAL EmbeddingGemma model (needs Ollama running).

1. Loads the sample PDF (and the scanned one, to test OCR).
2. Indexes it in a temporary ChromaDB folder.
3. Runs supported and unsupported questions and prints similarity scores.

Use the printed scores to pick MIN_RELEVANCE: a value between the lowest
"supported" score and the highest "unsupported" score.

Run from the project folder:  python scripts/check_retrieval.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.embeddings import OllamaEmbedder  # noqa: E402
from src.errors import NoteWiseError  # noqa: E402
from src.retrieval import NoteIndex  # noqa: E402

# (question, expected page or None if the notes don't answer it)
QUESTIONS = [
    ("Why do leaves look green?", 2),
    ("Where does the oxygen released in photosynthesis come from?", 3),
    ("What does RuBisCO do?", 4),
    ("How does temperature affect the rate of photosynthesis?", 5),
    ("How do CAM plants save water?", 6),
    ("What is the GDP of India?", None),
    ("Who won the cricket world cup in 2011?", None),
    ("Explain Newton's second law of motion.", None),
    ("Who discovered photosynthesis?", None),  # on-topic but missing: the hard case
]


def main() -> None:
    sample = ROOT / "sample_data" / "photosynthesis_notes.pdf"
    scanned = ROOT / "sample_data" / "photosynthesis_notes_scanned.pdf"

    print("== Loading PDFs ==")
    doc = load_document(sample.read_bytes(), sample.name)
    print(f"{doc.name}: {doc.page_count} pages, {doc.char_count} chars, OCR pages: {doc.ocr_pages}")
    start = time.time()
    scan = load_document(scanned.read_bytes(), scanned.name)
    print(f"{scan.name}: {scan.page_count} pages, OCR pages: {scan.ocr_pages} ({time.time() - start:.1f}s)")
    print("OCR sample (page 2):", scan.pages[1].text[:160].replace("\n", " "), "...")

    print("\n== Indexing with", config.EMBED_MODEL, "==")
    index = NoteIndex(OllamaEmbedder(), persist_dir=tempfile.mkdtemp(prefix="notewise_check_"))
    start = time.time()
    n = index.add_document(doc)
    print(f"Indexed {n} chunks in {time.time() - start:.1f}s")

    print(f"\n== Questions (current MIN_RELEVANCE = {config.MIN_RELEVANCE}) ==")
    supported_scores, unsupported_scores = [], []
    for question, expected in QUESTIONS:
        result = index.search(question, [doc.doc_id], min_score=0.0)
        top = result.chunks[0]
        tag = "IN NOTES " if expected else "NOT IN NOTES"
        page_ok = "" if expected is None else ("  page OK" if top.chunk.page_number == expected else f"  page WRONG (expected {expected})")
        print(f"[{tag}] score={top.score:.3f}  top page={top.chunk.page_number}{page_ok}  | {question}")
        (supported_scores if expected else unsupported_scores).append(top.score)

    clear_unsupported = unsupported_scores[:-1]  # exclude the 'who discovered' hard case
    print("\n== Suggested threshold ==")
    print(f"Lowest supported score:          {min(supported_scores):.3f}")
    print(f"Highest off-topic score:         {max(clear_unsupported):.3f}")
    if min(supported_scores) > max(clear_unsupported):
        suggestion = round((min(supported_scores) + max(clear_unsupported)) / 2, 2)
        print(f"Suggested MIN_RELEVANCE:         {suggestion}")
    else:
        print("Scores overlap: send this output to Claude to tune chunking/threshold.")
    print(f"'Who discovered...' score:       {unsupported_scores[-1]:.3f}  (Phase 2 handles this case)")


if __name__ == "__main__":
    try:
        main()
    except NoteWiseError as exc:
        print("ERROR:", exc)
        sys.exit(1)
