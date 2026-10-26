"""Phase 2 check with REAL Gemma 4 + EmbeddingGemma (needs Ollama running).

Runs the demo questions through the full pipeline and prints the answer,
its status, the cited pages and how long Gemma took.

Run from the project folder:  python scripts/check_tutor.py
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
from src.gemma_client import GemmaClient, check_health  # noqa: E402
from src.retrieval import NoteIndex  # noqa: E402
from src.tutor import answer_question  # noqa: E402

# (question, what we expect)
QUESTIONS = [
    ("Why do leaves look green?", "answered"),
    ("Where does the oxygen released by plants come from?", "answered"),
    ("Where do the light reactions and the Calvin cycle each take place?", "answered (2 passages)"),
    ("What is the GDP of India?", "not_in_notes"),
    ("Who discovered photosynthesis?", "not_in_notes or partial"),
    ("What is the role of chlorophyll, and who discovered it?", "partial"),
]


def main() -> None:
    health = check_health()
    print("Ollama:", health.message)
    if not health.ok:
        sys.exit(1)

    sample = ROOT / "sample_data" / "photosynthesis_notes.pdf"
    doc = load_document(sample.read_bytes(), sample.name)
    index = NoteIndex(OllamaEmbedder(), persist_dir=tempfile.mkdtemp(prefix="notewise_tutor_"))
    index.add_document(doc)
    gemma = GemmaClient()

    print(f"\nModel: {config.CHAT_MODEL}  |  threshold: {config.MIN_RELEVANCE}\n")
    print("(The first answer is slower while Gemma loads into memory.)\n")
    for question, expected in QUESTIONS:
        start = time.time()
        ans = answer_question(question, [doc.doc_id], index, gemma)
        took = time.time() - start
        pages = sorted({c.page_number for c in ans.citations})
        print("=" * 78)
        print(f"Q: {question}")
        print(f"Expected: {expected}  |  Got: {ans.status}  |  top score {ans.top_score:.2f}  |  {took:.1f}s")
        print(f"Cited pages: {pages or '-'}")
        print(f"A: {ans.text}")
        if ans.raw_text and ans.raw_text != ans.text:
            print(f"(raw: {ans.raw_text[:300]})")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except NoteWiseError as exc:
        print("ERROR:", exc)
        sys.exit(1)
