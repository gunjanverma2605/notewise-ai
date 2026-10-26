"""Check Nova with REAL Gemma 4 + EmbeddingGemma (needs Ollama running).

Asks the same question in each Nova mode and follow-up style, and checks that
off-topic requests are still refused. Run:  python scripts/check_mentor.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import mentor  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.embeddings import OllamaEmbedder  # noqa: E402
from src.errors import NoteWiseError  # noqa: E402
from src.gemma_client import GemmaClient, check_health  # noqa: E402
from src.retrieval import NoteIndex  # noqa: E402
from src.study_mode import page_options  # noqa: E402


def show(title: str, ans, took: float) -> None:
    pages = sorted({c.page_number for c in ans.citations})
    print("=" * 78)
    print(f"{title}\nStatus: {ans.status}  |  pages {pages or '-'}  |  {took:.1f}s\n{ans.text}")


def main() -> None:
    health = check_health()
    print("Ollama:", health.message)
    if not health.ok:
        sys.exit(1)
    sample = ROOT / "sample_data" / "photosynthesis_notes.pdf"
    doc = load_document(sample.read_bytes(), sample.name)
    index = NoteIndex(OllamaEmbedder(), persist_dir=tempfile.mkdtemp(prefix="notewise_nova_"))
    index.add_document(doc)
    gemma = GemmaClient()
    profile = mentor.build_profile([], [], [], [], page_options(index.get_chunks([doc.doc_id])))
    q = "What happens in the Calvin cycle?"
    history: list[dict] = []
    for mode in ("explain", "doubt"):
        t = time.time()
        ans = mentor.ask(q, mode, profile, [doc.doc_id], index, gemma)
        show(f"[{mode}] {q}", ans, time.time() - t)
        history = [{"role": "user", "content": q}, {"role": "assistant", "content": ans.text}]
    for key, (label, text, _) in mentor.FOLLOW_UPS.items():
        t = time.time()
        ans = mentor.ask(text, "explain", profile, [doc.doc_id], index, gemma, history=history, follow_up=key)
        show(f"[follow-up: {label}]", ans, time.time() - t)
    t = time.time()
    ans = mentor.ask("Who won the cricket world cup?", "explain", profile, [doc.doc_id], index, gemma)
    show("[off-topic, expect not_in_notes]", ans, time.time() - t)
    print("=" * 78)
    print("Quiz pages for 'Calvin cycle':", mentor.pages_for_query("Calvin cycle", [doc.doc_id], index))
    print("Quiz pages for 'cricket':", mentor.pages_for_query("cricket world cup", [doc.doc_id], index), "(expect [])")


if __name__ == "__main__":
    try:
        main()
    except NoteWiseError as exc:
        print("ERROR:", exc)
        sys.exit(1)
