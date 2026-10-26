"""Check the mind map with REAL Gemma 4 + EmbeddingGemma (needs Ollama running).

Run from the project folder:  python scripts/check_mind_map.py
It builds the map for the sample notes, prints it as an outline and saves mind_map_check.svg.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.document_loader import load_document  # noqa: E402
from src.embeddings import OllamaEmbedder  # noqa: E402
from src.errors import NoteWiseError  # noqa: E402
from src.gemma_client import GemmaClient, check_health  # noqa: E402
from src.mind_map import as_outline, generate_mind_map, render_svg  # noqa: E402
from src.retrieval import NoteIndex  # noqa: E402


def main() -> None:
    health = check_health()
    print("Ollama:", health.message)
    if not health.ok:
        sys.exit(1)
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_data" / "photosynthesis_notes.pdf"
    doc = load_document(pdf.read_bytes(), pdf.name)
    index = NoteIndex(OllamaEmbedder(), persist_dir=tempfile.mkdtemp(prefix="notewise_map_"))
    index.add_document(doc)
    start = time.time()
    mm = generate_mind_map(index.get_chunks([doc.doc_id]), GemmaClient())
    print(f"\nBuilt in {time.time() - start:.1f}s\n")
    print(as_outline(mm))
    out = ROOT / "mind_map_check.svg"
    out.write_text(render_svg(mm), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    try:
        main()
    except NoteWiseError as exc:
        print("ERROR:", exc)
        sys.exit(1)
