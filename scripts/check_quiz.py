"""Phase 4 check with REAL Gemma 4 (needs Ollama running): demo scenario 2.

1. Generates a 5-question quiz from the sample notes and times it.
2. Prints every question with its answer key and source page.
3. Answers question 1 WRONG on purpose and prints the feedback a student sees.

Run from the project folder:  python scripts/check_quiz.py
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
from src.quiz_evaluator import QuizSession  # noqa: E402
from src.quiz_generator import LETTERS, generate_quiz  # noqa: E402
from src.retrieval import NoteIndex  # noqa: E402


def main() -> None:
    health = check_health()
    print("Ollama:", health.message)
    if not health.ok:
        sys.exit(1)

    sample = ROOT / "sample_data" / "photosynthesis_notes.pdf"
    doc = load_document(sample.read_bytes(), sample.name)
    index = NoteIndex(OllamaEmbedder(), persist_dir=tempfile.mkdtemp(prefix="notewise_quiz_"))
    index.add_document(doc)

    print("\nGenerating 5 questions (a writer call + a self-check call per question, plus retries)...")
    start = time.time()
    result = generate_quiz(index.get_chunks([doc.doc_id]), GemmaClient(), num_questions=5)
    print(f"{result.message}  |  skipped passages: {result.skipped}  |  rejected by self-check: {result.rejected_by_check}  |  {time.time() - start:.1f}s\n")

    for n, q in enumerate(result.questions, start=1):
        print("=" * 78)
        print(f"Q{n} (page {q.page_number}): {q.question}")
        for i, opt in enumerate(q.options):
            mark = "  <- correct" if i == q.correct_index else ""
            print(f"   {LETTERS[i]}. {opt}{mark}")
        print(f"   Explanation: {q.explanation}")
        print(f"   Hint: {q.hint}")

    if not result.questions:
        return
    session = QuizSession(result.questions)
    q1 = result.questions[0]
    wrong = (q1.correct_index + 1) % 4
    feedback = session.submit(q1.id, wrong)
    print("=" * 78)
    print(f"\nStudent picks {LETTERS[wrong]} on Q1 (wrong on purpose):")
    print(f"  {feedback.headline}")
    print(f"  Why your answer is wrong: {feedback.why_wrong}")
    print(f"  Explanation: {feedback.explanation}")
    print(f"  Hint: {feedback.hint}")
    print(f"  Source: page {feedback.page_number} -> \"{feedback.source_text[:160]}...\"")
    print(f"\nScore so far: {session.score}/{session.answered}  |  pages to revise: {session.pages_to_revise()}")


if __name__ == "__main__":
    try:
        main()
    except NoteWiseError as exc:
        print("ERROR:", exc)
        sys.exit(1)
