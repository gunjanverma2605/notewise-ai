"""Phase 3 check with REAL Gemma 4 (needs Ollama running): demo scenario 4.

1. Starts a chat and asks a question, then follow-ups like "explain that simpler".
2. Prints how each follow-up was rewritten for search.
3. Starts a second chat, then reopens the first one from the database.

Uses a temporary folder, so your real data/ folder is not touched.
Run from the project folder:  python scripts/check_conversation.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.conversation_manager import ConversationManager  # noqa: E402
from src.database import Database  # noqa: E402
from src.document_loader import load_document  # noqa: E402
from src.embeddings import OllamaEmbedder  # noqa: E402
from src.errors import NoteWiseError  # noqa: E402
from src.gemma_client import GemmaClient, check_health  # noqa: E402
from src.retrieval import NoteIndex  # noqa: E402

CHAT_1 = [
    "What happens in the light-dependent reactions?",
    "Explain its second step in more detail",
    "Explain that in simpler words",
    "Which gas is released in it?",
]


def main() -> None:
    health = check_health()
    print("Ollama:", health.message)
    if not health.ok:
        sys.exit(1)

    tmp = Path(tempfile.mkdtemp(prefix="notewise_chat_"))
    db_path = tmp / "notewise.db"
    index = NoteIndex(OllamaEmbedder(), persist_dir=tmp / "chroma")
    manager = ConversationManager(Database(db_path), index, upload_dir=tmp / "uploads")
    gemma = GemmaClient()

    sample = ROOT / "sample_data" / "photosynthesis_notes.pdf"
    data = sample.read_bytes()
    doc = load_document(data, sample.name)

    chat1 = manager.new_chat()
    manager.add_document(chat1, doc, data)
    print("\n== Chat 1 ==")
    for question in CHAT_1:
        start = time.time()
        ans = manager.ask(chat1, question, gemma)
        pages = sorted({c.page_number for c in ans.citations})
        print("-" * 78)
        print(f"Student:      {question}")
        if ans.search_query != question:
            print(f"Searched as:  {ans.search_query}")
        print(f"Status:       {ans.status}  |  pages {pages or '-'}  |  {time.time() - start:.1f}s")
        print(f"NoteWise:     {ans.text}")

    print("\n== New chat, then reopen chat 1 ==")
    chat2 = manager.new_chat()
    manager.add_document(chat2, doc, data)
    manager.ask(chat2, "Why do leaves look green?", gemma)

    reopened = ConversationManager(Database(db_path), index, upload_dir=tmp / "uploads")  # like a browser refresh
    for conv in reopened.db.list_conversations():
        print(f"Saved chat: '{conv.title}'  ({reopened.db.message_count(conv.id)} messages)")
    msgs = reopened.messages(chat1)
    print(f"Reopened chat 1: {len(msgs)} messages, first: \"{msgs[0].content}\"")
    ans = reopened.ask(chat1, "Summarise what we discussed in one line", gemma)
    print(f"Continued chat 1 -> status {ans.status}: {ans.text}")
    print(f"\nTemporary data folder: {tmp}")


if __name__ == "__main__":
    try:
        main()
    except NoteWiseError as exc:
        print("ERROR:", exc)
        sys.exit(1)
