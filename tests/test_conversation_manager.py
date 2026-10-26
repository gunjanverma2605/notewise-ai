import pytest

from src.conversation_manager import (
    REWRITE_PROMPT,
    TITLE_PROMPT,
    ConversationManager,
    fallback_title,
    needs_rewrite,
    rewrite_query,
)
from src.database import Database
from src.document_loader import load_document
from src.errors import OllamaUnavailableError
from src.retrieval import NoteIndex

SUPPORTED_Q = "Which enzyme joins carbon dioxide to RuBP in phase 1 of the Calvin cycle?"


class ScriptedGemma:
    """Answers differently depending on which job it is asked to do."""

    def __init__(self, rewrite="Explain RuBisCO carbon fixation RuBP Calvin cycle in simple words",
                 title="Calvin Cycle Revision", answer="RuBisCO fixes carbon dioxide [S1]."):
        self.rewrite, self.title, self.answer = rewrite, title, answer
        self.calls: list[str] = []

    def chat(self, messages, json_mode=False, temperature=None):
        system = messages[0]["content"]
        if system == REWRITE_PROMPT:
            self.calls.append("rewrite")
            return self.rewrite
        if system == TITLE_PROMPT:
            self.calls.append("title")
            return self.title
        self.calls.append("answer")
        return self.answer


class BrokenGemma:
    def chat(self, *args, **kwargs):
        raise OllamaUnavailableError("down")


@pytest.fixture
def env(tmp_path, fake_embedder, sample_pdf_bytes):
    db = Database(tmp_path / "chat.db")
    index = NoteIndex(fake_embedder, persist_dir=tmp_path / "chroma")
    manager = ConversationManager(db, index, upload_dir=tmp_path / "uploads")
    doc = load_document(sample_pdf_bytes, "photosynthesis_notes.pdf")
    return manager, db, index, doc, sample_pdf_bytes, tmp_path, fake_embedder


def test_first_question_is_saved_answered_and_titled(env):
    manager, db, _, doc, data, *_ = env
    cid = manager.new_chat()
    manager.add_document(cid, doc, data)
    gemma = ScriptedGemma()
    ans = manager.ask(cid, SUPPORTED_Q, gemma)
    assert ans.status == "answered"
    assert gemma.calls == ["answer", "title"]            # first question: no rewrite needed
    msgs = db.get_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].citations[0]["page_number"] == ans.citations[0].page_number
    assert db.get_conversation(cid).title == "Calvin Cycle Revision"


def test_follow_up_is_rewritten_for_search_only(env):
    manager, db, _, doc, data, *_ = env
    cid = manager.new_chat()
    manager.add_document(cid, doc, data)
    gemma = ScriptedGemma()
    manager.ask(cid, SUPPORTED_Q, gemma)
    ans = manager.ask(cid, "Explain that in simpler words", gemma)
    assert gemma.calls[-2:] == ["rewrite", "answer"]
    assert ans.question == "Explain that in simpler words"
    assert "RuBisCO" in ans.search_query
    assert ans.status == "answered"
    user_msgs = [m for m in db.get_messages(cid) if m.role == "user"]
    assert user_msgs[-1].content == "Explain that in simpler words"   # student's words are kept


def test_title_only_set_once(env):
    manager, db, _, doc, data, *_ = env
    cid = manager.new_chat()
    manager.add_document(cid, doc, data)
    manager.ask(cid, SUPPORTED_Q, ScriptedGemma(title="First Title"))
    manager.ask(cid, "Explain that again", ScriptedGemma(title="Second Title"))
    assert db.get_conversation(cid).title == "First Title"


def test_chat_without_notes_asks_for_upload(env):
    manager, db, *_ = env
    cid = manager.new_chat()
    gemma = ScriptedGemma()
    ans = manager.ask(cid, "What is chlorophyll?", gemma)
    assert ans.status == "no_document" and gemma.calls == []
    assert db.message_count(cid) == 2


def test_chats_only_search_their_own_notes(env):
    manager, db, index, doc, data, tmp_path, _ = env
    from src.document_loader import load_text
    physics = load_text("Newton's second law: force equals mass times acceleration.", name="physics.txt")
    a = manager.new_chat()
    manager.add_document(a, doc, data)
    b = manager.new_chat()
    manager.add_document(b, physics, b"Newton's second law: force equals mass times acceleration.")
    ans = manager.ask(b, SUPPORTED_Q, ScriptedGemma())
    assert ans.status == "not_in_notes"                  # chat B never sees chat A's notes


def test_reopened_chat_restores_index_from_saved_file(env):
    manager, db, _, doc, data, tmp_path, fake_embedder = env
    cid = manager.new_chat()
    manager.add_document(cid, doc, data)
    manager.ask(cid, SUPPORTED_Q, ScriptedGemma())
    # simulate a lost vector index (e.g. data/chroma deleted)
    fresh_index = NoteIndex(fake_embedder, persist_dir=tmp_path / "chroma_new")
    reopened = ConversationManager(Database(tmp_path / "chat.db"), fresh_index, upload_dir=tmp_path / "uploads")
    status = reopened.document_status(cid)
    assert status[0].ready and status[0].restored
    assert len(reopened.messages(cid)) == 2
    assert reopened.ask(cid, "Explain that again", ScriptedGemma()).status == "answered"


def test_missing_file_asks_for_reupload(env):
    manager, db, _, doc, data, tmp_path, fake_embedder = env
    cid = manager.new_chat()
    manager.add_document(cid, doc, data)
    (tmp_path / "uploads" / f"{doc.doc_id}.pdf").unlink()
    fresh = ConversationManager(db, NoteIndex(fake_embedder, persist_dir=tmp_path / "empty"), upload_dir=tmp_path / "uploads")
    status = fresh.document_status(cid)
    assert not status[0].ready and "re-upload" in status[0].message
    gemma = ScriptedGemma()
    ans = fresh.ask(cid, SUPPORTED_Q, gemma)
    assert ans.status == "no_document" and "re-upload" in ans.text
    assert gemma.calls == []                              # never answers from other notes


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Explain its second step in more detail", True),
        ("Explain that in simpler words", True),
        ("Why?", True),
        ("Give me an example", True),
        ("Summarise what we discussed in one line", True),
        ("What happens in the Calvin cycle in the stroma?", False),
    ],
)
def test_needs_rewrite(question, expected):
    history = [{"role": "user", "content": "What is photosynthesis?"}, {"role": "assistant", "content": "..."}]
    assert needs_rewrite(question, history) is expected


def test_no_rewrite_without_history():
    assert not needs_rewrite("Explain that again", [])


def test_rewrite_falls_back_when_gemma_fails_or_rambles():
    history = [{"role": "user", "content": "What is photolysis?"}, {"role": "assistant", "content": "Splitting water [p. 3]."}]
    assert rewrite_query("Explain it simpler", history, BrokenGemma()) == "What is photolysis? Explain it simpler"
    rambling = ScriptedGemma(rewrite="x" * 500)
    assert rewrite_query("Explain it simpler", history, rambling) == "What is photolysis? Explain it simpler"
    labelled = ScriptedGemma(rewrite='Standalone question: "What is photolysis in simple words?"')
    assert rewrite_query("Explain it simpler", history, labelled) == "What is photolysis in simple words?"


def test_fallback_title():
    assert fallback_title("What is the Calvin cycle?") == "Calvin cycle"
    assert fallback_title("") == "New chat"
