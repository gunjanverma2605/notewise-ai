from src.database import Database


def make_db(tmp_path):
    return Database(tmp_path / "test.db")


def test_conversation_crud(tmp_path):
    db = make_db(tmp_path)
    cid = db.create_conversation("New chat")
    assert db.get_conversation(cid).title == "New chat"
    db.rename_conversation(cid, "  Photosynthesis Revision  ")
    assert db.get_conversation(cid).title == "Photosynthesis Revision"
    db.rename_conversation(cid, "   ")
    assert db.get_conversation(cid).title == "Untitled chat"
    assert [c.id for c in db.list_conversations()] == [cid]


def test_messages_persist_after_reopening_database(tmp_path):
    db = make_db(tmp_path)
    cid = db.create_conversation()
    db.add_message(cid, "user", "What is photosynthesis?")
    db.add_message(cid, "assistant", "It makes food [p. 1].", status="answered",
                   citations=[{"label": "S1", "page_number": 1}])
    reopened = Database(tmp_path / "test.db")          # simulates a browser refresh / app restart
    msgs = reopened.get_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].citations == [{"label": "S1", "page_number": 1}]
    assert msgs[1].status == "answered"


def test_same_message_id_is_stored_once(tmp_path):
    db = make_db(tmp_path)
    cid = db.create_conversation()
    db.add_message(cid, "user", "Hello", message_id="fixed-id")
    db.add_message(cid, "user", "Hello", message_id="fixed-id")   # Streamlit rerun
    assert db.message_count(cid) == 1


def test_messages_keep_order_and_limit(tmp_path):
    db = make_db(tmp_path)
    cid = db.create_conversation()
    for i in range(10):
        db.add_message(cid, "user" if i % 2 == 0 else "assistant", f"m{i}")
    assert [m.content for m in db.get_messages(cid)] == [f"m{i}" for i in range(10)]
    assert [m.content for m in db.get_messages(cid, limit=3)] == ["m7", "m8", "m9"]


def test_delete_conversation_removes_messages_and_links(tmp_path):
    db = make_db(tmp_path)
    cid = db.create_conversation()
    db.upsert_document("doc1", "notes.pdf", "pdf", 8, None)
    db.link_document(cid, "doc1")
    db.add_message(cid, "user", "hi")
    db.delete_conversation(cid)
    assert db.get_conversation(cid) is None
    assert db.get_messages(cid) == []
    assert db.conversation_documents(cid) == []
    assert db.get_document("doc1") is not None          # the notes themselves are kept


def test_document_link_is_idempotent_and_upsert_keeps_path(tmp_path):
    db = make_db(tmp_path)
    cid = db.create_conversation()
    db.upsert_document("doc1", "notes.pdf", "pdf", 8, str(tmp_path / "doc1.pdf"))
    db.upsert_document("doc1", "notes.pdf", "pdf", 8, None)
    db.link_document(cid, "doc1")
    db.link_document(cid, "doc1")
    docs = db.conversation_documents(cid)
    assert len(docs) == 1 and docs[0].file_path.endswith("doc1.pdf")
    assert not docs[0].file_available                    # file was never written


def test_quiz_attempts_and_reset(tmp_path):
    db = make_db(tmp_path)
    db.add_quiz_attempt(3, 5, doc_id="doc1", attempt_id="a1")
    db.add_quiz_attempt(3, 5, doc_id="doc1", attempt_id="a1")
    assert len(db.list_quiz_attempts()) == 1
    db.create_conversation()
    db.reset_all()
    assert db.list_conversations() == [] and db.list_quiz_attempts() == []


def test_stats_and_attempts_per_chat(tmp_path):
    db = make_db(tmp_path)
    a, b = db.create_conversation(), db.create_conversation()
    db.add_message(a, "user", "q1")
    db.add_message(a, "assistant", "x", status="answered")
    db.add_message(b, "user", "q2")
    db.add_quiz_attempt(4, 5, conversation_id=a)
    db.add_quiz_attempt(1, 5, conversation_id=b)
    assert db.stats()["quizzes"] == 2 and db.stats()["questions"] == 2
    sa = db.stats(a)
    assert sa["quizzes"] == 1 and sa["quiz_correct"] == 4 and sa["questions"] == 1 and sa["grounded_answers"] == 1
    assert [x["score"] for x in db.list_quiz_attempts(conversation_id=b)] == [1]
    assert len(db.list_quiz_attempts()) == 2
