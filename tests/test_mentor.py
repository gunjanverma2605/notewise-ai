import json
from datetime import date

from src import mentor
from src.database import Database
from src.study_mode import PageOption

TODAY = date(2026, 10, 24)


def attempt(when, details):
    ok = sum(d["is_correct"] for d in details)
    return {"created_at": when, "score": ok, "total": len(details), "doc_id": "d1",
            "details_json": json.dumps(details)}


def row(page, ok, topic, yours="Roots", right="Chlorophyll"):
    return {"doc_name": "notes.pdf", "page_number": page, "is_correct": ok, "topic": topic,
            "question": "Q?", "your_answer": yours, "correct_answer": right, "source_text": "t"}


PAGES = [PageOption("d1", "notes.pdf", 1, "Intro"), PageOption("d1", "notes.pdf", 2, "Light")]


def profile_with_mistakes():
    a = attempt("2026-10-24T10:00:00+00:00", [
        row(4, False, "Calvin cycle"), row(4, False, "Calvin cycle"),
        row(2, True, "Light reactions"), row(2, True, "Light reactions"), row(6, True, "Limiting factors")])
    return mentor.build_profile([a], ["2026-10-23T10:00:00+00:00", "2026-10-24T09:00:00+00:00"],
                                ["explain in simpler words", "simpler please", "what is ATP"],
                                [{"mood": "tired"}], PAGES, today=TODAY)


def test_empty_profile_and_greeting():
    p = mentor.build_profile([], [], [], [], [], today=TODAY)
    assert not p.has_data and p.accuracy is None and p.weakest is None
    head, lines = mentor.greeting(p, has_notes=False)
    assert "Nova" in head and "Add your notes" in lines[0]
    _, lines = mentor.greeting(p, has_notes=True)
    assert any("Quiz Me" in l for l in lines)


def test_profile_from_real_data():
    p = profile_with_mistakes()
    assert p.streak == 2 and p.quizzes == 1 and p.accuracy == 60
    assert p.weakest.topic == "Calvin cycle" and p.weakest.mistakes == 2
    assert "Light reactions" in p.strengths and p.last_mood == "tired" and p.habits["simpler"] == 2
    _, lines = mentor.greeting(p, has_notes=True)
    text = " ".join(lines)
    assert "2-day study streak" in text and "Calvin cycle (p. 4)" in text and "twice" in text


def test_style_uses_mode_follow_up_habits_and_mistakes():
    p = profile_with_mistakes()
    s = mentor.mentor_style("doubt", p, "I'm confused about the Calvin cycle")
    assert "confused" in s and "simpler words" in s and 'answered "Roots"' in s
    s = mentor.mentor_style("explain", p, "x", follow_up="exam")
    assert "model exam answer" in s and "simpler words" not in s
    assert "ONLY if the notes contain one" in mentor.mentor_style("explain", p, "x", follow_up="example")


def test_plan_quick_prompts_and_tips_are_personal():
    p = profile_with_mistakes()
    plan = mentor.study_plan(p)
    assert "Calvin cycle (p. 4)" in plan[0][1] and any("tired" in s for _, s in plan)
    labels = [q.label for q in mentor.quick_prompts(p)]
    assert labels[0] == "Explain Calvin cycle simply" and "Quiz me on my weak areas" in labels
    titles = [t for t, _ in mentor.tips(p)]
    assert titles[0] == "Fix your weak spot" and "Active recall" in titles and "Keep the streak" in titles


def test_new_student_gets_page_prompts_and_starter_plan():
    p = mentor.build_profile([], [], [], [], PAGES, today=TODAY)
    prompts = mentor.quick_prompts(p)
    assert prompts[0].kind == "page" and prompts[0].page_key == "d1:1"
    assert "page 1" in mentor.study_plan(p)[0][1]
    assert [t for t, _ in mentor.tips(p)][0] == "Active recall"


class StyleGemma:
    def __init__(self):
        self.calls = []

    def chat(self, messages, json_mode=False, temperature=None):
        self.calls.append(messages)
        return "RuBisCO fixes carbon dioxide [S1]."


def test_ask_is_grounded_and_passes_style(tmp_path, fake_embedder):
    from src.retrieval import NoteIndex
    from src.document_loader import load_text
    index = NoteIndex(fake_embedder, persist_dir=tmp_path / "c")
    doc = load_text("Chlorophyll absorbs red and blue light.\fWater is split in photolysis.\f"
                    "Glucose is stored as starch.\fRuBisCO fixes carbon dioxide to RuBP in the Calvin cycle.")
    index.add_document(doc)
    g = StyleGemma()
    p = profile_with_mistakes()
    ans = mentor.ask("What does RuBisCO fix in the Calvin cycle?", "explain", p, [doc.doc_id], index, g)
    assert ans.status == "answered" and ans.citations[0].page_number == 4
    assert "(How to answer:" in g.calls[-1][-1]["content"]
    off = mentor.ask("Who won the cricket world cup?", "explain", p, [doc.doc_id], index, g)
    assert off.status == "not_in_notes"
    assert mentor.pages_for_query("RuBisCO Calvin cycle carbon dioxide", [doc.doc_id], index) == [(doc.doc_id, 4)]
    assert mentor.pages_for_query("cricket world cup", [doc.doc_id], index) == []


def test_mentor_messages_storage_and_streak(tmp_path):
    db = Database(tmp_path / "n.db")
    cid = db.create_conversation("x")
    db.add_mentor_message(cid, "user", "hi", mode="explain", message_id="m1")
    db.add_mentor_message(cid, "user", "hi", mode="explain", message_id="m1")     # idempotent
    db.add_mentor_message(cid, "assistant", "hello", mode="explain", status="answered",
                          citations=[{"label": "S1", "page_number": 2}])
    msgs = db.get_mentor_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["citations"][0]["page_number"] == 2
    assert len(db.activity_dates(cid)) == 1
    db.clear_mentor_messages(cid)
    assert db.get_mentor_messages(cid) == []
