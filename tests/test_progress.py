import json

from src.progress import summarise


def attempt(when, score, total, details, doc="d1"):
    return {"created_at": when, "score": score, "total": total, "doc_id": doc,
            "details_json": json.dumps(details)}


def row(page, ok, topic=None):
    d = {"doc_name": "notes.pdf", "page_number": page, "is_correct": ok}
    if topic:
        d["topic"] = topic
    return d


def test_empty():
    s = summarise([])
    assert s.quizzes == 0 and s.accuracy == 0 and s.change is None and s.pages == []


def test_totals_history_and_change():
    a1 = attempt("2026-10-19T10:00", 2, 5, [row(3, True), row(3, False), row(5, False), row(5, False), row(8, True)])
    a2 = attempt("2026-10-19T11:00", 4, 5, [row(3, True), row(3, True), row(5, False), row(8, True), row(8, True)], doc="d2")
    s = summarise([a2, a1])                       # database gives newest first
    assert s.quizzes == 2 and s.correct == 6 and s.answered == 10 and s.accuracy == 60
    assert s.history == [40, 80] and s.change == 40
    assert s.notes == 2


def test_page_accuracy_strengths_and_weaknesses():
    a = attempt("t", 3, 6, [row(3, True, "Light reactions"), row(3, True), row(5, False, "ATP and NADPH"),
                             row(5, True), row(8, False), row(8, False)])
    s = summarise([a])
    by_page = {p.page_number: p for p in s.pages}
    assert by_page[3].accuracy == 100 and by_page[3].topic == "Light reactions"
    assert by_page[5].accuracy == 50 and by_page[5].topic == "ATP and NADPH"
    assert by_page[8].topic == "Page 8"           # old quizzes without topics
    assert [p.page_number for p in s.strengths] == [3]
    assert [p.page_number for p in s.to_improve] == [8, 5]   # weakest first


def test_zero_total_attempt_is_ignored():
    assert summarise([attempt("t", 0, 0, [])]).quizzes == 0


from datetime import date

from src.progress import mistake_memory, study_habits, study_streak


def wrong(page, q="Q?", yours="Roots", right="Chlorophyll", topic=None, ok=False):
    return {"doc_name": "notes.pdf", "page_number": page, "is_correct": ok, "question": q,
            "your_answer": yours, "correct_answer": right, "topic": topic or f"Topic {page}",
            "source_text": f"text of page {page}"}


def test_mistake_memory_groups_counts_and_status():
    a1 = attempt("2026-10-20T10:00:00+00:00", 0, 3, [wrong(3), wrong(3), wrong(5)])
    a2 = attempt("2026-10-21T10:00:00+00:00", 2, 3, [wrong(3, ok=True), wrong(5, ok=True), wrong(8, ok=True)])
    memo = mistake_memory([a2, a1])
    by_page = {m.page_number: m for m in memo}
    assert set(by_page) == {3, 5}                       # page 8 never wrong -> not listed
    assert by_page[3].mistakes == 2 and by_page[3].attempts == 3
    assert by_page[3].status == "Improving"             # last attempt was right
    assert by_page[5].status == "Improving" and by_page[5].accuracy == 50
    assert by_page[3].examples[0]["your_answer"] == "Roots"
    assert by_page[3].source_text == "text of page 3"


def test_needs_work_listed_first():
    a = attempt("2026-10-20T10:00:00+00:00", 1, 4, [wrong(2), wrong(2), wrong(4), wrong(4, ok=True)])
    memo = mistake_memory([a])
    assert memo[0].page_number == 2 and memo[0].status == "Needs work"


def test_old_details_without_answer_text_still_work():
    a = attempt("2026-10-20T10:00:00+00:00", 0, 1,
                [{"doc_name": "n.pdf", "page_number": 1, "is_correct": False, "selected": "B", "correct": "C"}])
    m = mistake_memory([a])[0]
    assert m.examples[0]["your_answer"] == "B" and m.topic == "Page 1"


def test_streak():
    today = date(2026, 10, 24)
    days = ["2026-10-24T09:00:00", "2026-10-23T09:00:00", "2026-10-22T20:00:00", "2026-10-20T09:00:00"]
    assert study_streak(days, today) == 3
    assert study_streak(days[1:], today) == 2           # nothing today yet: streak still counts
    assert study_streak(["2026-10-20T09:00:00"], today) == 0
    assert study_streak([], today) == 0


def test_study_habits():
    h = study_habits(["Explain that in simpler words", "Explain it in more detail",
                      "Give an example from my notes", "What is RuBisCO?"])
    assert h == {"simpler": 1, "detail": 1, "example": 1, "questions": 4}
