import pytest

from src.database import Database
from src.quiz_evaluator import QuizSession, evaluate_answer
from src.quiz_generator import QuizQuestion


def make_q(qid="q1", correct=2, page=3, reasons=None):
    return QuizQuestion(
        id=qid,
        question="Where does the oxygen released by plants come from?",
        options=["Carbon dioxide", "Glucose", "Water", "Chlorophyll"],
        correct_index=correct,
        explanation="Photolysis splits water, which releases oxygen.",
        hint="Think about what is split by light.",
        wrong_reasons=reasons or ["The notes say oxygen comes from water, not CO2.", "Glucose is a product.", "", ""],
        page_number=page,
        doc_name="notes.pdf",
        source_text="The oxygen released by plants comes from water, not from carbon dioxide.",
    )


def test_correct_answer():
    f = evaluate_answer(make_q(), 2)
    assert f.is_correct and f.headline == "Correct!" and f.why_wrong == ""
    assert f.page_number == 3 and "water" in f.source_text


def test_incorrect_answer_explains_mistake():
    f = evaluate_answer(make_q(), 0)
    assert not f.is_correct
    assert f.headline.startswith("Not quite, the correct answer is C: Water")
    assert "not CO2" in f.why_wrong
    assert f.explanation and f.hint and f.page_number == 3


def test_missing_wrong_reason_gets_fallback():
    f = evaluate_answer(make_q(reasons=["", "", "", ""]), 3)
    assert "page 3" in f.why_wrong


@pytest.mark.parametrize("bad", [-1, 4, None, "A"])
def test_invalid_selection(bad):
    with pytest.raises(ValueError):
        evaluate_answer(make_q(), bad)


def test_answer_key_is_independent_of_explanation_text():
    q = make_q()
    q.explanation = "Carbon dioxide is definitely the right answer!"   # misleading text
    assert not evaluate_answer(q, 0).is_correct


def test_session_scoring_and_progress():
    s = QuizSession([make_q("a", page=3), make_q("b", page=4), make_q("c", page=5)])
    s.submit("a", 2)
    s.submit("b", 0)
    assert s.score == 1 and s.answered == 2 and not s.finished
    s.submit("c", 1)
    assert s.finished and s.score == 1
    assert s.pages_to_revise() == [4, 5]


def test_resubmitting_cannot_change_score():
    s = QuizSession([make_q("a")])
    s.submit("a", 0)
    again = s.submit("a", 2)
    assert not again.is_correct and s.score == 0


def test_unknown_question():
    with pytest.raises(KeyError):
        QuizSession([make_q("a")]).submit("zzz", 0)


def test_attempt_saved_with_details(tmp_path):
    s = QuizSession([make_q("a", page=3), make_q("b", page=4)])
    s.submit("a", 2)
    s.submit("b", 1)
    db = Database(tmp_path / "q.db")
    db.add_quiz_attempt(s.score, s.total, doc_id="d", details=s.details(), attempt_id="att1")
    saved = db.list_quiz_attempts()[0]
    assert saved["score"] == 1 and saved["total"] == 2
    assert '"page_number": 4' in saved["details_json"]
