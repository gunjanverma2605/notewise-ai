import pytest

from src.study_mode import (
    MOODS,
    build_explain_messages,
    choose_pages,
    chunks_for,
    explain,
    make_plan,
    page_options,
)
from src.text_chunker import Chunk


def chunk(page, i=0, doc="d1", name="notes.pdf", text=None):
    return Chunk(f"{doc}-p{page}-c{i}", doc, name, page, i,
                 text or f"Page {page} passage {i} about chlorophyll and light energy in leaves.")


CHUNKS = [chunk(p, i) for p in range(1, 6) for i in range(2)]


def test_plan_depends_on_mood_goal_and_length():
    tired = make_plan("tired", "learn", 5)
    fresh = make_plan("fresh", "learn", 30)
    assert tired.questions == 2 and fresh.questions == 6
    assert tired.pages == 1 and fresh.pages == 2
    assert tired.explain == "teach"
    assert make_plan("fresh", "revise", 15).explain == "recap"
    assert make_plan("revising", "learn", 15).explain == "recap"
    assert make_plan("fresh", "quiz", 15).explain is None
    assert any("weak" in s.lower() or "wrong" in s.lower() for s in make_plan("fresh", "weak", 15).steps)


def test_unknown_inputs_fall_back_safely():
    plan = make_plan("???", "???", 999)
    assert plan.mood.key == "unsure" and plan.goal == "learn" and plan.minutes == 999
    assert 2 <= plan.questions <= 6


def test_page_options_one_per_page_with_preview():
    opts = page_options(CHUNKS)
    assert [o.page_number for o in opts] == [1, 2, 3, 4, 5]
    assert opts[0].preview.startswith("Page 1 passage 0")
    assert "page 1" in opts[0].label


def test_choose_pages_from_selected_page_stays_in_document():
    opts = page_options(CHUNKS + [chunk(1, doc="d2", name="other.pdf")])
    plan = make_plan("fresh", "learn", 30)            # 2 pages
    picked = choose_pages(plan, opts, "d1:4", None)
    assert [(p.doc_id, p.page_number) for p in picked] == [("d1", 4), ("d1", 5)]
    last = choose_pages(plan, opts, "d1:5", None)
    assert [p.page_number for p in last] == [5]       # never spills into another document


def test_choose_weak_pages_from_quiz_history():
    opts = page_options(CHUNKS)
    plan = make_plan("fresh", "weak", 30)
    picked = choose_pages(plan, opts, None, [("notes.pdf", 3), ("notes.pdf", 1), ("missing.pdf", 2)])
    assert [p.page_number for p in picked] == [3, 1]
    assert choose_pages(plan, opts, None, []) == []


def test_chunks_for_pages():
    opts = page_options(CHUNKS)
    got = chunks_for([opts[1]], CHUNKS)
    assert {c.page_number for c in got} == {2} and len(got) == 2


class FakeGemma:
    def __init__(self, reply):
        self.reply = reply
        self.sent = None

    def chat(self, messages, json_mode=False, temperature=None):
        self.sent = messages
        return self.reply


def test_explanation_is_grounded_and_mood_changes_prompt():
    gemma = FakeGemma("1. Chlorophyll captures light energy [S1].\n2. Leaves use it [S2].")
    out = explain(chunks_for(page_options(CHUNKS)[:1], CHUNKS), MOODS["tired"], "teach", gemma)
    assert out.status == "answered"
    assert "[p. 1]" in out.text and [c.page_number for c in out.citations] == [1, 1]
    assert "tired" in gemma.sent[0]["content"]
    assert "[S1] (from \"notes.pdf\", page 1)" in gemma.sent[1]["content"]


def test_explanation_with_invented_source_is_not_trusted():
    out = explain(CHUNKS[:1], MOODS["fresh"], "recap", FakeGemma("Plants are green [S7]."))
    assert out.status == "unverified" and out.citations == []


def test_explanation_with_no_chunks():
    assert explain([], MOODS["fresh"], "teach", FakeGemma("x")).status == "no_document"


@pytest.mark.parametrize("kind, word", [("teach", "numbered"), ("recap", "bullet")])
def test_task_wording(kind, word):
    msgs = build_explain_messages(CHUNKS[:1], MOODS["fresh"], kind)
    assert word in msgs[0]["content"]
