import pytest

from src import config
from src.document_loader import load_document
from src.gemma_client import strip_thinking
from src.retrieval import NoteIndex
from src.tutor import (
    NOT_IN_NOTES_MESSAGE,
    SYSTEM_PROMPT,
    answer_question,
    build_messages,
)

SUPPORTED_Q = "Which enzyme joins carbon dioxide to RuBP in phase 1 of the Calvin cycle?"


class FakeGemma:
    """Returns a canned reply and records what it was sent."""

    def __init__(self, reply: str = "RuBisCO joins carbon dioxide to RuBP [S1]."):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, json_mode=False, temperature=None):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def setup(tmp_path, fake_embedder, sample_pdf_bytes):
    index = NoteIndex(fake_embedder, persist_dir=tmp_path / "chroma")
    doc = load_document(sample_pdf_bytes, "photosynthesis_notes.pdf")
    index.add_document(doc)
    return index, doc


def test_supported_question_gets_page_citation(setup):
    index, doc = setup
    gemma = FakeGemma()
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, gemma)
    assert ans.status == "answered" and ans.grounded
    assert ans.citations[0].page_number == 4
    assert "[p. 4]" in ans.text and "[S1]" not in ans.text
    assert len(gemma.calls) == 1


def test_off_topic_question_never_reaches_gemma(setup):
    index, doc = setup
    gemma = FakeGemma()
    ans = answer_question("What is the GDP of India?", [doc.doc_id], index, gemma)
    assert ans.status == "not_in_notes"
    assert ans.text == NOT_IN_NOTES_MESSAGE
    assert ans.citations == []
    assert gemma.calls == []           # safeguard 1


@pytest.mark.parametrize("reply", ["NOT_IN_NOTES", "  not_in_notes.", "NOT IN NOTES"])
def test_gemma_refusal_is_respected(setup, reply):
    index, doc = setup
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, FakeGemma(reply))
    assert ans.status == "not_in_notes"   # safeguard 2
    assert ans.citations == []


def test_invented_citation_is_removed(setup):
    index, doc = setup
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, FakeGemma("RuBisCO does it [S9]."))
    assert ans.status == "unverified"     # safeguard 3
    assert "[S9]" not in ans.text and ans.citations == []


def test_mixed_valid_and_invalid_citations(setup):
    index, doc = setup
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, FakeGemma("Fact one [S1]. Fact two [S42]."))
    assert ans.status == "answered"
    assert [c.label for c in ans.citations] == ["S1"]
    assert "[S42]" not in ans.text


def test_repeated_page_labels_are_merged(setup):
    index, doc = setup
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, FakeGemma("RuBisCO [S1][S1] fixes carbon."))
    assert ans.text.count("[p. 4]") == 1


def test_no_document_selected(setup):
    index, _ = setup
    gemma = FakeGemma()
    ans = answer_question("What is chlorophyll?", [], index, gemma)
    assert ans.status == "no_document" and gemma.calls == []


def test_prompt_contains_passages_and_limits_history(setup):
    index, doc = setup
    evidence = index.search(SUPPORTED_Q, [doc.doc_id]).evidence
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(20)]
    msgs = build_messages(SUPPORTED_Q, evidence, history)
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len(msgs) == 1 + config.MAX_HISTORY_TURNS + 1
    assert "[S1] (from \"photosynthesis_notes.pdf\", page" in msgs[-1]["content"]
    assert SUPPORTED_Q in msgs[-1]["content"]


def test_search_query_used_for_retrieval_but_question_kept(setup):
    index, doc = setup
    gemma = FakeGemma()
    ans = answer_question("explain that simpler", [doc.doc_id], index, gemma, search_query=SUPPORTED_Q)
    assert ans.question == "explain that simpler" and ans.search_query == SUPPORTED_Q
    assert ans.status == "answered"
    assert "explain that simpler" in gemma.calls[0][-1]["content"]


@pytest.mark.parametrize(
    "reply",
    [
        "The provided notes do not state who discovered photosynthesis.",
        "The passages don't mention the discoverer.",
        "This is not mentioned in the notes.",
        "There is no information about who discovered it.",
    ],
)
def test_refusal_in_own_words_is_not_in_notes(setup, reply):
    index, doc = setup
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, FakeGemma(reply))
    assert ans.status == "not_in_notes"


@pytest.mark.parametrize(
    "reply",
    [
        "RuBisCO fixes carbon [S1]. The notes do not mention who discovered it.",
        "The provided notes do not state who discovered photosynthesis. However, the Calvin cycle was named after Melvin Calvin [S2].",
    ],
)
def test_cited_answer_with_gap_is_partial(setup, reply):
    index, doc = setup
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, FakeGemma(reply))
    assert ans.status == "partial" and ans.grounded
    assert ans.citations


@pytest.mark.parametrize("tag", ["[S1, S2]", "[S1,S2]", "[S1 and S2]", "[s1, s2]"])
def test_grouped_citations_are_converted(setup, tag):
    index, doc = setup
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, FakeGemma(f"RuBisCO fixes carbon {tag}."))
    assert "[S" not in ans.text.upper().replace("[P.", "")
    assert "[p. " in ans.text
    assert ans.status == "answered"


def test_strip_thinking():
    assert strip_thinking("<think>reasoning here</think>Final answer.") == "Final answer."
    assert strip_thinking("Plain answer.") == "Plain answer."


def test_latex_is_turned_into_plain_text(setup):
    index, doc = setup
    reply = r"Photolysis makes electrons, hydrogen ions ($\text{H}^+$) and $\text{O}_2$ [S1]."
    ans = answer_question(SUPPORTED_Q, [doc.doc_id], index, FakeGemma(reply))
    assert "hydrogen ions (H+)" in ans.text and "O2" in ans.text and "$" not in ans.text


def test_resolved_meaning_is_given_to_gemma(setup):
    index, doc = setup
    gemma = FakeGemma()
    answer_question("Explain that in simpler words", [doc.doc_id], index, gemma, search_query=SUPPORTED_Q)
    last = gemma.calls[0][-1]["content"]
    assert "Explain that in simpler words" in last and f"The student means: {SUPPORTED_Q}" in last


def test_no_meaning_line_for_normal_questions(setup):
    index, doc = setup
    gemma = FakeGemma()
    answer_question(SUPPORTED_Q, [doc.doc_id], index, gemma)
    assert "The student means" not in gemma.calls[0][-1]["content"]


from src.tutor import is_overview_question, overview_evidence  # noqa: E402


@pytest.mark.parametrize("q", ["explain me this pdf", "name the topics mention in pdf", "summarize my notes",
                               "What is this pdf about?", "list the main topics in these notes"])
def test_whole_document_questions_are_detected(q):
    assert is_overview_question(q)


@pytest.mark.parametrize("q", ["what is higher order function", "Explain the Calvin cycle from my notes",
                               "summarize the Calvin cycle", "What is the GDP of India?", "Explain that in simpler words"])
def test_specific_questions_are_not_overview(q):
    assert not is_overview_question(q)


def test_overview_question_uses_passages_from_every_page(setup):
    index, doc = setup
    gemma = FakeGemma("- Light reactions make ATP [S1].\n- The Calvin cycle makes sugar [S5].")
    ans = answer_question("explain me this pdf", [doc.doc_id], index, gemma)
    assert ans.status == "answered" and ans.citations
    sent = gemma.calls[0][-1]["content"]
    assert all(f"page {p})" in sent for p in range(1, 9))                  # every page of the notes
    assert "overview of their whole notes" in sent


def test_overview_evidence_is_capped_and_spread(setup):
    index, doc = setup
    chunks = index.get_chunks([doc.doc_id])
    ev = overview_evidence(chunks, limit=4)
    assert len(ev) == 4 and len({e.chunk.page_number for e in ev}) == 4
    assert ev[0].chunk.page_number == 1 and ev[-1].chunk.page_number > 4
