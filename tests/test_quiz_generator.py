import json
import random

import pytest

from src.quiz_generator import (
    LETTERS,
    InvalidQuestion,
    extract_json,
    generate_quiz,
    is_duplicate,
    pick_passages,
    shuffle_options,
    validate_question,
)
from src.text_chunker import Chunk

PASSAGE = Chunk(
    chunk_id="d-p2-c0", doc_id="d", doc_name="notes.pdf", page_number=2, chunk_index=0,
    text="Chlorophyll absorbs mainly blue and red light. It reflects green light, which is why "
         "leaves look green. Carotenoids are yellow and orange accessory pigments.",
)


def good(**overrides) -> dict:
    data = {
        "question": "Why do leaves look green?",
        "options": ["Chlorophyll reflects green light", "Leaves absorb green light",
                    "Carotenoids are green", "Stomata are green"],
        "correct_index": 0,
        "explanation": "Chlorophyll reflects green light, so leaves look green.",
        "hint": "Think about which colour is reflected.",
        "wrong_reasons": ["", "Green is reflected, not absorbed.", "Carotenoids are yellow and orange.", "Stomata are pores."],
    }
    data.update(overrides)
    return data


# ---------- parsing ----------
@pytest.mark.parametrize(
    "reply",
    [
        json.dumps(good()),
        "```json\n" + json.dumps(good()) + "\n```",
        "Here is your question:\n" + json.dumps(good()) + "\nHope it helps!",
        json.dumps([good()]),
    ],
)
def test_extract_json_variants(reply):
    assert extract_json(reply)["question"] == "Why do leaves look green?"


@pytest.mark.parametrize("reply", ["", "no json here", '{"question": "broken",', "[1, 2, 3]"])
def test_malformed_json_raises(reply):
    with pytest.raises(InvalidQuestion):
        extract_json(reply)


def new_format(**overrides) -> dict:
    data = {
        "question": "Why do leaves look green?",
        "options": [
            {"text": "Leaves absorb green light", "correct": False, "why_wrong": "Green is reflected, not absorbed."},
            {"text": "Chlorophyll reflects green light", "correct": True, "why_wrong": ""},
            {"text": "Carotenoids are green", "correct": False, "why_wrong": "Carotenoids are yellow and orange."},
            {"text": "Stomata are green", "correct": False, "why_wrong": "Stomata are pores."},
        ],
        "explanation": "Chlorophyll reflects green light, so leaves look green.",
        "hint": "Think about which colour is reflected.",
    }
    data.update(overrides)
    return data


# ---------- validation ----------
def test_new_format_with_correct_flags():
    q = validate_question(new_format(), PASSAGE)
    assert q.correct_option == "Chlorophyll reflects green light"
    assert q.wrong_reasons[0] == "Green is reflected, not absorbed." and q.wrong_reasons[1] == ""


@pytest.mark.parametrize("flags", [[False, False, False, False], [True, True, False, False]])
def test_new_format_needs_exactly_one_correct(flags):
    data = new_format()
    for opt, flag in zip(data["options"], flags):
        opt["correct"] = flag
    with pytest.raises(InvalidQuestion, match="exactly one"):
        validate_question(data, PASSAGE)


def test_wrong_answer_key_is_caught_by_explanation_check():
    """Real failure seen on the Mac: the key marked one option, the explanation described another."""
    passage = Chunk("x", "d", "notes.pdf", 2, 0,
                    "Chlorophyll a is the primary pigment that takes part directly in the light reactions. "
                    "Chlorophyll b is an accessory pigment that captures extra light and passes the energy to "
                    "chlorophyll a. Chlorophyll absorbs mainly blue and red light.")
    data = {
        "question": "What is the specific function of chlorophyll b?",
        "options": ["It is the primary pigment directly involved in the light reactions.",
                    "It absorbs blue and red light, causing the green color of leaves.",
                    "It is responsible for reflecting green light during photosynthesis.",
                    "It captures extra light energy and transfers it to chlorophyll a."],
        "correct_index": 1,
        "explanation": "Chlorophyll b is an accessory pigment that captures extra light and passes the energy to chlorophyll a.",
    }
    with pytest.raises(InvalidQuestion, match="different option"):
        validate_question(data, passage)
    data["correct_index"] = 3
    assert validate_question(data, passage).correct_index == 3


@pytest.mark.parametrize("text", ["Option 1 is wrong because of X.", "As option B shows, leaves are green.", "Answer (C) is right."])
def test_explanation_mentioning_option_numbers_is_rejected(text):
    with pytest.raises(InvalidQuestion, match="option number"):
        validate_question(new_format(explanation=text + " Chlorophyll reflects green light."), PASSAGE)


def test_hint_mentioning_option_letter_is_replaced():
    q = validate_question(new_format(hint="It is not option A."), PASSAGE)
    assert "page 2" in q.hint

def test_valid_question():
    q = validate_question(good(), PASSAGE)
    assert q.correct_option == "Chlorophyll reflects green light"
    assert q.page_number == 2 and q.source_text == PASSAGE.text
    assert q.wrong_reasons[0] == ""


@pytest.mark.parametrize(
    "answer, expected",
    [("B", 1), ("b", 1), ("B) Leaves absorb green light", 1), ("2", 2), ("Leaves absorb green light", 1)],
)
def test_correct_answer_formats(answer, expected):
    data = good(options=["Chlorophyll reflects green light", "Leaves absorb green light",
                         "Green light is reflected by chlorophyll", "Blue light is reflected"])
    del data["correct_index"]
    data["answer"] = answer
    data["explanation"] = "This is stated directly on page two of the passage."
    assert validate_question(data, PASSAGE).correct_index == expected


def test_option_letters_are_stripped_and_dict_options_accepted():
    data = good(options={"A": "A) Chlorophyll reflects green light", "B": "B. Leaves absorb green light",
                         "C": "C: Carotenoids are green", "D": "D) Stomata are green"})
    q = validate_question(data, PASSAGE)
    assert q.options[0] == "Chlorophyll reflects green light" and q.options[1] == "Leaves absorb green light"


@pytest.mark.parametrize(
    "bad",
    [
        good(question=""),
        good(options=["a", "b", "c"]),
        good(options=["Same", "same", "Other", "Another"]),
        good(options=["Chlorophyll reflects green light", "All of the above", "x y z", "p q r"]),
        good(options=["Chlorophyll reflects green light", "None of the above", "x y z", "p q r"]),
        good(correct_index=7),
        good(explanation=""),
        good(options=["Mitochondria produce ATP", "Leaves absorb green light", "Carotenoids are green", "Stomata are green"]),
    ],
)
def test_invalid_questions_rejected(bad):
    with pytest.raises(InvalidQuestion):
        validate_question(bad, PASSAGE)


def test_hint_that_reveals_answer_is_replaced():
    q = validate_question(good(hint="The answer is: Chlorophyll reflects green light"), PASSAGE)
    assert "Chlorophyll reflects green light" not in q.hint and "page 2" in q.hint


def test_shuffle_keeps_answer_key_aligned():
    for seed in range(20):
        q = shuffle_options(validate_question(good(), PASSAGE), random.Random(seed))
        assert q.correct_option == "Chlorophyll reflects green light"
        assert q.wrong_reasons[q.correct_index] == ""
        assert q.wrong_reasons[q.options.index("Leaves absorb green light")] == "Green is reflected, not absorbed."


def test_duplicate_detection():
    a = validate_question(good(), PASSAGE)
    b = validate_question(good(question="Why do leaves appear green?"), PASSAGE)
    c = validate_question(good(question="What colours does chlorophyll absorb?",
                               options=["Blue and red", "Green only", "Yellow", "Orange"],
                               explanation="Chlorophyll absorbs mainly blue and red light."), PASSAGE)
    assert is_duplicate(b, [a])
    assert not is_duplicate(c, [a])


# ---------- generation loop ----------
def make_chunks(pages=8, per_page=2):
    return [
        Chunk(f"d-p{p}-c{i}", "d", "notes.pdf", p, i,
              f"Page {p} passage {i}. Chlorophyll reflects green light and absorbs blue and red light "
              f"so leaves look green. Extra words to make this passage long enough for a quiz question here.")
        for p in range(1, pages + 1) for i in range(per_page)
    ]


def test_pick_passages_spreads_across_pages():
    picked = pick_passages(make_chunks(), 5, random.Random(1))
    assert len({c.page_number for c in picked[:8]}) == 8


class ScriptedGemma:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, json_mode=False, temperature=None):
        self.calls += 1
        assert json_mode is True
        return self.replies.pop(0) if self.replies else "garbage"


_TOPICS = [
    ("Which colour does chlorophyll reflect?", "Green light"),
    ("Which colours does chlorophyll absorb most?", "Blue and red"),
    ("Why do leaves appear the colour they do?", "Chlorophyll reflects green"),
    ("How long must a passage be for the quiz?", "Long enough"),
    ("What do extra words in a passage help with?", "Quiz question"),
]


def unique(i):
    question, answer = _TOPICS[i - 1]
    return json.dumps(new_format(
        question=question,
        options=[{"text": answer, "correct": True, "why_wrong": ""},
                 {"text": f"Mitochondria {i}", "correct": False, "why_wrong": "Not in the notes."},
                 {"text": f"Ribosomes {i}", "correct": False, "why_wrong": "Not in the notes."},
                 {"text": f"Nucleus {i}", "correct": False, "why_wrong": "Not in the notes."}],
        explanation=f"The passage says: {answer}.",
    ))


def test_generate_quiz_retries_malformed_output():
    gemma = ScriptedGemma(["not json", unique(1), unique(2), unique(3), unique(4), unique(5)])
    result = generate_quiz(make_chunks(), gemma, num_questions=5, seed=3, verify=False)
    assert len(result.questions) == 5
    assert gemma.calls == 6                      # one retry for the malformed reply
    assert "Created 5 questions" in result.message


def test_generate_quiz_skips_duplicates_and_reports_shortfall():
    same = json.dumps(good())
    gemma = ScriptedGemma([same] * 40)
    result = generate_quiz(make_chunks(), gemma, num_questions=5, seed=3, verify=False)
    assert len(result.questions) == 1           # every later one is a duplicate
    assert "1 of 5" in result.message


def test_generate_quiz_all_bad_gives_helpful_message():
    result = generate_quiz(make_chunks(), ScriptedGemma([]), num_questions=5, seed=3, verify=False)
    assert result.questions == [] and "couldn't create" in result.message


def test_generate_quiz_with_no_chunks():
    result = generate_quiz([], ScriptedGemma([]), num_questions=5, verify=False)
    assert result.questions == []


def test_question_roundtrip_dict():
    from src.quiz_generator import QuizQuestion
    q = validate_question(good(), PASSAGE)
    assert QuizQuestion.from_dict(q.to_dict()) == q


# ---------- self-check ----------
from src.quiz_generator import parse_verdict, verify_question, VERIFY_PROMPT  # noqa: E402


@pytest.mark.parametrize(
    "reply, expected",
    [("B", 1), ("b", 1), ("The answer is C.", 2), ("D. On the thylakoid membranes", 3),
     ("NONE", None), ("None of them", None), ("", None), ("I am not sure", None)],
)
def test_parse_verdict(reply, expected):
    assert parse_verdict(reply) == expected


class VerifyingGemma:
    """Writes questions, and answers the self-check with a fixed rule."""

    def __init__(self, check_answer):
        self.check_answer = check_answer      # function(prompt text, last correct answer) -> reply
        self.n = 0
        self.checks = 0
        self.last_answer = ""

    def chat(self, messages, json_mode=False, temperature=None):
        if messages[0]["content"] == VERIFY_PROMPT:
            self.checks += 1
            return self.check_answer(messages[1]["content"], self.last_answer)
        self.n += 1
        i = (self.n - 1) % 5 + 1
        self.last_answer = _TOPICS[i - 1][1]
        return unique(i)


def _letter_of(content, text):
    for line in content.splitlines():
        if line[3:].strip() == text:
            return line[0]
    return "NONE"


def test_self_check_keeps_questions_it_agrees_with():
    gemma = VerifyingGemma(lambda content, answer: _letter_of(content, answer))
    progress = []
    result = generate_quiz(make_chunks(), gemma, num_questions=5, seed=3, verify=True,
                           on_progress=lambda d, t: progress.append((d, t)))
    assert len(result.questions) == 5 and result.rejected_by_check == 0
    assert gemma.checks == 5
    assert progress[-1] == (5, 5)


def test_self_check_rejects_disagreement_and_ambiguity():
    result = generate_quiz(make_chunks(), VerifyingGemma(lambda c, a: "NONE"), num_questions=3, seed=3, verify=True)
    assert result.questions == [] and result.rejected_by_check > 0


def test_verify_question_compares_with_key():
    q = validate_question(new_format(), PASSAGE)
    right = LETTERS[q.correct_index]
    wrong = LETTERS[(q.correct_index + 1) % 4]
    assert verify_question(q, VerifyingGemma(lambda c, a: right))
    assert not verify_question(q, VerifyingGemma(lambda c, a: wrong))


def test_inferred_answer_is_rejected():
    """Real case from the Mac: the notes say 'another factor becomes limiting', Gemma keyed CO2."""
    passage = Chunk("x", "d", "notes.pdf", 5, 0,
                    "As light intensity increases, the rate of photosynthesis increases. At high light intensity "
                    "the rate levels off, because another factor becomes limiting. More carbon dioxide increases "
                    "the rate of carbon fixation.")
    data = new_format(
        question="Why does the rate of photosynthesis level off at high light intensity?",
        options=[{"text": "The availability of carbon dioxide becomes the limiting factor", "correct": True, "why_wrong": ""},
                 {"text": "The light reactions stop working", "correct": False, "why_wrong": "x"},
                 {"text": "Enzymes are exhausted", "correct": False, "why_wrong": "x"},
                 {"text": "The Calvin cycle is saturated with energy", "correct": False, "why_wrong": "x"}],
        explanation="Another factor has become limiting, which in the context of the notes implies that carbon dioxide is the limiting factor.",
    )
    with pytest.raises(InvalidQuestion, match="inferred"):
        validate_question(data, passage)


def test_topic_is_kept_or_falls_back_to_page():
    assert validate_question(new_format(topic="calvin cycle"), PASSAGE).topic == "Calvin cycle"
    assert validate_question(new_format(), PASSAGE).topic == "Page 2"
    assert validate_question(new_format(topic="x " * 30), PASSAGE).topic == "Page 2"


def test_old_saved_question_without_topic_still_loads():
    from src.quiz_generator import QuizQuestion
    d = validate_question(new_format(), PASSAGE).to_dict()
    d.pop("topic")
    assert QuizQuestion.from_dict(d).topic == ""
