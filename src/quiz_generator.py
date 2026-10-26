"""Generate multiple-choice questions from the student's own notes.

Design choices (and why):
* One question per passage. A small model writes better questions from one
  short passage than from a whole document, and every question then has an
  exact source page.
* Passages are picked from different pages so the quiz covers the notes,
  not just one section.
* Gemma returns JSON. Everything is validated before a student sees it:
  4 distinct options, a valid answer, an explanation, no "all of the above",
  no duplicate questions, and the correct answer must share words with the
  source passage. Bad output is retried, then skipped.
* Options are shuffled in code, because small models tend to put the right
  answer first. The answer key is stored by index, separately from anything
  the student selects.
"""

from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import asdict, dataclass, field

from src import config
from src.text_chunker import Chunk

LETTERS = "ABCD"

QUIZ_PROMPT = """You write ONE multiple-choice question to test a student on their own study notes.

Rules:
- Use ONLY the passage below. The correct answer must be STATED in the passage, not
  inferred, implied or guessed. If the passage only says "another factor", do not name one.
- Ask about an important idea, not a trivial detail.
- Give exactly 4 options. Exactly ONE has "correct": true. The 3 wrong options must be
  believable but clearly wrong according to the passage.
- Never use "All of the above", "None of the above" or "Both A and B".
- In the explanation and hint, never refer to options by number or letter
  (no "option 1" or "option B"); the options will be shuffled.
- Write plain text, no LaTeX or $ signs.

Return ONLY a JSON object, with no other text, in exactly this shape:
{
  "question": "the question, ending with ?",
  "options": [
    {"text": "an option", "correct": false, "why_wrong": "why this is wrong, using the passage"},
    {"text": "the right answer", "correct": true, "why_wrong": ""},
    {"text": "an option", "correct": false, "why_wrong": "why this is wrong, using the passage"},
    {"text": "an option", "correct": false, "why_wrong": "why this is wrong, using the passage"}
  ],
  "explanation": "1-2 sentences: why the correct answer is right, based on the passage",
  "hint": "a short clue that helps without giving the answer away",
  "topic": "the topic of the question in 2 to 4 words, e.g. Calvin Cycle"
}"""

VERIFY_PROMPT = """You are checking a quiz question for a student. Use ONLY the passage.
Decide which ONE option is correct according to the passage.
If no option is correct, or more than one option could be correct, answer NONE.
Reply with only one word: A, B, C, D or NONE."""

_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "it", "its", "by", "for",
    "on", "with", "as", "be", "at", "from", "that", "this", "which", "into", "they", "their",
    "used", "uses", "use", "make", "makes", "made",
}
_BANNED_OPTIONS = re.compile(r"\b(all|none|both) of the (above|options)\b|^both [a-d] and [a-d]$", re.I)


@dataclass
class QuizQuestion:
    id: str
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    hint: str
    wrong_reasons: list[str]
    page_number: int
    doc_name: str
    source_text: str
    topic: str = ""            # short label for the Progress page, e.g. "Calvin Cycle"

    @property
    def correct_letter(self) -> str:
        return LETTERS[self.correct_index]

    @property
    def correct_option(self) -> str:
        return self.options[self.correct_index]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QuizQuestion":
        return cls(**data)


@dataclass
class QuizResult:
    questions: list[QuizQuestion] = field(default_factory=list)
    requested: int = 0
    skipped: int = 0            # passages whose output failed validation twice
    rejected_by_check: int = 0  # questions dropped because Gemma's blind answer disagreed

    @property
    def message(self) -> str:
        if not self.questions:
            return "I couldn't create quiz questions from these notes. Try again, or upload notes with more explanation text."
        if len(self.questions) < self.requested:
            return f"Created {len(self.questions)} of {self.requested} questions; the notes didn't support more good questions."
        return f"Created {len(self.questions)} questions from your notes."


class InvalidQuestion(ValueError):
    pass


# ---------- parsing and validation ----------
def extract_json(reply: str) -> dict:
    """Accept clean JSON, JSON in ```fences```, or JSON with chatter around it."""
    text = reply.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise InvalidQuestion("no JSON object found")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise InvalidQuestion(f"malformed JSON: {exc}") from exc
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise InvalidQuestion("JSON is not an object")
    return data


def _strip_letter(option: str) -> str:
    return re.sub(r"^\s*[A-Da-d][\).:\-]\s+", "", option).strip()


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+]+", text.lower()) if w not in _STOP and len(w) > 2}


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _resolve_correct_index(data: dict, options: list[str]) -> int:
    for key in ("correct_index", "answer_index", "correct", "answer", "correct_answer"):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            v = value.strip()
            if v.isdigit():
                return int(v)
            if len(v) == 1 and v.upper() in LETTERS:
                return LETTERS.index(v.upper())
            letter = re.match(r"^\(?([A-Da-d])[\).:]", v)
            if letter:
                return LETTERS.index(letter.group(1).upper())
            for i, opt in enumerate(options):
                if _normalise(v) == _normalise(opt):
                    return i
    raise InvalidQuestion("missing or unreadable correct answer")


# An explanation that hedges means the answer was inferred, not stated in the notes.
_HEDGING = re.compile(
    r"\b(impl(y|ies|ied)|suggests?|likely|probably|presumably|can be inferred|inferred|"
    r"assume[ds]?|it seems|may be|might be|could be)\b",
    re.I,
)

_OPTION_REFERENCE = re.compile(r"\b(options?|choices?|answers?)\s*\(?([1-4]|[A-D])\b\)?", re.I)


def _parse_options(data: dict) -> tuple[list[str], list[str], int | None]:
    """Return (texts, why_wrong, correct_index or None). Accepts the new shape
    (list of {"text", "correct", "why_wrong"}) and the older shapes (list of
    strings or {"A": ...} dict, answer given separately)."""
    raw = data.get("options")
    if isinstance(raw, dict):                                   # {"A": "...", "B": "..."}
        raw = [raw[k] for k in sorted(raw)]
    if not isinstance(raw, list) or len(raw) != 4:
        raise InvalidQuestion("need exactly 4 options")

    if all(isinstance(o, dict) for o in raw):
        texts = [_strip_letter(str(o.get("text") or o.get("option") or "")) for o in raw]
        reasons = [str(o.get("why_wrong") or o.get("reason") or "").strip() for o in raw]
        flags = [o.get("correct") is True or str(o.get("correct")).strip().lower() == "true" for o in raw]
        if sum(flags) != 1:
            raise InvalidQuestion("exactly one option must be marked correct")
        return texts, reasons, flags.index(True)

    texts = [_strip_letter(str(o)) for o in raw]
    reasons = data.get("wrong_reasons") or []
    reasons = [str(r).strip() for r in reasons][:4] if isinstance(reasons, list) else []
    reasons += [""] * (4 - len(reasons))
    return texts, reasons, None


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a) if a else 0.0


def validate_question(data: dict, chunk: Chunk) -> QuizQuestion:
    """Turn Gemma's JSON into a QuizQuestion, or raise InvalidQuestion."""
    question = str(data.get("question", "")).strip()
    if len(question) < 10:
        raise InvalidQuestion("question is missing or too short")
    if not question.endswith("?"):
        question += "?"

    options, reasons, correct = _parse_options(data)
    if any(not o for o in options):
        raise InvalidQuestion("empty option")
    if len({_normalise(o) for o in options}) != 4:
        raise InvalidQuestion("duplicate options")
    if any(_BANNED_OPTIONS.search(o) for o in options):
        raise InvalidQuestion("'all/none of the above' style option")

    if correct is None:
        correct = _resolve_correct_index(data, options)
    if not 0 <= correct < 4:
        raise InvalidQuestion("correct answer index out of range")

    explanation = str(data.get("explanation", "")).strip()
    if len(explanation) < 10:
        raise InvalidQuestion("explanation is missing")
    if _OPTION_REFERENCE.search(explanation):
        raise InvalidQuestion("explanation refers to an option number, which breaks after shuffling")
    if _HEDGING.search(explanation):
        raise InvalidQuestion("explanation hedges, so the answer is inferred rather than stated in the notes")

    # Grounding check: the right answer must use words that appear in the passage.
    answer_words = _content_words(options[correct])
    if answer_words and not (answer_words & _content_words(chunk.text)):
        raise InvalidQuestion("correct answer is not supported by the passage")

    # Consistency check: the explanation must describe the marked answer, not another
    # option. Catches a wrong answer key (e.g. the model confusing option numbers).
    explanation_words = _content_words(explanation)
    scores = [_overlap(_content_words(o), explanation_words) for o in options]
    best_other = max(score for i, score in enumerate(scores) if i != correct)
    if best_other > scores[correct] + 0.15:
        raise InvalidQuestion("explanation supports a different option than the answer key")

    reasons[correct] = ""
    reasons = ["" if _OPTION_REFERENCE.search(r) else r for r in reasons]

    hint = str(data.get("hint", "")).strip()
    if (
        not hint
        or _normalise(options[correct]) in _normalise(hint)
        or _OPTION_REFERENCE.search(hint)
    ):
        hint = f"Look at page {chunk.page_number} of your notes."   # hint must not give the answer away

    topic = re.sub(r"[^\w\s&+,'-]", "", str(data.get("topic", ""))).strip()
    if not topic or len(topic) > 40 or len(topic.split()) > 6:
        topic = f"Page {chunk.page_number}"

    return QuizQuestion(
        id=uuid.uuid4().hex,
        question=question,
        options=options,
        correct_index=correct,
        explanation=explanation,
        hint=hint,
        wrong_reasons=reasons,
        page_number=chunk.page_number,
        doc_name=chunk.doc_name,
        source_text=chunk.text,
        topic=topic[:1].upper() + topic[1:],
    )


def shuffle_options(q: QuizQuestion, rng: random.Random) -> QuizQuestion:
    order = list(range(4))
    rng.shuffle(order)
    q.options = [q.options[i] for i in order]
    q.wrong_reasons = [q.wrong_reasons[i] for i in order]
    q.correct_index = order.index(q.correct_index)
    return q


def is_duplicate(q: QuizQuestion, existing: list[QuizQuestion], threshold: float = 0.7) -> bool:
    words = _content_words(q.question) | _content_words(q.correct_option)
    for other in existing:
        if _normalise(q.question) == _normalise(other.question):
            return True
        other_words = _content_words(other.question) | _content_words(other.correct_option)
        union = words | other_words
        if union and len(words & other_words) / len(union) >= threshold:
            return True
    return False


# ---------- choosing passages ----------
def pick_passages(chunks: list[Chunk], count: int, rng: random.Random, min_words: int = 25) -> list[Chunk]:
    """Spread questions across pages: first one passage per page, then extras."""
    usable = [c for c in chunks if c.word_count >= min_words] or list(chunks)
    by_page: dict[tuple[str, int], list[Chunk]] = {}
    for c in usable:
        by_page.setdefault((c.doc_id, c.page_number), []).append(c)
    pages = list(by_page)
    rng.shuffle(pages)
    picked: list[Chunk] = []
    for key in pages:
        picked.append(rng.choice(by_page[key]))
    rest = [c for c in usable if c not in picked]
    rng.shuffle(rest)
    return picked + rest          # ordered: diverse first, extras as backups


# ---------- self-check ----------
def parse_verdict(reply: str) -> int | None:
    """Return the option index Gemma picked, or None for NONE / unreadable."""
    text = reply.strip().upper()
    if not text or text.startswith("NONE") or re.search(r"\bNONE\b", text[:20]):
        return None
    match = re.search(r"\b([ABCD])\b", text)
    return LETTERS.index(match.group(1)) if match else None


def verify_question(q: QuizQuestion, gemma) -> bool:
    """Ask Gemma to answer the question WITHOUT the answer key. Keep the question
    only if it independently picks the same single option. Catches wrong keys
    and ambiguous questions where two options could be right."""
    options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(q.options))
    reply = gemma.chat(
        [
            {"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": f"Passage:\n{q.source_text}\n\nQuestion: {q.question}\n{options}\n\nAnswer:"},
        ],
        temperature=0.0,
    )
    return parse_verdict(reply) == q.correct_index


# ---------- main entry point ----------
def generate_quiz(
    chunks: list[Chunk],
    gemma,
    num_questions: int = 5,
    seed: int | None = None,
    max_attempts_per_passage: int = 2,
    max_passages: int | None = None,
    verify: bool | None = None,
    on_progress=None,
    style: str = "",
) -> QuizResult:
    """on_progress(done, total) is called after each accepted question (used by the UI).
    `style` is an extra instruction, e.g. "Make it an easy question about a main idea." (Study Mode)."""
    verify = config.QUIZ_VERIFY if verify is None else verify
    rng = random.Random(seed)
    result = QuizResult(requested=num_questions)
    candidates = pick_passages(chunks, num_questions, rng)
    if max_passages is not None:
        candidates = candidates[:max_passages]
    else:
        # A short page may have fewer passages than questions: reuse passages (the duplicate
        # check still blocks repeated questions), then cap the time spent on retries.
        while candidates and len(candidates) < num_questions * 2:
            candidates = candidates + candidates[: num_questions * 2 - len(candidates)]
        candidates = candidates[: num_questions * 2]

    for chunk in candidates:
        if len(result.questions) >= num_questions:
            break
        question = None
        for _ in range(max_attempts_per_passage):
            try:
                reply = gemma.chat(
                    [
                        {"role": "system", "content": QUIZ_PROMPT},
                        {"role": "user", "content": f"Passage (page {chunk.page_number}):\n{chunk.text}"
                         + (f"\n\n{style}" if style else "")},
                    ],
                    json_mode=True,
                    temperature=0.4,
                )
                candidate = validate_question(extract_json(reply), chunk)
            except InvalidQuestion:
                continue
            if is_duplicate(candidate, result.questions):
                continue
            candidate = shuffle_options(candidate, rng)
            if verify and not verify_question(candidate, gemma):
                result.rejected_by_check += 1
                continue
            question = candidate
            break
        if question:
            result.questions.append(question)
            if on_progress:
                on_progress(len(result.questions), num_questions)
        else:
            result.skipped += 1
    return result
