"""Mood-aware Study Mode: turns (mood, goal, length) into a real study session.

Every choice changes something concrete:
* mood   -> the tone and depth of Gemma's explanation, and quiz difficulty/size
* goal   -> which pages are used and whether the session teaches, recaps or only tests
* length -> how many pages and how many practice questions

The explanation is grounded exactly like the tutor: Gemma only sees passages
from the student's notes and must cite them; the same citation check is reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval import RetrievedChunk
from src.text_chunker import Chunk
from src.tutor import Citation, parse_reply


@dataclass(frozen=True)
class Mood:
    key: str
    emoji: str
    label: str
    sub: str
    tone: str            # instruction for Gemma's explanation
    quiz_style: str      # instruction for quiz questions
    question_delta: int  # +/- practice questions
    adapts: tuple[str, ...]


MOODS: dict[str, Mood] = {m.key: m for m in [
    Mood("fresh", "😄", "Feeling fresh", "Ready to learn!",
         "The student is fresh and focused. Explain clearly and completely, and mention how ideas connect.",
         "", +1, ("Normal difficulty", "More practice questions", "Connections between ideas")),
    Mood("tired", "😴", "A bit tired", "Short & easy session",
         "The student is tired. Keep it very short: only the 3 most important points, in simple words.",
         "Ask an easy question about a main idea that is stated directly.", -1,
         ("Shorter explanations", "Only the key points", "Fewer, easier questions")),
    Mood("stressed", "😟", "Stressed", "Need gentle practice",
         "The student feels stressed. Use a calm, encouraging tone. Go step by step and keep each step short.",
         "Ask an easy, confidence-building question about a main idea.", 0,
         ("Gentle, supportive tone", "Small steps", "Easier questions first")),
    Mood("difficult", "🤔", "Finding it difficult", "More explanations",
         "The student finds this topic difficult. Explain step by step in very simple language, "
         "define every key term, and use an example only if the passages give one.",
         "Ask a clear question that checks understanding of one key term or step.", 0,
         ("Step-by-step explanations", "Key terms defined", "Focused practice")),
    Mood("revising", "📚", "Just revising", "Quick practice",
         "The student is revising. Give a quick, well-organised recap of the key facts.",
         "", 0, ("Quick recap", "Straight to practice", "Normal difficulty")),
    Mood("unsure", "🙂", "Not sure", "Let me explore",
         "Explain the main ideas in a friendly, simple way.", "", 0,
         ("Balanced session", "Simple explanations", "A few questions")),
]}

GOALS: dict[str, str] = {
    "learn": "Learn a new topic",
    "revise": "Revise a topic",
    "quiz": "Take a short quiz",
    "weak": "Practise my weak topics",
}

LENGTHS: dict[int, tuple[str, str, int, int]] = {
    # minutes: (title, subtitle, pages, base questions)
    5: ("5 minutes", "Quick concept recap", 1, 2),
    15: ("15 minutes", "Focused learning session", 1, 3),
    30: ("30 minutes", "Deep practice with quiz", 2, 5),
}


@dataclass
class SessionPlan:
    mood: Mood
    goal: str
    minutes: int
    pages: int
    questions: int
    explain: str | None          # "teach", "recap" or None (quiz only)
    steps: list[str] = field(default_factory=list)


def make_plan(mood_key: str, goal: str, minutes: int) -> SessionPlan:
    mood = MOODS.get(mood_key, MOODS["unsure"])
    goal = goal if goal in GOALS else "learn"
    title, _, pages, base = LENGTHS.get(minutes, LENGTHS[15])
    questions = max(2, min(6, base + mood.question_delta))

    if goal == "quiz":
        explain = None
    elif goal == "revise" or mood.key == "revising":
        explain = "recap"
    else:
        explain = "teach"

    steps: list[str] = []
    if goal == "weak":
        steps.append("Focus on the pages you got wrong most often in quizzes")
    if explain == "teach":
        steps.append("Learn the key ideas from your notes, " +
                     ("step by step" if mood.key in ("difficult", "stressed") else "with page references"))
    elif explain == "recap":
        steps.append("Quick recap of the key points, with page references")
    steps.append(f"{questions} practice question{'s' if questions != 1 else ''}"
                 + (" (easier)" if mood.quiz_style and mood.key != "difficult" else ""))
    steps.append("Score and what to revise next")
    return SessionPlan(mood, goal, minutes, pages, questions, explain, steps)


# ---------- pages ----------
@dataclass
class PageOption:
    doc_id: str
    doc_name: str
    page_number: int
    preview: str

    @property
    def key(self) -> str:
        return f"{self.doc_id}:{self.page_number}"

    @property
    def label(self) -> str:
        return f"{self.doc_name} · page {self.page_number} — {self.preview}"


def page_options(chunks: list[Chunk]) -> list[PageOption]:
    """One option per page, with the first words of the page as a preview."""
    seen: dict[tuple[str, int], PageOption] = {}
    for c in sorted(chunks, key=lambda c: (c.doc_name, c.page_number, c.chunk_index)):
        key = (c.doc_id, c.page_number)
        if key not in seen:
            words = c.text.split()
            preview = " ".join(words[:8]) + ("…" if len(words) > 8 else "")
            seen[key] = PageOption(c.doc_id, c.doc_name, c.page_number, preview)
    return list(seen.values())


def choose_pages(
    plan: SessionPlan,
    options: list[PageOption],
    selected_key: str | None,
    weak_pages: list[tuple[str, int]] | None = None,
) -> list[PageOption]:
    """Pages for this session. Weak-topic sessions use quiz history (by document
    name and page); the others start at the page the student picked."""
    if not options:
        return []
    if plan.goal == "weak":
        by_name = {(o.doc_name, o.page_number): o for o in options}
        picked = [by_name[w] for w in (weak_pages or []) if w in by_name]
        return picked[: plan.pages]
    start = next((i for i, o in enumerate(options) if o.key == selected_key), 0)
    first = options[start]
    same_doc = [o for o in options[start:] if o.doc_id == first.doc_id]
    return same_doc[: plan.pages]


def chunks_for(pages: list[PageOption], chunks: list[Chunk]) -> list[Chunk]:
    wanted = {(p.doc_id, p.page_number) for p in pages}
    return [c for c in chunks if (c.doc_id, c.page_number) in wanted]


# ---------- grounded explanation ----------
EXPLAIN_PROMPT = """You are NoteWise, a friendly study mentor. You help a student study THEIR OWN notes.

Rules:
1. Use ONLY the numbered note passages. Do not add facts from your own knowledge.
2. After every sentence that uses a passage, cite it like [S1] or [S2]. Only use passage numbers that exist.
3. {task}
4. {tone}
5. Write plain text. No LaTeX, no $ signs, no headings with #."""

TASKS = {
    "teach": "Teach the key ideas of these passages so the student understands them. "
             "Use a short numbered list of steps or points.",
    "recap": "Give a quick recap of the most important facts as 3 to 6 short bullet points starting with '- '.",
}


@dataclass
class Explanation:
    status: str
    text: str
    citations: list[Citation]


def build_explain_messages(chunks: list[Chunk], mood: Mood, kind: str) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[S{i}] (from \"{c.doc_name}\", page {c.page_number})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )
    system = EXPLAIN_PROMPT.format(task=TASKS.get(kind, TASKS["teach"]), tone=mood.tone)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Note passages:\n\n{passages}\n\nPlease help me study these notes."},
    ]


def explain(chunks: list[Chunk], mood: Mood, kind: str, gemma, max_passages: int = 6) -> Explanation:
    chunks = chunks[:max_passages]
    if not chunks:
        return Explanation("no_document", "There are no notes for this page.", [])
    reply = gemma.chat(build_explain_messages(chunks, mood, kind))
    evidence = [RetrievedChunk(chunk=c, score=1.0) for c in chunks]
    status, text, citations = parse_reply(reply, evidence)
    return Explanation(status, text, citations)
