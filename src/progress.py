"""Turn saved quiz attempts into the numbers shown on the Progress page.

Pure Python, no AI: everything here is counted from the quiz_attempts table,
so every number on screen can be traced back to real answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class PageStat:
    doc_name: str
    page_number: int
    topic: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> int:
        return round(100 * self.correct / self.total) if self.total else 0


@dataclass
class ProgressSummary:
    quizzes: int = 0
    correct: int = 0
    answered: int = 0
    notes: int = 0
    history: list[int] = field(default_factory=list)      # % per quiz, oldest first
    pages: list[PageStat] = field(default_factory=list)   # most-asked first

    @property
    def accuracy(self) -> int:
        return round(100 * self.correct / self.answered) if self.answered else 0

    @property
    def change(self) -> int | None:
        """Change in % between the last two quizzes, or None with fewer than two."""
        return self.history[-1] - self.history[-2] if len(self.history) >= 2 else None

    @property
    def strengths(self) -> list[PageStat]:
        return sorted([p for p in self.pages if p.accuracy >= 75],
                      key=lambda p: (-p.accuracy, -p.total))

    @property
    def to_improve(self) -> list[PageStat]:
        return sorted([p for p in self.pages if p.accuracy < 75],
                      key=lambda p: (p.accuracy, -p.total))


def summarise(attempts: list[dict]) -> ProgressSummary:
    """`attempts` as returned by Database.list_quiz_attempts (newest first)."""
    s = ProgressSummary()
    pages: dict[tuple[str, int], PageStat] = {}
    docs: set[str] = set()
    for a in sorted(attempts, key=lambda a: a["created_at"]):     # oldest first
        if not a["total"]:
            continue
        s.quizzes += 1
        s.correct += a["score"]
        s.answered += a["total"]
        s.history.append(round(100 * a["score"] / a["total"]))
        if a.get("doc_id"):
            docs.add(a["doc_id"])
        for d in json.loads(a.get("details_json") or "[]"):
            key = (d.get("doc_name", ""), int(d.get("page_number", 0)))
            stat = pages.setdefault(key, PageStat(key[0], key[1], ""))
            stat.topic = d.get("topic") or stat.topic or f"Page {key[1]}"   # newest label wins
            stat.total += 1
            stat.correct += 1 if d.get("is_correct") else 0
    s.notes = len(docs)
    s.pages = sorted(pages.values(), key=lambda p: (-p.total, p.page_number))
    return s


# ---------- Mistake Memory ----------
@dataclass
class MistakeTopic:
    doc_name: str
    page_number: int
    topic: str
    mistakes: int = 0
    attempts: int = 0
    last_attempt: str = ""
    last_correct: bool = False
    examples: list[dict] = field(default_factory=list)   # newest first
    source_text: str = ""

    @property
    def accuracy(self) -> int:
        return round(100 * (self.attempts - self.mistakes) / self.attempts) if self.attempts else 0

    @property
    def status(self) -> str:
        if self.accuracy >= 75:
            return "Good"
        if self.accuracy >= 50 or self.last_correct:
            return "Improving"
        return "Needs work"


def mistake_memory(attempts: list[dict]) -> list[MistakeTopic]:
    """Pages where the student has got at least one question wrong, worst first."""
    topics: dict[tuple[str, int], MistakeTopic] = {}
    for a in sorted(attempts, key=lambda a: a["created_at"]):
        for d in json.loads(a.get("details_json") or "[]"):
            key = (d.get("doc_name", ""), int(d.get("page_number", 0)))
            t = topics.setdefault(key, MistakeTopic(key[0], key[1], ""))
            t.topic = d.get("topic") or t.topic or f"Page {key[1]}"
            t.attempts += 1
            t.last_attempt = a["created_at"]
            t.last_correct = bool(d.get("is_correct"))
            if d.get("source_text"):
                t.source_text = d["source_text"]
            if not d.get("is_correct"):
                t.mistakes += 1
                t.examples.insert(0, {
                    "question": d.get("question", ""),
                    "your_answer": d.get("your_answer") or d.get("selected") or "",
                    "correct_answer": d.get("correct_answer") or d.get("correct") or "",
                    "explanation": d.get("explanation", ""),
                    "date": a["created_at"],
                })
    found = [t for t in topics.values() if t.mistakes]
    return sorted(found, key=lambda t: (t.status != "Needs work", -t.mistakes, t.accuracy))


# ---------- streak and study habits ----------
def study_streak(timestamps: list[str], today=None) -> int:
    """Consecutive days with any activity, ending today (or yesterday, so the streak
    isn't lost before the student has studied today)."""
    from datetime import date, datetime, timedelta

    days = set()
    for ts in timestamps:
        try:
            days.add(datetime.fromisoformat(ts).astimezone().date())
        except ValueError:
            continue
    today = today or date.today()
    day = today if today in days else today - timedelta(days=1)
    streak = 0
    while day in days:
        streak += 1
        day -= timedelta(days=1)
    return streak


HABITS = {
    "simpler": ("Asked for simpler explanations", ("simpler", "simple words", "easy words")),
    "detail": ("Asked for more detail", ("more detail", "in detail", "elaborate", "explain it more")),
    "example": ("Asked for examples", ("example",)),
}


def study_habits(user_messages: list[str]) -> dict[str, int]:
    """Counts of how the student actually asks for help (from their real messages)."""
    counts = {k: 0 for k in HABITS}
    counts["questions"] = len(user_messages)
    for msg in user_messages:
        low = msg.lower()
        for key, (_, words) in HABITS.items():
            if any(w in low for w in words):
                counts[key] += 1
    return counts
