"""Check quiz answers. Pure Python, no AI involved.

The score comes only from comparing the student's choice with the stored
answer key. A convincing explanation can never turn a wrong answer right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.quiz_generator import LETTERS, QuizQuestion


@dataclass
class AnswerFeedback:
    question_id: str
    selected_index: int
    correct_index: int
    is_correct: bool
    headline: str                # "Correct!" / "Not quite, the correct answer is B."
    explanation: str             # why the correct answer is right
    why_wrong: str               # why the student's choice is wrong (empty if correct)
    hint: str                    # a nudge for revision
    page_number: int
    doc_name: str
    source_text: str


def evaluate_answer(question: QuizQuestion, selected_index: int) -> AnswerFeedback:
    if not isinstance(selected_index, int) or not 0 <= selected_index < len(question.options):
        raise ValueError("Please choose one of the options.")
    correct = selected_index == question.correct_index
    if correct:
        headline = "Correct!"
        why_wrong = ""
    else:
        headline = f"Not quite, the correct answer is {LETTERS[question.correct_index]}: {question.correct_option}"
        why_wrong = question.wrong_reasons[selected_index] or (
            f"\"{question.options[selected_index]}\" is not what your notes say on page {question.page_number}."
        )
    return AnswerFeedback(
        question_id=question.id,
        selected_index=selected_index,
        correct_index=question.correct_index,
        is_correct=correct,
        headline=headline,
        explanation=question.explanation,
        why_wrong=why_wrong,
        hint=question.hint,
        page_number=question.page_number,
        doc_name=question.doc_name,
        source_text=question.source_text,
    )


@dataclass
class QuizSession:
    """Tracks one quiz attempt. Each question can be answered once."""

    questions: list[QuizQuestion]
    answers: dict[str, AnswerFeedback] = field(default_factory=dict)

    def submit(self, question_id: str, selected_index: int) -> AnswerFeedback:
        if question_id in self.answers:
            return self.answers[question_id]          # re-clicking can't change the score
        question = next((q for q in self.questions if q.id == question_id), None)
        if question is None:
            raise KeyError("Unknown question.")
        feedback = evaluate_answer(question, selected_index)
        self.answers[question_id] = feedback
        return feedback

    @property
    def score(self) -> int:
        return sum(1 for f in self.answers.values() if f.is_correct)

    @property
    def answered(self) -> int:
        return len(self.answers)

    @property
    def total(self) -> int:
        return len(self.questions)

    @property
    def finished(self) -> bool:
        return self.total > 0 and self.answered == self.total

    def pages_to_revise(self) -> list[int]:
        return sorted({f.page_number for f in self.answers.values() if not f.is_correct})

    def details(self) -> list[dict]:
        """Saved to the database with the attempt (used by the Progress page)."""
        rows = []
        for q in self.questions:
            f = self.answers.get(q.id)
            rows.append(
                {
                    "question": q.question,
                    "page_number": q.page_number,
                    "doc_name": q.doc_name,
                    "topic": q.topic or f"Page {q.page_number}",
                    "selected": None if f is None else LETTERS[f.selected_index],
                    "correct": q.correct_letter,
                    "is_correct": bool(f and f.is_correct),
                    # kept so Mistake Memory can show what went wrong later
                    "your_answer": None if f is None else q.options[f.selected_index],
                    "correct_answer": q.correct_option,
                    "explanation": q.explanation,
                    "source_text": q.source_text[:800],
                }
            )
        return rows
