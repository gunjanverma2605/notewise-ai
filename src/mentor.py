"""Nova, the personalised study mentor.

Nova does not invent a personality from nothing. Everything personal it says comes
from the student's own saved data: quiz results (Mistake Memory), activity (streak),
how they ask questions (study habits) and Study Mode sessions. Every answer about
the subject still goes through the normal grounded tutor (retrieval gate, page
citations, "not in your notes" refusals). Nova only changes HOW it answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src import config
from src.conversation_manager import needs_rewrite, rewrite_query
from src.progress import MistakeTopic, mistake_memory, study_habits, study_streak, summarise
from src.study_mode import PageOption
from src.tutor import TutorAnswer, answer_question


# ---------- the student's profile (only real data) ----------
@dataclass
class MentorProfile:
    streak: int = 0
    quizzes: int = 0
    answered: int = 0
    accuracy: int | None = None              # None until the first quiz
    weak: list[MistakeTopic] = field(default_factory=list)    # worst first, status != Good
    strengths: list[str] = field(default_factory=list)        # topic names
    habits: dict[str, int] = field(default_factory=dict)
    sessions: int = 0
    last_mood: str = ""
    pages: list[PageOption] = field(default_factory=list)

    @property
    def weakest(self) -> MistakeTopic | None:
        return self.weak[0] if self.weak else None

    @property
    def has_data(self) -> bool:
        return bool(self.quizzes or self.habits.get("questions") or self.sessions)


def build_profile(
    attempts: list[dict],
    activity: list[str],
    user_messages: list[str],
    sessions: list[dict],
    pages: list[PageOption],
    today=None,
) -> MentorProfile:
    summary = summarise(attempts)
    weak = [t for t in mistake_memory(attempts) if t.status != "Good"]
    return MentorProfile(
        streak=study_streak(activity, today=today),
        quizzes=summary.quizzes,
        answered=summary.answered,
        accuracy=summary.accuracy if summary.answered else None,
        weak=weak,
        strengths=[p.topic for p in summary.strengths][:3],
        habits=study_habits(user_messages),
        sessions=len(sessions),
        last_mood=(sessions[0].get("mood") or "") if sessions else "",
        pages=pages,
    )


def times(n: int) -> str:
    return "once" if n == 1 else "twice" if n == 2 else f"{n} times"


def page_label(topic: MistakeTopic) -> str:
    return f"{topic.topic} (p. {topic.page_number})"


# ---------- greeting ----------
def greeting(profile: MentorProfile, has_notes: bool) -> tuple[str, list[str]]:
    """A short, honest welcome built only from real data. Returns (headline, lines)."""
    if not has_notes:
        return ("Hi! I'm Nova, your study mentor 💜",
                ["Add your notes in the sidebar first. I only teach from YOUR notes, "
                 "so everything I say can be checked against a page."])
    lines: list[str] = []
    if profile.streak >= 2:
        lines.append(f"You're on a {profile.streak}-day study streak. Let's keep it going! 🔥")
    elif profile.streak == 1:
        lines.append("You studied today. Nice, every day counts.")
    if profile.weakest:
        w = profile.weakest
        lines.append(f"I noticed you missed **{page_label(w)}** {times(w.mistakes)}. Want me to explain it in a simpler way?")
    elif profile.accuracy is not None:
        lines.append(f"Your quiz accuracy is {profile.accuracy}%. "
                     + ("Great work, let's go deeper." if profile.accuracy >= 75 else "Let's lift it together."))
    else:
        lines.append("Ask me anything from your notes, or tap **Quiz Me** and I'll find out what you already know.")
    if profile.strengths:
        lines.append(f"You're strong in {', '.join(profile.strengths)}.")
    return ("Hi! I'm Nova, your study mentor 💜", lines)


# ---------- answering ----------
MODES = {
    "explain": ("Explain", "💡", "Explain the idea clearly for a student, in short paragraphs or a short numbered list."),
    "doubt": ("Doubt Solve", "❓",
              "The student is confused. Start with the one core idea in one sentence, then clear up the confusion "
              "step by step, and end with one short question the student can answer to check understanding."),
    "quiz": ("Quiz Me", "📝", ""),
    "plan": ("Study Plan", "🗓️", ""),
}

FOLLOW_UPS = {
    "simple": ("Explain simply", "Explain that again in very simple words",
               "Use very simple words and short sentences, as if to a 12-year-old. Keep every fact from the notes."),
    "example": ("Give an example", "Give me an example of that",
                "Give an example ONLY if the notes contain one. If the notes have no example, say so in one sentence "
                "and restate the key fact instead. Never make up an example."),
    "steps": ("Step by step", "Explain that step by step",
              "Answer as a numbered list of steps, one idea per step."),
    "exam": ("Exam answer", "How should I write that in an exam?",
             "Write it as a model exam answer: one definition sentence, then the key points as '- ' bullets, "
             "using the exact terms from the notes."),
}


def weak_context(profile: MentorProfile, question: str) -> str:
    """If the question is about a topic the student got wrong before, tell Nova what they confused."""
    q = question.lower()
    for t in profile.weak:
        words = [w for w in t.topic.lower().split() if len(w) > 3 and w != "basics"]
        if t.examples and words and any(w in q for w in words):
            ex = t.examples[0]
            return (f"Earlier in a quiz the student answered \"{ex['your_answer']}\" but the notes say "
                    f"\"{ex['correct_answer']}\". Gently point out the difference.")
    return ""


def mentor_style(mode: str, profile: MentorProfile, question: str, follow_up: str = "") -> str:
    parts = []
    if follow_up in FOLLOW_UPS:
        parts.append(FOLLOW_UPS[follow_up][2])
    else:
        parts.append(MODES.get(mode, MODES["explain"])[2])
    if profile.habits.get("simpler", 0) >= 2 and follow_up != "exam":
        parts.append("This student often asks for simpler words, so keep the language easy.")
    if mode == "doubt":
        hint = weak_context(profile, question)
        if hint:
            parts.append(hint)
    return " ".join(p for p in parts if p)


def ask(
    question: str,
    mode: str,
    profile: MentorProfile,
    doc_ids: list[str],
    index,
    gemma,
    history: list[dict[str, str]] | None = None,
    follow_up: str = "",
) -> TutorAnswer:
    """Grounded answer in Nova's style. Follow-ups ("explain that simply") are rewritten
    into a standalone search question first, exactly like the main tutor."""
    history = (history or [])[-config.MAX_HISTORY_TURNS:]
    search = question
    if follow_up or needs_rewrite(question, history):
        search = rewrite_query(question, history, gemma)
    style = mentor_style(mode, profile, search, follow_up)
    return answer_question(question, doc_ids, index, gemma, history=history, search_query=search, style=style)


def pages_for_query(query: str, doc_ids: list[str], index, limit: int = 2) -> list[tuple[str, int]]:
    """Which pages of the notes a request like 'quiz me on the Calvin cycle' is about.
    Empty when the topic is not in the notes."""
    if not doc_ids:
        return []
    result = index.search(query, doc_ids)
    if not result.supported:
        return []
    found: list[tuple[str, int]] = []
    for e in result.evidence:
        key = (e.chunk.doc_id, e.chunk.page_number)
        if key not in found:
            found.append(key)
    return found[:limit]


# ---------- study plan (from data only, no model call) ----------
def study_plan(profile: MentorProfile, minutes: int = 30) -> list[tuple[str, str]]:
    """A short plan for today as (time, step). Built from the student's real results,
    so it is instant and never hallucinated."""
    steps: list[tuple[str, str]] = []
    weak = profile.weak[:2]
    if weak:
        w = weak[0]
        steps.append(("10 min", f"Re-learn **{page_label(w)}** with Study Mode (you missed it {times(w.mistakes)})."))
        steps.append(("5 min", f"Ask me a doubt about {w.topic}, then tap *Exam answer* and write it from memory."))
        if len(weak) > 1:
            steps.append(("5 min", f"Quick recap of **{page_label(weak[1])}**."))
        steps.append(("10 min", "Take a 3-question follow-up quiz on those pages to lock it in."))
    elif profile.quizzes:
        steps.append(("10 min", "Pick a page you haven't been quizzed on and learn it in Study Mode."))
        steps.append(("10 min", "Take a 5-question quiz to find new weak spots."))
        steps.append(("10 min", "Explain one idea out loud without looking, then check it against the page."))
    elif profile.pages:
        first = profile.pages[0]
        steps.append(("10 min", f"Learn **page {first.page_number}** of {first.doc_name} in Study Mode."))
        steps.append(("10 min", "Take your first 5-question quiz so I can learn what you know."))
        steps.append(("10 min", "Ask me about anything that felt unclear."))
    else:
        steps.append(("2 min", "Add your notes in the sidebar. Then I can build a real plan for you."))
    if profile.last_mood in ("tired", "stressed"):
        steps.append(("Tip", "You felt " + profile.last_mood + " last session, so take a 5-minute break halfway."))
    return steps


# ---------- quick prompts and tips ----------
@dataclass
class QuickPrompt:
    label: str
    kind: str          # "ask" | "page" | "quiz" | "plan"
    mode: str = "explain"
    text: str = ""
    page_key: str = ""


def quick_prompts(profile: MentorProfile) -> list[QuickPrompt]:
    prompts: list[QuickPrompt] = []
    w = profile.weakest
    if w:
        prompts.append(QuickPrompt(f"Explain {w.topic} simply", "ask", "explain",
                                   f"Explain {w.topic} in simple words"))
        prompts.append(QuickPrompt(f"I'm confused about {w.topic}", "ask", "doubt",
                                   f"I'm confused about {w.topic}. Can you help?"))
    elif profile.pages:
        p = profile.pages[0]
        prompts.append(QuickPrompt(f"Teach me page {p.page_number}", "page", page_key=p.key))
        if len(profile.pages) > 1:
            p2 = profile.pages[1]
            prompts.append(QuickPrompt(f"Teach me page {p2.page_number}", "page", page_key=p2.key))
    prompts.append(QuickPrompt("Quiz me on my weak areas" if w else "Quiz me on my notes", "quiz"))
    prompts.append(QuickPrompt("Make my study plan for today", "plan"))
    return prompts


TECHNIQUES = [
    ("Active recall", "Close your notes and write what you remember. Then check. Testing yourself beats re-reading."),
    ("Spaced repetition", "Review a topic after 1 day, 3 days and 1 week. Short reviews, spread out, stick longest."),
    ("Feynman technique", "Explain the idea in simple words as if teaching a friend. Where you get stuck is your gap."),
    ("Pomodoro", "Study for 25 minutes, then rest for 5. After four rounds, take a longer break."),
    ("Interleaving", "Mix two or three topics in one session instead of one topic for hours."),
]


def tips(profile: MentorProfile) -> list[tuple[str, str]]:
    """Personal tips first (from real data), then proven study techniques."""
    out: list[tuple[str, str]] = []
    if profile.weakest:
        w = profile.weakest
        out.append(("Fix your weak spot", f"You missed {page_label(w)} {times(w.mistakes)}. "
                    "Explain it out loud in your own words, then take a 3-question quiz on it."))
    if profile.habits.get("simpler", 0) >= 2:
        out.append(("You like simple words", "Start with *Explain simply*, then tap *Exam answer* "
                    "so you also learn the exact terms examiners look for."))
    if profile.accuracy is not None and profile.accuracy >= 75:
        out.append(("You're doing well", f"{profile.accuracy}% accuracy. Push yourself with harder pages "
                    "instead of repeating the easy ones."))
    if profile.streak >= 2:
        out.append(("Keep the streak", f"{profile.streak} days in a row! Even 5 minutes tomorrow keeps it alive."))
    elif profile.has_data:
        out.append(("Build a streak", "Study a little every day. A 5-minute Study Mode session counts."))
    return out + TECHNIQUES
