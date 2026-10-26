"""End-to-end tests of the Streamlit app with fake models (no Ollama needed).

Streamlit's AppTest runs app.py in-process and lets us click buttons and
type into widgets, then inspect what is on screen.
"""

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import services
from src.conversation_manager import ConversationManager, REWRITE_PROMPT, TITLE_PROMPT
from src.database import Database
from src.gemma_client import HealthStatus
from src.mind_map import MINDMAP_PROMPT
from src.quiz_generator import QUIZ_PROMPT, VERIFY_PROMPT
from src.retrieval import NoteIndex

APP = str(Path(__file__).resolve().parent.parent / "app.py")

QUIZ_QUESTIONS = [
    "Which colour does chlorophyll reflect?",
    "Where do the light-dependent reactions take place?",
    "Which enzyme fixes carbon dioxide in the Calvin cycle?",
    "Where does the oxygen released by plants come from?",
    "What do CAM plants do at night?",
    "What limits the rate of photosynthesis?",
    "Why is photosynthesis important for food chains?",
    "What happens in photolysis?",
]
ANSWER_KEY: dict[str, str] = {}      # question -> correct option, filled by the fake


class AppGemma:
    """Fake Gemma that knows each job the app gives it."""

    def __init__(self):
        self.n = 0
        self.last_answer = ""

    def chat(self, messages, json_mode=False, temperature=None):
        system = messages[0]["content"]
        if system == REWRITE_PROMPT:
            return "What does RuBisCO do in the Calvin cycle?"
        if system == TITLE_PROMPT:
            return "Calvin Cycle Basics"
        if system == QUIZ_PROMPT:
            question = QUIZ_QUESTIONS[self.n % len(QUIZ_QUESTIONS)]
            passage = messages[1]["content"].split("\n", 1)[1]
            answer = " ".join(passage.split()[:6])          # taken from the passage, like real Gemma
            self.n += 1
            self.last_answer = answer
            ANSWER_KEY[question] = answer
            return json.dumps({
                "question": question,
                "options": [
                    {"text": answer, "correct": True, "why_wrong": ""},
                    {"text": f"Mitochondria {self.n}", "correct": False, "why_wrong": "The notes never say this."},
                    {"text": f"Ribosomes {self.n}", "correct": False, "why_wrong": "The notes never say this."},
                    {"text": f"Nucleus {self.n}", "correct": False, "why_wrong": "The notes never say this."},
                ],
                "explanation": f"The notes say: {answer}.",
                "hint": "Read the passage again carefully.",
                "topic": question.split()[-1].rstrip("?").capitalize() + " basics",
            })
        if system == MINDMAP_PROMPT:
            import re as _re
            blocks = _re.findall(r"\[S(\d+)\] \(from [^)]*\)\n(.+?)(?=\n\n\[S|\n\nMake)", messages[1]["content"], _re.S)
            branches = []
            for n, text in blocks[:5]:
                w = text.split()
                branches.append({"topic": " ".join(w[:3]), "source": f"S{n}",
                                 "points": [" ".join(w[:6]), " ".join(w[6:12]), "made up fact about volcanoes"]})
            return json.dumps({"title": "Photosynthesis", "branches": branches})
        if system == VERIFY_PROMPT:
            for line in messages[1]["content"].splitlines():
                if line[3:].strip() == self.last_answer:
                    return line[0]
            return "NONE"
        return "RuBisCO joins carbon dioxide to RuBP [S1]."


@pytest.fixture
def app(tmp_path, fake_embedder, monkeypatch):
    def fake_services():
        db = Database(tmp_path / "notewise.db")
        index = NoteIndex(fake_embedder, persist_dir=tmp_path / "chroma")
        manager = ConversationManager(db, index, upload_dir=tmp_path / "uploads")
        return services.Services(db=db, index=index, manager=manager, gemma=AppGemma())

    monkeypatch.setattr(services, "build_services", fake_services)
    monkeypatch.setattr(services, "health", lambda: HealthStatus(True, True, True, "Ready"))
    monkeypatch.setattr("src.config.MIN_RELEVANCE", 0.2)       # threshold tuned for the fake embedder
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["entered"] = True          # skip the landing page
    at.run()
    yield at
    st.cache_resource.clear()
    st.cache_data.clear()


def click(at, label):
    for b in at.button:
        if b.label == label:
            b.click()
            return at.run()
    raise AssertionError(f"No button {label!r}; have {[b.label for b in at.button]}")


def all_markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_starts_with_welcome_screen(app):
    assert not app.exception
    assert "Study from your" in all_markdown(app) and "own notes" in all_markdown(app)
    assert app.chat_input[0].disabled            # can't chat before adding notes


def test_sample_notes_then_supported_question(app):
    click(app, "Use sample notes (Photosynthesis)")
    assert not app.exception
    assert any("8 page(s)" in s.value for s in app.success)
    app.chat_input[0].set_value("Which enzyme joins carbon dioxide to RuBP in phase 1 of the Calvin cycle?").run()
    text = all_markdown(app)
    assert "From your notes" in text
    assert 'class="nw-page">p. 4' in text        # page chip from the [S1] citation
    assert any("Source: page" in e.label for e in app.expander)


def test_unsupported_question_is_refused(app):
    click(app, "Use sample notes (Photosynthesis)")
    app.chat_input[0].set_value("What is the GDP of India?").run()
    text = all_markdown(app)
    assert "Not in your notes" in text
    assert "couldn&#x27;t find enough information" in text


def test_follow_up_button_and_chat_title(app):
    click(app, "Use sample notes (Photosynthesis)")
    app.chat_input[0].set_value("Which enzyme joins carbon dioxide to RuBP in phase 1 of the Calvin cycle?").run()
    click(app, "Explain that in simpler words")
    assert not app.exception
    users = [m for m in app.chat_message if m.name == "user"]
    assert len(users) == 2
    assert "Calvin Cycle Basics" in all_markdown(app)              # title from the first question


def test_new_chat_and_reopen_old_chat(app):
    click(app, "Use sample notes (Photosynthesis)")
    app.chat_input[0].set_value("Which enzyme joins carbon dioxide to RuBP in phase 1 of the Calvin cycle?").run()
    click(app, "New chat")
    assert len(app.chat_message) == 0
    click(app, "Calvin Cycle Basics")
    assert len(app.chat_message) == 2                               # restored from SQLite


def test_pasted_notes(app):
    app.text_area[0].set_value("Osmosis is the movement of water across a semi-permeable membrane.").run()
    click(app, "Add pasted notes")
    assert not app.exception
    assert "Pasted notes" in all_markdown(app)
    assert not app.chat_input[0].disabled


def test_quiz_wrong_answer_feedback_and_progress(app):
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Quiz")
    click(app, "Generate 5-question quiz")
    assert not app.exception
    radio = app.radio[0]
    wrong = next(o for o in radio.options if "Mitochondria" in o)
    radio.set_value(wrong).run()
    click(app, "Check answer")
    text = all_markdown(app)
    assert "Not quite, the correct answer is" in text
    assert "The notes never say this." in text
    assert any("Source:" in e.label for e in app.expander)

    for _ in range(4):                                              # answer the rest correctly
        click(app, "Next question →")
        radio = app.radio[0]
        q_text = all_markdown(app)
        answer = next(a for q, a in ANSWER_KEY.items() if q in q_text)
        radio.set_value(next(o for o in radio.options if o.endswith(answer))).run()
        click(app, "Check answer")
    click(app, "See results →")
    assert "4 / 5" in all_markdown(app)

    click(app, "My Learning DNA")
    assert not app.exception
    text = all_markdown(app)
    assert "80%" in text and "4 of 5 correct" in text
    assert "Mistake Memory" in text and "Your Next Best Step" in text


def test_ollama_down_shows_clear_message(app, monkeypatch):
    monkeypatch.setattr(services, "health", lambda: HealthStatus(False, False, False, "Ollama is not running."))
    st.cache_data.clear()
    app.run()
    assert any("brew services start ollama" in e.value for e in app.error)
    assert app.chat_input[0].disabled


def test_new_chat_on_empty_chat_does_not_pile_up(app):
    before = sum(1 for b in app.button if b.key and b.key.startswith("chat_"))
    click(app, "New chat")
    click(app, "New chat")
    after = sum(1 for b in app.button if b.key and b.key.startswith("chat_"))
    assert after == before == 1


def test_progress_is_per_chat_with_all_chats_option(app):
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Quiz")
    click(app, "Generate 5-question quiz")
    for n in range(5):
        if n:
            click(app, "Next question →")
        app.radio[0].set_value(app.radio[0].options[0]).run()
        click(app, "Check answer")
    click(app, "See results →")
    click(app, "My Learning DNA")
    assert "of 5 correct" in all_markdown(app)
    click(app, "New chat")
    click(app, "My Learning DNA")
    assert "Your Learning DNA is empty" in all_markdown(app)
    app.segmented_control(key="progress_scope").set_value("All chats").run()
    assert "of 5 correct" in all_markdown(app)


def answer_all(app, pick_wrong=False):
    """Answer every question of the quiz on screen; returns the number answered."""
    n = 0
    while True:
        radio = app.radio(key=[r.key for r in app.radio if r.key and r.key.startswith("choice_")][0])
        text = all_markdown(app)
        answer = next((a for q, a in ANSWER_KEY.items() if q in text), None)
        options = radio.options
        good = next((o for o in options if answer and answer in o), options[0])
        radio.set_value(next(o for o in options if o != good) if pick_wrong else good).run()
        click(app, "Check answer")
        n += 1
        labels = [b.label for b in app.button]
        if "See results →" in labels:
            click(app, "See results →")
            return n
        click(app, "Next question →")


def test_study_mode_full_session(app):
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Study Mode")
    assert not app.exception
    text = all_markdown(app)
    assert "How are you feeling today?" in text and "Your Study Plan" in text
    app.radio(key="study_mood").set_value("tired").run()
    app.radio(key="study_len").set_value(5).run()
    assert "2 practice questions" in all_markdown(app)            # tired + 5 min -> 2 questions
    click(app, "Start session")
    assert not app.exception
    assert "From your notes" in all_markdown(app)                 # grounded explanation with citation
    click(app, "Continue to practice →")
    assert answer_all(app) == 2
    text = all_markdown(app)
    assert "Session complete" in text and "Rest well" in text     # tired-mood message
    click(app, "Open My Learning DNA")
    assert "1 study session" in all_markdown(app)


def test_mistake_memory_review_and_follow_up_quiz(app):
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Quiz")
    click(app, "Generate 5-question quiz")
    answer_all(app, pick_wrong=True)
    click(app, "My Learning DNA")
    text = all_markdown(app)
    assert "Mistake Memory" in text and "Needs work" in text and "Your Next Best Step" in text
    app.button(key="review_0").click().run()
    text = all_markdown(app)
    assert "You answered" in text and "Correct:" in text and "Relevant passage from your notes" in text
    app.button(key="review_quiz").click().run()
    assert not app.exception
    assert "Follow-up quiz" in all_markdown(app)                  # now on the Quiz tab with 3 questions
    assert answer_all(app) == 3


def test_next_best_step_opens_study_mode_on_that_page(app):
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Quiz")
    click(app, "Generate 5-question quiz")
    answer_all(app, pick_wrong=True)
    click(app, "My Learning DNA")
    app.button(key="next_explain").click().run()
    assert not app.exception
    assert "Your Study Plan" in all_markdown(app)
    assert app.radio(key="study_mood").value == "difficult"


def nova_input(app):
    return next(c for c in app.chat_input if c.key == "nova_input")


def test_nova_greets_teaches_and_follows_up(app):
    click(app, "Meet Nova")
    assert not app.exception
    text = all_markdown(app)
    assert "Meet Nova" in text and "Add your notes in the sidebar first" in text
    assert nova_input(app).disabled
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Meet Nova")
    assert "What Nova knows about you" in all_markdown(app)
    click(app, "Teach me page 1")                                   # grounded page explanation
    assert not app.exception
    assert "From your notes" in all_markdown(app)
    click(app, "Explain simply")                                    # follow-up chip
    assert not app.exception
    assert "Explain that again in very simple words" in all_markdown(app)
    app.segmented_control(key="nova_mode").set_value("doubt").run()
    nova_input(app).set_value("Which enzyme joins carbon dioxide to RuBP in the Calvin cycle?").run()
    assert not app.exception
    text = all_markdown(app)
    assert "Doubt Solve" in text and 'class="nw-page">p. 4' in text
    click(app, "Next ›")
    assert "Nova&#x27;s Study Tips" in all_markdown(app) or "Nova's Study Tips" in all_markdown(app)


def test_nova_uses_mistakes_for_greeting_plan_and_quiz(app):
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Quiz")
    click(app, "Generate 5-question quiz")
    answer_all(app, pick_wrong=True)
    click(app, "Meet Nova")
    text = all_markdown(app)
    assert "I noticed you missed" in text and "Fix your weak spot" in text
    click(app, "Make my study plan for today")
    text = all_markdown(app)
    assert "Built from your progress" in text and "Study Mode" in text
    click(app, "Quiz me on my weak areas")
    assert not app.exception
    assert "Nova&#x27;s quiz" in all_markdown(app) or "Nova's quiz" in all_markdown(app)
    assert answer_all(app) == 3


def test_nova_refuses_quiz_on_topic_not_in_notes(app):
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Meet Nova")
    app.segmented_control(key="nova_mode").set_value("quiz").run()
    nova_input(app).set_value("GDP of India").run()
    assert not app.exception
    assert "couldn" in all_markdown(app) and "in your notes" in all_markdown(app)


def test_mind_map_tab_builds_checked_map(app):
    click(app, "Mind Map")
    assert "Add your notes in the sidebar first" in "\n".join(i.value for i in app.info)
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Mind Map")
    click(app, "Create mind map")
    assert not app.exception
    text = all_markdown(app)
    assert "<svg" in text and "Photosynthesis" in text and "p. 1" in text
    assert "volcanoes" not in text                     # unsupported point was dropped
    click(app, "Quick quiz")
    assert not app.exception
    assert "Follow-up quiz" in all_markdown(app)


def test_revision_book_builds_from_mind_map(app):
    click(app, "Use sample notes (Photosynthesis)")
    click(app, "Book")
    click(app, "Create revision book")
    assert not app.exception
    text = all_markdown(app)
    assert "Revision Book" in text and "Chapter 1" in text and "Contents" in text and "bk-page" in text
    assert any(b.label == "Download book" for b in app.get("download_button"))


def test_landing_page_then_start(tmp_path, fake_embedder, monkeypatch):
    def fake_services():
        db = Database(tmp_path / "notewise.db")
        index = NoteIndex(fake_embedder, persist_dir=tmp_path / "chroma")
        return services.Services(db=db, index=index, manager=ConversationManager(db, index, tmp_path / "up"),
                                 gemma=AppGemma())

    monkeypatch.setattr(services, "build_services", fake_services)
    monkeypatch.setattr(services, "health", lambda: HealthStatus(True, True, True, "Ready"))
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert "Study smarter from" in all_markdown(at) and len(at.sidebar.button) == 0
    click(at, "Start studying →")
    assert any(b.label == "New chat" for b in at.button)


def test_dark_mode_toggle(app):
    app.toggle(key="dark_mode").set_value(True).run()
    assert not app.exception
    assert "AUTO-GENERATED" in all_markdown(app)


def test_study_timer_start_pause_stop(app):
    app.selectbox(key="timer_pick").set_value("5 hours").run()
    app.button(key="timer_start").click().run()
    assert not app.exception
    assert app.session_state["timer"]["total"] == 5 * 3600
    app.button(key="timer_pause").click().run()
    assert app.session_state["timer"]["paused_left"] is not None
    app.button(key="timer_resume").click().run()
    app.button(key="timer_stop").click().run()
    assert "timer" not in app.session_state
