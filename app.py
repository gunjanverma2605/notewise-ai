"""NoteWise AI — Streamlit app.

Run:  streamlit run app.py

This file only handles the screen. All the real work lives in src/:
loading notes, retrieval, the Gemma tutor, chat history and quizzes.
Streamlit reruns this whole script on every click, so anything that must
survive a refresh is stored in SQLite (src/database.py), and session_state
only remembers what is on screen right now.
"""

from __future__ import annotations

import html
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="NoteWise AI", page_icon="📘", layout="wide")

from src import book, config, mentor, mind_map, services, study_timer  # noqa: E402
from src.document_loader import LoadedDocument, compute_doc_id, load_document, load_text  # noqa: E402
from src.errors import DocumentError, NoteWiseError  # noqa: E402
from src.quiz_evaluator import QuizSession  # noqa: E402
from src.progress import HABITS, mistake_memory, study_habits, study_streak, summarise  # noqa: E402
from src.study_mode import (  # noqa: E402
    GOALS, LENGTHS, MOODS, PageOption, choose_pages, chunks_for, explain, make_plan, page_options,
)
from src.quiz_generator import LETTERS, QuizQuestion, generate_quiz  # noqa: E402

ROOT = Path(__file__).resolve().parent
SAMPLE_PDF = ROOT / "sample_data" / "photosynthesis_notes.pdf"

TUTOR_AVATAR = str(ROOT / "static" / "avatar_tutor.png")
STUDENT_AVATAR = str(ROOT / "static" / "avatar_student.png")

LOGO = """<svg width="46" height="46" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="nwg" x1="0" y1="0" x2="34" y2="34"><stop stop-color="#7B5CFF"/><stop offset="1" stop-color="#4B2FD6"/></linearGradient></defs>
<rect width="34" height="34" rx="9" fill="url(#nwg)"/>
<path d="M17 11.5c-2.2-1.6-5-2.2-8-1.8v13c3-.4 5.8.2 8 1.8 2.2-1.6 5-2.2 8-1.8v-13c-3-.4-5.8.2-8 1.8Z"
 fill="#fff"/><path d="M17 11.5v13" stroke="#6D4AFF" stroke-width="1.4"/>
<circle cx="25.5" cy="8.5" r="2" fill="#FFD66B"/></svg>"""


def file_icon(name: str) -> str:
    """Small coloured file badge: red for PDF, blue for text."""
    is_pdf = name.lower().endswith(".pdf")
    color, bg, label = ("#E5484D", "#FDECEE", "PDF") if is_pdf else ("#3B82F6", "#E8F1FF", "TXT")
    return (f'<svg width="34" height="40" viewBox="0 0 34 40"><rect x="1" y="1" width="32" height="38" rx="7" fill="{bg}"/>'
            f'<path d="M9 8h11l6 6v18H9z" fill="#fff" stroke="{color}" stroke-width="1.6" stroke-linejoin="round"/>'
            f'<rect x="6" y="21" width="22" height="9" rx="2.5" fill="{color}"/>'
            f'<text x="17" y="28" text-anchor="middle" font-size="6.5" font-weight="800" fill="#fff" '
            f'font-family="sans-serif">{label}</text></svg>')


ICON_NOTES = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M7 3h7l5 5v13H7z" fill="#fff" stroke="#7B5CFF" stroke-width="1.8" stroke-linejoin="round"/><path d="M10 12h6M10 15.5h6M10 8.5h3" stroke="#7B5CFF" stroke-width="1.8" stroke-linecap="round"/></svg>"""
ICON_CHAT = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M4 6.5A3.5 3.5 0 0 1 7.5 3h9A3.5 3.5 0 0 1 20 6.5v6a3.5 3.5 0 0 1-3.5 3.5H11l-4.5 4v-4A3.5 3.5 0 0 1 4 12.5z" fill="#fff"/><circle cx="8.6" cy="9.6" r="1.3" fill="#3B82F6"/><circle cx="12" cy="9.6" r="1.3" fill="#3B82F6"/><circle cx="15.4" cy="9.6" r="1.3" fill="#3B82F6"/></svg>"""
ICON_CHART = """<svg width="26" height="26" viewBox="0 0 24 24" fill="none"><rect x="4" y="12" width="4" height="8" rx="1.5" fill="#fff"/><rect x="10" y="8" width="4" height="12" rx="1.5" fill="#fff"/><rect x="16" y="4" width="4" height="16" rx="1.5" fill="#fff"/></svg>"""

STATUS_LABELS = {
    "answered": "✓ From your notes",
    "partial": "◐ Your notes only partly cover this",
    "unverified": "⚠ Couldn't match this to a page — check the passages",
    "not_in_notes": "Not in your notes",
    "no_document": "No notes yet",
    "error": "Something went wrong",
}

QUICK_FOLLOW_UPS = ["Explain that in simpler words", "Explain it in more detail", "Give an example from my notes"]


# ---------- setup ----------
@st.cache_resource(show_spinner=False)
def get_services() -> services.Services:
    return services.build_services()


@st.cache_data(ttl=15, show_spinner=False)
def get_health():
    return services.health()


def load_css() -> None:
    css = (ROOT / "assets" / "style.css").read_text(encoding="utf-8")
    if st.session_state.get("dark_mode"):
        css += (ROOT / "assets" / "dark.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def format_answer(text: str) -> str:
    """Escape the model's text, then turn [p. 3] into a small page chip."""
    safe = html.escape(text)
    return re.sub(r"\[p\. ([\d, ]+)\]", lambda m: f'<span class="nw-page">p. {m.group(1)}</span>', safe)


def rich(text: str) -> str:
    """format_answer plus **bold** and *italic*, for text placed inside HTML blocks."""
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", format_answer(text))
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", out)


def reset_quiz() -> None:
    st.session_state.pop("quiz", None)


def open_chat(conversation_id: str) -> None:
    st.session_state.conv_id = conversation_id
    reset_quiz()
    st.session_state.pop("review", None)


# ---------- notes ----------
def add_notes(svc: services.Services, conv_id: str, data: bytes, name: str, pasted: bool = False) -> None:
    linked = {d.doc_id for d in svc.db.conversation_documents(conv_id)}
    if compute_doc_id(data) in linked:
        return          # same file again, or Streamlit rerunning with the uploader still filled
    try:
        if pasted:
            doc: LoadedDocument = load_text(data.decode("utf-8"), name=name, source_type="paste")
        else:
            doc = load_document(data, name)
    except DocumentError as exc:
        st.session_state.notice = ("error", str(exc))
        return

    try:
        with st.spinner(f"Reading and indexing {name} with EmbeddingGemma…"):
            chunks = svc.manager.add_document(conv_id, doc, data)
    except NoteWiseError as exc:
        st.session_state.notice = ("error", str(exc))
        return
    st.session_state.setdefault("reports", {})[doc.doc_id] = {
        "name": doc.name,
        "pages": doc.page_count,
        "ocr_pages": doc.ocr_pages,
        "warnings": doc.warnings,
        "chunks": chunks,
        "page_text": [(p.page_number, p.method, p.text) for p in doc.pages],
    }
    st.session_state.notice = ("success", f"Added “{doc.name}”: {doc.page_count} page(s), {chunks} passages indexed.")
    reset_quiz()


def render_sidebar(svc: services.Services, conv_id: str) -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="nw-brand">{LOGO}<div class="nw-brand-name">NoteWise <span>AI</span></div></div>'
            '<div class="nw-tagline">Your Notes. Your Learning Companion.</div>',
            unsafe_allow_html=True,
        )
        health = get_health()
        if health.ok:
            st.markdown(f'<span class="nw-status ok">● Gemma 4 ready · runs offline</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="nw-status bad">● Gemma 4 not available</span>', unsafe_allow_html=True)
            st.caption(health.message)

        st.toggle("Dark mode", key="dark_mode", help="Switch between light and dark theme")
        render_study_timer()
        if st.button("New chat", icon=":material/add:", use_container_width=True, type="primary", key="new_chat"):
            # Reuse the open chat if it's still empty, so the list doesn't fill with blank chats.
            empty = svc.db.message_count(conv_id) == 0 and not svc.db.conversation_documents(conv_id)
            if not empty:
                open_chat(svc.manager.new_chat())
            st.rerun()

        # ----- notes for this chat -----
        st.markdown('<div class="nw-section">Notes in this chat</div>', unsafe_allow_html=True)
        statuses = svc.manager.document_status(conv_id)
        if not statuses:
            st.caption("No notes yet. Add a PDF, a .txt file or paste text below.")
        for s in statuses:
            kind = "" if s.ready else " missing"
            extra = f"{s.record.page_count} page(s)" if s.ready else html.escape(s.message)
            if s.restored:
                extra += " · restored"
            st.markdown(
                f'<div class="nw-doc{kind}">{file_icon(s.record.name)}'
                f'<div class="nw-doc-text"><b title="{html.escape(s.record.name)}">{html.escape(s.record.name)}</b>'
                f'<small>{extra}</small></div></div>',
                unsafe_allow_html=True,
            )

        uploaded = st.file_uploader(
            "Upload notes", type=["pdf", "txt", "md"], key=f"uploader_{conv_id}",
            help=f"PDF (also scanned) or text file, up to {config.MAX_PAGES} pages / {config.MAX_FILE_MB} MB.",
        )
        if uploaded is not None:
            add_notes(svc, conv_id, uploaded.getvalue(), uploaded.name)

        with st.expander("Paste notes instead"):
            pasted = st.text_area("Your notes", height=140, key=f"paste_{conv_id}", label_visibility="collapsed",
                                  placeholder="Paste lecture notes here…")
            title = st.text_input("Name", value="Pasted notes", key=f"paste_name_{conv_id}")
            if st.button("Add pasted notes", use_container_width=True, disabled=not pasted.strip()):
                add_notes(svc, conv_id, pasted.encode("utf-8"), title.strip() or "Pasted notes", pasted=True)
                st.rerun()

        if SAMPLE_PDF.exists() and not statuses:
            if st.button("Use sample notes (Photosynthesis)", icon=":material/description:", use_container_width=True):
                add_notes(svc, conv_id, SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)
                st.rerun()

        # ----- saved chats -----
        st.markdown('<div class="nw-section">Your chats</div>', unsafe_allow_html=True)
        with st.container(key="chatlist"):
            for conv in svc.db.list_conversations(limit=30):
                current = conv.id == conv_id
                if st.button(conv.title, key=f"chat_{conv.id}", icon=":material/chat_bubble:",
                             use_container_width=True, type="secondary", disabled=current):
                    open_chat(conv.id)
                    st.rerun()

        with st.expander("Chat options"):
            current = svc.db.get_conversation(conv_id)
            new_title = st.text_input("Rename this chat", value=current.title if current else "", key=f"rename_{conv_id}")
            if st.button("Save name", use_container_width=True):
                svc.db.rename_conversation(conv_id, new_title)
                st.rerun()
            confirm = st.checkbox("Yes, delete this chat", key=f"confirm_delete_{conv_id}")
            if st.button("Delete chat", use_container_width=True, disabled=not confirm):
                svc.db.delete_conversation(conv_id)
                st.session_state.pop("conv_id", None)
                reset_quiz()
                st.rerun()

        with st.expander("Settings & help"):
            st.caption(
                f"Model: **{config.CHAT_MODEL}** · Embeddings: **{config.EMBED_MODEL}**  \n"
                f"Relevance threshold: {config.MIN_RELEVANCE} · Passages per answer: {config.TOP_K}  \n"
                "Everything runs on this laptop through Ollama. Your notes are stored in the "
                "project's `data/` folder and never leave the device."
            )
            really = st.checkbox("I want to erase all chats, notes and quiz scores", key="confirm_reset")
            if st.button("Reset demo data", use_container_width=True, disabled=not really):
                for record in svc.db.list_documents():
                    svc.index.delete_document(record.doc_id)
                    if record.file_path:
                        Path(record.file_path).unlink(missing_ok=True)
                svc.db.reset_all()
                for key in ("conv_id", "quiz", "reports", "confirm_reset"):
                    st.session_state.pop(key, None)
                st.rerun()

        if st.button("Home page", icon=":material/home:", use_container_width=True, key="go_home"):
            st.session_state.entered = False
            st.rerun()

        st.markdown(
            '<div class="nw-side-girl"><img src="app/static/sidebar_girl.jpg" alt="">'
            '<div class="nw-side-quote">Same notes.<br>Better learning. 💜</div></div>',
            unsafe_allow_html=True,
        )


# ---------- study timer ----------
def render_study_timer() -> None:
    """Sidebar focus timer: pick 25 minutes to 5 hours, then pause, resume or stop."""
    st.markdown('<div class="nw-section">Study timer</div>', unsafe_allow_html=True)
    state = st.session_state.get("timer")
    with st.container(key="timerbox"):
        if not state:
            options = list(study_timer.PRESETS) + ["Custom"]
            pick = st.selectbox("Study for", options, index=1, key="timer_pick", label_visibility="collapsed")
            minutes = study_timer.PRESETS.get(pick)
            if minutes is None:
                minutes = st.number_input("Minutes", min_value=5, max_value=600, value=90, step=5, key="timer_custom")
            breaks = st.toggle("Break reminder every 50 min", value=True, key="timer_breaks")
            if st.button("Start timer", icon=":material/timer:", use_container_width=True, key="timer_start"):
                st.session_state.timer = study_timer.start(minutes, breaks)
                st.rerun()
            return
        components.html(study_timer.clock_html(state, dark=bool(st.session_state.get("dark_mode"))), height=104)
        c1, c2 = st.columns(2)
        if state.get("paused_left") is None:
            if c1.button("Pause", icon=":material/pause:", use_container_width=True, key="timer_pause"):
                st.session_state.timer = study_timer.pause(state)
                st.rerun()
        elif c1.button("Resume", icon=":material/play_arrow:", use_container_width=True, key="timer_resume"):
            st.session_state.timer = study_timer.resume(state)
            st.rerun()
        if c2.button("Stop", icon=":material/stop:", use_container_width=True, key="timer_stop"):
            st.session_state.pop("timer", None)
            st.rerun()


# ---------- tutor ----------
def render_message(m) -> None:
    avatar = STUDENT_AVATAR if m.role == "user" else TUTOR_AVATAR
    with st.chat_message(m.role, avatar=avatar):
        if m.role == "user":
            st.markdown(f'<div class="nw-user-msg">{html.escape(m.content)}</div>', unsafe_allow_html=True)
            return
        st.markdown('<span class="nw-bot-msg"></span>', unsafe_allow_html=True)
        if m.status in STATUS_LABELS and m.status != "answered":
            st.markdown(f'<span class="nw-badge {m.status}">{STATUS_LABELS[m.status]}</span>', unsafe_allow_html=True)
        elif m.status == "answered":
            st.markdown(f'<span class="nw-badge answered">{STATUS_LABELS["answered"]}</span>', unsafe_allow_html=True)
        st.markdown(format_answer(m.content), unsafe_allow_html=True)
        if m.citations:
            pages = sorted({c["page_number"] for c in m.citations})
            with st.expander("Source: page " + ", ".join(map(str, pages)), icon=":material/menu_book:"):
                for c in m.citations:
                    st.markdown(
                        f'<div class="nw-source"><b>{html.escape(c["doc_name"])} · page {c["page_number"]}</b><br>'
                        f'{html.escape(c["text"])}</div>',
                        unsafe_allow_html=True,
                    )


def render_upload_reports(svc: services.Services, conv_id: str) -> None:
    reports = st.session_state.get("reports", {})
    linked = {d.doc_id for d in svc.db.conversation_documents(conv_id)}
    for doc_id, r in reports.items():
        if doc_id not in linked:
            continue
        ocr = f" · OCR used on page(s) {', '.join(map(str, r['ocr_pages']))}" if r["ocr_pages"] else ""
        with st.expander(f"📄 {r['name']} — {r['pages']} page(s) processed{ocr}"):
            for w in r["warnings"]:
                st.warning(w)
            st.caption(f"{r['chunks']} passages indexed. This is the text NoteWise extracted, page by page:")
            for number, method, text in r["page_text"]:
                tag = " (OCR)" if method == "ocr" else ""
                st.markdown(f"**Page {number}{tag}**")
                st.text(text[:1500] + ("…" if len(text) > 1500 else ""))


def render_hero(compact: bool = False) -> None:
    """Banner with the NoteWise illustration. Full size on an empty chat, slim once chatting."""
    features = "" if compact else f"""
        <div class="nw-features">
          <div class="nw-feature"><div class="nw-ficon purple">{ICON_NOTES}</div><b>Add your notes</b>
            <span>Upload a PDF (even scanned), a .txt file, or paste text.</span></div>
          <div class="nw-feature"><div class="nw-ficon blue">{ICON_CHAT}</div><b>Ask questions</b>
            <span>Get clear explanations with page references.</span></div>
          <div class="nw-feature"><div class="nw-ficon orange">{ICON_CHART}</div><b>Practice &amp; track</b>
            <span>Take quizzes and see your progress.</span></div>
        </div>"""
    sub = ("Ask anything about your notes. Every answer shows the page it came from." if compact else
           "Upload your notes, ask questions, get clear answers, practice with quizzes, "
           "and track your progress — all in one place.")
    # Wide screens: the full scene on the right. Narrow screens and the chat banner:
    # a round portrait, so the picture is never squeezed or cut off.
    face = '<img class="nw-hero-face" src="app/static/hero_face.jpg" alt="">'
    art = face if compact else '<img class="nw-hero-art" src="app/static/hero.jpg" alt="">' + face
    st.markdown(
        f"""<div class="nw-hero2{' compact' if compact else ''}">{art}
        <div class="nw-hero-text">
          <div class="nw-hello">Hello! 👋</div>
          <div class="nw-h1">Study from your <em class="nw-accent">own notes</em></div>
          <p>{sub}</p>
        </div>{features}</div>""",
        unsafe_allow_html=True,
    )


def render_tutor(svc: services.Services, conv_id: str, ready: bool, health_ok: bool) -> None:
    render_upload_reports(svc, conv_id)
    messages = svc.manager.messages(conv_id)
    render_hero(compact=bool(messages))
    if not messages and ready:
        st.markdown(
            '<div class="nw-card nw-ready"><b>Your notes are ready.</b> Try asking '
            '<i>“What topics are in my notes?”</i> or ask about any idea in them.</div>',
            unsafe_allow_html=True,
        )
    for m in messages:
        render_message(m)

    last = messages[-1] if messages else None
    if ready and health_ok and last and last.role == "assistant" and last.status in ("answered", "partial"):
        with st.container(key="followups"):
            cols = st.columns([1.1, 1.3, 1.3, 1.5])
            for col, text in zip(cols[1:], QUICK_FOLLOW_UPS):
                if col.button(text, key=f"quick_{text}", use_container_width=True):
                    st.session_state.pending_prompt = text
                    st.rerun()

    placeholder = "Ask a question about your notes…" if ready else "Add your notes first (sidebar)"
    typed = st.chat_input(placeholder, disabled=not (ready and health_ok))
    prompt = typed or st.session_state.pop("pending_prompt", None)
    if prompt:
        render_message(SimpleNamespace(role="user", content=prompt))
        with st.chat_message("assistant", avatar=TUTOR_AVATAR):
            with st.spinner("Searching your notes and asking Gemma 4…"):
                try:
                    svc.manager.ask(conv_id, prompt, svc.gemma,
                                    user_message_id=uuid.uuid4().hex, assistant_message_id=uuid.uuid4().hex)
                except NoteWiseError as exc:
                    svc.db.add_message(conv_id, "assistant", str(exc), status="error")
        st.rerun()


# ---------- quiz ----------
QUIZ_TIPS = """<div class="nw-tips"><div class="nw-tips-title">💡 Tips</div><ul>
<li>Every question is written from your uploaded notes.</li>
<li>Each question shows the page it came from.</li>
<li>Stuck? Open the hint before you answer.</li>
<li>After you answer, you'll see why, and which page to revise.</li></ul></div>"""


def quiz_state(conv_id: str) -> dict | None:
    quiz = st.session_state.get("quiz")
    return quiz if quiz and quiz.get("conv_id") == conv_id else None


def render_quiz_banner(sub: str) -> None:
    face = '<img class="nw-hero-face" src="app/static/quiz_face.jpg" alt="">'
    art = '<img class="nw-hero-art" src="app/static/quiz_hero.jpg" alt="">' + face
    st.markdown(
        f"""<div class="nw-hero2 quiz">{art}
        <div class="nw-hero-text">
          <div class="nw-hello">✨ Quiz Time! 🎯</div>
          <div class="nw-h1">Practice from your <em class="nw-accent">own notes</em></div>
          <p>{sub}</p>
        </div></div>""",
        unsafe_allow_html=True,
    )


def score_ring(value: int, total: int, label: str, size: int = 92) -> str:
    """Circular progress ring drawn with SVG."""
    r = size / 2 - 8
    circumference = 2 * 3.14159 * r
    filled = circumference * (value / total if total else 0)
    c = size / 2
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="nw-ring">'
        f'<defs><linearGradient id="nwring" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#7B5CFF"/>'
        f'<stop offset="1" stop-color="#A35CFF"/></linearGradient></defs>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="#EEE9FF" stroke-width="8"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="url(#nwring)" stroke-width="8" stroke-linecap="round" '
        f'stroke-dasharray="{filled:.1f} {circumference:.1f}" transform="rotate(-90 {c} {c})"/>'
        f'<text x="{c}" y="{c + 6}" text-anchor="middle" font-size="{size // 4.6:.0f}" font-weight="800" '
        f'fill="#4B2FD6" font-family="Plus Jakarta Sans, sans-serif">{label}</text></svg>'
    )


def progress_card(questions: list[QuizQuestion], session: QuizSession, current: int) -> str:
    dots = []
    for k, q in enumerate(questions):
        f = session.answers.get(q.id)
        cls = "" if f is None else ("right" if f.is_correct else "wrong")
        mark = str(k + 1) if f is None else ("✓" if f.is_correct else "✗")
        here = " current" if k == current else ""
        dots.append(f'<span class="nw-dot {cls}{here}" title="Question {k + 1}">{mark}</span>')
    return (
        '<div class="nw-side-card"><div class="nw-side-title">Quiz Progress</div>'
        f'<div class="nw-ring-row">{score_ring(session.answered, session.total, str(session.answered))}'
        f'<div><div class="nw-ring-big">{session.answered} / {session.total}</div>'
        f'<div class="nw-ring-sub">Questions answered · Score {session.score}</div></div></div>'
        f'<div class="nw-dots">{"".join(dots)}</div></div>'
    )


def build_quiz(svc: services.Services, conv_id: str, doc_id: str, chunks, num: int,
               style: str = "", origin: str = "") -> dict | None:
    """Generate questions with a progress bar and return a quiz state dict (or None after showing an error)."""
    bar = st.progress(0.0, text="Writing and checking questions with Gemma 4…")

    def progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Question {done} of {total} ready…")

    try:
        result = generate_quiz(chunks, svc.gemma, num_questions=num, on_progress=progress, style=style)
    except NoteWiseError as exc:
        st.error(str(exc))
        return None
    if not result.questions:
        st.error(result.message)
        return None
    return {
        "conv_id": conv_id,
        "doc_id": doc_id,
        "questions": [q.to_dict() for q in result.questions],
        "answers": {},
        "current": 0,
        "attempt_id": uuid.uuid4().hex,
        "message": result.message,
        "origin": origin,
    }


def render_quiz(svc: services.Services, conv_id: str, doc_ids: list[str], health_ok: bool) -> None:
    quiz = quiz_state(conv_id)
    render_quiz_banner("Test your understanding with questions written from your uploaded notes. "
                       "Every question shows the page it came from.")
    if not doc_ids:
        st.info("Add your notes in the sidebar first. The quiz is built only from them.")
        return

    if quiz is None:
        with st.container(key="quizlayout"):
            left, right = st.columns([2.35, 1], gap="medium")
            with left, st.container(border=True, key="quizcard"):
                st.markdown(
                    f"""<div class="nw-q">Ready to test yourself?</div>
                    <div class="nw-steps2">
                      <div><span>1</span>NoteWise picks passages from different pages of your notes.</div>
                      <div><span>2</span>Gemma 4 writes {config.QUIZ_QUESTIONS} multiple-choice questions.</div>
                      <div><span>3</span>Each question is double-checked: Gemma answers it again without the key,
                        and unclear questions are replaced.</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button(f"Generate {config.QUIZ_QUESTIONS}-question quiz", type="primary",
                             icon=":material/auto_awesome:", disabled=not health_ok):
                    quiz = build_quiz(svc, conv_id, doc_ids[0], svc.index.get_chunks(doc_ids), config.QUIZ_QUESTIONS)
                    if quiz:
                        st.session_state.quiz = quiz
                        st.rerun()
            with right:
                st.markdown(QUIZ_TIPS, unsafe_allow_html=True)
        return

    session = render_quiz_runner(quiz)
    if session.finished and quiz.get("show_results"):
        render_quiz_summary(svc, conv_id, quiz, session)


def render_quiz_runner(quiz: dict) -> QuizSession:
    """Shows the current question (options, hint, feedback). Used by the Quiz tab, Study Mode
    and follow-up quizzes. Returns the scored session; the caller shows the results."""
    questions = [QuizQuestion.from_dict(d) for d in quiz["questions"]]
    session = QuizSession(questions)
    for qid, selected in quiz["answers"].items():
        session.submit(qid, selected)

    if session.finished and quiz.get("show_results"):
        return session
    if quiz.get("origin"):
        st.markdown(f'<div class="nw-origin">{html.escape(quiz["origin"])}</div>', unsafe_allow_html=True)

    i = quiz["current"]
    q = questions[i]
    answered = q.id in quiz["answers"]

    with st.container(key="quizlayout"):
        left, right = st.columns([2.35, 1], gap="medium")
        with right:
            st.markdown(progress_card(questions, session, i), unsafe_allow_html=True)
            st.markdown(QUIZ_TIPS, unsafe_allow_html=True)
        with left, st.container(border=True, key="quizcard"):
            pct = round(100 * (i + 1) / session.total)
            st.markdown(
                f'<div class="nw-qhead"><span>Question {i + 1} of {session.total}</span>'
                f'<span class="nw-chip">Source: page {q.page_number}</span></div>'
                f'<div class="nw-qbar"><div style="width:{pct}%"></div></div>'
                f'<div class="nw-q">{html.escape(q.question)}</div>',
                unsafe_allow_html=True,
            )
            choice = None
            if not answered:
                with st.container(key="quizoptions"):
                    choice = st.radio("Choose an answer", q.options, key=f"choice_{q.id}",
                                      label_visibility="collapsed", index=None)
                with st.expander("Need a hint?", icon=":material/lightbulb:"):
                    st.write(q.hint)
            else:
                picked = quiz["answers"][q.id]
                rows = []
                for k, option in enumerate(q.options):
                    cls, tag = "", ""
                    if k == q.correct_index:
                        cls, tag = "right", "Correct answer"
                    elif k == picked:
                        cls, tag = "wrong", "Your answer"
                    if k == picked and k == q.correct_index:
                        tag = "Your answer ✓"
                    tag_html = f'<span class="nw-opt-tag">{tag}</span>' if tag else ""
                    rows.append(f'<div class="nw-opt {cls}"><span class="nw-letter">{LETTERS[k]}</span>'
                                f'<span class="nw-opt-text">{html.escape(option)}</span>{tag_html}</div>')
                st.markdown("".join(rows), unsafe_allow_html=True)

            with st.container(key="quiznav"):
                back, middle, forward = st.columns([1, 1, 1.3])
                if back.button("Previous", icon=":material/arrow_back:", disabled=i == 0, use_container_width=True):
                    quiz["current"] = i - 1
                    st.rerun()
                if not answered:
                    if forward.button("Check answer", type="primary", disabled=choice is None, use_container_width=True):
                        quiz["answers"][q.id] = q.options.index(choice)
                        st.rerun()
                elif i == session.total - 1 and session.finished:
                    if forward.button("See results →", type="primary", use_container_width=True):
                        quiz["show_results"] = True
                        st.rerun()
                elif forward.button("Next question →", type="primary", disabled=i == session.total - 1,
                                    use_container_width=True):
                    quiz["current"] = i + 1
                    st.rerun()

    if answered:
        f = session.answers[q.id]
        chip = f'<span class="nw-chip">Source: page {f.page_number}</span>'
        if f.is_correct:
            st.markdown(
                f'<div class="nw-fb right"><div class="nw-fb-icon">✓</div><div class="nw-fb-body">'
                f'<div class="nw-fb-head"><span class="nw-fb-title">Correct!</span>{chip}</div>'
                f'<p>{html.escape(f.explanation)}</p></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="nw-fb wrong"><div class="nw-fb-icon">✗</div><div class="nw-fb-body">'
                f'<div class="nw-fb-head"><span class="nw-fb-title">{html.escape(f.headline)}</span>{chip}</div>'
                f"<p><b>Why your answer isn't right:</b> {html.escape(f.why_wrong)}</p>"
                f"<p><b>Explanation:</b> {html.escape(f.explanation)}</p>"
                f"<p><b>Revise:</b> {html.escape(f.hint)}</p></div></div>",
                unsafe_allow_html=True,
            )
        with st.expander(f"Source: {f.doc_name} · page {f.page_number}", icon=":material/menu_book:"):
            st.markdown(f'<div class="nw-source">{html.escape(f.source_text)}</div>', unsafe_allow_html=True)
    return session


def render_quiz_summary(svc: services.Services, conv_id: str, quiz: dict, session: QuizSession) -> None:
    svc.db.add_quiz_attempt(                         # INSERT OR IGNORE: saved once even on reruns
        session.score, session.total, doc_id=quiz["doc_id"], conversation_id=conv_id,
        details=session.details(), attempt_id=quiz["attempt_id"],
    )
    pct = round(100 * session.score / session.total)
    revise = session.pages_to_revise()
    chips = "".join(f'<span class="nw-chip">page {p}</span>' for p in revise)
    revise_html = (f'<div class="nw-revise">Pages to revise: {chips}</div>' if revise
                   else '<div class="nw-revise">No pages to revise — great work! 🎉</div>')
    message = ("Excellent! You know these notes well." if pct >= 80 else
               "Good effort! Review the pages below and try again." if pct >= 40 else
               "Keep going! Revise the pages below, then take a new quiz.")
    rows = []
    for n, row in enumerate(session.details(), start=1):
        cls = "right" if row["is_correct"] else "wrong"
        mark = "✓" if row["is_correct"] else "✗"
        rows.append(
            f'<div class="nw-result {cls}"><span class="nw-dot {cls}">{mark}</span>'
            f'<span class="nw-result-q"><b>Q{n}.</b> {html.escape(row["question"])}</span>'
            f'<span class="nw-result-meta">page {row["page_number"]} · you {row["selected"]} · '
            f'answer {row["correct"]}</span></div>'
        )
    st.markdown(
        f'<div class="nw-summary">{score_ring(session.score, session.total, f"{pct}%", size=128)}'
        f'<div><div class="nw-summary-score">{session.score} / {session.total} correct</div>'
        f'<div class="nw-summary-msg">{message}</div>{revise_html}</div></div>'
        f'<div class="nw-results">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    if a.button("Review answers", icon=":material/visibility:", use_container_width=True):
        quiz["show_results"] = False
        quiz["current"] = 0
        st.rerun()
    if b.button("New quiz", type="primary", icon=":material/refresh:", use_container_width=True):
        reset_quiz()
        st.rerun()


# ---------- study mode ----------
def study_state(conv_id: str) -> dict:
    """Setup choices and the running session for this chat."""
    state = st.session_state.get("study")
    if not state or state.get("conv_id") != conv_id:
        state = {"conv_id": conv_id, "stage": "setup"}
        st.session_state.study = state
    return state


def render_study_banner() -> None:
    hero = "study_hero.jpg" if (ROOT / "static" / "study_hero.jpg").exists() else "hero.jpg"
    face = "study_face.jpg" if (ROOT / "static" / "study_face.jpg").exists() else "hero_face.jpg"
    st.markdown(
        f"""<div class="nw-hero2 quiz"><img class="nw-hero-art" src="app/static/{hero}" alt="">
        <img class="nw-hero-face" src="app/static/{face}" alt="">
        <div class="nw-hero-text">
          <div class="nw-hello">💜 Your feelings matter</div>
          <div class="nw-h1">Mood-Aware <em class="nw-accent">Study Mode</em></div>
          <p>Tell NoteWise how you're feeling and what you want to do. It builds a study session
          from your own notes that suits your energy today.</p>
        </div></div>""",
        unsafe_allow_html=True,
    )


def step_title(n: int, title: str, sub: str) -> str:
    return (f'<div class="nw-step-head"><span class="nw-step-no">{n}</span><div><div class="nw-step-title">{title}</div>'
            f'<div class="nw-step-sub">{sub}</div></div></div>')


def weak_pages_for(svc: services.Services, conv_id: str) -> list[tuple[str, int]]:
    memo = mistake_memory(svc.db.list_quiz_attempts(limit=500, conversation_id=conv_id))
    return [(m.doc_name, m.page_number) for m in memo if m.status != "Good"]


def render_study(svc: services.Services, conv_id: str, doc_ids: list[str], health_ok: bool) -> None:
    render_study_banner()
    if not doc_ids:
        st.info("Add your notes in the sidebar first. Study sessions are built only from them.")
        return
    state = study_state(conv_id)
    if state["stage"] == "setup":
        render_study_setup(svc, conv_id, doc_ids, health_ok, state)
    else:
        render_study_session(svc, conv_id, state)


def render_study_setup(svc, conv_id, doc_ids, health_ok, state) -> None:
    chunks = svc.index.get_chunks(doc_ids)
    options = page_options(chunks)

    with st.container(border=True, key="studymood"):
        st.markdown(step_title(1, "How are you feeling today?",
                               "This sets the tone, depth and difficulty of your session."), unsafe_allow_html=True)
        mood = st.radio(
            "Mood", list(MOODS), key="study_mood", horizontal=True, label_visibility="collapsed",
            index=list(MOODS).index(state.get("mood", "fresh")),
            format_func=lambda k: f"{MOODS[k].emoji}  \n**{MOODS[k].label}**  \n{MOODS[k].sub}",
        )

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        with st.container(border=True, key="studygoal"):
            st.markdown(step_title(2, "Select your study goal", "What do you want to do now?"), unsafe_allow_html=True)
            goal = st.radio("Goal", list(GOALS), key="study_goal", format_func=GOALS.get,
                            index=list(GOALS).index(state.get("goal", "learn")), label_visibility="collapsed")
            weak = weak_pages_for(svc, conv_id) if goal == "weak" else []
            page_key = None
            if goal == "weak":
                if weak:
                    st.caption("Using your weakest pages from quiz history: "
                               + ", ".join(f"page {p}" for _, p in weak[:3]))
                else:
                    st.caption("No weak topics yet. Take a quiz first, or pick another goal.")
            else:
                keys = [o.key for o in options]
                default = state.get("page_key")
                page_key = st.selectbox(
                    "Which part of your notes?", keys, key="study_page",
                    index=keys.index(default) if default in keys else 0,
                    format_func=lambda k: next(o.label for o in options if o.key == k),
                )
        with st.container(border=True, key="studylen"):
            st.markdown(step_title(3, "Choose session length",
                                   "Sets how many pages and practice questions you get."), unsafe_allow_html=True)
            minutes = st.radio("Length", list(LENGTHS), key="study_len", horizontal=True,
                               index=list(LENGTHS).index(state.get("minutes", 15)), label_visibility="collapsed",
                               format_func=lambda m: f"**{LENGTHS[m][0]}**  \n{LENGTHS[m][1]}")

    plan = make_plan(mood, goal, minutes)
    pages = choose_pages(plan, options, page_key, weak)
    with right, st.container(border=True, key="studyplan"):
        st.markdown(step_title(4, "Your Study Plan", "Based on your mood, goal and time:"), unsafe_allow_html=True)
        steps = "".join(f'<div class="nw-plan-step"><span>✓</span>{html.escape(t)}</div>' for t in plan.steps)
        where = ", ".join(f"page {p.page_number}" for p in pages) or "—"
        st.markdown(
            f'<div class="nw-plan"><div class="nw-plan-title">Recommended for you</div>{steps}</div>'
            f'<div class="nw-plan-meta"><div><b>{plan.mood.emoji} Mood</b>{plan.mood.label}</div>'
            f'<div><b>🎯 Goal</b>{GOALS[plan.goal]}</div><div><b>⏱ Time</b>about {minutes} min</div>'
            f'<div><b>📄 Notes</b>{html.escape(where)}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Start session", type="primary", icon=":material/play_arrow:", use_container_width=True,
                     disabled=not (health_ok and pages)):
            state.update(stage="learn", mood=mood, goal=goal, minutes=minutes, page_key=page_key,
                         pages=[(p.doc_id, p.doc_name, p.page_number) for p in pages],
                         session_id=uuid.uuid4().hex, explanation=None, quiz=None)
            st.rerun()

    adapt = "".join(
        f'<div class="nw-adapt {k}"><div class="nw-adapt-emoji">{MOODS[k].emoji}</div><div>'
        f'<b>If you\'re {MOODS[k].label.lower()}</b><ul>'
        + "".join(f"<li>{html.escape(a)}</li>" for a in MOODS[k].adapts) + "</ul></div></div>"
        for k in ("fresh", "tired", "stressed", "difficult")
    )
    st.markdown(f'<div class="nw-panel"><div class="nw-panel-title">🧠 How NoteWise adapts to your mood</div>'
                f'<div class="nw-adapts">{adapt}</div></div>', unsafe_allow_html=True)


def render_study_session(svc, conv_id, state) -> None:
    plan = make_plan(state["mood"], state["goal"], state["minutes"])
    pages = [PageOption(d, n, p, "") for d, n, p in state["pages"]]
    chunks = chunks_for(pages, svc.index.get_chunks(sorted({p.doc_id for p in pages})))
    stage = state["stage"]
    if stage == "learn" and plan.explain is None:
        stage = state["stage"] = "practice"

    labels = (["Learn"] if plan.explain else []) + ["Practice", "Summary"]
    current = {"learn": "Learn", "practice": "Practice", "done": "Summary"}[stage]
    track = "".join(
        f'<span class="nw-track-step{" on" if lbl == current else ""}{" done" if labels.index(lbl) < labels.index(current) else ""}">'
        f'{i}. {lbl}</span>' for i, lbl in enumerate(labels, start=1)
    )
    where = ", ".join(f"page {p.page_number}" for p in pages)
    st.markdown(
        f'<div class="nw-session-head"><div><b>{plan.mood.emoji} {html.escape(GOALS[plan.goal])}</b>'
        f'<span> · {plan.mood.label} · about {plan.minutes} min · {html.escape(pages[0].doc_name)}, {where}</span></div>'
        f'<div class="nw-track">{track}</div></div>',
        unsafe_allow_html=True,
    )

    if stage == "learn":
        if state.get("explanation") is None:
            with st.spinner("Gemma 4 is preparing your explanation from your notes…"):
                try:
                    ex = explain(chunks, plan.mood, plan.explain, svc.gemma)
                    state["explanation"] = {"status": ex.status, "text": ex.text,
                                            "citations": [c.__dict__ for c in ex.citations]}
                except NoteWiseError as exc:
                    st.error(str(exc))
                    return
        ex = state["explanation"]
        render_message(SimpleNamespace(role="assistant", content=ex["text"], status=ex["status"],
                                       citations=ex["citations"]))
        a, b, c = st.columns([1, 1, 1.4])
        if a.button("End session", icon=":material/close:", use_container_width=True):
            state["stage"] = "setup"
            st.rerun()
        if b.button("Ask about this", icon=":material/chat:", use_container_width=True):
            st.session_state.view = "Tutor"
            st.rerun()
        if c.button("Continue to practice →", type="primary", use_container_width=True):
            state["stage"] = "practice"
            st.rerun()
        return

    if stage == "practice":
        if state.get("quiz") is None:
            quiz = build_quiz(svc, conv_id, pages[0].doc_id, chunks, plan.questions, style=plan.mood.quiz_style,
                              origin=f"Practice from {pages[0].doc_name}, {where}")
            if not quiz:
                if st.button("Back to setup"):
                    state["stage"] = "setup"
                    st.rerun()
                return
            state["quiz"] = quiz
            st.rerun()
        session = render_quiz_runner(state["quiz"])
        if session.finished and state["quiz"].get("show_results"):
            state["stage"] = "done"
            st.rerun()
        return

    # summary
    quiz = state["quiz"]
    questions = [QuizQuestion.from_dict(d) for d in quiz["questions"]]
    session = QuizSession(questions)
    for qid, sel in quiz["answers"].items():
        session.submit(qid, sel)
    svc.db.add_quiz_attempt(session.score, session.total, doc_id=quiz["doc_id"], conversation_id=conv_id,
                            details=session.details(), attempt_id=quiz["attempt_id"])
    svc.db.add_study_session(plan.mood.key, plan.goal, plan.minutes,
                             [{"doc_name": n, "page_number": p} for _, n, p in state["pages"]],
                             session.score, session.total, conversation_id=conv_id, session_id=state["session_id"])
    pct = round(100 * session.score / session.total)
    revise = session.pages_to_revise()
    cheer = {"tired": "Well done for studying even when tired. Rest well! 🌙",
             "stressed": "You did it, one step at a time. Be proud of that. 💜"}.get(
        plan.mood.key, "Great session! Keep the streak going. 🔥")
    nxt = (f"Next: revise page{'s' if len(revise) > 1 else ''} {', '.join(map(str, revise))}, "
           "then try a Practise-my-weak-topics session." if revise else
           "Next: try a new page, or a 30-minute session for deeper practice.")
    st.markdown(
        f'<div class="nw-summary">{score_ring(session.score, session.total, f"{pct}%", size=128)}'
        f'<div><div class="nw-summary-score">Session complete · {session.score} / {session.total} correct</div>'
        f'<div class="nw-summary-msg">{cheer}</div><div class="nw-revise">{html.escape(nxt)}</div></div></div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    if a.button("Open My Learning DNA", icon=":material/insights:", use_container_width=True):
        st.session_state.view = "My Learning DNA"
        st.rerun()
    if b.button("New session", type="primary", icon=":material/refresh:", use_container_width=True):
        state["stage"] = "setup"
        st.rerun()


# ---------- progress ----------
BAR_COLORS = ["#2FBF8A", "#8B5CF6", "#4C8DF6", "#F5B83D", "#F0609E", "#A78BFA"]
ICON_BOOK = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 6.5C10 5 7 4.6 4 5v13c3-.4 6 0 8 1.5 2-1.5 5-1.9 8-1.5V5c-3-.4-6 0-8 1.5Z" fill="#fff"/><path d="M12 6.5v13" stroke="#12805C" stroke-width="1.6"/></svg>"""
ICON_STAR = """<svg width="24" height="24" viewBox="0 0 24 24"><path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9 6.8 19.6l1-5.8L3.5 9.7l5.9-.9z" fill="#fff"/></svg>"""
ICON_ASK = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M4 6.5A3.5 3.5 0 0 1 7.5 3h9A3.5 3.5 0 0 1 20 6.5v6a3.5 3.5 0 0 1-3.5 3.5H11l-4.5 4v-4A3.5 3.5 0 0 1 4 12.5z" fill="#fff"/><path d="M10 8.2a2 2 0 1 1 2.6 1.9c-.4.2-.6.5-.6.9v.5" stroke="#6D4AFF" stroke-width="1.6" stroke-linecap="round"/><circle cx="12" cy="13.2" r=".9" fill="#6D4AFF"/></svg>"""


def line_chart(history: list[int]) -> str:
    """Quiz scores over time as an SVG line chart (last 10 quizzes)."""
    data = history[-10:]
    first = len(history) - len(data) + 1
    w, h, left, right, top, bottom = 600, 270, 52, 30, 30, 40
    pw, ph = w - left - right, h - top - bottom
    step = pw / (len(data) - 1) if len(data) > 1 else 0
    pad = 24
    step = (pw - 2 * pad) / (len(data) - 1) if len(data) > 1 else 0
    xs = [left + pad + (i * step if len(data) > 1 else (pw - 2 * pad) / 2) for i in range(len(data))]
    ys = [top + ph * (1 - v / 100) for v in data]
    grid = "".join(
        f'<line x1="{left}" x2="{w - right}" y1="{top + ph * (1 - g / 100):.1f}" y2="{top + ph * (1 - g / 100):.1f}" '
        f'stroke="#EEE9FF"/><text x="{left - 8}" y="{top + ph * (1 - g / 100) + 4:.1f}" text-anchor="end" '
        f'class="nw-axis">{g}%</text>'
        for g in (0, 25, 50, 75, 100)
    )
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    area = (f'<path d="{path} L{xs[-1]:.1f},{top + ph} L{xs[0]:.1f},{top + ph} Z" fill="url(#nwarea)"/>'
            if len(data) > 1 else "")
    points = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#6D4AFF" stroke="#fff" stroke-width="2"/>'
        f'<text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" class="nw-pt">{v}%</text>'
        f'<text x="{x:.1f}" y="{h - 10}" text-anchor="middle" class="nw-axis">Quiz {first + i}</text>'
        for i, (x, y, v) in enumerate(zip(xs, ys, data))
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" class="nw-chart" preserveAspectRatio="xMidYMid meet">'
        '<defs><linearGradient id="nwarea" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#8B5CF6" stop-opacity=".22"/>'
        '<stop offset="1" stop-color="#8B5CF6" stop-opacity="0"/></linearGradient></defs>'
        f'{grid}{area}<path d="{path}" fill="none" stroke="#6D4AFF" stroke-width="3" stroke-linejoin="round"/>'
        f"{points}</svg>"
    )


ICON_FILE = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M7 3h7l5 5v13H7z" fill="#fff"/><path d="M10 12h6M10 15.5h6" stroke="#6D4AFF" stroke-width="1.8" stroke-linecap="round"/></svg>"""
ICON_FIRE = """<svg width="24" height="24" viewBox="0 0 24 24"><path d="M12 2.8c.6 3-1.6 4.4-2.8 6.3-1 1.6-1.4 3.1-.6 4.7-1.5-.4-2.4-1.6-2.6-3-1.7 1.8-2.3 4.1-1.7 6.2C5.1 20 8.2 22 12 22s7-2.4 7-6.6c0-3.5-2.2-5.4-3.4-7.6-.5 1.3-1.3 2.2-2.4 2.6.8-2.9.2-5.6-1.2-7.6z" fill="#fff"/></svg>"""
STATUS_CLASS = {"Needs work": "bad", "Improving": "mid", "Good": "good"}


def render_dna_banner() -> None:
    hero = "dna_hero.jpg" if (ROOT / "static" / "dna_hero.jpg").exists() else "progress_hero.jpg"
    face = "dna_face.jpg" if (ROOT / "static" / "dna_face.jpg").exists() else "progress_face.jpg"
    st.markdown(
        f"""<div class="nw-hero2 quiz"><img class="nw-hero-art" src="app/static/{hero}" alt="">
        <img class="nw-hero-face" src="app/static/{face}" alt="">
        <div class="nw-hero-text">
          <div class="nw-hello">🧬 Understand your learning, grow faster</div>
          <div class="nw-h1">My Learning <em class="nw-accent">DNA</em></div>
          <p>A personal profile built from your notes, quizzes and study sessions: what you know,
          the mistakes you repeat, and what to do next.</p>
        </div></div>""",
        unsafe_allow_html=True,
    )


def find_page_chunks(svc: services.Services, doc_name: str, page: int):
    """Chunks of one page, found by document name (works across chats)."""
    for rec in svc.db.list_documents():
        if rec.name == doc_name and svc.index.has_document(rec.doc_id):
            found = [c for c in svc.index.get_chunks([rec.doc_id]) if c.page_number == page]
            if found:
                return rec.doc_id, found
    return None, []


def start_followup_quiz(svc, conv_id: str, doc_name: str, page: int, topic: str) -> None:
    doc_id, chunks = find_page_chunks(svc, doc_name, page)
    if not chunks:
        st.error(f"'{doc_name}' isn't available any more. Re-upload it to practise this page.")
        return
    quiz = build_quiz(svc, conv_id, doc_id, chunks, 3,
                      style="Ask about an idea a student could easily misunderstand.",
                      origin=f"Follow-up quiz · {topic} · {doc_name}, page {page}")
    if quiz:
        st.session_state.quiz = quiz
        st.session_state.view = "Quiz"
        st.rerun()


def start_study_on(svc, conv_id: str, doc_name: str, page: int) -> None:
    doc_id, chunks = find_page_chunks(svc, doc_name, page)
    if not chunks:
        st.error(f"'{doc_name}' isn't available any more. Re-upload it to study this page.")
        return
    st.session_state.study = {"conv_id": conv_id, "stage": "setup", "goal": "learn", "mood": "difficult",
                              "minutes": 15, "page_key": f"{doc_id}:{page}"}
    for k in ("study_mood", "study_goal", "study_len", "study_page"):
        st.session_state.pop(k, None)
    st.session_state.view = "Study Mode"
    st.rerun()


def fmt_date(ts: str) -> str:
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts).astimezone().strftime("%d %b %Y")
    except ValueError:
        return ts[:10]


def render_dna(svc: services.Services, conv_id: str, health_ok: bool) -> None:
    render_dna_banner()
    with st.container(key="scope"):
        scope = st.segmented_control(
            "Show for", ["This chat", "All chats"], default="This chat",
            key="progress_scope", label_visibility="collapsed",
        ) or "This chat"
    chat_only = scope == "This chat"
    cid = conv_id if chat_only else None
    attempts = svc.db.list_quiz_attempts(limit=500, conversation_id=cid)
    s = summarise(attempts)
    memo = mistake_memory(attempts)
    streak = study_streak(svc.db.activity_dates(cid))
    habits = study_habits(svc.db.user_messages(cid))
    sessions = svc.db.list_study_sessions(cid)
    notes = (len(svc.db.conversation_documents(conv_id)) if chat_only else len(svc.db.list_documents()))

    if s.change is None:
        delta = ""
    else:
        cls, arrow = ("up", "↑") if s.change >= 0 else ("down", "↓")
        delta = f'<span class="nw-delta {cls}">{arrow} {abs(s.change)}% from last quiz</span>'
    acc = f"{s.accuracy}%" if s.answered else "—"
    st.markdown(
        '<div class="nw-stats">'
        f'<div class="nw-stat"><div class="nw-stat-icon purple">{ICON_FILE}</div><div>'
        f'<div class="nw-stat-num">{notes}</div><div class="nw-stat-label">Notes uploaded</div></div></div>'
        f'<div class="nw-stat"><div class="nw-stat-icon green">{ICON_BOOK}</div><div>'
        f'<div class="nw-stat-num">{s.answered}</div><div class="nw-stat-label">Questions attempted</div>'
        f'<div class="nw-stat-sub">{s.quizzes} quiz{"zes" if s.quizzes != 1 else ""} · '
        f'{len(sessions)} study session{"s" if len(sessions) != 1 else ""}</div></div></div>'
        f'<div class="nw-stat big">{score_ring(s.correct, s.answered or 1, acc, size=96)}<div>'
        f'<div class="nw-stat-label">Overall accuracy</div><div class="nw-stat-sub">{s.correct} of {s.answered} correct</div>'
        f'{delta}</div></div>'
        f'<div class="nw-stat"><div class="nw-stat-icon fire">{ICON_FIRE}</div><div>'
        f'<div class="nw-stat-num">{streak}</div><div class="nw-stat-label">Day study streak</div>'
        f'<div class="nw-stat-sub">{"Keep it going!" if streak else "Study today to start one"}</div></div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if not s.quizzes and not habits["questions"]:
        st.markdown(
            '<div class="nw-panel nw-empty"><div class="nw-panel-title">Your Learning DNA is empty for now</div>'
            "<p>Ask questions in <b>Tutor</b>, take a <b>Quiz</b> or start a <b>Study Mode</b> session. "
            "NoteWise builds your profile from what you actually do."
            + (" Switch to <b>All chats</b> to see your other chats." if chat_only else "") + "</p></div>",
            unsafe_allow_html=True,
        )
        return

    # ---- topic mastery + next best step ----
    bars = "".join(
        f'<div class="nw-topic"><span class="nw-topic-name" title="{html.escape(p.doc_name)}, page {p.page_number}">'
        f'{html.escape(p.topic)} <small>p. {p.page_number}</small></span>'
        f'<span class="nw-bar"><span style="width:{max(p.accuracy, 3)}%;background:{BAR_COLORS[k % len(BAR_COLORS)]}">'
        f'</span></span><span class="nw-topic-pct">{p.accuracy}%</span></div>'
        for k, p in enumerate(sorted(s.pages[:7], key=lambda p: (p.doc_name, p.page_number)))
    ) or '<p class="nw-muted">Take a quiz to see how well you know each topic.</p>'
    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        st.markdown(f'<div class="nw-panel tall"><div class="nw-panel-title">📊 Topic Mastery</div>'
                    f'<div class="nw-panel-sub">From your quiz answers, topic by topic.</div>{bars}</div>',
                    unsafe_allow_html=True)
    with right, st.container(border=True, key="nextstep"):
        st.markdown('<div class="nw-panel-title">🎯 Your Next Best Step</div>', unsafe_allow_html=True)
        target = next((m for m in memo if m.status != "Good"), None)
        if target:
            st.markdown(
                f'<div class="nw-next"><div class="nw-next-title">Revise: {html.escape(target.topic)} '
                f'(page {target.page_number})</div><div class="nw-next-sub">You answered {target.accuracy}% '
                f'of questions on this page correctly ({target.attempts - target.mistakes} of {target.attempts}).'
                "</div></div>",
                unsafe_allow_html=True,
            )
            a, b = st.columns(2)
            if a.button("Explain this topic", key="next_explain", use_container_width=True, disabled=not health_ok):
                start_study_on(svc, conv_id, target.doc_name, target.page_number)
            if b.button("Take a quiz", key="next_quiz", type="primary", use_container_width=True, disabled=not health_ok):
                start_followup_quiz(svc, conv_id, target.doc_name, target.page_number, target.topic)
        else:
            st.markdown('<div class="nw-next"><div class="nw-next-title">You\'re on track 🎉</div>'
                        '<div class="nw-next-sub">No weak pages right now. Try a quiz on a new part of your notes.'
                        "</div></div>", unsafe_allow_html=True)

        rows = [(HABITS[k][0], habits[k]) for k in ("simpler", "detail", "example")]
        habit_html = "".join(f'<div class="nw-habit"><span>{html.escape(label)}</span><b>{n}×</b></div>'
                             for label, n in rows)
        moods = {}
        for sess in sessions:
            moods[sess["mood"]] = moods.get(sess["mood"], 0) + 1
        mood_html = (" · ".join(f'{MOODS[m].emoji} {MOODS[m].label} {n}×' for m, n in moods.items() if m in MOODS)
                     or "No Study Mode sessions yet")
        st.markdown(
            f'<div class="nw-habits"><div class="nw-habits-title">How you study</div>'
            f'<div class="nw-habit"><span>Questions asked to your tutor</span><b>{habits["questions"]}</b></div>'
            f'{habit_html}<div class="nw-habit-moods">{mood_html}</div></div>',
            unsafe_allow_html=True,
        )

    # ---- mistake memory ----
    st.markdown('<div class="nw-section-title">❗ Mistake Memory</div>'
                '<div class="nw-panel-sub">Pages where you got questions wrong, worst first. Review a mistake to '
                "see what you answered, the passage from your notes, and practise it again.</div>",
                unsafe_allow_html=True)
    if not memo:
        st.markdown('<div class="nw-panel"><p class="nw-muted">No mistakes recorded yet. 🎉 '
                    "Wrong quiz answers will be remembered here.</p></div>", unsafe_allow_html=True)
    else:
        with st.container(key="mistakes"):
            head = st.columns([3, 1, 1.5, 1.4, 1.1])
            for col, label in zip(head, ["Topic", "Mistakes", "Last attempt", "Status", ""]):
                col.markdown(f'<div class="nw-th">{label}</div>', unsafe_allow_html=True)
            for k, m in enumerate(memo[:8]):
                c = st.columns([3, 1, 1.5, 1.4, 1.1], vertical_alignment="center")
                c[0].markdown(f'<div class="nw-td-topic"><b>{html.escape(m.topic)}</b>'
                              f'<small>{html.escape(m.doc_name)} · page {m.page_number}</small></div>',
                              unsafe_allow_html=True)
                c[1].markdown(f'<div class="nw-td-num">{m.mistakes}</div>', unsafe_allow_html=True)
                c[2].markdown(f'<div class="nw-td">{fmt_date(m.last_attempt)}</div>', unsafe_allow_html=True)
                c[3].markdown(f'<span class="nw-status {STATUS_CLASS[m.status]}">{m.status}</span>',
                              unsafe_allow_html=True)
                if c[4].button("Review →", key=f"review_{k}", use_container_width=True):
                    st.session_state.review = (m.doc_name, m.page_number)
                    st.rerun()

        selected = st.session_state.get("review")
        m = next((x for x in memo if (x.doc_name, x.page_number) == selected), None)
        if m:
            with st.container(border=True, key="review"):
                st.markdown(f'<div class="nw-panel-title">📖 Review mistake: {html.escape(m.topic)}'
                            f' <span class="nw-chip">page {m.page_number}</span></div>', unsafe_allow_html=True)
                ex_html = "".join(
                    f'<div class="nw-mistake"><div class="nw-mistake-q">{html.escape(e["question"])}</div>'
                    f'<div class="nw-mistake-row bad">✗ You answered: {html.escape(str(e["your_answer"]))}</div>'
                    f'<div class="nw-mistake-row good">✓ Correct: {html.escape(str(e["correct_answer"]))}</div>'
                    + (f'<div class="nw-mistake-ex">{html.escape(e["explanation"])}</div>' if e["explanation"] else "")
                    + "</div>"
                    for e in m.examples[:3]
                )
                passage = (f'<div class="nw-panel-sub"><b>Relevant passage from your notes</b> · '
                           f'{html.escape(m.doc_name)}, page {m.page_number}</div>'
                           f'<div class="nw-source">{html.escape(m.source_text)}</div>') if m.source_text else ""
                st.markdown(
                    f'<div class="nw-review-head">You got this page wrong <b>{m.mistakes} time'
                    f'{"s" if m.mistakes != 1 else ""}</b> out of {m.attempts}.</div>{ex_html}{passage}',
                    unsafe_allow_html=True,
                )
                a, b, c = st.columns([1, 1.2, 1.3])
                if a.button("Close", key="review_close", use_container_width=True):
                    st.session_state.pop("review", None)
                    st.rerun()
                if b.button("Explain this topic", key="review_explain", use_container_width=True,
                            disabled=not health_ok):
                    start_study_on(svc, conv_id, m.doc_name, m.page_number)
                if c.button("Follow-up quiz (3 questions)", key="review_quiz", type="primary",
                            use_container_width=True, disabled=not health_ok):
                    start_followup_quiz(svc, conv_id, m.doc_name, m.page_number, m.topic)

    # ---- performance + strengths ----
    if s.quizzes:
        strengths = "".join(f'<div class="nw-sw-item"><span class="nw-sw-mark good">✓</span>{html.escape(p.topic)}'
                            f' <small>· p. {p.page_number} · {p.accuracy}%</small></div>' for p in s.strengths[:4]) \
            or '<p class="nw-muted">Topics you get right 75%+ of the time appear here.</p>'
        st.markdown(
            f'<div class="nw-grid2"><div class="nw-panel"><div class="nw-panel-title">📈 Your Progress Over Time</div>'
            f'{line_chart(s.history)}</div><div class="nw-panel"><div class="nw-panel-title">💪 Your Strengths</div>'
            f'<div class="nw-sw-box good">{strengths}</div></div></div>',
            unsafe_allow_html=True,
        )


# ---------- page ----------
# ---------- Nova: personalised study mentor ----------
NOVA_AVATAR = str(ROOT / "static" / "avatar_nova.png")
NOVA_STATUS = {
    "answered": ("answered", "✓ From your notes"),
    "partial": ("partial", "Partly in your notes"),
    "unverified": ("unverified", "Check the page"),
    "not_in_notes": ("not_in_notes", "Not in your notes"),
    "no_document": ("no_document", "No notes yet"),
    "plan": ("answered", "🗓️ Built from your progress"),
    "quiz": ("answered", "📝 Quiz ready"),
    "error": ("not_in_notes", "Something went wrong"),
}
NOVA_CAPS = [("📖", "Explains from your notes"), ("🧠", "Remembers your mistakes"),
             ("💜", "Adapts to how you learn"), ("🗓️", "Plans your study"), ("🔒", "100% offline")]


def render_nova_banner() -> None:
    hero = "nova_hero.jpg" if (ROOT / "static" / "nova_hero.jpg").exists() else "study_hero.jpg"
    face = "nova_face.jpg" if (ROOT / "static" / "nova_face.jpg").exists() else "study_face.jpg"
    caps = "".join(f'<span class="nw-cap">{i} {html.escape(t)}</span>' for i, t in NOVA_CAPS)
    st.markdown(
        f"""<div class="nw-hero2 quiz nova"><img class="nw-hero-art" src="app/static/{hero}" alt="">
        <img class="nw-hero-face" src="app/static/{face}" alt="">
        <div class="nw-hero-text">
          <div class="nw-hello">✦ Meet Nova</div>
          <div class="nw-h1">Your Personalised <em class="nw-accent">Study Mentor</em></div>
          <p>Nova knows how you learn: your weak pages, your mistakes and how you like things explained.
          Everything it teaches still comes from your own notes, with page references.</p>
          <div class="nw-caps">{caps}</div>
        </div></div>""",
        unsafe_allow_html=True,
    )


def nova_profile(svc: services.Services, conv_id: str, doc_ids: list[str]) -> mentor.MentorProfile:
    chunks = svc.index.get_chunks(doc_ids) if doc_ids else []
    return mentor.build_profile(
        attempts=svc.db.list_quiz_attempts(limit=500, conversation_id=conv_id),
        activity=svc.db.activity_dates(),                    # the streak counts every chat
        user_messages=svc.db.user_messages(conv_id) + [
            m["content"] for m in svc.db.get_mentor_messages(conv_id) if m["role"] == "user"],
        sessions=svc.db.list_study_sessions(conversation_id=conv_id),
        pages=page_options(chunks),
    )


def nova_bubble(role: str, content: str, status: str | None = None, citations=None, mode: str | None = None) -> None:
    avatar = STUDENT_AVATAR if role == "user" else NOVA_AVATAR
    with st.chat_message(role, avatar=avatar):
        if role == "user":
            tag = mentor.MODES.get(mode or "", ("", "", ""))
            chip = f'<span class="nw-mode-tag">{tag[1]} {tag[0]}</span>' if tag[0] else ""
            st.markdown(f'<div class="nw-user-msg">{chip}{html.escape(content)}</div>', unsafe_allow_html=True)
            return
        st.markdown('<span class="nw-bot-msg nova"></span>', unsafe_allow_html=True)
        if status in NOVA_STATUS:
            cls, label = NOVA_STATUS[status]
            st.markdown(f'<span class="nw-badge {cls}">{label}</span>', unsafe_allow_html=True)
        st.markdown(format_answer(content) if status not in ("plan", "quiz") else content, unsafe_allow_html=True)
        if citations:
            pages = sorted({c["page_number"] for c in citations})
            with st.expander("Source: page " + ", ".join(map(str, pages)), icon=":material/menu_book:"):
                for c in citations:
                    st.markdown(
                        f'<div class="nw-source"><b>{html.escape(c["doc_name"])} · page {c["page_number"]}</b><br>'
                        f'{html.escape(c["text"])}</div>', unsafe_allow_html=True)


def nova_save_answer(svc, conv_id: str, mode: str, answer) -> None:
    from dataclasses import asdict
    svc.db.add_mentor_message(conv_id, "assistant", answer.text, mode=mode, status=answer.status,
                              citations=[asdict(c) for c in answer.citations])


def nova_plan_text(profile: mentor.MentorProfile) -> str:
    rows = "".join(f'<div class="nw-plan-row"><span>{html.escape(t)}</span><div>{rich(s)}</div></div>'
                   for t, s in mentor.study_plan(profile))
    return f'<div class="nw-nova-plan"><b>Here’s your plan for today:</b>{rows}</div>'


def nova_quiz(svc, conv_id: str, doc_ids: list[str], profile, request: str) -> str | None:
    """Make a quiz for a Nova request. Returns Nova's reply, or None if a quiz was opened."""
    target = []
    if request.strip():
        target = mentor.pages_for_query(request, doc_ids, svc.index)
        if not target:
            return mentor_not_found(request)
    elif profile.weakest:
        doc_id, chunks = find_page_chunks(svc, profile.weakest.doc_name, profile.weakest.page_number)
        if chunks:
            target = [(doc_id, profile.weakest.page_number)]
    if target:
        doc_id = target[0][0]
        pages = {p for d, p in target if d == doc_id}
        chunks = [c for c in svc.index.get_chunks([doc_id]) if c.page_number in pages]
        label = ", ".join(f"p. {p}" for p in sorted(pages))
        origin = f"Nova's quiz · {label}"
        num = 3
    else:
        doc_id = doc_ids[0]
        chunks = svc.index.get_chunks([doc_id])
        label, origin, num = "your notes", "Nova's quiz · your notes", 5
    quiz = build_quiz(svc, conv_id, doc_id, chunks, num, origin=origin,
                      style="Ask about an idea a student could easily misunderstand." if target else "")
    if not quiz:
        return "I couldn't write a quiz from those pages. Try another topic."
    svc.db.add_mentor_message(conv_id, "assistant",
                              f"I made a {len(quiz['questions'])}-question quiz on {label}. Good luck! 💜",
                              mode="quiz", status="quiz")
    st.session_state.quiz = quiz
    st.session_state.view = "Quiz"
    return None


def mentor_not_found(request: str) -> str:
    return (f"I couldn't find “{request}” in your notes, so I can't make a fair quiz on it. "
            "Try a topic from your notes.")


def nova_handle(svc, conv_id: str, doc_ids: list[str], profile, req: dict) -> None:
    """Run one request to Nova and save both sides of the conversation."""
    kind, mode, text = req.get("kind", "ask"), req.get("mode", "explain"), req.get("text", "")
    shown = req.get("label") or text
    svc.db.add_mentor_message(conv_id, "user", shown, mode=mode if kind == "ask" else kind)
    nova_bubble("user", shown, mode=mode if kind == "ask" else kind)
    with st.chat_message("assistant", avatar=NOVA_AVATAR):
        with st.spinner("Nova is reading your notes…"):
            try:
                if kind == "plan":
                    svc.db.add_mentor_message(conv_id, "assistant", nova_plan_text(profile), mode="plan", status="plan")
                elif kind == "quiz":
                    reply = nova_quiz(svc, conv_id, doc_ids, profile, text)
                    if reply:
                        svc.db.add_mentor_message(conv_id, "assistant", reply, mode="quiz", status="not_in_notes")
                elif kind == "page":
                    doc_id, page = req["page_key"].rsplit(":", 1)
                    chunks = [c for c in svc.index.get_chunks([doc_id]) if c.page_number == int(page)]
                    mood = MOODS["difficult"] if profile.habits.get("simpler", 0) >= 2 else MOODS["fresh"]
                    result = explain(chunks, mood, "teach", svc.gemma)
                    nova_save_answer(svc, conv_id, "explain", result)
                else:
                    history = [{"role": m["role"], "content": m["content"]}
                               for m in svc.db.get_mentor_messages(conv_id)[:-1]
                               if m.get("status") not in ("plan", "quiz")]
                    ans = mentor.ask(text, mode, profile, doc_ids, svc.index, svc.gemma,
                                     history=history, follow_up=req.get("follow_up", ""))
                    nova_save_answer(svc, conv_id, mode, ans)
            except NoteWiseError as exc:
                svc.db.add_mentor_message(conv_id, "assistant", str(exc), mode=mode, status="error")
    st.rerun()


def render_nova(svc: services.Services, conv_id: str, doc_ids: list[str], health_ok: bool) -> None:
    render_nova_banner()
    profile = nova_profile(svc, conv_id, doc_ids)
    ready = bool(doc_ids) and health_ok
    with st.container(key="novalayout"):
        left, right = st.columns([2.15, 1], gap="large")

    with right:
        render_nova_side(svc, conv_id, profile, bool(doc_ids))

    with left, st.container(key="novachat"):
        head, lines = mentor.greeting(profile, bool(doc_ids))
        with st.chat_message("assistant", avatar=NOVA_AVATAR):
            st.markdown('<span class="nw-bot-msg nova"></span>', unsafe_allow_html=True)
            st.markdown(f'<div class="nw-nova-hi"><b>{head}</b>'
                        + "".join(f"<div class=\"nw-hi-line\">{rich(l)}</div>" for l in lines) + "</div>",
                        unsafe_allow_html=True)
        msgs = svc.db.get_mentor_messages(conv_id)
        for m in msgs:
            nova_bubble(m["role"], m["content"], m.get("status"), m.get("citations"), m.get("mode"))

        last = msgs[-1] if msgs else None
        if ready and last and last["role"] == "assistant" and last.get("status") in ("answered", "partial"):
            with st.container(key="novafollow"):
                cols = st.columns(4)
                for col, (key, (label, ask_text, _)) in zip(cols, mentor.FOLLOW_UPS.items()):
                    if col.button(label, key=f"nova_fu_{key}", use_container_width=True):
                        st.session_state.nova_pending = {"kind": "ask", "mode": last.get("mode") or "explain",
                                                         "text": ask_text, "follow_up": key}
                        st.rerun()

        if ready:
            with st.container(key="novaquick"):
                st.markdown('<div class="nw-nova-label">Try asking</div>', unsafe_allow_html=True)
                prompts = mentor.quick_prompts(profile)
                cols = st.columns(2)
                for i, q in enumerate(prompts):
                    col = cols[i % 2]
                    if col.button(q.label, key=f"nova_q{i}", use_container_width=True):
                        st.session_state.nova_pending = {"kind": q.kind, "mode": q.mode, "text": q.text,
                                                         "label": q.label, "page_key": q.page_key}
                        st.rerun()

        with st.container(key="novamode"):
            mode = st.segmented_control(
                "Mode", options=list(mentor.MODES), default="explain", key="nova_mode",
                format_func=lambda k: f"{mentor.MODES[k][1]} {mentor.MODES[k][0]}", label_visibility="collapsed",
            ) or "explain"
        hints = {"explain": "Ask Nova to explain anything from your notes…",
                 "doubt": "What's confusing you? Nova will clear it up step by step…",
                 "quiz": "Type a topic from your notes to be quizzed on…",
                 "plan": "Ask for a plan, e.g. “plan my study for today”…"}
        placeholder = hints[mode] if ready else ("Add your notes first (sidebar)" if health_ok else "Gemma 4 is offline")
        typed = st.chat_input(placeholder, key="nova_input", disabled=not ready)
        req = st.session_state.pop("nova_pending", None)
        if typed is not None and not req and ready:
            kind = mode if mode in ("quiz", "plan") else "ask"
            label = typed.strip() or ("Quiz me on my weak areas" if kind == "quiz" else "Make my study plan")
            req = {"kind": kind, "mode": mode, "text": typed.strip(), "label": label}
        if req and ready and (req.get("text") or req["kind"] != "ask"):
            nova_handle(svc, conv_id, doc_ids, profile, req)


def render_nova_side(svc, conv_id: str, profile: mentor.MentorProfile, has_notes: bool) -> None:
    # What Nova knows (transparency: every line comes from saved data)
    facts = []
    facts.append(("🔥", "Study streak", f"{profile.streak} day{'s' if profile.streak != 1 else ''}"))
    facts.append(("🎯", "Quiz accuracy", f"{profile.accuracy}%" if profile.accuracy is not None else "No quiz yet"))
    w = profile.weakest
    facts.append(("🧩", "Weakest page", mentor.page_label(w) if w else "None yet"))
    if profile.strengths:
        facts.append(("⭐", "Strong in", ", ".join(profile.strengths)))
    habit = max((k for k in HABITS), key=lambda k: profile.habits.get(k, 0))
    if profile.habits.get(habit, 0):
        facts.append(("💬", "You often", HABITS[habit][0].lower()))
    rows = "".join(f'<div class="nw-know"><span>{i}</span><div><small>{html.escape(k)}</small>'
                   f'<b>{html.escape(v)}</b></div></div>' for i, k, v in facts)
    st.markdown(f'<div class="nw-side-card nova"><div class="nw-side-title">What Nova knows about you</div>{rows}'
                '<div class="nw-know-note">Only from your quizzes, chats and study sessions in this chat. '
                'Nothing leaves your laptop.</div></div>', unsafe_allow_html=True)

    if has_notes and w:
        c1, c2 = st.columns(2)
        if c1.button("Study it", icon="💜", key="nova_study", use_container_width=True):
            start_study_on(svc, conv_id, w.doc_name, w.page_number)
        if c2.button("Quick quiz", icon="📝", key="nova_fquiz", use_container_width=True):
            start_followup_quiz(svc, conv_id, w.doc_name, w.page_number, w.topic)

    # Tips carousel
    tips = mentor.tips(profile)
    i = st.session_state.get("nova_tip", 0) % len(tips)
    title, body = tips[i]
    dots = "".join(f'<i class="{"on" if j == i else ""}"></i>' for j in range(len(tips)))
    with st.container(key="novatips"):
        st.markdown(f'<div class="nw-tipcard"><div class="nw-side-title">💡 Nova\'s Study Tips</div>'
                    f'<b>{html.escape(title)}</b><p>{rich(body)}</p><div class="nw-dots">{dots}</div></div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("‹ Prev", key="nova_tip_prev", use_container_width=True):
            st.session_state.nova_tip = (i - 1) % len(tips)
            st.rerun()
        if c2.button("Next ›", key="nova_tip_next", use_container_width=True):
            st.session_state.nova_tip = (i + 1) % len(tips)
            st.rerun()

    if svc.db.get_mentor_messages(conv_id):
        if st.button("Clear Nova chat", icon=":material/delete_sweep:", key="nova_clear", use_container_width=True):
            svc.db.clear_mentor_messages(conv_id)
            st.rerun()


# ---------- Mind map ----------
def pick_notes(statuses, label: str, key: str, why: str) -> list[str]:
    """Which notes of this chat to use: all of them, or one file when there are several."""
    ready = [s.record for s in statuses if s.ready]
    if not ready:
        st.info(f"Add your notes in the sidebar first. {why}")
        return []
    if len(ready) == 1:
        return [ready[0].doc_id]
    options = ["All notes in this chat"] + [r.name for r in ready]
    choice = st.selectbox(label, options, key=key)
    if choice == "All notes in this chat":
        return [r.doc_id for r in ready]
    return [r.doc_id for r in ready if r.name == choice][:1]


def ensure_mind_map(svc, conv_id: str, ids: list[str]) -> mind_map.MindMap:
    key = mind_map.doc_key(ids)
    saved = svc.db.get_mind_map(conv_id, key)
    if saved:
        return mind_map.MindMap.from_dict(saved)
    mm = mind_map.generate_mind_map(svc.index.get_chunks(ids), svc.gemma)
    svc.db.save_mind_map(conv_id, key, mm.to_dict())
    return mm


# ---------- Revision Book ----------
def render_book(svc: services.Services, conv_id: str, statuses, health_ok: bool) -> None:
    st.markdown(
        """<div class="nw-hero2 quiz"><img class="nw-hero-art" src="app/static/study_hero.jpg" alt="">
        <img class="nw-hero-face" src="app/static/study_face.jpg" alt="">
        <div class="nw-hero-text">
          <div class="nw-hello">📖 Your notes, as a book</div>
          <div class="nw-h1">Revision <em class="nw-accent">Book</em></div>
          <p>One click turns your PDF into a short revision book: a chapter for each main topic,
          explained by Gemma 4 only from your notes, with key points and the page to revise.</p>
        </div></div>""",
        unsafe_allow_html=True,
    )
    ids = pick_notes(statuses, "Book from", "bk_scope", "The book is written only from them.")
    if not ids:
        return
    key = mind_map.doc_key(ids)
    saved = svc.db.get_book(conv_id, key)
    label = "Rewrite book" if saved else "Create revision book"
    if st.button(label, type="secondary" if saved else "primary", icon=":material/menu_book:",
                 disabled=not health_ok, key="bk_make"):
        try:
            with st.spinner("Finding the main topics…"):
                mm = ensure_mind_map(svc, conv_id, ids)
            bar = st.progress(0.0, text="Writing chapter 1…")

            def progress(done: int, total: int) -> None:
                bar.progress(done / total, text=f"Chapter {done} of {total} written…")

            def page_chunks(doc_name: str, page: int):
                return find_page_chunks(svc, doc_name, page)[1]

            bk = book.generate_book(mm, page_chunks, svc.gemma, on_progress=progress)
            svc.db.save_book(conv_id, key, bk.to_dict())
            st.rerun()
        except NoteWiseError as exc:
            st.error(str(exc))
    if not saved:
        st.markdown('<div class="nw-card nw-ready">Click <b>Create revision book</b>. Gemma writes one chapter per '
                    'topic, so it takes a minute or two.</div>', unsafe_allow_html=True)
        return
    bk = book.Book.from_dict(saved)
    c1, c2 = st.columns([1, 1])
    c1.download_button("Download book", book.render_download(bk), file_name="revision_book.html", mime="text/html",
                       icon=":material/download:", use_container_width=True, key="bk_dl")
    c2.markdown('<div class="nw-bk-tip">Open the file and press <b>Cmd + P</b> → <b>Save as PDF</b>.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<style>{book.BOOK_CSS}</style><div class="bk nw-book">{book.render_body(bk)}</div>',
                unsafe_allow_html=True)


def render_mind_map(svc: services.Services, conv_id: str, statuses, health_ok: bool) -> None:
    hero = "hero.jpg" if (ROOT / "static" / "hero.jpg").exists() else "study_hero.jpg"
    st.markdown(
        f"""<div class="nw-hero2 quiz"><img class="nw-hero-art" src="app/static/{hero}" alt="">
        <img class="nw-hero-face" src="app/static/hero_face.jpg" alt="">
        <div class="nw-hero-text">
          <div class="nw-hello">🧠 Revise at a glance</div>
          <div class="nw-h1">Mind Map of your <em class="nw-accent">notes</em></div>
          <p>Gemma 4 finds the main topics of your PDF and their key points. Every point is checked
          against your notes, and every topic shows the page to revise.</p>
        </div></div>""",
        unsafe_allow_html=True,
    )
    ids = pick_notes(statuses, "Map for", "mm_scope", "The mind map is built only from them.")
    if not ids:
        return
    key = mind_map.doc_key(ids)
    saved = svc.db.get_mind_map(conv_id, key)

    label = "Remake mind map" if saved else "Create mind map"
    if st.button(label, type="secondary" if saved else "primary", icon=":material/account_tree:",
                 disabled=not health_ok, key="mm_make"):
        with st.spinner("Gemma 4 is reading every page and drawing your mind map…"):
            try:
                mm = mind_map.generate_mind_map(svc.index.get_chunks(ids), svc.gemma)
                svc.db.save_mind_map(conv_id, key, mm.to_dict())
                st.rerun()
            except NoteWiseError as exc:
                st.error(str(exc))
    if not saved:
        st.markdown('<div class="nw-card nw-ready">Click <b>Create mind map</b>. It takes a little while because '
                    'Gemma reads passages from every page.</div>', unsafe_allow_html=True)
        return

    mm = mind_map.MindMap.from_dict(saved)
    st.markdown(f'<div class="nw-mm-wrap">{mind_map.render_svg(mm)}</div>', unsafe_allow_html=True)
    st.caption("Each topic shows its page. Points were kept only if their words appear in that page of your notes.")

    with st.container(key="mmactions"):
        st.markdown('<div class="nw-side-title">Revise a topic</div>', unsafe_allow_html=True)
        names = [f"{b.topic} (p. {b.page_number})" for b in mm.branches]
        pick = st.selectbox("Topic", range(len(names)), format_func=lambda i: names[i], key="mm_topic",
                            label_visibility="collapsed")
        b = mm.branches[pick]
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Study this page", icon="💜", key="mm_study", use_container_width=True):
            start_study_on(svc, conv_id, b.doc_name, b.page_number)
        if c2.button("Quick quiz", icon="📝", key="mm_quiz", use_container_width=True, disabled=not health_ok):
            start_followup_quiz(svc, conv_id, b.doc_name, b.page_number, b.topic)
        c3.download_button("Download map", mind_map.render_svg(mm), file_name="mind_map.svg",
                           mime="image/svg+xml", icon=":material/download:", use_container_width=True, key="mm_svg")
        c4.download_button("Download notes", mind_map.as_outline(mm), file_name="mind_map.txt",
                           mime="text/plain", icon=":material/description:", use_container_width=True, key="mm_txt")


# ---------- landing page ----------
LANDING_FEATURES = [
    ("💬", "AI Tutor", "Answers only from your notes, with the page for every sentence."),
    ("📝", "Smart Quiz", "Questions from your PDF, double-checked by Gemma before you see them."),
    ("💜", "Mood-Aware Study", "Tired or fresh? Sessions adapt to how you feel today."),
    ("🧬", "Learning DNA", "Accuracy, streak, strong and weak topics, all from your real activity."),
    ("🧩", "Mistake Memory", "Remembers every wrong answer and re-tests just those pages."),
    ("✨", "Nova Mentor", "A personal mentor that knows your weak pages and how you learn."),
    ("🧠", "Mind Map", "Your whole PDF as one map of topics, with page numbers."),
    ("📖", "Revision Book", "One click turns your notes into a printable revision book."),
]


def render_landing() -> None:
    cards = "".join(
        f'<div class="nw-lf" style="animation-delay:{0.08 * i:.2f}s"><div class="nw-lf-icon">{icon}</div>'
        f'<b>{title}</b><span>{text}</span></div>'
        for i, (icon, title, text) in enumerate(LANDING_FEATURES)
    )
    st.markdown(
        f"""<div class="nw-land">
        <div class="nw-land-top"><div class="nw-brand">{LOGO}<div class="nw-brand-name">NoteWise <span>AI</span></div></div>
          <span class="nw-pill">● Gemma 4 · runs 100% offline</span></div>
        <div class="nw-land-hero">
          <div class="nw-land-text">
            <div class="nw-kicker">Hacktoberfest Hackday 2026 · Team Neural Nomads</div>
            <h1 class="nw-land-h1">Study smarter from<br><span class="nw-grad">your own notes</span></h1>
            <div class="nw-type"><span>Ask. Practise. Revise. With page proof.</span></div>
            <p>Upload any PDF, even a scanned one. NoteWise answers only from your notes, shows the page,
            and says <b>"not in your notes"</b> when it doesn't know. No internet, no API key, no cost.</p>
          </div>
          <div class="nw-land-art"><img src="app/static/hero.jpg" alt=""></div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    with st.container(key="landcta"):
        if st.button("Start studying →", type="primary", key="landing_start"):
            st.session_state.entered = True
            st.rerun()
    st.markdown(
        f"""<div class="nw-land-stats">
          <div><b>100%</b><span>offline</span></div><div><b>0</b><span>API keys</span></div>
          <div><b>Every</b><span>answer cited</span></div><div><b>Gemma 4</b><span>+ EmbeddingGemma</span></div>
        </div>
        <div class="nw-land-h2">Everything you need to revise</div>
        <div class="nw-lf-grid">{cards}</div>
        <div class="nw-land-h2">How it works</div>
        <div class="nw-steps3">
          <div><span>1</span><b>Add your notes</b><p>PDF, scanned pages or text. Every page keeps its number.</p></div>
          <div><span>2</span><b>Ask and practise</b><p>Grounded answers, quizzes, Study Mode and Nova.</p></div>
          <div><span>3</span><b>Revise smarter</b><p>Mind map, revision book and your Learning DNA show what to study next.</p></div>
        </div>
        <div class="nw-footer">NoteWise AI · Team Neural Nomads · IIMT University, Meerut ·
        Built with Gemma 4 + EmbeddingGemma via Ollama, running fully on this laptop.</div>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    load_css()
    if not st.session_state.get("entered"):
        render_landing()
        return
    svc = get_services()

    conv_id = st.session_state.get("conv_id")
    if not conv_id or svc.db.get_conversation(conv_id) is None:
        existing = svc.db.list_conversations(limit=1)
        conv_id = existing[0].id if existing else svc.manager.new_chat()
        open_chat(conv_id)

    render_sidebar(svc, conv_id)
    if "conv_id" not in st.session_state:             # the chat was just deleted
        st.rerun()

    notice = st.session_state.pop("notice", None)
    if notice:
        getattr(st, notice[0])(notice[1])

    health_ok = get_health().ok
    if not health_ok:
        st.error(
            "Gemma 4 isn't reachable, so NoteWise can't answer right now. "
            "Start Ollama with `brew services start ollama` and make sure the models are downloaded: "
            f"`ollama pull {config.CHAT_MODEL}` and `ollama pull {config.EMBED_MODEL}`."
        )

    statuses = svc.manager.document_status(conv_id)
    doc_ids = [s.record.doc_id for s in statuses if s.ready]
    conv = svc.db.get_conversation(conv_id)

    view = st.session_state.get("view", "Tutor")
    with st.container(key="nav"):
        cols = st.columns([0.95, 0.85, 1.25, 1.6, 1.25, 1.2, 0.9])
        for col, (name, icon) in zip(cols, [("Tutor", "💬"), ("Quiz", "📝"), ("Study Mode", "💜"),
                                            ("My Learning DNA", "🧬"), ("Meet Nova", "✨"), ("Mind Map", "🧠"),
                                            ("Book", "📖")]):
            if col.button(name, icon=icon, key=f"nav_{name}", use_container_width=True,
                          type="primary" if view == name else "secondary"):
                st.session_state.view = name
                st.rerun()
    names = ", ".join(s.record.name for s in statuses if s.ready) or "no notes yet"
    st.markdown(
        f'<div class="nw-context"><b>{html.escape(conv.title if conv else "New chat")}</b>'
        f'<span>·</span>Answering from: {html.escape(names)}</div>',
        unsafe_allow_html=True,
    )

    if view == "Progress":                       # old name, kept for saved sessions
        view = st.session_state.view = "My Learning DNA"
    if view == "Quiz":
        render_quiz(svc, conv_id, doc_ids, health_ok)
    elif view == "Study Mode":
        render_study(svc, conv_id, doc_ids, health_ok)
    elif view == "My Learning DNA":
        render_dna(svc, conv_id, health_ok)
    elif view == "Meet Nova":
        render_nova(svc, conv_id, doc_ids, health_ok)
    elif view == "Mind Map":
        render_mind_map(svc, conv_id, statuses, health_ok)
    elif view == "Book":
        render_book(svc, conv_id, statuses, health_ok)
    else:
        render_tutor(svc, conv_id, ready=bool(doc_ids), health_ok=health_ok)

    st.markdown(
        '<div class="nw-footer">NoteWise AI · Team Neural Nomads · IIMT University, Meerut · '
        "Built with Gemma 4 + EmbeddingGemma via Ollama, running fully on this laptop.</div>",
        unsafe_allow_html=True,
    )


main()
