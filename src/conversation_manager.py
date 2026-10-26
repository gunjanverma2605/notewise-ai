"""Chat memory: saved conversations, follow-up questions and linked notes.

Kept separate on purpose:
* database.py  stores things
* retrieval.py finds passages
* tutor.py     answers from passages
* this module  decides WHAT to ask (resolving "explain that simpler") and
               WHICH notes a chat may use, then saves the exchange.

Follow-up handling: "Explain its second step" can't be searched as-is, so we
ask Gemma to rewrite it into a standalone question ("Explain the second step
of the light-dependent reactions"). That rewritten query is used ONLY for
retrieval; the student still sees, and Gemma still answers, their own words.
Fresh evidence is retrieved every turn, so an earlier wrong answer is never
treated as a source.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config
from src.database import Database, DocumentRecord, Message
from src.document_loader import LoadedDocument, load_document, save_upload
from src.errors import NoteWiseError
from src.retrieval import NoteIndex
from src.tutor import TutorAnswer, answer_question

# Words that usually mean "this question depends on the previous turn".
_REFERENCE = re.compile(
    r"\b(it|its|it's|that|this|these|those|they|them|their|there|he|she|his|her|"
    r"above|previous|same|again|simpler|simple words|more detail|in detail|elaborate|"
    r"example|why not|what about|and then|discussed|summari[sz]e|summary|recap|so far|we talked|conversation)\b",
    re.IGNORECASE,
)

REWRITE_PROMPT = """You rewrite a student's latest message into ONE standalone search question.
Use the earlier conversation only to replace words like "it", "that" or "this" with the real topic.
If the student asks to summarise or recap the conversation, ask for a summary of the main topic discussed.
Do not answer the question. Do not add new topics. Output only the rewritten question, nothing else."""

TITLE_PROMPT = """Write a short, plain title (2 to 5 words) for a study chat that starts with the question below.
Name the topic, like "Light Reactions Revision" or "Calvin Cycle Basics".
No clickbait words such as Secrets, Unveiled, Magic or Amazing.
Output only the title, without quotes or punctuation at the end."""


@dataclass
class DocumentStatus:
    record: DocumentRecord
    ready: bool          # indexed and searchable now
    restored: bool       # was re-indexed from the saved file just now
    message: str


def needs_rewrite(question: str, history: list[dict[str, str]]) -> bool:
    if not any(turn["role"] == "user" for turn in history):
        return False
    return len(question.split()) <= 4 or bool(_REFERENCE.search(question))


def _clean_line(text: str, max_len: int) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"^(standalone (search )?question|rewritten question|title)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip("\"'`*").strip()
    return text if 2 < len(text) <= max_len else ""


def rewrite_query(question: str, history: list[dict[str, str]], gemma) -> str:
    """Turn a follow-up into a standalone search question. Never raises:
    if Gemma fails, fall back to 'previous question + this question'."""
    last_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
    fallback = f"{last_user} {question}".strip()
    transcript = "\n".join(
        f"{'Student' if t['role'] == 'user' else 'Tutor'}: {t['content'][:300]}"
        for t in history[-4:]
    )
    try:
        reply = gemma.chat(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}\n\nStandalone question:"},
            ],
            temperature=0.0,
        )
    except NoteWiseError:
        return fallback
    return _clean_line(reply, 200) or fallback


def fallback_title(question: str) -> str:
    text = re.sub(r"[?!.]+$", "", question.strip())
    text = re.sub(
        r"^(what|why|how|explain|describe|define|tell me about)\s+((is|are|does|do)\s+)?((the|a|an)\s+)?",
        "", text, flags=re.I,
    )
    words = text.split()[:5]
    if not words:
        return "New chat"
    title = " ".join(words)
    return title[0].upper() + title[1:]


def generate_title(question: str, gemma) -> str:
    try:
        reply = gemma.chat(
            [{"role": "system", "content": TITLE_PROMPT}, {"role": "user", "content": question}],
            temperature=0.0,
        )
    except NoteWiseError:
        return fallback_title(question)
    title = _clean_line(reply, 60).rstrip(".")
    return title or fallback_title(question)


class ConversationManager:
    def __init__(self, db: Database, index: NoteIndex, upload_dir: str | Path | None = None) -> None:
        self.db = db
        self.index = index
        self.upload_dir = Path(upload_dir or config.UPLOAD_DIR)

    # ---------- chats ----------
    def new_chat(self, doc_ids: list[str] | None = None) -> str:
        cid = self.db.create_conversation()
        for doc_id in doc_ids or []:
            self.db.link_document(cid, doc_id)
        return cid

    def messages(self, conversation_id: str) -> list[Message]:
        return self.db.get_messages(conversation_id)

    def history_for_prompt(self, conversation_id: str) -> list[dict[str, str]]:
        """Only the last few turns, so the prompt never grows without limit."""
        recent = self.db.get_messages(conversation_id, limit=config.MAX_HISTORY_TURNS)
        return [{"role": m.role, "content": m.content} for m in recent]

    # ---------- notes ----------
    def add_document(self, conversation_id: str, doc: LoadedDocument, data: bytes) -> int:
        """Save the file, register it, index it (once) and attach it to the chat."""
        path = save_upload(data, doc, self.upload_dir)
        self.db.upsert_document(doc.doc_id, doc.name, doc.source_type, doc.page_count, str(path))
        chunks = self.index.add_document(doc)
        self.db.link_document(conversation_id, doc.doc_id)
        return chunks

    def document_status(self, conversation_id: str) -> list[DocumentStatus]:
        """For each note linked to this chat: is it searchable, can it be restored
        from the saved copy, or must the student re-upload it?"""
        statuses: list[DocumentStatus] = []
        for rec in self.db.conversation_documents(conversation_id):
            if self.index.has_document(rec.doc_id):
                statuses.append(DocumentStatus(rec, True, False, "Ready"))
                continue
            if rec.file_available:
                try:
                    data = Path(rec.file_path).read_bytes()
                    doc = load_document(data, Path(rec.file_path).name)
                    doc.name = rec.name
                    if doc.doc_id == rec.doc_id:
                        self.index.add_document(doc)
                        statuses.append(DocumentStatus(rec, True, True, "Restored from saved copy"))
                        continue
                except NoteWiseError:
                    pass
            statuses.append(
                DocumentStatus(rec, False, False, f"'{rec.name}' is missing. Please re-upload it to continue this chat.")
            )
        return statuses

    # ---------- asking ----------
    def ask(
        self,
        conversation_id: str,
        question: str,
        gemma,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
    ) -> TutorAnswer:
        question = question.strip()
        history = self.history_for_prompt(conversation_id)       # before saving the new question
        is_first_question = not any(t["role"] == "user" for t in history)

        statuses = self.document_status(conversation_id)
        doc_ids = [s.record.doc_id for s in statuses if s.ready]
        missing = [s.message for s in statuses if not s.ready]

        self.db.add_message(conversation_id, "user", question, message_id=user_message_id)

        if not doc_ids:
            text = " ".join(missing) if missing else "Please upload or paste your notes first, then ask your question."
            answer = TutorAnswer(question, question, "no_document", text)
        else:
            search_query = rewrite_query(question, history, gemma) if needs_rewrite(question, history) else question
            answer = answer_question(
                question, doc_ids, self.index, gemma, history=history, search_query=search_query
            )

        self.db.add_message(
            conversation_id,
            "assistant",
            answer.text,
            search_query=answer.search_query,
            status=answer.status,
            citations=[asdict(c) for c in answer.citations],
            message_id=assistant_message_id,
        )
        if is_first_question and answer.status != "no_document":
            self.db.rename_conversation(conversation_id, generate_title(question, gemma))
        return answer
