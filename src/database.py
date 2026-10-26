"""SQLite storage for chats, messages, documents and quiz scores.

Where things live (all inside data/, which is git-ignored):
    data/notewise.db   this database
    data/uploads/      a copy of every uploaded file, named <doc_id>.<ext>
    data/chroma/       the vector index (retrieval.py)

Streamlit session state only holds "which chat is open right now". Everything
the student should get back after a refresh is stored here.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    page_count   INTEGER NOT NULL,
    file_path    TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_documents (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    PRIMARY KEY (conversation_id, doc_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    search_query    TEXT,
    status          TEXT,
    citations_json  TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS mentor_messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    mode            TEXT,
    status          TEXT,
    citations_json  TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mind_maps (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    doc_key         TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (conversation_id, doc_key)
);

CREATE TABLE IF NOT EXISTS books (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    doc_key         TEXT NOT NULL,
    data_json       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (conversation_id, doc_key)
);

CREATE TABLE IF NOT EXISTS study_sessions (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    mood            TEXT NOT NULL,
    goal            TEXT NOT NULL,
    minutes         INTEGER NOT NULL,
    pages_json      TEXT,
    score           INTEGER,
    total           INTEGER,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    doc_id          TEXT,
    score           INTEGER NOT NULL,
    total           INTEGER NOT NULL,
    details_json    TEXT,
    created_at      TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class DocumentRecord:
    doc_id: str
    name: str
    source_type: str
    page_count: int
    file_path: str | None
    created_at: str

    @property
    def file_available(self) -> bool:
        return bool(self.file_path) and Path(self.file_path).exists()


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    search_query: str | None
    status: str | None
    citations: list[dict]
    created_at: str


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # A new short-lived connection per call is the simplest safe pattern
        # with Streamlit, which reruns the script on different threads.
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _run(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self._connect()
        try:
            with conn:
                return conn.execute(sql, params)
        finally:
            conn.close()

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    # ---------- documents ----------
    def upsert_document(
        self, doc_id: str, name: str, source_type: str, page_count: int, file_path: str | None
    ) -> None:
        self._run(
            """INSERT INTO documents (doc_id, name, source_type, page_count, file_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_id) DO UPDATE SET
                   name = excluded.name,
                   file_path = COALESCE(excluded.file_path, documents.file_path)""",
            (doc_id, name, source_type, page_count, file_path, _now()),
        )

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        rows = self._all("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return DocumentRecord(**dict(rows[0])) if rows else None

    def list_documents(self) -> list[DocumentRecord]:
        return [DocumentRecord(**dict(r)) for r in self._all("SELECT * FROM documents ORDER BY created_at DESC")]

    # ---------- conversations ----------
    def create_conversation(self, title: str = "New chat", conversation_id: str | None = None) -> str:
        cid = conversation_id or new_id()
        now = _now()
        self._run(
            "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (cid, title, now, now),
        )
        return cid

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        rows = self._all("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        return Conversation(**dict(rows[0])) if rows else None

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        rows = self._all("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [Conversation(**dict(r)) for r in rows]

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        title = title.strip()[:80] or "Untitled chat"
        self._run("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    def delete_conversation(self, conversation_id: str) -> None:
        self._run("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def touch_conversation(self, conversation_id: str) -> None:
        self._run("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id))

    def link_document(self, conversation_id: str, doc_id: str) -> None:
        self._run(
            "INSERT OR IGNORE INTO conversation_documents (conversation_id, doc_id) VALUES (?, ?)",
            (conversation_id, doc_id),
        )

    def unlink_document(self, conversation_id: str, doc_id: str) -> None:
        self._run(
            "DELETE FROM conversation_documents WHERE conversation_id = ? AND doc_id = ?",
            (conversation_id, doc_id),
        )

    def conversation_documents(self, conversation_id: str) -> list[DocumentRecord]:
        rows = self._all(
            """SELECT d.* FROM documents d
               JOIN conversation_documents cd ON cd.doc_id = d.doc_id
               WHERE cd.conversation_id = ? ORDER BY d.created_at""",
            (conversation_id,),
        )
        return [DocumentRecord(**dict(r)) for r in rows]

    # ---------- messages ----------
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        search_query: str | None = None,
        status: str | None = None,
        citations: list[dict] | None = None,
        message_id: str | None = None,
    ) -> str:
        """Pass the same message_id twice (e.g. a Streamlit rerun) and only one row is stored."""
        mid = message_id or new_id()
        self._run(
            """INSERT OR IGNORE INTO messages
               (id, conversation_id, role, content, search_query, status, citations_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (mid, conversation_id, role, content, search_query, status,
             json.dumps(citations or []), _now()),
        )
        self.touch_conversation(conversation_id)
        return mid

    def get_messages(self, conversation_id: str, limit: int | None = None) -> list[Message]:
        sql = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid"
        rows = self._all(sql, (conversation_id,))
        if limit is not None:
            rows = rows[-limit:]
        return [
            Message(
                id=r["id"], conversation_id=r["conversation_id"], role=r["role"],
                content=r["content"], search_query=r["search_query"], status=r["status"],
                citations=json.loads(r["citations_json"] or "[]"), created_at=r["created_at"],
            )
            for r in rows
        ]

    def message_count(self, conversation_id: str) -> int:
        return self._all(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (conversation_id,)
        )[0]["n"]

    # ---------- quiz attempts (used in Phase 4) ----------
    def add_quiz_attempt(
        self, score: int, total: int, doc_id: str | None = None,
        conversation_id: str | None = None, details: list[dict] | None = None,
        attempt_id: str | None = None,
    ) -> str:
        aid = attempt_id or new_id()
        self._run(
            """INSERT OR IGNORE INTO quiz_attempts
               (id, conversation_id, doc_id, score, total, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (aid, conversation_id, doc_id, score, total, json.dumps(details or []), _now()),
        )
        return aid

    def list_quiz_attempts(self, limit: int = 20, conversation_id: str | None = None) -> list[dict]:
        """Newest first. Pass conversation_id to get only that chat's quizzes."""
        if conversation_id is None:
            rows = self._all("SELECT * FROM quiz_attempts ORDER BY created_at DESC LIMIT ?", (limit,))
        else:
            rows = self._all(
                "SELECT * FROM quiz_attempts WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            )
        return [dict(r) for r in rows]

    # ---------- mind maps ----------
    def save_mind_map(self, conversation_id: str, doc_key: str, data: dict) -> None:
        self._run(
            """INSERT OR REPLACE INTO mind_maps (conversation_id, doc_key, data_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, doc_key, json.dumps(data), _now()),
        )

    def get_mind_map(self, conversation_id: str, doc_key: str) -> dict | None:
        rows = self._all("SELECT data_json FROM mind_maps WHERE conversation_id = ? AND doc_key = ?",
                         (conversation_id, doc_key))
        return json.loads(rows[0]["data_json"]) if rows else None

    def save_book(self, conversation_id: str, doc_key: str, data: dict) -> None:
        self._run(
            """INSERT OR REPLACE INTO books (conversation_id, doc_key, data_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, doc_key, json.dumps(data), _now()),
        )

    def get_book(self, conversation_id: str, doc_key: str) -> dict | None:
        rows = self._all("SELECT data_json FROM books WHERE conversation_id = ? AND doc_key = ?",
                         (conversation_id, doc_key))
        return json.loads(rows[0]["data_json"]) if rows else None

    # ---------- Nova (mentor) chat ----------
    def add_mentor_message(
        self, conversation_id: str, role: str, content: str, mode: str | None = None,
        status: str | None = None, citations: list[dict] | None = None, message_id: str | None = None,
    ) -> str:
        mid = message_id or new_id()
        self._run(
            """INSERT OR IGNORE INTO mentor_messages
               (id, conversation_id, role, content, mode, status, citations_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (mid, conversation_id, role, content, mode, status, json.dumps(citations or []), _now()),
        )
        return mid

    def get_mentor_messages(self, conversation_id: str) -> list[dict]:
        rows = self._all("SELECT * FROM mentor_messages WHERE conversation_id = ? ORDER BY created_at, rowid",
                         (conversation_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["citations"] = json.loads(d.pop("citations_json") or "[]")
            out.append(d)
        return out

    def clear_mentor_messages(self, conversation_id: str) -> None:
        self._run("DELETE FROM mentor_messages WHERE conversation_id = ?", (conversation_id,))

    # ---------- study sessions (Study Mode) ----------
    def add_study_session(
        self, mood: str, goal: str, minutes: int, pages: list[dict], score: int | None, total: int | None,
        conversation_id: str | None = None, session_id: str | None = None,
    ) -> str:
        sid = session_id or new_id()
        self._run(
            """INSERT OR IGNORE INTO study_sessions
               (id, conversation_id, mood, goal, minutes, pages_json, score, total, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sid, conversation_id, mood, goal, minutes, json.dumps(pages), score, total, _now()),
        )
        return sid

    def list_study_sessions(self, conversation_id: str | None = None, limit: int = 200) -> list[dict]:
        if conversation_id is None:
            rows = self._all("SELECT * FROM study_sessions ORDER BY created_at DESC LIMIT ?", (limit,))
        else:
            rows = self._all(
                "SELECT * FROM study_sessions WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            )
        return [dict(r) for r in rows]

    def activity_dates(self, conversation_id: str | None = None) -> list[str]:
        """Timestamps of everything the student did (questions, quizzes, sessions), for the streak."""
        where, params = ("", ()) if conversation_id is None else (" WHERE conversation_id = ?", (conversation_id,))
        rows = self._all(
            f"""SELECT created_at FROM messages{where + (" AND" if where else " WHERE")} role = 'user'
                UNION ALL SELECT created_at FROM quiz_attempts{where}
                UNION ALL SELECT created_at FROM study_sessions{where}
                UNION ALL SELECT created_at FROM mentor_messages{where + (" AND" if where else " WHERE")} role = 'user'""",
            params * 4,
        )
        return [r["created_at"] for r in rows]

    def user_messages(self, conversation_id: str | None = None) -> list[str]:
        if conversation_id is None:
            rows = self._all("SELECT content FROM messages WHERE role = 'user'")
        else:
            rows = self._all("SELECT content FROM messages WHERE role = 'user' AND conversation_id = ?",
                             (conversation_id,))
        return [r["content"] for r in rows]

    # ---------- progress page ----------
    def stats(self, conversation_id: str | None = None) -> dict:
        """Counts for the Progress page: all chats, or one chat when conversation_id is given."""
        if conversation_id is None:
            where, params = "", ()
        else:
            where, params = " AND conversation_id = ?", (conversation_id,)
        row = self._all(
            f"""SELECT
                 (SELECT COUNT(*) FROM conversations) AS chats,
                 (SELECT COUNT(*) FROM messages WHERE role = 'user'{where}) AS questions,
                 (SELECT COUNT(*) FROM messages WHERE role = 'assistant'
                    AND status IN ('answered', 'partial'){where}) AS grounded_answers,
                 (SELECT COUNT(*) FROM messages WHERE role = 'assistant'
                    AND status = 'not_in_notes'{where}) AS not_in_notes,
                 (SELECT COUNT(*) FROM quiz_attempts WHERE 1=1{where}) AS quizzes,
                 (SELECT COALESCE(SUM(score), 0) FROM quiz_attempts WHERE 1=1{where}) AS quiz_correct,
                 (SELECT COALESCE(SUM(total), 0) FROM quiz_attempts WHERE 1=1{where}) AS quiz_total""",
            params * 6,
        )[0]
        return dict(row)

    # ---------- demo reset ----------
    def reset_all(self) -> None:
        conn = self._connect()
        try:
            with conn:
                for table in ("messages", "mentor_messages", "mind_maps", "books", "conversation_documents", "quiz_attempts", "study_sessions",
                              "conversations", "documents"):
                    conn.execute(f"DELETE FROM {table}")
        finally:
            conn.close()
