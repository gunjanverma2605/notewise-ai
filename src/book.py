"""Revision Book: turns the mind map of a PDF into a short book.

Each mind-map topic becomes a chapter. Gemma 4 teaches that topic from the passages
of its page (the same grounded, cited explanation used in Study Mode), and the
chapter keeps the checked key points from the mind map. If Gemma's explanation for
a chapter is not grounded, the chapter shows only the key points.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass, field
from datetime import date

from src.mind_map import MindMap
from src.study_mode import MOODS, explain


@dataclass
class Chapter:
    title: str
    doc_name: str
    page_number: int
    text: str
    points: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)      # pages cited in the explanation
    grounded: bool = True


@dataclass
class Book:
    title: str
    chapters: list[Chapter]
    made_on: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "made_on": self.made_on, "chapters": [asdict(c) for c in self.chapters]}

    @classmethod
    def from_dict(cls, data: dict) -> "Book":
        return cls(data.get("title", "My notes"), [Chapter(**c) for c in data.get("chapters", [])],
                   data.get("made_on", ""))


def _plain(text: str) -> str:
    """'[p. 3]' chips become '(p. 3)' in the book."""
    return re.sub(r"\s*\[p\. ([\d, ]+)\]", r" (p. \1)", text).strip()


def generate_book(mm: MindMap, page_chunks, gemma, on_progress=None) -> Book:
    """`page_chunks(doc_name, page)` returns the chunks of one page."""
    chapters: list[Chapter] = []
    total = len(mm.branches)
    for i, branch in enumerate(mm.branches, start=1):
        chunks = page_chunks(branch.doc_name, branch.page_number)
        text, pages, grounded = "", [], False
        if chunks:
            result = explain(chunks, MOODS["fresh"], "teach", gemma)
            if result.status in ("answered", "partial") and result.citations:
                text = _plain(result.text)
                pages = sorted({c.page_number for c in result.citations})
                grounded = True
        if not grounded:
            text = "Key points from your notes:"
            pages = [branch.page_number]
        chapters.append(Chapter(branch.topic, branch.doc_name, branch.page_number, text,
                                list(branch.points), pages, grounded))
        if on_progress:
            on_progress(i, total)
    return Book(mm.title, chapters, date.today().strftime("%d %b %Y"))


# ---------- rendering ----------
def _paragraphs(text: str) -> str:
    """Plain text with '1.' or '-' lines -> HTML paragraphs and lists."""
    out, items, kind = [], [], None

    def flush():
        nonlocal items, kind
        if items:
            tag = "ol" if kind == "ol" else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{tag}>")
        items, kind = [], None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            flush()
            continue
        m_ol = re.match(r"^\d+[.)]\s+(.*)", line)
        m_ul = re.match(r"^[-•*]\s+(.*)", line)
        if m_ol or m_ul:
            new_kind = "ol" if m_ol else "ul"
            if kind and kind != new_kind:
                flush()
            kind = new_kind
            items.append(_inline((m_ol or m_ul).group(1)))
        else:
            flush()
            out.append(f"<p>{_inline(line)}</p>")
    flush()
    return "".join(out)


def _inline(text: str) -> str:
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return re.sub(r"\(p\. ([\d, ]+)\)", r'<span class="bk-page">p. \1</span>', safe)


def render_body(book: Book) -> str:
    """The book's inner HTML (cover, contents, chapters). Used in the app and the download."""
    toc = "".join(
        f'<li><span>{i}. {html.escape(c.title)}</span><span class="bk-dots"></span><span>p. {c.page_number}</span></li>'
        for i, c in enumerate(book.chapters, start=1)
    )
    chapters = []
    for i, c in enumerate(book.chapters, start=1):
        points = "".join(f"<li>{html.escape(p)}</li>" for p in c.points)
        src = f"From {html.escape(c.doc_name)}, page {', '.join(map(str, c.pages or [c.page_number]))}"
        chapters.append(
            f'<section class="bk-chapter"><div class="bk-num">Chapter {i}</div>'
            f'<h2>{html.escape(c.title)}</h2><div class="bk-src">{src}</div>'
            f'<div class="bk-text">{_paragraphs(c.text)}</div>'
            + (f'<div class="bk-keys"><b>Key points</b><ul>{points}</ul></div>' if points else "")
            + "</section>"
        )
    return (
        f'<div class="bk-cover"><div class="bk-kicker">NoteWise AI · Revision Book</div>'
        f'<h1>{html.escape(book.title)}</h1>'
        f'<div class="bk-sub">Made from your own notes · {len(book.chapters)} chapters · {html.escape(book.made_on)}</div>'
        f'<div class="bk-note">Every chapter is explained only from your notes and shows the page to revise.</div></div>'
        f'<div class="bk-toc"><h3>Contents</h3><ol>{toc}</ol></div>'
        + "".join(chapters)
    )


BOOK_CSS = """
.bk { font-family: Georgia, 'Times New Roman', serif; color: #2a2540; background: #FFFDF8; line-height: 1.65; }
.bk-cover { text-align: center; padding: 48px 28px 36px; border-bottom: 2px solid #E9E2FF;
  background: linear-gradient(160deg, #F4F0FF 0%, #FFFDF8 70%); }
.bk-kicker { font-family: -apple-system, Arial, sans-serif; font-size: 12px; letter-spacing: .14em;
  text-transform: uppercase; color: #6D4AFF; font-weight: 700; }
.bk-cover h1 { font-size: 40px; margin: 14px 0 8px; color: #1d1840; }
.bk-sub { color: #6b6890; font-size: 15px; }
.bk-note { margin: 18px auto 0; max-width: 460px; font-size: 14px; color: #4b4670; font-style: italic; }
.bk-toc { padding: 24px 40px; border-bottom: 1px solid #EEE8FF; }
.bk-toc h3 { font-size: 18px; margin: 0 0 10px; color: #4B2FD6; }
.bk-toc ol { list-style: none; padding: 0; margin: 0; }
.bk-toc li { display: flex; gap: 8px; padding: 5px 0; font-size: 16px; }
.bk-dots { flex: 1; border-bottom: 1px dotted #b9b2d9; margin-bottom: 6px; }
.bk-chapter { padding: 30px 40px 26px; border-bottom: 1px solid #EEE8FF; break-inside: avoid-page; }
.bk-num { font-family: -apple-system, Arial, sans-serif; font-size: 12px; letter-spacing: .12em;
  text-transform: uppercase; color: #8B6CFF; font-weight: 700; }
.bk-chapter h2 { font-size: 26px; margin: 4px 0 2px; color: #1d1840; }
.bk-src { font-size: 13px; color: #8a86a8; margin-bottom: 12px; font-style: italic; }
.bk-text p { margin: 0 0 10px; font-size: 16.5px; }
.bk-text ol, .bk-text ul { margin: 0 0 10px 22px; padding: 0; font-size: 16.5px; }
.bk-text li { margin-bottom: 6px; }
.bk-page { font-family: -apple-system, Arial, sans-serif; font-size: 11px; font-weight: 700; color: #4B2FD6;
  background: #EEE9FF; border-radius: 6px; padding: 1px 6px; white-space: nowrap; }
.bk-keys { background: #F7F4FF; border-left: 4px solid #8B6CFF; border-radius: 10px; padding: 12px 16px; margin-top: 12px; }
.bk-keys b { font-family: -apple-system, Arial, sans-serif; font-size: 13px; color: #4B2FD6; }
.bk-keys ul { margin: 6px 0 0 18px; padding: 0; font-size: 15.5px; }
"""


def render_download(book: Book) -> str:
    """A standalone HTML file: open it and press Cmd+P -> Save as PDF."""
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>{html.escape(book.title)} – Revision Book</title><style>"
        + BOOK_CSS
        + "body{margin:0;background:#eee9f7}.bk{max-width:820px;margin:24px auto;box-shadow:0 8px 30px rgba(0,0,0,.1)}"
        "@media print{body{background:#fff}.bk{box-shadow:none;margin:0;max-width:none}"
        ".bk-chapter{break-before:page}.bk-chapter:first-of-type{break-before:auto}}"
        "</style></head><body><div class='bk'>" + render_body(book) + "</div></body></html>"
    )
