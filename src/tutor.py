"""The grounded tutor: retrieve evidence, ask Gemma 4, check the answer.

Three safeguards against made-up answers (none is perfect on its own):
1. Retrieval gate   - if no passage is relevant enough, Gemma is never called.
2. Refusal token    - Gemma is told to reply NOT_IN_NOTES when the passages
                      don't contain the answer (catches on-topic-but-missing
                      questions like "Who discovered photosynthesis?").
3. Citation check   - every [S#] tag Gemma writes must point to a passage we
                      actually sent. Invented tags are removed, and an answer
                      with no valid citation is flagged as "unverified".
                      An answer that cites the notes but also says part of the
                      question isn't covered is marked "partial".

Remaining limitation: a valid citation shows WHERE related text was found; it
does not prove every sentence of the answer is supported. The UI therefore
always shows the passages so the student can check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config
from src.retrieval import NoteIndex, RetrievalResult, RetrievedChunk

NOT_IN_NOTES_TOKEN = "NOT_IN_NOTES"
NOT_IN_NOTES_MESSAGE = (
    "I couldn't find enough information in your uploaded notes to answer this confidently. "
    "Try asking about a topic covered in the document."
)
NO_DOCUMENT_MESSAGE = "Please upload or select your notes first, then ask your question."

# Matches [S1], [S1, S2], [S1,S3], [S1 and S2], [S1 & S2]
_CITATION = re.compile(r"\[\s*(S\s*\d+(?:\s*(?:,|and|&)\s*S?\s*\d+)*)\s*\]", re.IGNORECASE)

# Gemma sometimes refuses in its own words instead of the exact token.
_REFUSAL = re.compile(
    r"\b(notes?|passages?|sources?|provided (text|information))\b[^.]{0,40}\b("
    r"do(es)? not|don't|doesn't|did not)\b[^.]{0,20}\b("
    r"state|mention|contain|cover|provide|say|include|specify|give|answer|explain)"
    r"|\bnot (mentioned|stated|covered|included|provided) in the (notes?|passages?)"
    r"|\bno information (about|on|regarding)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = f"""You are NoteWise, a friendly tutor who helps a student understand THEIR OWN notes.

Rules:
1. Use ONLY the numbered note passages provided. Do not add facts from your own knowledge.
2. After every sentence that uses a passage, cite it like [S1] or [S2]. Only use passage numbers that exist.
3. If the passages do not contain the answer, reply with exactly: {NOT_IN_NOTES_TOKEN}
4. If the passages answer only part of the question, answer that part and clearly say what the notes do not cover.
5. Explain in simple, student-friendly language. Keep it short (2-6 sentences) unless the student asks for more detail.
6. Earlier chat messages only help you understand what the student means. They are NOT evidence; only the passages are.
7. Write plain text. Do not use LaTeX, $ signs or math formatting (write H+ and CO2, not formulas)."""


def plain_text(text: str) -> str:
    """Remove LaTeX that small models sometimes add, e.g. $\\text{H}^+$ -> H+."""
    def clean(math: str) -> str:
        math = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", math)
        math = re.sub(r"\^\{?([+\-0-9]+)\}?", r"\1", math)
        math = re.sub(r"_\{?([0-9]+)\}?", r"\1", math)
        math = math.replace("\\rightarrow", "->").replace("\\to", "->")
        return math.replace("{", "").replace("}", "").strip()

    # only touch text inside $...$ so normal sentences are never changed
    return re.sub(r"\$+([^$]{1,60})\$+", lambda m: clean(m.group(1)), text)


@dataclass
class Citation:
    label: str                # "S1"
    page_number: int
    doc_name: str
    text: str
    score: float


@dataclass
class TutorAnswer:
    question: str
    search_query: str
    status: str               # "answered" | "partial" | "unverified" | "not_in_notes" | "no_document"
    text: str                 # what the student sees ([S1] replaced by page labels)
    raw_text: str = ""        # Gemma's original reply, kept for debugging
    citations: list[Citation] = field(default_factory=list)
    evidence: list[Citation] = field(default_factory=list)   # all passages sent to Gemma
    top_score: float = 0.0

    @property
    def grounded(self) -> bool:
        """Answer is backed by at least one valid passage citation."""
        return self.status in ("answered", "partial")


def build_messages(
    question: str,
    evidence: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
    search_query: str | None = None,
    style: str = "",
) -> list[dict[str, str]]:
    """`style` is an extra instruction about HOW to answer (used by Nova); it never adds evidence."""
    passages = "\n\n".join(
        f"[S{i}] (from \"{r.chunk.doc_name}\", page {r.chunk.page_number})\n{r.chunk.text}"
        for i, r in enumerate(evidence, start=1)
    )
    meaning = ""
    if search_query and search_query.strip().lower() != question.strip().lower():
        meaning = f"\n(The student means: {search_query.strip()}. Answer about that topic, in the way the student asked.)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-config.MAX_HISTORY_TURNS:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"Note passages:\n\n{passages}\n\nStudent's question: {question}" + meaning
            + (f"\n(How to answer: {style})" if style else ""),
        }
    )
    return messages


def parse_reply(reply: str, evidence: list[RetrievedChunk]) -> tuple[str, str, list[Citation]]:
    """Return (status, display_text, citations) after validating the [S#] tags."""
    cleaned = reply.strip()
    if not cleaned or NOT_IN_NOTES_TOKEN in cleaned.upper().replace(" ", "_"):
        return "not_in_notes", NOT_IN_NOTES_MESSAGE, []

    used: dict[int, Citation] = {}

    def replace(match: re.Match) -> str:
        pages: list[int] = []
        for n in (int(x) for x in re.findall(r"\d+", match.group(1))):
            if not 1 <= n <= len(evidence):
                continue                                # invented source: drop it
            r = evidence[n - 1]
            used.setdefault(
                n,
                Citation(f"S{n}", r.chunk.page_number, r.chunk.doc_name, r.chunk.text, r.score),
            )
            if r.chunk.page_number not in pages:
                pages.append(r.chunk.page_number)
        if not pages:
            return ""
        return "[p. " + ", ".join(str(p) for p in sorted(pages)) + "]"

    text = plain_text(_CITATION.sub(replace, cleaned))

    # A reply with no valid citation that says "the notes don't cover this" is a refusal.
    if not used and _REFUSAL.search(text):
        return "not_in_notes", NOT_IN_NOTES_MESSAGE, []
    text = re.sub(r"(\[p\. (\d+)\])(\s*\[p\. \2\])+", r"\1", text)   # merge repeats like [p. 3][p. 3]
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    citations = [used[k] for k in sorted(used)]
    if not citations:
        status = "unverified"
    elif _REFUSAL.search(text):
        status = "partial"        # cites the notes but says part of the question isn't covered
    else:
        status = "answered"
    return status, text, citations


# ---------- whole-document questions ----------
# "Explain this pdf" or "name the topics in my notes" are about the WHOLE document, so they
# have no specific words to search for and similarity search scores them low. For these we
# show Gemma passages from every page instead. Still grounded: only real passages, with citations.
_DOC_WORDS = {"pdf", "pdfs", "notes", "note", "document", "documents", "doc", "file", "files", "chapter",
              "chapters", "slides", "slide", "ppt", "material", "lecture", "unit", "syllabus"}
_OVERVIEW_WORDS = {"topics", "topic", "summary", "summarize", "summarise", "summarised", "summarized",
                   "overview", "recap", "contents", "content", "about", "cover", "covers", "covered"}
_FILLER = {
    "explain", "me", "us", "this", "that", "these", "those", "the", "a", "an", "my", "our", "your", "in", "of",
    "on", "from", "given", "all", "main", "key", "important", "major", "point", "points", "name", "names",
    "list", "mention", "mentioned", "mentions", "what", "whats", "is", "are", "it", "its", "give", "tell",
    "please", "pls", "plz", "everything", "whole", "entire", "full", "uploaded", "discussed", "does", "do",
    "which", "there", "can", "could", "you", "briefly", "short", "brief", "quick", "describe", "teach",
    "simple", "simply", "words", "and", "or", "to", "for", "i", "want", "know", "ideas", "idea", "concepts",
    "concept", "things", "learn", "study", "so", "about", "with", "how", "many", "detail", "details",
    "easy", "language", "way", "each", "every", "one", "by", "here", "provided", "inside", "within",
    "contain", "contains", "include", "includes", "have", "has", "be", "write", "show", "discuss",
} | _DOC_WORDS | _OVERVIEW_WORDS
OVERVIEW_PASSAGES = 12
OVERVIEW_STYLE = ("The student wants an overview of their whole notes. Name each main topic the passages cover, "
                  "with one or two simple sentences about it, as a short list starting with '- '. "
                  "Cover passages from all pages, not just one.")


def is_overview_question(question: str) -> bool:
    """True for questions about the whole document rather than one idea."""
    words = re.findall(r"[a-z]+", question.lower())
    if not words or not any(w in _DOC_WORDS or w in _OVERVIEW_WORDS for w in words):
        return False
    return all(w in _FILLER for w in words)


def overview_evidence(chunks: list, limit: int = OVERVIEW_PASSAGES) -> list[RetrievedChunk]:
    """Passages spread over every page (and every file), in page order."""
    by_page: dict[tuple[str, int], list] = {}
    for c in chunks:
        by_page.setdefault((c.doc_id, c.page_number), []).append(c)
    pages = list(by_page.values())
    if len(pages) > limit:                                    # long notes: evenly spaced pages
        step = len(pages) / limit
        pages = [pages[int(i * step)] for i in range(limit)]
    per_page = max(1, limit // max(1, len(pages)))
    picked = [c for page in pages for c in page[:per_page]][:limit]
    return [RetrievedChunk(chunk=c, score=1.0) for c in picked]


def answer_question(
    question: str,
    doc_ids: list[str],
    index: NoteIndex,
    gemma,
    history: list[dict[str, str]] | None = None,
    search_query: str | None = None,
    style: str = "",
) -> TutorAnswer:
    """`search_query` is the standalone version of a follow-up question (Phase 3).
    The student's original wording is still what Gemma answers."""
    query = (search_query or question).strip()
    if not doc_ids:
        return TutorAnswer(question, query, "no_document", NO_DOCUMENT_MESSAGE)

    if is_overview_question(question) or is_overview_question(query):
        evidence = overview_evidence(index.get_chunks(doc_ids))
        if evidence:
            style = (style + " " + OVERVIEW_STYLE).strip()
            retrieval = RetrievalResult(query=query, chunks=evidence, min_score=0.0)
        else:
            retrieval = index.search(query, doc_ids)
    else:
        retrieval = index.search(query, doc_ids)
    evidence = retrieval.evidence
    evidence_view = [
        Citation(f"S{i}", r.chunk.page_number, r.chunk.doc_name, r.chunk.text, r.score)
        for i, r in enumerate(evidence, start=1)
    ]

    # Safeguard 1: nothing relevant enough -> don't ask Gemma at all
    if not retrieval.supported:
        return TutorAnswer(
            question, query, "not_in_notes", NOT_IN_NOTES_MESSAGE, top_score=retrieval.top_score
        )

    reply = gemma.chat(build_messages(question, evidence, history, search_query=query, style=style))
    status, text, citations = parse_reply(reply, evidence)       # safeguards 2 and 3
    return TutorAnswer(
        question=question,
        search_query=query,
        status=status,
        text=text,
        raw_text=reply,
        citations=citations,
        evidence=evidence_view,
        top_score=retrieval.top_score,
    )
