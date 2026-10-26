"""Mind map of the student's notes, for quick revision.

Gemma 4 reads passages from every page and returns the main topics with a few short
points each, citing the passage it used. Code then keeps only points whose words are
really in that passage, attaches the page number, and draws the map as SVG.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field

from src.errors import NoteWiseError
from src.quiz_generator import InvalidQuestion, _content_words, extract_json
from src.tutor import overview_evidence

MAX_BRANCHES = 7
MAX_POINTS = 3
MAP_PASSAGES = 14

MINDMAP_PROMPT = """You make a revision mind map from a student's OWN notes.

Rules:
1. Use ONLY the numbered passages. Do not add facts from your own knowledge.
2. Find the 4 to 7 main topics of the notes. For each topic give 2 or 3 key points.
3. Each point is a short phrase of at most 8 words, copied or closely paraphrased from ONE passage.
4. For each topic, "source" is the passage number it mainly comes from, like "S3".
5. "title" is the overall subject of the notes in at most 5 words.
6. Reply with JSON only, in this shape:
{"title": "...", "branches": [{"topic": "...", "source": "S1", "points": ["...", "..."]}]}"""


@dataclass
class Branch:
    topic: str
    doc_name: str
    page_number: int
    points: list[str] = field(default_factory=list)


@dataclass
class MindMap:
    title: str
    branches: list[Branch]

    def to_dict(self) -> dict:
        return {"title": self.title, "branches": [asdict(b) for b in self.branches]}

    @classmethod
    def from_dict(cls, data: dict) -> "MindMap":
        return cls(data.get("title", "My notes"), [Branch(**b) for b in data.get("branches", [])])


def build_messages(evidence) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[S{i}] (from \"{e.chunk.doc_name}\", page {e.chunk.page_number})\n{e.chunk.text}"
        for i, e in enumerate(evidence, start=1)
    )
    return [
        {"role": "system", "content": MINDMAP_PROMPT},
        {"role": "user", "content": f"Note passages:\n\n{passages}\n\nMake the mind map JSON."},
    ]


def _clean(text: str, limit: int) -> str:
    text = " ".join(str(text).replace("*", "").split()).strip(" -•.;:")
    return text[:limit].rstrip()


def _supported(point: str, passage: str) -> bool:
    """At least half of the point's content words must appear in the passage."""
    words = _content_words(point)
    if not words:
        return False
    return len(words & _content_words(passage)) / len(words) >= 0.5


def validate(data: dict, evidence) -> MindMap:
    """Keep only what the notes support. Raises InvalidQuestion if too little is left."""
    branches: list[Branch] = []
    seen: set[str] = set()
    for raw in data.get("branches") or []:
        if not isinstance(raw, dict):
            continue
        topic = _clean(raw.get("topic", ""), 40)
        if not topic or topic.lower() in seen:
            continue
        label = str(raw.get("source", "")).strip().upper().strip("[]")
        idx = int(label[1:]) - 1 if label.startswith("S") and label[1:].isdigit() else -1
        source = evidence[idx] if 0 <= idx < len(evidence) else None
        points: list[str] = []
        for p in raw.get("points") or []:
            p = _clean(p, 70)
            if not p or p.lower() in (q.lower() for q in points):
                continue
            if source is not None and _supported(p, source.chunk.text):
                points.append(p)
            elif source is None:                      # no valid source: find the passage that supports it
                match = next((e for e in evidence if _supported(p, e.chunk.text)), None)
                if match is not None:
                    source = match
                    points.append(p)
        if source is None or not points:
            continue
        seen.add(topic.lower())
        branches.append(Branch(topic, source.chunk.doc_name, source.chunk.page_number, points[:MAX_POINTS]))
        if len(branches) == MAX_BRANCHES:
            break
    if len(branches) < 2:
        raise InvalidQuestion("too few topics supported by the notes")
    title = _clean(data.get("title", ""), 40) or "My notes"
    return MindMap(title, branches)


def generate_mind_map(chunks, gemma, attempts: int = 2) -> MindMap:
    """Ask Gemma for the map, check it against the passages; one retry on bad output."""
    evidence = overview_evidence(chunks, limit=MAP_PASSAGES)
    if not evidence:
        raise NoteWiseError("There are no notes to map yet.")
    last = None
    for _ in range(attempts):
        reply = gemma.chat(build_messages(evidence), json_mode=True)
        try:
            return validate(extract_json(reply), evidence)
        except InvalidQuestion as exc:
            last = exc
    raise NoteWiseError("Gemma couldn't build a reliable mind map from these notes. Please try again.") from last


# ---------- drawing ----------
COLORS = ["#7B5CFF", "#2FBF8A", "#F08A3C", "#3B82F6", "#E8559A", "#C9A227", "#14A3B8"]


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines or [""]


def _text(lines: list[str], x: float, y: float, size: float, weight: int, color: str, anchor: str = "middle") -> str:
    out = []
    for i, line in enumerate(lines):
        out.append(f'<text x="{x:.0f}" y="{y + i * size * 1.3:.0f}" font-size="{size}" font-weight="{weight}" '
                   f'fill="{color}" text-anchor="{anchor}">{html.escape(line)}</text>')
    return "".join(out)


def render_svg(mm: MindMap) -> str:
    """A two-sided mind map: title in the middle, topics left and right, points outside."""
    center_w, branch_w, leaf_w = 200, 200, 250
    gap_side = 30
    W = center_w + 2 * (2 * gap_side + branch_w + leaf_w) + 40
    cx = W / 2
    leaf_h_line, branch_line = 20, 21
    half = (len(mm.branches) + 1) // 2
    sides = [("right", mm.branches[:half]), ("left", mm.branches[half:])]

    def leaf_height(p: str) -> int:
        return 16 + len(_wrap(p, 25)) * leaf_h_line

    def block_height(b: Branch) -> int:
        leaves = sum(leaf_height(p) for p in b.points) + 8 * (len(b.points) - 1)
        box = 30 + len(_wrap(b.topic, 17)) * branch_line + 20
        return max(leaves, box)

    side_heights = [sum(block_height(b) for b in bs) + 26 * max(0, len(bs) - 1) for _, bs in sides]
    H = max(max(side_heights) + 80, 300)
    cy = H / 2
    parts: list[str] = []
    nodes: list[str] = []

    for (side, bs), total in zip(sides, side_heights):
        sign = 1 if side == "right" else -1
        bx = cx + sign * (center_w / 2 + gap_side) + (0 if sign > 0 else -branch_w)   # branch box left edge
        lx = bx + branch_w + gap_side if sign > 0 else bx - gap_side - leaf_w          # leaf box left edge
        y = cy - total / 2
        for i, b in enumerate(bs):
            color = COLORS[(mm.branches.index(b)) % len(COLORS)]
            bh = block_height(b)
            mid = y + bh / 2
            t_lines = _wrap(b.topic, 17)
            box_h = 34 + len(t_lines) * branch_line
            # curve from centre to the branch
            sx = cx + sign * center_w / 2
            ex = bx if sign > 0 else bx + branch_w
            parts.append(f'<path d="M{sx:.0f} {cy:.0f} C{(sx + ex) / 2:.0f} {cy:.0f}, {(sx + ex) / 2:.0f} {mid:.0f}, '
                         f'{ex:.0f} {mid:.0f}" stroke="{color}" stroke-width="3" fill="none" opacity="0.55"/>')
            nodes.append(f'<rect x="{bx:.0f}" y="{mid - box_h / 2:.0f}" width="{branch_w}" height="{box_h:.0f}" rx="14" '
                         f'fill="{color}" fill-opacity="0.14" stroke="{color}" stroke-width="2"/>')
            nodes.append(_text(t_lines, bx + branch_w / 2, mid - box_h / 2 + 24, 17, 700, "#1d1b33"))
            nodes.append(f'<text x="{bx + branch_w / 2:.0f}" y="{mid + box_h / 2 - 9:.0f}" font-size="13" font-weight="700" '
                         f'fill="{color}" text-anchor="middle">p. {b.page_number}</text>')
            # leaves
            leaves_total = sum(leaf_height(p) for p in b.points) + 8 * (len(b.points) - 1)
            ly = mid - leaves_total / 2
            for p in b.points:
                lh = leaf_height(p)
                lmid = ly + lh / 2
                fx = bx + branch_w if sign > 0 else bx
                tx = lx if sign > 0 else lx + leaf_w
                parts.append(f'<path d="M{fx:.0f} {mid:.0f} C{(fx + tx) / 2:.0f} {mid:.0f}, {(fx + tx) / 2:.0f} {lmid:.0f}, '
                             f'{tx:.0f} {lmid:.0f}" stroke="{color}" stroke-width="1.6" fill="none" opacity="0.5"/>')
                nodes.append(f'<rect x="{lx:.0f}" y="{ly:.0f}" width="{leaf_w}" height="{lh:.0f}" rx="10" fill="#fff" '
                             f'stroke="{color}" stroke-opacity="0.45"/>')
                nodes.append(_text(_wrap(p, 25), lx + 12, ly + 23, 15.5, 500, "#2b2950", anchor="start"))
                ly += lh + 8
            y += bh + 26

    t_lines = _wrap(mm.title, 16)
    ch = 36 + len(t_lines) * 26
    nodes.append(f'<rect x="{cx - center_w / 2:.0f}" y="{cy - ch / 2:.0f}" width="{center_w}" height="{ch:.0f}" rx="22" '
                 f'fill="url(#nwmm)"/>')
    nodes.append(_text(t_lines, cx, cy - ch / 2 + 33, 21, 800, "#ffffff"))
    return (f'<svg class="nw-mindmap" viewBox="0 0 {W} {H:.0f}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="Plus Jakarta Sans, -apple-system, Arial, sans-serif" role="img" '
            f'aria-label="Mind map: {html.escape(mm.title)}">'
            '<defs><linearGradient id="nwmm" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#8B6CFF"/>'
            '<stop offset="1" stop-color="#5B3BF0"/></linearGradient></defs>'
            + "".join(parts) + "".join(nodes) + "</svg>")


def as_outline(mm: MindMap) -> str:
    """Plain-text version for download/printing."""
    lines = [mm.title, ""]
    for b in mm.branches:
        lines.append(f"{b.topic}  (p. {b.page_number})")
        lines += [f"  - {p}" for p in b.points]
        lines.append("")
    return "\n".join(lines)


def doc_key(doc_ids: list[str]) -> str:
    return ",".join(sorted(doc_ids))


def dumps(mm: MindMap) -> str:
    return json.dumps(mm.to_dict())
