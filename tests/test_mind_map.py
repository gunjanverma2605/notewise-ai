import json

import pytest

from src.errors import NoteWiseError
from src.mind_map import MindMap, as_outline, generate_mind_map, render_svg, validate
from src.retrieval import RetrievedChunk
from src.text_chunker import Chunk

TEXTS = ["Chlorophyll absorbs red and blue light and reflects green light.",
         "The Calvin cycle takes place in the stroma where RuBisCO fixes carbon dioxide.",
         "Light reactions split water in the thylakoid and release oxygen gas."]
CHUNKS = [Chunk(f"c{i}", "d1", "bio.pdf", i + 1, 0, t) for i, t in enumerate(TEXTS)]
EV = [RetrievedChunk(c, 1.0) for c in CHUNKS]


def test_validate_keeps_only_supported_points_and_maps_pages():
    data = {"title": "Photosynthesis", "branches": [
        {"topic": "Chlorophyll", "source": "S1", "points": ["Absorbs red and blue light", "Invented by Newton in 1700"]},
        {"topic": "Calvin cycle", "source": "[S2]", "points": ["Takes place in the stroma"]},
        {"topic": "Ghost topic", "source": "S9", "points": ["Volcanoes erupt lava"]},
        {"topic": "Light reactions", "source": "S7", "points": ["Water is split in the thylakoid"]}]}
    mm = validate(data, EV)
    assert [b.topic for b in mm.branches] == ["Chlorophyll", "Calvin cycle", "Light reactions"]
    assert mm.branches[0].points == ["Absorbs red and blue light"] and mm.branches[0].page_number == 1
    assert mm.branches[2].page_number == 3             # bad source -> found the passage that supports it


def test_validate_rejects_maps_with_too_little_support():
    with pytest.raises(Exception):
        validate({"branches": [{"topic": "X", "source": "S1", "points": ["Volcanoes erupt lava"]}]}, EV)


class FakeGemma:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, json_mode=False, temperature=None):
        return self.replies.pop(0)


def test_generate_retries_then_fails_cleanly():
    good = json.dumps({"title": "Bio", "branches": [
        {"topic": "Chlorophyll", "source": "S1", "points": ["absorbs red and blue light"]},
        {"topic": "Calvin cycle", "source": "S2", "points": ["RuBisCO fixes carbon dioxide"]}]})
    mm = generate_mind_map(CHUNKS, FakeGemma(["not json", good]))
    assert len(mm.branches) == 2
    with pytest.raises(NoteWiseError):
        generate_mind_map(CHUNKS, FakeGemma(["nope", "nope"]))


def test_render_escapes_text_and_round_trips():
    mm = MindMap("A <b> & C", [])
    mm = MindMap.from_dict(validate({"title": "A <b> & C", "branches": [
        {"topic": "Chlorophyll <x>", "source": "S1", "points": ["absorbs red and blue light"]},
        {"topic": "Calvin", "source": "S2", "points": ["RuBisCO fixes carbon dioxide"]}]}, EV).to_dict())
    svg = render_svg(mm)
    assert svg.startswith("<svg") and "&lt;x&gt;" in svg and "<x>" not in svg and "p. 2" in svg
    assert "Chlorophyll <x>  (p. 1)" in as_outline(mm)


def test_book_chapters_fall_back_to_key_points():
    from src.book import generate_book, render_body, render_download
    mm = validate({"title": "Bio", "branches": [
        {"topic": "Chlorophyll", "source": "S1", "points": ["absorbs red and blue light"]},
        {"topic": "Calvin cycle", "source": "S2", "points": ["RuBisCO fixes carbon dioxide"]}]}, EV)

    class G:
        def __init__(self):
            self.n = 0

        def chat(self, messages, json_mode=False, temperature=None):
            self.n += 1
            return "Chlorophyll absorbs red and blue light [S1]." if self.n == 1 else "No citation here."

    by_page = {c.page_number: [c] for c in CHUNKS}
    bk = generate_book(mm, lambda name, page: by_page.get(page, []), G())
    assert bk.chapters[0].grounded and "(p. 1)" in bk.chapters[0].text
    assert not bk.chapters[1].grounded and bk.chapters[1].points == ["RuBisCO fixes carbon dioxide"]
    body = render_body(bk)
    assert "Chapter 2" in body and 'class="bk-page">p. 1' in body
    assert render_download(bk).startswith("<!doctype html>")
