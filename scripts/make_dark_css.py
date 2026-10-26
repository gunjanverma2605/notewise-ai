"""Builds assets/dark.css from assets/style.css: every rule that paints a light
background or border gets a dark-theme override. Run after editing style.css:
    python scripts/make_dark_css.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "assets" / "style.css").read_text()

LIGHT = {  # light colour -> dark colour
    "#fff": "#1C1934", "#ffffff": "#1C1934", "#fbfaff": "#121027", "#faf8ff": "#1A1731", "#f3efff": "#1A1731",
    "#f4f1ff": "#25204A", "#f3eeff": "#1F1B3D", "#e9e2ff": "#2A2452", "#eee9ff": "#2E2758", "#eee8ff": "#2E2758",
    "#f6f2ff": "#211D40", "#fbf9ff": "#1E1A38", "#f7f5ff": "#1E1A38", "#f7f4ff": "#211D40", "#efeefa": "#25204A",
    "#eff0f6": "#2A2747", "#f1eefa": "#2A2550", "#fffdf8": "#1C1934", "#e6e0ff": "#34305A", "#e6e1fa": "#34305A",
    "#d6ccff": "#4A4180", "#dee0ea": "#3A3760", "#c9ebda": "#1F5A43", "#f6ddaa": "#6B5222", "#e6f6ee": "#143A2D",
    "#fff3db": "#3D3016", "#fdecef": "#45202A", "#dcd3ff": "#3E3670", "#cfc2ff": "#5A4FA0", "#d8cdff": "#5A4FA0",
    "#b9a5ff": "#7A66E0", "#eeeaff": "#2E2758",
}
PROPS = ("background", "background-color", "border", "border-color", "border-left", "border-top", "border-bottom",
         "border-right", "fill")
HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")


def blocks(css):
    """Yield (prelude, body) for top-level blocks, recursing one level into @media."""
    i, n = 0, len(css)
    while i < n:
        j = css.find("{", i)
        if j < 0:
            return
        prelude = css[i:j].strip()
        depth, k = 1, j + 1
        while depth and k < n:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        yield prelude, css[j + 1:k - 1]
        i = k


def lum(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def dark_bg(m):
    h = m.group(0)
    if h.lower() in LIGHT:
        return LIGHT[h.lower()]
    return "#241F44" if lum(h) > 0.82 else h


def dark_border(m):
    h = m.group(0)
    if h.lower() in LIGHT:
        return LIGHT[h.lower()]
    return "#3A3466" if lum(h) > 0.75 else h


RGBA_WHITE = re.compile(r"rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*([0-9.]+)\s*\)")


def convert(body):
    out = []
    for decl in body.split(";"):
        if ":" not in decl:
            continue
        prop, val = decl.split(":", 1)
        prop = prop.strip()
        val = val.replace("!important", "").strip()
        if prop in PROPS:
            fn = dark_border if prop.startswith("border") else dark_bg
            new = HEX.sub(fn, val)
            new = RGBA_WHITE.sub(lambda m: f"rgba(28, 25, 52, {m.group(1)})", new)
            if new != val:
                out.append(f"{prop}: {new} !important")
        elif prop == "color":
            hexes = HEX.findall(val)
            if hexes and lum(hexes[0]) < 0.3:                 # dark text -> light text
                out.append("color: #E6E1FF !important")
    return out


rules = []
for prelude, body in blocks(re.sub(r"/\*.*?\*/", "", SRC, flags=re.S)):
    prelude = re.sub(r"\s+", " ", prelude)
    if prelude.startswith("@font-face") or prelude.startswith("@keyframes"):
        continue
    if prelude.startswith("@media"):
        inner = []
        for p2, b2 in blocks(body):
            d = convert(b2)
            if d:
                inner.append(f"  {re.sub(chr(10), ' ', p2)} {{ {'; '.join(d)}; }}")
        if inner:
            rules.append(prelude + " {\n" + "\n".join(inner) + "\n}")
        continue
    d = convert(body)
    if d:
        rules.append(f"{prelude} {{ {'; '.join(d)}; }}")

MANUAL = """
/* ---- base dark theme (hand-written) ---- */
:root { --nw-ink: #ECE9FF; --nw-muted: #A9A4CC; --nw-line: #34305A; --nw-lavender: #25204A; --nw-purple-soft: #2E2758;
  --nw-purple-dark: #B3A1FF; --nw-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
  --nw-green: #5FD3A6; --nw-green-bg: #143A2D; --nw-amber: #F3C266; --nw-amber-bg: #3D3016; --nw-red: #FF8A9C; --nw-red-bg: #45202A; }
.stApp { background: #121027 !important; color: var(--nw-ink); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1A1636 0%, #15122C 100%) !important; }
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"] p { color: var(--nw-ink); }
.stApp [data-testid="stCaptionContainer"], .stApp small { color: var(--nw-muted) !important; }
.nw-hero2, .nw-land-hero { background: linear-gradient(120deg, #1A1734 0%, #221C46 60%, #2B2357 100%) !important; }
.stApp .stButton button, .stApp .stDownloadButton button { background: #221E3F; color: var(--nw-ink); border-color: #3A3466; }
.stApp .stButton button[kind="primary"] { background: linear-gradient(135deg, #7B5CFF 0%, #5B3BF0 100%) !important; color: #fff !important; }
.stApp .stButton button:disabled { background: #2A2550 !important; color: #CFC8FF !important; }
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"], .stApp [data-baseweb="input"] input,
.stApp textarea, .stApp [data-baseweb="textarea"] { background: #1C1934 !important; color: var(--nw-ink) !important;
  border-color: #3A3466 !important; }
.stApp [data-baseweb="popover"] ul, .stApp [data-baseweb="menu"] { background: #1C1934 !important; }
.stApp [data-baseweb="menu"] li { color: var(--nw-ink) !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] textarea { background: #1C1934 !important; color: var(--nw-ink) !important; }
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div { background: #121027 !important; }
.stApp [data-testid="stExpander"] details { background: #1C1934 !important; border-color: #34305A !important; }
.stApp [data-testid="stExpander"] summary:hover { background: #241F44 !important; }
.stApp [data-testid="stFileUploaderDropzone"] { background: #1C1934 !important; }
.stApp [data-testid="stAlertContainer"] { background: #241F44 !important; color: var(--nw-ink) !important; }
.stApp [data-testid="stButtonGroup"] button { background: #221E3F; color: var(--nw-ink); border-color: #3A3466; }
.stApp [data-testid="stButtonGroup"] button[kind$="Active"] { background: #3A2F7A !important; color: #fff !important; }
.stApp [data-testid="stChatMessage"] { background: transparent; }
.stApp code { background: #2A2550; color: #E6E0FF; }
.nw-page, .bk-page { background: #3A2F7A !important; color: #E6E0FF !important; }
.nw-user-msg, [data-testid="stChatMessage"]:has(.nw-user-msg) [data-testid="stChatMessageContent"] { color: var(--nw-ink); }
.nw-footer { color: var(--nw-muted) !important; }
/* keep "paper" exhibits light so their dark text stays readable */
.nw-mm-wrap { background: #F7F5FF !important; }
.nw-book, .nw-book * { --x: 0; }
.nw-book .bk { background: #FFFDF8 !important; }
.nw-book .bk p, .nw-book .bk li, .nw-book .bk span, .nw-book .bk h1, .nw-book .bk h2, .nw-book .bk h3 { color: inherit; }
.nw-book .bk-cover h1, .nw-book .bk-chapter h2 { color: #1d1840 !important; }
.nw-book .bk-text p, .nw-book .bk-text li, .nw-book .bk-keys li, .nw-book .bk-toc li span { color: #2a2540 !important; }
.nw-book .bk-sub, .nw-book .bk-src { color: #6b6890 !important; }
.nw-book .bk-keys b, .nw-book .bk-toc h3, .nw-book .bk-kicker, .nw-book .bk-num { color: #6D4AFF !important; }
.nw-book .bk-note { color: #4b4670 !important; }
.nw-book .bk-toc, .nw-book .bk-chapter { background: #FFFDF8 !important; }
.nw-book .bk-cover { background: linear-gradient(160deg, #F4F0FF 0%, #FFFDF8 70%) !important; }
.nw-book .bk-keys { background: #F7F4FF !important; }
.nw-book .bk-page { background: #EEE9FF !important; color: #4B2FD6 !important; }
.nw-side-quote { color: #3a2f7a !important; background: #fff !important; }
.stApp .nw-book .bk h1, .stApp .nw-book .bk h2 { color: #1d1840 !important; }
.stApp .nw-book .bk h3 { color: #6D4AFF !important; }
.stApp [data-testid="stFileUploaderDropzone"] button { background: #221E3F !important; color: var(--nw-ink) !important;
  border-color: #3A3466 !important; }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] div { background: #1C1934 !important; border-color: #3A3466 !important; }
.nw-feature { background: #221E3F !important; }
.stApp textarea::placeholder, .stApp input::placeholder { color: #8F89B8 !important; }
"""

out = "/* AUTO-GENERATED by scripts/make_dark_css.py from style.css, then the hand-written base below. */\n" \
      + "\n".join(rules) + "\n" + MANUAL
(ROOT / "assets" / "dark.css").write_text(out)
print(len(rules), "rules")
