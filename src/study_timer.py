"""Study timer: the student picks how long to study (25 minutes to 5 hours).

Python keeps the state (end time, paused time); a small HTML/JS clock in the
sidebar counts down every second in the browser, so the page doesn't need to
rerun. Optional break reminder every 50 minutes.
"""

from __future__ import annotations

import json
import time

PRESETS = {"25 minutes": 25, "1 hour": 60, "2 hours": 120, "3 hours": 180, "5 hours": 300}
BREAK_EVERY_MIN = 50


def start(minutes: int, breaks: bool, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    total = int(minutes) * 60
    return {"total": total, "end": now + total, "paused_left": None, "breaks": bool(breaks)}


def remaining(state: dict, now: float | None = None) -> int:
    now = time.time() if now is None else now
    if state.get("paused_left") is not None:
        return max(0, int(state["paused_left"]))
    return max(0, int(round(state["end"] - now)))


def pause(state: dict, now: float | None = None) -> dict:
    if state.get("paused_left") is None:
        state = {**state, "paused_left": remaining(state, now)}
    return state


def resume(state: dict, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    if state.get("paused_left") is not None:
        state = {**state, "end": now + state["paused_left"], "paused_left": None}
    return state


def fmt(seconds: int) -> str:
    h, rest = divmod(max(0, int(seconds)), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def clock_html(state: dict, dark: bool = False) -> str:
    """Self-contained countdown. Only stable values go in, so Streamlit doesn't
    re-create the clock on every rerun."""
    cfg = json.dumps({
        "end": state["end"] * 1000,
        "total": state["total"],
        "paused": state.get("paused_left"),
        "breaks": state.get("breaks", False),
        "breakEvery": BREAK_EVERY_MIN * 60,
    })
    bg, card, ink, muted, line = (("transparent", "#1C1934", "#ECE9FF", "#A9A4CC", "#34305A") if dark
                                  else ("transparent", "#FFFFFF", "#17143A", "#6B6890", "#E6E0FF"))
    return f"""<!doctype html><html><head><style>
body {{ margin:0; background:{bg}; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
.c {{ background:{card}; border:1px solid {line}; border-radius:18px; padding:12px 14px; display:flex; gap:14px;
  align-items:center; box-shadow:0 6px 20px rgba(76,52,190,.10); }}
.t {{ font-size:28px; font-weight:800; color:{ink}; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
.s {{ font-size:12px; color:{muted}; margin-top:2px; }}
.s.brk {{ color:#12A06C; font-weight:700; }}
.s.done {{ color:#7B5CFF; font-weight:800; animation:p 1.2s ease-in-out infinite; }}
@keyframes p {{ 50% {{ opacity:.55; }} }}
body {{ padding:2px 4px 10px; }}
</style></head><body>
<div class="c"><svg width="64" height="64" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="27" fill="none" stroke="{line}" stroke-width="7"/>
<circle id="ring" cx="32" cy="32" r="27" fill="none" stroke="url(#g)" stroke-width="7" stroke-linecap="round"
 transform="rotate(-90 32 32)" stroke-dasharray="169.6" stroke-dashoffset="0"/>
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#9B7BFF"/><stop offset="1" stop-color="#5B3BF0"/></linearGradient></defs>
<text id="pct" x="32" y="37" text-anchor="middle" font-size="13" font-weight="700" fill="{ink}">0%</text></svg>
<div><div class="t" id="t">--:--</div><div class="s" id="s">Focus time left</div></div></div>
<script>
const C = {cfg};
let beeped = false;
function fmt(x) {{ x = Math.max(0, Math.round(x)); const h = Math.floor(x/3600), m = Math.floor(x%3600/60), s = x%60;
  return (h ? h + ':' + String(m).padStart(2,'0') : String(m).padStart(2,'0')) + ':' + String(s).padStart(2,'0'); }}
function beep() {{ try {{ const a = new (window.AudioContext||window.webkitAudioContext)(); [0,0.35,0.7].forEach(d => {{
  const o = a.createOscillator(), g = a.createGain(); o.frequency.value = 880; o.connect(g); g.connect(a.destination);
  g.gain.setValueAtTime(0.15, a.currentTime + d); g.gain.exponentialRampToValueAtTime(0.001, a.currentTime + d + 0.3);
  o.start(a.currentTime + d); o.stop(a.currentTime + d + 0.3); }}); }} catch (e) {{}} }}
function tick() {{
  const left = C.paused !== null ? C.paused : (C.end - Date.now()) / 1000;
  const done = Math.max(0, C.total - left);
  document.getElementById('t').textContent = fmt(left);
  const sub = document.getElementById('s');
  const frac = Math.min(1, done / C.total);
  document.getElementById('ring').setAttribute('stroke-dashoffset', (169.6 * frac).toFixed(1));
  document.getElementById('pct').textContent = Math.round(frac * 100) + '%';
  const onBreak = C.breaks && left > 0 && C.paused === null && done >= C.breakEvery && (done % C.breakEvery) < 60;
  if (left <= 0) {{ sub.textContent = "Time's up! Great session 🎉"; sub.className = 's done'; }}
  else if (C.paused !== null) {{ sub.textContent = 'Paused'; sub.className = 's'; }}
  else if (onBreak) {{ sub.textContent = 'Take a 10-min break 🌿'; sub.className = 's brk'; }}
  else {{ sub.textContent = 'Focus time left'; sub.className = 's'; }}
  if (left <= 0 && !beeped) {{ beeped = true; beep(); }}
}}
tick(); setInterval(tick, 1000);
</script></body></html>"""
