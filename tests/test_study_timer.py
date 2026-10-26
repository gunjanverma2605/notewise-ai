from src import study_timer as t


def test_start_pause_resume_and_format():
    s = t.start(60, True, now=1000.0)
    assert t.remaining(s, now=1000.0) == 3600 and s["breaks"]
    s = t.pause(s, now=1600.0)
    assert t.remaining(s, now=9999.0) == 3000              # paused: time doesn't move
    s = t.resume(s, now=2000.0)
    assert t.remaining(s, now=2100.0) == 2900
    assert t.remaining(s, now=99999.0) == 0
    assert t.fmt(3600 * 5) == "5:00:00" and t.fmt(25 * 60) == "25:00"


def test_clock_html_is_stable_and_self_contained():
    s = t.start(25, False, now=1000.0)
    a, b = t.clock_html(s), t.clock_html(s)
    assert a == b and "<script>" in a and '"total": 1500' in a
    assert "#1C1934" in t.clock_html(s, dark=True)
