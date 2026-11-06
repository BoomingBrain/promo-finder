from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.signals import compute_signals, is_visual


def _ep1():
    return load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes[0]


def test_signal_hits():
    s = compute_signals(_ep1(), load_config(Path("config")).signals)
    assert "kw:slap" in s["E01-B003"].hints
    assert "emo:furious" in s["E01-B004"].hints and "exclamation" in s["E01-B004"].hints
    assert {"sfx:swell", "kw:cliffhanger", "ep_end"} <= set(s["E01-B012"].hints)
    assert 0 <= s["E01-B012"].score <= 10 and s["E01-B012"].score > s["E01-B001"].score


def test_visual_filter():
    cfg = load_config(Path("config")).signals
    b = _ep1().beats
    assert is_visual(b[1], cfg)        # "Close-up on Mara..."
    assert not is_visual(b[0], cfg)    # "Wide shot..."
    assert is_visual(b[3], cfg)        # dialogue always kept
