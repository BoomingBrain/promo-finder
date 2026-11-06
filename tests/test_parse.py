from pathlib import Path
import pytest
from promofinder.parse import load_show, show_slug, ParseHealthError
from promofinder.readers.base import ReaderError

A = Path("tests/fixtures/show_a")


def test_reads_both_formats_and_ignores_non_txt():
    r = load_show(A, "kuku_v1", 0.05, False)
    assert [e.ep for e in r.episodes] == [1, 2]
    e1 = r.episodes[0]
    assert e1.duration_s == 55.0 and len(e1.beats) == 13 and len(e1.scenes) == 2
    b4 = e1.beats[3]
    assert (b4.beat_id, b4.type, b4.speaker, b4.emotion_tags, b4.start_s) == ("E01-B004", "DIALOGUE", "VICTOR", ["Furious"], 6.5)
    assert e1.beats[11].type == "SFX" and e1.beats[11].scene == 2
    s = r.episodes[1].scenes[0]
    assert (s.heading, s.end_s, r.episodes[1].duration_s) == ("EXT. GARDEN – DAY", 15.0, 15.0)


def test_fail_loud_and_force(tmp_path):
    bad = tmp_path / "show"; bad.mkdir()
    txt = (A / "episode-002-Episode 9002.txt").read_text().replace("Start: (00:02.00) | End: (00:04.00)\n", "")
    (bad / "episode-002.txt").write_text(txt)
    with pytest.raises(ParseHealthError) as ei:
        load_show(bad, "kuku_v1", 0.05, False)
    assert "episode-002.txt" in ei.value.report
    r = load_show(bad, "kuku_v1", 0.05, True)
    assert r.health[0].n_missing_ts == 1


def test_episode_number_fallback_and_error(tmp_path):
    d = tmp_path / "s"; d.mkdir()
    body = (A / "episode-002-Episode 9002.txt").read_text().replace("EPISODE NUMBER: Episode 2\n", "")
    (d / "ep7.txt").write_text(body)
    assert load_show(d, "kuku_v1", 0.05, False).episodes[0].ep == 7
    (d / "intro.txt").write_text(body)
    with pytest.raises(ReaderError, match="intro.txt"):
        load_show(d, "kuku_v1", 0.05, False)


def test_slug():
    assert show_slug(Path("My Show. Part Two (Final)")) == "my-show-part-two-final"
