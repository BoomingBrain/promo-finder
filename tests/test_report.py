import csv
import json
from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.meta import RunMeta
from promofinder.models import Candidate, Segment, Promo
from promofinder.invariants import check_promo
from promofinder.report import write_outputs, segments_str

CFG = load_config(Path("config"))
READ = load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False)
EPS = {e.ep: e for e in READ.episodes}


def P(start, end, hook="E01-B003", cliff="E01-B008", segs=None, kind="single"):
    c = Candidate(cand_id=f"S-{start}", kind=kind, segments=segs or [Segment(ep=1, start_s=start, end_s=end)], hook_beat_id=hook,
                  cliff_beat_id=cliff, hook_score=9, cliff_score=8, story_importance=5, rule_score=5, length_fit=10, penalties=0,
                  window_score=8, scene_key="E01-S01")
    return Promo(rank=1, candidate=c, judgement=None, final_score=7.0)


def test_invariants():
    assert check_promo(P(4.0, 30.0), EPS, {}, CFG.scoring) == []
    v = check_promo(P(4.0, 22.5), EPS, {}, CFG.scoring)
    assert "mid_line_cut" in v and "not_on_boundary" in v and "cliff_cut" in v


def test_stitched_invariants():
    ok = P(0, 0, segs=[Segment(ep=1, start_s=4.0, end_s=20.0), Segment(ep=1, start_s=42.0, end_s=52.0)], cliff="E01-B012", kind="stitched")
    assert check_promo(ok, EPS, {}, CFG.scoring) == []
    bad = P(0, 0, segs=[Segment(ep=1, start_s=42.0, end_s=52.0), Segment(ep=1, start_s=4.0, end_s=6.0)], kind="stitched")
    assert {"not_chronological", "too_short_segment", "total_duration"} <= set(check_promo(bad, EPS, {}, CFG.scoring))


def test_outputs(tmp_path):
    write_outputs(tmp_path, [P(4.0, 30.0)], EPS, {}, READ, RunMeta(), CFG)
    row = next(csv.DictReader(open(tmp_path / "promos.csv", encoding="utf-8")))
    assert row["segments"] == "E01 00:04.00–00:30.00" and row["editor_verdict"] == "" and row["rule_score"] == "5.0"
    assert json.loads((tmp_path / "promos.json").read_text())[0]["segments_ts"][0]["start"] == "00:04.00"
    md = (tmp_path / "report.md").read_text()
    assert "Invariant pass rate: 100%" in md
    assert "prompt_versions" in json.loads((tmp_path / "run_meta.json").read_text())["values"]
