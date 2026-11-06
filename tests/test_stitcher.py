from pathlib import Path
from promofinder.config import load_config
from promofinder.meta import RunMeta
from promofinder.models import CliffTag, EpisodeTags, HookTag, Segment
from promofinder.parse import load_show
from promofinder.stitcher import stitch, validate_combo
from tests.fakes import FakeLLM

CFG = load_config(Path("config"))
EPS = {e.ep: e for e in load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes}
S = lambda ep, a, b: Segment(ep=ep, start_s=a, end_s=b)


def test_validate_combo_rules():
    assert validate_combo([S(1, 4, 12), S(1, 44, 52)], ["hook", "cliff"], CFG) == ["total_duration"]
    assert "not_chronological" in validate_combo([S(1, 42, 52), S(1, 4, 20)], ["hook", "cliff"], CFG)
    assert "too_short_segment" in validate_combo([S(1, 4, 7), S(1, 30, 52)], ["hook", "cliff"], CFG)
    assert "bad_roles" in validate_combo([S(1, 4, 20), S(1, 42, 52)], ["cliff", "hook"], CFG)
    assert validate_combo([S(1, 4, 20), S(1, 42, 52)], ["hook", "cliff"], CFG) == []


def test_stitch_keeps_only_valid_combo():
    tags = {1: EpisodeTags(ep=1, hooks=[HookTag(beat_id="E01-B003", hook_score=9, story_importance=6)],
                           cliffhangers=[CliffTag(beat_id="E01-B012", cliff_score=8, story_importance=7)])}
    reply = {"combos": [
        {"segments": [{"beat_id": "E01-B003", "role": "hook", "after_s": 10}, {"beat_id": "E01-B012", "role": "cliff"}], "why": "ok"},
        {"segments": [{"beat_id": "E01-B012", "role": "cliff"}, {"beat_id": "E01-B003", "role": "hook"}], "why": "bad"}]}
    meta = RunMeta()
    out = stitch(EPS, tags, None, FakeLLM([reply]), CFG, meta)
    assert [c.cand_id for c in out] == ["M-001"]
    assert out[0].kind == "stitched" and out[0].segments == [S(1, 4.0, 20.0), S(1, 42.0, 52.0)]
    assert any("stitch" in w for w in meta.warnings)


def test_stitch_empty_combo_is_dropped_not_crash():
    tags = {1: EpisodeTags(ep=1, hooks=[HookTag(beat_id="E01-B003", hook_score=9, story_importance=6)],
                           cliffhangers=[CliffTag(beat_id="E01-B012", cliff_score=8, story_importance=7)])}
    assert stitch(EPS, tags, None, FakeLLM([{"combos": [{"segments": [], "why": "x"}]}]), CFG, RunMeta()) == []
