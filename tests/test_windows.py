from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.signals import compute_signals
from promofinder.models import EpisodeTags, HookTag, CliffTag
from promofinder.windows import snap_start, snap_end, length_fit, build_candidates, check_clip_rules

CFG = load_config(Path("config"))
SHOW = load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False)
E1, E2 = SHOW.episodes
B = {b.beat_id: b for b in E1.beats}


def H(b, s=9):
    return HookTag(beat_id=b, hook_score=s, story_importance=5, kind="slap", why="w", quote="q")


def C(b, s=9, res=None):
    return CliffTag(beat_id=b, cliff_score=s, story_importance=5, open_question="?", resolved_by=res, spoiler_level="none", quote="q")


def test_snapping():
    assert snap_start(E1, B["E01-B003"], 3.0) == 4.0
    assert snap_end(E1, B["E01-B012"], None) == 52.0
    assert snap_end(E1, B["E01-B008"], None) == 30.0


def test_snap_end_stops_before_resolution():
    # shot containing B010's end (35.0) runs to 42.0, past a resolution starting at 40.0 -> cut at the cliff line's end
    assert snap_end(E1, B["E01-B010"], resolved_start=40.0) == 35.0
    assert snap_end(E1, B["E01-B010"], resolved_start=None) == 42.0


def test_length_fit():
    w = CFG.scoring.window
    assert length_fit(35, w) == 10 and length_fit(20, w) == 5 and length_fit(60, w) == 5 and length_fit(61, w) == 0


def test_clip_rules():
    w = CFG.scoring.window
    assert check_clip_rules(E1, 4.0, 30.0, B["E01-B003"], B["E01-B008"], None, w) == []
    v = check_clip_rules(E1, 0.0, 22.5, B["E01-B003"], B["E01-B008"], 10.0, w)
    assert {"hook_lead", "mid_line_cut", "resolution_inside", "cliff_cut"} <= set(v)


def test_build_keeps_valid_rejects_short_and_dedupes():
    tags = EpisodeTags(ep=1, hooks=[H("E01-B003")], cliffhangers=[C("E01-B006"), C("E01-B008"), C("E01-B012")], context_beats=[])
    cands = build_candidates(E1, tags, compute_signals(E1, CFG.signals), CFG.scoring)
    ids = {c.cand_id for c in cands}
    assert "S-E01-B003-E01-B006" not in ids                       # 16 s, too short
    assert ids <= {"S-E01-B003-E01-B008", "S-E01-B003-E01-B012"} and ids
    assert len(cands) == 1                                          # the two overlap >50% -> NMS keeps one
    for c in cands:
        assert 20 <= c.duration_s <= 60 and c.segments[0].start_s == 4.0 and c.scene_key == "E01-S01"


def test_low_hook_score_ignored():
    tags = EpisodeTags(ep=1, hooks=[H("E01-B003", s=4)], cliffhangers=[C("E01-B012")], context_beats=[])
    assert build_candidates(E1, tags, compute_signals(E1, CFG.signals), CFG.scoring) == []


def test_short_episode_yields_nothing():
    tags = EpisodeTags(ep=2, hooks=[H("E02-B002")], cliffhangers=[C("E02-B003")], context_beats=[])
    assert build_candidates(E2, tags, compute_signals(E2, CFG.signals), CFG.scoring) == []
