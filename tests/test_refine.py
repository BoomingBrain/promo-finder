from pathlib import Path
from promofinder.config import load_config
from promofinder.meta import RunMeta
from promofinder.models import Candidate, CliffTag, EpisodeTags, HookTag, Promo, Segment
from promofinder.parse import load_show
from promofinder.refine import options, refine_promo
from promofinder.windows import check_clip_rules
from tests.fakes import FakeLLM

CFG = load_config(Path("config"))
EPS = {e.ep: e for e in load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes}
TAGS = {1: EpisodeTags(ep=1, hooks=[HookTag(beat_id="E01-B003", hook_score=9, story_importance=5, kind="slap", why="w", quote="q")],
                       cliffhangers=[CliffTag(beat_id="E01-B008", cliff_score=8, story_importance=5, open_question="?",
                                              resolved_by=None, spoiler_level="none", quote="q")])}


def cand():
    return Candidate(cand_id="S-E01-B003-E01-B008", kind="single", segments=[Segment(ep=1, start_s=4.0, end_s=30.0)],
                     hook_beat_id="E01-B003", cliff_beat_id="E01-B008", hook_score=9, cliff_score=8, story_importance=5,
                     rule_score=1, length_fit=10, penalties=0, window_score=8, scene_key="E01-S01")


def promo():
    return Promo(rank=1, candidate=cand(), final_score=8)


def test_options_original_first_and_all_valid():
    ops = options(cand(), EPS[1], TAGS[1], CFG.scoring)
    assert ops[0] == Segment(ep=1, start_s=4.0, end_s=30.0) and len(ops) <= 6
    by_id = {b.beat_id: b for b in EPS[1].beats}
    for o in ops:
        assert check_clip_rules(EPS[1], o.start_s, o.end_s, by_id["E01-B003"], by_id["E01-B008"], None, CFG.scoring.window) == []


def test_choice_b_replaces_segments():
    ops = options(cand(), EPS[1], TAGS[1], CFG.scoring)
    assert len(ops) >= 2
    p = refine_promo(promo(), EPS, TAGS, FakeLLM([{"choice": "B", "why": "w"}]), CFG, RunMeta())
    assert p.original_segments == [Segment(ep=1, start_s=4.0, end_s=30.0)] and p.candidate.segments == [ops[1]]


def test_invalid_letter_keeps_original():
    meta = RunMeta()
    p = refine_promo(promo(), EPS, TAGS, FakeLLM([{"choice": "Z", "why": "w"}]), CFG, meta)
    assert p.candidate.segments == [Segment(ep=1, start_s=4.0, end_s=30.0)] and p.original_segments is None
    assert meta.counters["invalid_refine_picks"] == 1
