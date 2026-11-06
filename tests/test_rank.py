from pathlib import Path
from promofinder.config import load_config
from promofinder.models import Candidate, Segment, Judgement, Promo, Bible, Act
from promofinder.rank import final_score, diversify, finalists

CFG = load_config(Path("config")).scoring


def cand(i, ep=1, scene="E01-S01", h=8, c=8):
    return Candidate(cand_id=f"S-{i}", kind="single", segments=[Segment(ep=ep, start_s=0, end_s=30)], hook_beat_id="h", cliff_beat_id="c",
                     hook_score=h, cliff_score=c, story_importance=5, rule_score=5, length_fit=10, penalties=0, window_score=8, scene_key=scene)


J = Judgement(cand_id="x", standalone=8, confusing_bits=[], hook_text="", cliffhanger_text="", why_it_works="",
              spoiler_risk=False, resolution_leak=True, hook3s=3)


def test_final_score_flags():
    s, flags = final_score(cand(1, ep=20), J, "major", CFG)
    base = .30 * 8 + .30 * 8 + .25 * 8 + .05 * 5 + .10 * 5
    assert abs(s - (base - 1.5 - 3.0 - 1.0)) < 1e-9 and set(flags) == {"spoiler_major", "resolution_leak", "hook3s_low"}
    s2, f2 = final_score(cand(2, ep=3), None, "none", CFG)
    assert "early_episode" in f2 and "unjudged" in f2


def test_diversify_caps_per_scene():
    ps = [Promo(rank=0, candidate=cand(i), judgement=None, final_score=10 - i) for i in range(4)]
    ps.append(Promo(rank=0, candidate=cand(9, scene="E02-S01", ep=2), judgement=None, final_score=1))
    out = diversify(ps, None, CFG)
    assert [p.candidate.cand_id for p in out][:3] == ["S-0", "S-1", "S-9"] and [p.rank for p in out] == [1, 2, 3, 4, 5]


def test_diversify_caps_per_act():
    bible = Bible(premise="p", characters={}, arc=[Act(name="A1", ep_start=1, ep_end=9, summary=""),
                                                    Act(name="A2", ep_start=10, ep_end=20, summary="")],
                  episode_summaries={}, major_reveals=[])
    ps = [Promo(rank=0, candidate=cand(i, ep=1 + i, scene=f"E{1+i:02d}-S01"), judgement=None, final_score=10 - i) for i in range(4)]
    ps.append(Promo(rank=0, candidate=cand(9, ep=12, scene="E12-S01"), judgement=None, final_score=1))
    out = diversify(ps, bible, CFG)
    assert [p.candidate.cand_id for p in out][:4] == ["S-0", "S-1", "S-2", "S-9"]


def test_finalists_split():
    ps = [Promo(rank=i + 1, candidate=cand(i), judgement=None, final_score=1) for i in range(5)]
    a, b = finalists(ps, 3)
    assert len(a) == 3 and len(b) == 2
