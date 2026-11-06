from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.meta import RunMeta
from promofinder.llm import LLMStageError
from promofinder.models import Candidate, Segment
from promofinder.cliptext import clip_text
from promofinder.judge import judge_all
from tests.fakes import FakeLLM

CFG = load_config(Path("config"))
EPS = {e.ep: e for e in load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes}
CAND = Candidate(cand_id="S-x", kind="single", segments=[Segment(ep=1, start_s=4.0, end_s=30.0)], hook_beat_id="E01-B003",
                 cliff_beat_id="E01-B008", hook_score=9, cliff_score=8, story_importance=5, rule_score=5, length_fit=10,
                 penalties=0, window_score=8, scene_key="E01-S01")
JO = {"standalone": 8, "confusing_bits": [], "hook_text": "slap", "cliffhanger_text": "ring in bag?", "why_it_works": "w",
      "spoiler_risk": False, "resolution_leak": False}


def test_clip_text_no_ids_or_times_and_first3():
    t = clip_text(CAND, EPS, CFG.signals)
    assert "E01-B" not in t and "00:" not in t and 'VICTOR (Furious): "You stole my mother’s ring!"' in t
    f3 = clip_text(CAND, EPS, CFG.signals, first_seconds=3.0)
    assert "slaps" in f3 and "never seen that ring" not in f3   # window [4.0, 7.0): B003 (5.0) in, B006 (10.0) out


def test_judge_all_merges_hook3s_and_is_blind():
    llm = FakeLLM([JO, {"score": 9, "why": "slap"}])
    cfg = CFG.model_copy(deep=True); cfg.models.concurrency = 1
    out = judge_all([CAND], EPS, llm, cfg, RunMeta())
    assert out["S-x"].standalone == 8 and out["S-x"].hook3s == 9
    judge_msgs = [m for s, m, *_ in llm.seen if s == "judge"][0]
    assert "Mara is publicly accused" not in str(judge_msgs)   # no logline/show context leaks to the judge


def test_failed_judge_omitted():
    cfg = CFG.model_copy(deep=True); cfg.models.concurrency = 1
    m = RunMeta()
    out = judge_all([CAND], EPS, FakeLLM([LLMStageError("judge"), {"score": 9, "why": "x"}]), cfg, m)
    assert out == {} and m.warnings
