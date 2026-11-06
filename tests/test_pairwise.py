from pathlib import Path
from promofinder.config import load_config
from promofinder.meta import RunMeta
from promofinder.pairwise import elo_update, rank_pairwise, swiss_pairs
from promofinder.parse import load_show
from tests.fakes import FakeLLM
from tests.test_refine import cand
from promofinder.models import Promo, Segment

CFG = load_config(Path("config"))
EPS = {e.ep: e for e in load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes}


def test_elo_update():
    assert elo_update(1000, 1000, True, 32) == (1016.0, 984.0)


def test_swiss_pairs_bye():
    assert swiss_pairs(["a", "b", "c"], {"a": 1000, "b": 1000, "c": 1000}) == ([("a", "b")], "c")


class PreferSlap(FakeLLM):
    """Always picks whichever option contains the slap (the strongest candidate's hook)."""
    def complete_json(self, stage, messages, schema, vote_index=0, temperature=None):
        txt = messages[0]["content"]
        a, b = txt.split("OPTION B:")
        self.seen.append(stage)
        return schema.model_validate({"winner": "A" if "slaps" in a else "B", "why": "the slap stops the scroll"})


def _p(cid, s, e, score):
    c = cand().model_copy(update={"cand_id": cid, "segments": [Segment(ep=1, start_s=s, end_s=e)]})
    return Promo(rank=0, candidate=c, final_score=score)


def test_rank_pairwise_strongest_wins():
    fins = [_p("weak1", 30.0, 52.0, 9.0), _p("strong", 4.0, 30.0, 5.0), _p("weak2", 30.0, 55.0, 8.0)]
    meta = RunMeta()
    out = rank_pairwise(fins, EPS, PreferSlap([]), CFG, meta)
    assert out[0].candidate.cand_id == "strong" and out[0].post_first_because and out[0].elo is not None
    assert 0 <= meta.values["pairwise_first_position_rate"] <= 1


def test_rank_pairwise_single_unchanged():
    one = [_p("x", 4.0, 30.0, 5.0)]
    assert rank_pairwise(one, EPS, PreferSlap([]), CFG, RunMeta()) == one
