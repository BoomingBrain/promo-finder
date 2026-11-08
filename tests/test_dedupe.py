from pathlib import Path
from promofinder.config import load_config
from promofinder.dedupe import dedupe_finalists
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Judgement
from tests.fakes import FakeLLM
from tests.test_report import P

CFG = load_config(Path("config"))


def fins():
    out = []
    for i, (h, c) in enumerate([("burns book", "iron at her eye"), ("accused of poison", "dragged away"),
                                ("rants at captor", "iron at her eye again")], 1):
        p = P(4.0, 30.0); p.candidate = p.candidate.model_copy(update={"cand_id": f"C{i}"}); p.rank = i
        p.judgement = Judgement(cand_id=f"C{i}", standalone=7, hook_text=h, cliffhanger_text=c)
        out.append(p)
    return out


def test_lower_ranked_duplicate_is_moved_out():
    meta = RunMeta()
    kept, dropped = dedupe_finalists(fins(), FakeLLM([{"duplicates": [{"keep": 1, "drop": 3, "why": "same iron cliff"}]}]), CFG, meta)
    assert [p.candidate.cand_id for p in kept] == ["C1", "C2"] and [p.candidate.cand_id for p in dropped] == ["C3"]
    assert "duplicate_of_C1" in dropped[0].flags and meta.counters["dedupe_dropped"] == 1


def test_bad_indices_and_failures_keep_everything():
    for reply in ({"duplicates": [{"keep": 3, "drop": 1}]},        # would drop the higher-ranked one: ignored
                  {"duplicates": [{"keep": 1, "drop": 9}]},        # out of range: ignored
                  LLMStageError("down")):
        kept, dropped = dedupe_finalists(fins(), FakeLLM([reply]), CFG, RunMeta())
        assert len(kept) == 3 and dropped == []
