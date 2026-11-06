import json
from pathlib import Path
from promofinder.config import load_config
from promofinder.experiments import check_naive_clip, hindsight, plan_a
from promofinder.meta import RunMeta
from promofinder.models import Act, Bible
from promofinder.parse import load_show
from tests.fakes import FakeLLM
from tests.test_pipeline import EMPTY, JO, TAG

CFG = load_config(Path("config"))
EPS = {e.ep: e for e in load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes}
W = CFG.scoring.window


def test_check_naive_clip():
    ep = EPS[1]
    assert check_naive_clip(ep, "00:04.00", "00:30.00", W) == []
    bad = check_naive_clip(ep, "00:04.30", "00:22.50", W)
    assert "not_on_boundary" in bad and "mid_line_cut" in bad
    assert "beyond_episode" in check_naive_clip(ep, "00:50.00", "01:30.00", W)
    assert check_naive_clip(ep, "abc", "00:30.00", W) == ["unparseable"]


def test_plan_a_counts_errors_and_saves_raw(tmp_path):
    llm = FakeLLM([{"clips": [{"start": "00:04.00", "end": "00:30.00", "why": "w"},
                              {"start": "00:04.30", "end": "00:22.50", "why": "w"}]}])
    s = plan_a(EPS, [1], llm, CFG, RunMeta(), tmp_path)
    assert s["total_clips"] == 2 and s["valid_clips"] == 1 and s["errors"]["mid_line_cut"] == 1
    assert (tmp_path / "experiments" / "exp_a_raw" / "ep_001.json").exists()
    assert (tmp_path / "experiments" / "exp_a_summary.json").exists()
    assert llm.seen[0][0] == "judge" and "Victor" in str(llm.seen[0][1])


class Ctx(FakeLLM):
    def complete_json(self, stage, messages, schema, vote_index=0, temperature=None):
        self.seen.append((stage, messages, vote_index, temperature))
        if stage == "tagger":
            r = json.loads(json.dumps(TAG))
            if "Premise: p" not in str(messages):
                r["hooks"][0]["hook_score"] = 5
            r = r if "Mara is publicly accused" in str(messages) else EMPTY
        elif stage == "judge":
            r = JO
        else:
            r = {"score": 8, "why": "w"}
        return schema.model_validate(r)


def test_hindsight_flags_context_inflated_beats(tmp_path):
    bible = Bible(premise="p", characters={"Mara": "heroine"}, arc=[Act(name="A", ep_start=1, ep_end=2)],
                  episode_summaries={1: "a", 2: "b"})
    llm = Ctx([])
    rows = hindsight(EPS, [1], bible, llm, CFG, RunMeta(), tmp_path)
    assert rows and rows[0]["beat_id"] == "E01-B003" and rows[0]["score_with"] == 9 and rows[0]["score_without"] == 5
    assert rows[0]["standalone"] == 8
    assert sorted({v for s, m, v, t in llm.seen if s == "tagger"}) == [100, 101]
    assert (tmp_path / "experiments" / "exp_b_hindsight.json").exists()


def test_experiment_cli_needs_key(monkeypatch):
    from promofinder.cli import main
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr("promofinder.cli.load_dotenv", lambda: None)
    assert main(["experiment", "plan-a", "--input", "tests/fixtures/show_a", "--episodes", "1"]) == 3


class CtxOnly(Ctx):
    """Without context the tagger finds no hooks at all — absent is not the same as a score of 0."""
    def complete_json(self, stage, messages, schema, vote_index=0, temperature=None):
        if stage == "tagger" and "Premise: p" not in str(messages):
            self.seen.append((stage, messages, vote_index, temperature))
            return schema.model_validate(EMPTY)
        return super().complete_json(stage, messages, schema, vote_index, temperature)


def test_hindsight_separates_context_only_picks(tmp_path):
    bible = Bible(premise="p", characters={"Mara": "heroine"}, arc=[Act(name="A", ep_start=1, ep_end=2)],
                  episode_summaries={1: "a", 2: "b"})
    rows = hindsight(EPS, [1], bible, CtxOnly([]), CFG, RunMeta(), tmp_path)
    assert rows and all(r["tagged_without"] is False and r["score_without"] is None for r in rows)
    s = json.loads((tmp_path / "experiments" / "exp_b_summary.json").read_text())
    assert s["context_only"] >= 1 and s["score_rose"] == 0
