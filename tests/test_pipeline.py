import json
from pathlib import Path
from promofinder.pipeline import parse_stages, run, RunOptions
from promofinder.cli import main
from tests.fakes import FakeLLM

TAG = {"hooks": [{"beat_id": "E01-B003", "hook_score": 9, "story_importance": 5, "kind": "slap", "why": "w", "quote": "Victor slaps Mara across"}],
       "cliffhangers": [{"beat_id": "E01-B012", "cliff_score": 9, "story_importance": 6, "open_question": "?", "resolved_by": "next_episode",
                         "spoiler_level": "none", "quote": "Ominous string swell rises"}], "context_beats": []}
EMPTY = {"hooks": [], "cliffhangers": [], "context_beats": []}
JO = {"standalone": 8, "confusing_bits": [], "hook_text": "slap", "cliffhanger_text": "?", "why_it_works": "w",
      "spoiler_risk": False, "resolution_leak": False}


BIBLE = {"premise": "p", "characters": {"Mara": "accused heroine"}, "arc": [{"name": "Accusation", "ep_start": 1, "ep_end": 2}],
         "episode_summaries": {"1": "accused", "2": "garden aftermath"}, "major_reveals": []}


class Routed(FakeLLM):
    def complete_json(self, stage, messages, schema, vote_index=0, temperature=None):
        self.seen.append((stage, messages, vote_index, temperature))
        if stage == "tagger":
            r = TAG if "Mara is publicly accused" in str(messages) else EMPTY   # ep 1 logline
        elif stage == "judge":
            r = JO
        elif stage == "bible":
            r = BIBLE
        elif stage == "stitcher":
            r = {"combos": [{"segments": [{"beat_id": "E01-B003", "role": "hook", "after_s": 10},
                                          {"beat_id": "E01-B012", "role": "cliff"}], "why": "w"}]}
        elif stage == "refine":
            r = {"choice": "A", "why": "w"}
        elif stage == "pairwise":
            r = {"winner": "A", "why": "first is better"}
        else:
            r = {"score": 8, "why": "w"}
        return schema.model_validate(r)


def opts(tmp_path, **kw):
    return RunOptions(input=Path("tests/fixtures/show_a"), out_root=tmp_path, config_dir=Path("config"),
                      cache_dir=tmp_path / "cache", stages=parse_stages("1,2,4,5,7,10"), **kw)


def test_parse_stages():
    assert parse_stages("1-3,7") == {1, 2, 3, 7}


def test_end_to_end_and_resume(tmp_path):
    llm = Routed([])
    out = run(opts(tmp_path), client_factory=lambda *a: llm)
    promos = json.loads((out / "promos.json").read_text())
    assert promos[0]["candidate"]["cand_id"] == "S-E01-B003-E01-B012" and (out / "tags" / "ep_001.json").exists()
    for f in ["episodes.json", "beats.jsonl", "parse_health.txt", "signals.json", "candidates.json", "judgements.json",
              "promos.csv", "report.md", "run_meta.json"]:
        assert (out / f).exists(), f
    n = len(llm.seen)
    run(opts(tmp_path), client_factory=lambda *a: llm)
    assert len(llm.seen) == n                                   # resumed: no new LLM calls


def test_votes_use_distinct_vote_index(tmp_path):
    llm = Routed([])
    run(opts(tmp_path, votes=3), client_factory=lambda *a: llm)
    idx = sorted(v for s, m, v, t in llm.seen if s == "tagger" and "Mara is publicly accused" in str(m))
    assert idx == [0, 1, 2]


def test_dry_run_needs_no_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr("promofinder.cli.load_dotenv", lambda *a, **k: None)
    assert main(["run", "--input", "tests/fixtures/show_a", "--out", str(tmp_path), "--dry-run"]) == 0
    assert "tagger" in capsys.readouterr().out


def test_missing_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr("promofinder.cli.load_dotenv", lambda *a, **k: None)
    assert main(["run", "--input", "tests/fixtures/show_a", "--out", str(tmp_path)]) == 3


def test_parse_health_error_exit_code(tmp_path, monkeypatch):
    bad = tmp_path / "show"; bad.mkdir()
    txt = Path("tests/fixtures/show_a/episode-002-Episode 9002.txt").read_text().replace("Start: (00:02.00) | End: (00:04.00)\n", "")
    (bad / "ep2.txt").write_text(txt)
    monkeypatch.setattr("promofinder.cli.load_dotenv", lambda *a, **k: None)
    assert main(["run", "--input", str(bad), "--out", str(tmp_path / "o"), "--dry-run"]) == 2


def test_bible_stage_feeds_tagger_and_resumes(tmp_path):
    llm = Routed([])
    o = opts(tmp_path); o.stages = parse_stages("1-5,7,10")
    out = run(o, client_factory=lambda *a: llm)
    assert json.loads((out / "bible.json").read_text())["characters"] == {"Mara": "accused heroine"}
    tag_msgs = [str(m) for s, m, v, t in llm.seen if s == "tagger"]
    assert all("Mara — accused heroine" in m for m in tag_msgs)
    assert any("next episode: garden aftermath" in m for m in tag_msgs)
    n = sum(1 for s, *_ in llm.seen if s == "bible")
    run(o, client_factory=lambda *a: llm)
    assert n == 1 and sum(1 for s, *_ in llm.seen if s == "bible") == 1


def test_tagger_prompt_carries_anchors(tmp_path):
    llm = Routed([])
    run(opts(tmp_path), client_factory=lambda *a: llm)
    assert all("HOOK SCORE ANCHORS" in str(m) for s, m, v, t in llm.seen if s == "tagger")


def test_all_stages_write_their_files(tmp_path):
    llm = Routed([])
    o = opts(tmp_path); o.stages = parse_stages("1-10")
    out = run(o, client_factory=lambda *a: llm)
    for f in ["bible.json", "refine.json", "pairwise.json"]:
        assert (out / f).exists(), f
    promos = json.loads((out / "promos.json").read_text())
    assert any(p["candidate"]["kind"] == "stitched" for p in promos)
    assert all(p["elo"] is not None for p in promos[:2])
    stages = {s for s, *_ in llm.seen}
    assert {"stitcher", "pairwise"} <= stages


def test_feedback_csv_feeds_examples_to_tagger_and_judge(tmp_path):
    llm = Routed([])
    out = run(opts(tmp_path), client_factory=lambda *a: llm)
    import csv
    rows = list(csv.DictReader(open(out / "promos.csv", encoding="utf-8")))
    rows[0]["editor_verdict"] = "✓"; rows[0]["editor_note"] = "great slap open"
    fb = tmp_path / "marked.csv"
    with open(fb, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    gold = tmp_path / "gold.yaml"; gold.write_text("picks: []\n")
    llm2 = Routed([])
    run(opts(tmp_path, feedback=fb, fresh=True, gold_path=gold), client_factory=lambda *a: llm2)
    msgs = [str(m) for s, m, v, t in llm2.seen if s in ("tagger", "judge")]
    assert msgs and all("great slap open" in m for m in msgs)
    assert "F1" in gold.read_text()
