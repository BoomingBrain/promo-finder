import csv
import json
from pathlib import Path
from promofinder.evaluate import evaluate, overlap
from promofinder.cli import main
from promofinder.models import Segment
from promofinder.report import CSV_COLUMNS


def test_overlap_is_fraction_of_gold():
    assert overlap(Segment(ep=1, start_s=0, end_s=30), Segment(ep=1, start_s=10, end_s=40)) == 20 / 30
    assert overlap(Segment(ep=1, start_s=0, end_s=30), Segment(ep=2, start_s=0, end_s=30)) == 0


def _promo(rank, cid, ep, s, e):
    return {"rank": rank, "candidate": {"cand_id": cid, "kind": "single", "segments": [{"ep": ep, "start_s": s, "end_s": e}],
            "hook_beat_id": "x", "cliff_beat_id": "y", "hook_score": 5, "cliff_score": 5, "story_importance": 5,
            "rule_score": 0, "length_fit": 10, "penalties": 0, "window_score": 5, "scene_key": "E01-S01"},
            "judgement": None, "final_score": 5, "flags": []}


def make_run(d: Path, verdicts=("", "")):
    d.mkdir(parents=True, exist_ok=True)
    promos = [_promo(1, "A", 1, 0, 30), _promo(2, "B", 1, 100, 130)]
    (d / "promos.json").write_text(json.dumps(promos))
    (d / "episodes.json").write_text("[]")
    (d / "run_meta.json").write_text(json.dumps({"counters": {"quote_mismatches": 2}, "values": {"invariant_pass_rate": 1.0, "vote_agreement": 0.8}}))
    with open(d / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS); w.writeheader()
        for p, v in zip(promos, verdicts):
            w.writerow({k: "" for k in CSV_COLUMNS} | {"rank": p["rank"], "cand_id": p["candidate"]["cand_id"], "editor_verdict": v})
    gold = d.parent / "gold.yaml"
    gold.write_text('picks:\n  - {id: G1, segments: [{ep: 1, start: "00:12.00", end: "00:42.00"}], note: n}\n'
                    '  - {id: G2, segments: [{ep: 7, start: "00:00.00", end: "00:30.00"}], note: n}\n')
    return gold


def test_recall_and_precision(tmp_path):
    gold = make_run(tmp_path / "run", verdicts=("✓", ""))
    r = evaluate(tmp_path / "run", gold)
    assert r["recall_at_20"] == 0.5 and r["n_gold"] == 2      # G1 overlaps 18/30 = 60%; G2 is in another episode
    assert r["precision_at_10"] == 1.0
    assert r["vote_agreement"] == 0.8 and r["top10_overlap_with_compare"] is None
    assert list((tmp_path / "run").glob("eval_*.json"))


def test_precision_none_without_marks_and_compare(tmp_path):
    gold = make_run(tmp_path / "run")
    make_run(tmp_path / "other")
    r = evaluate(tmp_path / "run", gold, tmp_path / "other")
    assert r["precision_at_10"] is None and r["top10_overlap_with_compare"] == 1.0


def test_negative_marks_count(tmp_path):
    gold = make_run(tmp_path / "run", verdicts=("y", "no"))
    assert evaluate(tmp_path / "run", gold)["precision_at_10"] == 0.5


def test_eval_cli(tmp_path, capsys):
    gold = make_run(tmp_path / "run")
    assert main(["eval", "--run", str(tmp_path / "run"), "--gold", str(gold)]) == 0
    assert "recall_at_20" in capsys.readouterr().out
