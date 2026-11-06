import csv
import json
from pathlib import Path
import yaml
from promofinder.config import load_config
from promofinder.feedback import append_gold, examples_text, nudge_weights, read_marked
from promofinder.report import CSV_COLUMNS

CFG = load_config(Path("config"))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS); w.writeheader()
        for r in rows:
            w.writerow({k: "" for k in CSV_COLUMNS} | r)


def row(i, verdict, standalone=5):
    return {"rank": i, "cand_id": f"C{i}", "segments": f"E01 00:0{i}.00–00:3{i}.00", "hook": f"hook {i}",
            "cliffhanger": f"cliff {i}", "hook_score": 7, "cliff_score": 7, "standalone": standalone,
            "story_importance": 5, "rule_score": 2, "editor_verdict": verdict, "editor_note": f"note {i}"}


def test_few_marks_no_nudge_examples_without_timestamps(tmp_path):
    p = tmp_path / "p.csv"
    write_csv(p, [row(1, "✓"), row(2, "✗"), row(3, "y"), row(4, "")])
    m = read_marked(p)
    assert len(m) == 3 and nudge_weights(m, CFG.scoring) is None
    t = examples_text(m, CFG.scoring)
    assert "EDITOR LIKED" in t and "EDITOR REJECTED" in t and "00:" not in t


def test_nudge_moves_standalone_up_and_renormalises(tmp_path):
    p = tmp_path / "p.csv"
    write_csv(p, [row(i, "✓", standalone=9) for i in range(1, 6)] + [row(i, "✗", standalone=3) for i in range(6, 11)])
    out = nudge_weights(read_marked(p), CFG.scoring)["final_weights"]
    base = CFG.scoring.final_weights.model_dump()
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["standalone"] > base["standalone"] and out["standalone"] - base["standalone"] <= 0.05 + 1e-9


def test_append_gold_once(tmp_path):
    p = tmp_path / "p.csv"
    write_csv(p, [row(1, "✓"), row(2, "✗")])
    promos = [{"rank": 1, "candidate": {"cand_id": "C1", "segments": [{"ep": 1, "start_s": 1.0, "end_s": 31.0}]}},
              {"rank": 2, "candidate": {"cand_id": "C2", "segments": [{"ep": 1, "start_s": 2.0, "end_s": 32.0}]}}]
    pj = tmp_path / "promos.json"; pj.write_text(json.dumps(promos))
    gold = tmp_path / "gold.yaml"; gold.write_text("picks: []\n")
    m = read_marked(p)
    assert append_gold(m, pj, gold) == 1 and append_gold(m, pj, gold) == 0
    picks = yaml.safe_load(gold.read_text())["picks"]
    assert picks[0]["id"] == "F1" and picks[0]["segments"] == [{"ep": 1, "start": "00:01.00", "end": "00:31.00"}]


def test_append_gold_keeps_header_comments(tmp_path):
    p = tmp_path / "p.csv"; write_csv(p, [row(1, "✓")])
    pj = tmp_path / "promos.json"
    pj.write_text(json.dumps([{"rank": 1, "candidate": {"cand_id": "C1", "segments": [{"ep": 1, "start_s": 1.0, "end_s": 31.0}]}}]))
    gold = tmp_path / "gold.yaml"; gold.write_text("# keep me\nshow: s\npicks: []\n")
    append_gold(read_marked(p), pj, gold)
    assert gold.read_text().startswith("# keep me")
