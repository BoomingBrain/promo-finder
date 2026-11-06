"""Stage 11: editor ✓/✗ marks → prompt examples, small weight nudges, and feedback gold picks (F-ids).
F-picks are scored separately from the original human G-picks, so the tool never grades itself on its own output."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import yaml
from promofinder.config import AppConfig, ScoringConfig
from promofinder.timeutil import format_ts

POS, NEG = {"✓", "y", "yes"}, {"✗", "n", "no"}
COLS = {"hook": "hook_score", "cliff": "cliff_score", "standalone": "standalone",
        "story_importance": "story_importance", "rules": "rule_score"}


@dataclass
class Marked:
    cand_id: str
    verdict: bool
    note: str
    row: dict


def read_marked(csv_path: Path) -> list[Marked]:
    out = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("editor_verdict") or "").strip().lower()
            if v in POS or v in NEG:
                out.append(Marked(cand_id=r["cand_id"], verdict=v in POS, note=(r.get("editor_note") or "").strip(), row=r))
    return out


def examples_text(marked: list[Marked], cfg: ScoringConfig) -> str:
    fb = cfg.feedback
    pos = [m for m in marked if m.verdict][:fb.max_pos]
    neg = [m for m in marked if not m.verdict][:fb.max_neg]
    lines = [f"EDITOR LIKED: {m.row.get('hook', '')} / {m.row.get('cliffhanger', '')} — {m.note}".rstrip(" —") for m in pos]
    lines += [f"EDITOR REJECTED: {m.row.get('hook', '')} / {m.row.get('cliffhanger', '')} — {m.note}".rstrip(" —") for m in neg]
    return "\n".join(lines)


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def nudge_weights(marked: list[Marked], cfg: ScoringConfig) -> dict | None:
    fb = cfg.feedback
    if len(marked) < fb.min_marked:
        return None
    w = cfg.final_weights.model_dump()
    for key, col in COLS.items():
        p = [x for m in marked if m.verdict and (x := _num(m.row.get(col))) is not None]
        n = [x for m in marked if not m.verdict and (x := _num(m.row.get(col))) is not None]
        if p and n:
            d = sum(p) / len(p) - sum(n) / len(n)
            w[key] = max(0.0, w[key] + max(-fb.max_nudge, min(fb.max_nudge, d / 10)))
    total = sum(w.values())
    return {"final_weights": {k: v / total for k, v in w.items()}}


def append_gold(marked: list[Marked], promos_json: Path, gold_path: Path) -> int:
    promos = {p["candidate"]["cand_id"]: p for p in json.loads(Path(promos_json).read_text(encoding="utf-8"))}
    raw = Path(gold_path).read_text(encoding="utf-8")
    header = "".join(l + "\n" for l in raw.splitlines() if l.startswith("#") and raw.index(l) < raw.find("picks:"))
    data = yaml.safe_load(raw) or {}
    picks = data.get("picks") or []
    existing = [p["segments"] for p in picks]
    n_f = sum(1 for p in picks if str(p.get("id", "")).startswith("F"))
    added = 0
    for m in marked:
        if not m.verdict or m.cand_id not in promos:
            continue
        segs = [{"ep": s["ep"], "start": format_ts(s["start_s"]), "end": format_ts(s["end_s"])}
                for s in promos[m.cand_id]["candidate"]["segments"]]
        if segs in existing:
            continue
        n_f += 1; added += 1
        picks.append({"id": f"F{n_f}", "segments": segs, "note": m.note or "editor ✓ on tool output"})
        existing.append(segs)
    data["picks"] = picks
    Path(gold_path).write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=200), encoding="utf-8")
    return added


def apply(csv_path: Path, run_dir: Path, cfg: AppConfig, config_dir: Path, gold_path: Path) -> str:
    marked = read_marked(csv_path)
    nudged = nudge_weights(marked, cfg.scoring)
    if nudged is not None:
        (Path(config_dir) / "scoring.learned.yaml").write_text(yaml.safe_dump(nudged), encoding="utf-8")
    if (Path(run_dir) / "promos.json").exists():
        append_gold(marked, Path(run_dir) / "promos.json", gold_path)
    return examples_text(marked, cfg.scoring)
