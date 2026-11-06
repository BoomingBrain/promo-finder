"""`eval`: score a run against the hand-picked gold set plus editor marks and stability metrics."""
import csv
import json
from datetime import datetime
from pathlib import Path
import yaml
from promofinder.config import load_config
from promofinder.invariants import pass_rate
from promofinder.models import Beat, Episode, EpisodeTags, Promo, Segment
from promofinder.timeutil import parse_ts

POS, NEG = {"✓", "y", "yes"}, {"✗", "n", "no"}


def overlap(a: Segment, b: Segment) -> float:
    if a.ep != b.ep or b.end_s <= b.start_s:
        return 0.0
    return max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s)) / (b.end_s - b.start_s)


def load_gold(path: Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [{"id": p["id"], "segments": [Segment(ep=s["ep"], start_s=parse_ts(s["start"]), end_s=parse_ts(s["end"]))
                                         for s in p["segments"]]} for p in data.get("picks") or []]


def _promos(run_dir: Path) -> list[Promo]:
    return [Promo.model_validate(p) for p in json.loads((run_dir / "promos.json").read_text(encoding="utf-8"))]


def _verdict_precision(run_dir: Path, k: int = 10) -> float | None:
    path = run_dir / "promos.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["rank"] or 10**9))[:k]
    marks = [(r.get("editor_verdict") or "").strip().lower() for r in rows]
    pos, neg = sum(m in POS for m in marks), sum(m in NEG for m in marks)
    return pos / (pos + neg) if pos + neg else None


def _tags(run_dir: Path) -> dict[int, EpisodeTags]:
    out = {}
    for p in sorted((run_dir / "tags").glob("ep_*.json")) if (run_dir / "tags").exists() else []:
        t = EpisodeTags.model_validate_json(p.read_text(encoding="utf-8")); out[t.ep] = t
    return out


def _invariants(run_dir: Path, promos: list[Promo], tags, meta: dict, config_dir: Path) -> float | None:
    bpath = run_dir / "beats.jsonl"
    if not bpath.exists() or not promos:
        return meta.get("values", {}).get("invariant_pass_rate")
    beats: dict[int, list[Beat]] = {}
    for line in bpath.read_text(encoding="utf-8").splitlines():
        if line.strip():
            b = Beat.model_validate_json(line); beats.setdefault(b.ep, []).append(b)
    eps = {e["ep"]: Episode(**e, beats=beats.get(e["ep"], []))
           for e in json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))}
    rate, _ = pass_rate(promos, eps, tags, load_config(config_dir).scoring)
    return round(rate, 4)


def evaluate(run_dir: Path, gold_path: Path, compare_dir: Path | None = None, config_dir: Path = Path("config")) -> dict:
    run_dir = Path(run_dir)
    promos = sorted(_promos(run_dir), key=lambda p: p.rank)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8")) if (run_dir / "run_meta.json").exists() else {}
    gold = load_gold(gold_path)
    detail, hits = {}, 0
    for g in gold:
        best = (0.0, None)
        for p in promos:
            o = max((overlap(s, gs) for s in p.candidate.segments for gs in g["segments"]), default=0.0)
            if o >= 0.5 and (best[1] is None or p.rank < best[1]):
                best = (o, p.rank)
            elif best[1] is None and o > best[0]:
                best = (o, None)
        detail[g["id"]] = {"best_rank": best[1], "overlap": round(best[0], 3)}
        hits += best[1] is not None and best[1] <= 20
    tags = _tags(run_dir)
    qm = meta.get("counters", {}).get("quote_mismatches", 0)
    kept = sum(len(t.hooks) + len(t.cliffhangers) for t in tags.values())
    cmp = None
    if compare_dir is not None:
        a = {p.candidate.cand_id for p in promos[:10]}
        b = {p.candidate.cand_id for p in sorted(_promos(Path(compare_dir)), key=lambda p: p.rank)[:10]}
        cmp = len(a & b) / len(a | b) if a | b else None
    result = {
        "recall_at_20": hits / len(gold) if gold else None,
        "precision_at_10": _verdict_precision(run_dir),
        "invariant_pass_rate": _invariants(run_dir, promos, tags, meta, config_dir),
        "quote_mismatch_rate": round(qm / (qm + kept), 4) if qm + kept else 0.0,
        "vote_agreement": meta.get("values", {}).get("vote_agreement"),
        "top10_overlap_with_compare": cmp,
        "n_gold": len(gold),
        "gold_detail": detail,
    }
    (run_dir / f"eval_{datetime.now():%Y%m%d-%H%M%S}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
