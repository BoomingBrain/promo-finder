"""Stage 10 outputs: promos.json, promos.csv, report.md, run_meta.json."""
import csv
import json
from pathlib import Path
from promofinder.config import AppConfig
from promofinder.invariants import pass_rate
from promofinder.meta import RunMeta
from promofinder.models import Candidate, Promo
from promofinder.parse import health_report
from promofinder.prompts import prompt_versions
from promofinder.readers.base import ReadResult
from promofinder.timeutil import format_ts

CSV_COLUMNS = ["rank", "cand_id", "kind", "segments", "duration_s", "hook", "cliffhanger", "why_it_works",
               "post_first_because", "final_score", "elo", "hook_score", "cliff_score", "standalone",
               "story_importance", "rule_score", "hook3s", "flags", "editor_verdict", "editor_note"]


def segments_str(c: Candidate) -> str:
    return "; ".join(f"E{s.ep:02d} {format_ts(s.start_s)}–{format_ts(s.end_s)}" for s in c.segments)


def _row(p: Promo) -> dict:
    c, j = p.candidate, p.judgement
    return {"rank": p.rank, "cand_id": c.cand_id, "kind": c.kind, "segments": segments_str(c),
            "duration_s": round(c.duration_s, 2), "hook": j.hook_text if j else "", "cliffhanger": j.cliffhanger_text if j else "",
            "why_it_works": j.why_it_works if j else "", "post_first_because": p.post_first_because or "",
            "final_score": round(p.final_score, 3), "elo": "" if p.elo is None else round(p.elo, 1),
            "hook_score": c.hook_score, "cliff_score": c.cliff_score, "standalone": j.standalone if j else "",
            "story_importance": c.story_importance, "rule_score": c.rule_score,
            "hook3s": "" if not j or j.hook3s is None else j.hook3s, "flags": "|".join(p.flags),
            "editor_verdict": "", "editor_note": ""}


def write_outputs(out_dir: Path, promos: list[Promo], episodes, tags, read: ReadResult, meta: RunMeta, cfg: AppConfig) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for p in promos:
        d = p.model_dump()
        d["segments_ts"] = [{"ep": s.ep, "start": format_ts(s.start_s), "end": format_ts(s.end_s)} for s in p.candidate.segments]
        d["duration_s"] = round(p.candidate.duration_s, 2)
        data.append(d)
    (out_dir / "promos.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for p in promos:
            w.writerow(_row(p))
    rate, bad = pass_rate(promos, episodes, tags, cfg.scoring)
    summ = meta.summary()
    models = sorted({c["model"] for c in meta.calls})
    md = ["# Promo Finder report", "",
          f"Promos: {len(promos)} · LLM calls: {summ['total_calls']} ({summ['live_calls']} live) · "
          f"tokens: {summ['prompt_tokens']} in / {summ['completion_tokens']} out",
          f"Models used: {', '.join(models) or 'none'}", "",
          f"Invariant pass rate: {round(rate * 100)}%"]
    for cid, v in bad.items():
        md.append(f"- {cid}: {', '.join(v)}")
    md += ["", "## Ranked promos", ""]
    for p in promos:
        c, j = p.candidate, p.judgement
        md.append(f"### #{p.rank} — {segments_str(c)} ({c.duration_s:.1f} s)")
        if j:
            md += [f"- **Hook:** {j.hook_text}", f"- **Cliffhanger:** {j.cliffhanger_text}",
                   f"- **Why it works:** {j.why_it_works}"]
        if p.post_first_because and p.rank <= 5:
            md.append(f"- **Post first because:** {p.post_first_because}")
        scores = (f"final {p.final_score:.2f} · hook {c.hook_score:.1f} · cliff {c.cliff_score:.1f}"
                  + (f" · standalone {j.standalone:.1f}" if j else "")
                  + (f" · first-3s {j.hook3s:.1f}" if j and j.hook3s is not None else "")
                  + (f" · elo {p.elo:.0f}" if p.elo is not None else ""))
        md.append(f"- Scores: {scores}")
        if p.flags:
            md.append(f"- Flags: {', '.join(p.flags)}")
        if p.original_segments:
            md.append(f"- Refined from: {'; '.join(f'E{s.ep:02d} {format_ts(s.start_s)}–{format_ts(s.end_s)}' for s in p.original_segments)}")
        md.append("")
    md += ["## Parse health", "", "```", health_report(read), "```", ""]
    if meta.warnings:
        md += ["## Warnings", ""] + [f"- {w}" for w in meta.warnings[:200]]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    meta.set("prompt_versions", prompt_versions())
    meta.set("invariant_pass_rate", rate)
    meta.save(out_dir / "run_meta.json")
