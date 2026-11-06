"""Experiments for the solution doc (§13.2).
A: "Plan A" — let the LLM write timestamps itself, then check them.  B: hindsight bias of the show context."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from promofinder.bible import render_context
from promofinder.config import AppConfig, WindowCfg
from promofinder.judge import judge_all
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Bible, Episode
from promofinder.prompts import load_prompt, render_anchors
from promofinder.signals import compute_signals
from promofinder.tagger import tag_episode
from promofinder.timeutil import format_ts, parse_ts
from promofinder.windows import build_candidates, straddles_dialogue

ERRORS = ("unparseable", "not_on_boundary", "mid_line_cut", "duration", "beyond_episode")


class NaiveClip(BaseModel):
    model_config = ConfigDict(extra="ignore")
    start: str
    end: str
    why: str = ""


class NaiveClips(BaseModel):
    model_config = ConfigDict(extra="ignore")
    clips: list[NaiveClip] = []


def check_naive_clip(ep: Episode, start: str, end: str, w: WindowCfg) -> list[str]:
    """The only place LLM-written timestamps are parsed."""
    try:
        s, e = parse_ts(str(start)), parse_ts(str(end))
    except (ValueError, TypeError):
        return ["unparseable"]
    errs = []
    cams = [b for b in ep.beats if b.type == "CAMERA"]
    if not (any(abs(b.start_s - s) <= 0.01 for b in cams) and any(abs(b.end_s - e) <= 0.01 for b in cams)):
        errs.append("not_on_boundary")
    if straddles_dialogue(ep, s) or straddles_dialogue(ep, e):
        errs.append("mid_line_cut")
    if not (w.min_s - 1e-6 <= e - s <= w.max_s + 1e-6):
        errs.append("duration")
    if e > ep.duration_s + 0.01 or s < 0 or e <= s:
        errs.append("beyond_episode")
    return errs


def screenplay_text(ep: Episode) -> str:
    out = [f"EPISODE {ep.ep}: {ep.logline}"]
    for b in ep.beats:
        label = b.speaker if b.type == "DIALOGUE" and b.speaker else ("SOUND FX" if b.type == "SFX" else b.type)
        out.append(f"{label}: ({format_ts(b.start_s)}) - ({format_ts(b.end_s)}) {b.text}")
    return "\n".join(out)


def plan_a(episodes: dict[int, Episode], ep_nums: list[int], client, cfg: AppConfig, meta: RunMeta, out_dir: Path) -> dict:
    raw_dir = Path(out_dir) / "experiments" / "exp_a_raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    errors = {k: 0 for k in ERRORS}
    total = valid = 0
    per_ep = {}
    for n in ep_nums:
        ep = episodes[n]
        prompt = load_prompt("plan_a").safe_substitute(episode=screenplay_text(ep))
        try:
            out = client.complete_json("judge", [{"role": "user", "content": prompt}], NaiveClips)
        except LLMStageError as e:
            meta.warn(f"E{n:02d}: plan_a failed ({e})")
            continue
        rows = []
        for c in out.clips:
            errs = check_naive_clip(ep, c.start, c.end, cfg.scoring.window)
            total += 1; valid += not errs
            for k in errs:
                errors[k] += 1
            rows.append(c.model_dump() | {"errors": errs})
        per_ep[n] = rows
        (raw_dir / f"ep_{n:03d}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"episodes": ep_nums, "total_clips": total, "valid_clips": valid,
               "valid_rate": round(valid / total, 4) if total else None, "errors": errors}
    (Path(out_dir) / "experiments" / "exp_a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def hindsight(episodes: dict[int, Episode], ep_nums: list[int], bible: Bible | None, client, cfg: AppConfig,
              meta: RunMeta, out_dir: Path) -> list[dict]:
    anchors = render_anchors()
    rows, risen_cands = [], []
    def both(n):
        ep = episodes[n]; sig = compute_signals(ep, cfg.signals)
        ctx = render_context(bible, n)
        jobs = [(ctx, 100), ("", 101)]
        return n, sig, [tag_episode(ep, sig, client, cfg, meta, context=c, anchors=anchors, vote_index=v) for c, v in jobs]

    with ThreadPoolExecutor(max_workers=max(1, cfg.models.concurrency // 2)) as ex:
        tagged = list(ex.map(both, ep_nums))
    for n, sig, (with_ctx, without) in tagged:
        ep = episodes[n]
        if with_ctx is None or without is None:
            continue
        base = {h.beat_id: h.hook_score for h in without.hooks} | {f"c:{c.beat_id}": c.cliff_score for c in without.cliffhangers}
        risen = {}
        for h in with_ctx.hooks:
            if h.hook_score - base.get(h.beat_id, 0) >= 2:
                risen[h.beat_id] = (h.hook_score, base.get(h.beat_id))
        for c in with_ctx.cliffhangers:
            if c.cliff_score - base.get(f"c:{c.beat_id}", 0) >= 2:
                risen[c.beat_id] = (c.cliff_score, base.get(f"c:{c.beat_id}"))
        for cand in build_candidates(ep, with_ctx, sig, cfg.scoring):
            for bid in (cand.hook_beat_id, cand.cliff_beat_id):
                if bid in risen:
                    risen_cands.append((bid, risen[bid], cand))
    seen, todo = set(), []
    for _, _, cand in risen_cands:
        if cand.cand_id not in seen:
            seen.add(cand.cand_id); todo.append(cand)
    judged = judge_all(todo, episodes, client, cfg, meta) if todo else {}
    for bid, (sw, so), cand in risen_cands:
        j = judged.get(cand.cand_id)
        rows.append({"beat_id": bid, "score_with": sw, "score_without": so, "tagged_without": so is not None,
                     "cand_id": cand.cand_id, "standalone": j.standalone if j else None})
    path = Path(out_dir) / "experiments"; path.mkdir(parents=True, exist_ok=True)
    (path / "exp_b_hindsight.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    def mean_sa(rs):
        v = [r["standalone"] for r in rs if r["standalone"] is not None]
        return round(sum(v) / len(v), 2) if v else None
    rose = [r for r in rows if r["tagged_without"]]
    only = [r for r in rows if not r["tagged_without"]]
    summary = {"episodes": ep_nums, "score_rose": len({r["beat_id"] for r in rose}),
               "context_only": len({r["beat_id"] for r in only}), "mean_standalone_rose": mean_sa(rose),
               "mean_standalone_context_only": mean_sa(only),
               "standalone_le_4": sum(1 for r in rows if r["standalone"] is not None and r["standalone"] <= 4)}
    (path / "exp_b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows
