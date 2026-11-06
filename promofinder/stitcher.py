"""Stage 6 (experimental): LLM proposes 2-3 segment combos by beat ID; code builds and validates the segments."""
from typing import Literal
from pydantic import BaseModel, ConfigDict
from promofinder.bible import render_context
from promofinder.config import AppConfig
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Bible, Candidate, Episode, EpisodeTags, Segment
from promofinder.prompts import load_prompt
from promofinder.windows import length_fit, shot_at, snap_end, snap_start


class SegReq(BaseModel):
    model_config = ConfigDict(extra="ignore")
    beat_id: str
    role: Literal["hook", "bridge", "cliff"]
    before_s: float = 0
    after_s: float = 0


class Combo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    segments: list[SegReq]
    why: str = ""


class StitchOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    combos: list[Combo] = []


def _ep_of(beat_id: str) -> int | None:
    try:
        return int(beat_id.split("-")[0].lstrip("E"))
    except (ValueError, IndexError):
        return None


def build_segment(req: SegReq, ep: Episode, tags: EpisodeTags | None, cfg: AppConfig) -> Segment | None:
    b = next((x for x in ep.beats if x.beat_id == req.beat_id), None)
    if b is None:
        return None
    sc = cfg.scoring
    if req.role == "hook":
        start = snap_start(ep, b, sc.window.hook_lead_max_s)
        shot = shot_at(ep, start + max(req.after_s, sc.stitch.min_segment_s))
        return Segment(ep=ep.ep, start_s=start, end_s=shot.end_s) if shot else None
    if req.role == "cliff":
        end = snap_end(ep, b, None)
        shot = shot_at(ep, b.start_s - max(req.before_s, 0))
        return Segment(ep=ep.ep, start_s=shot.start_s, end_s=end) if shot else None
    s, e = shot_at(ep, b.start_s - req.before_s), shot_at(ep, b.end_s + req.after_s - 1e-6)
    return Segment(ep=ep.ep, start_s=s.start_s, end_s=e.end_s) if s and e else None


def validate_combo(segs: list[Segment], roles: list[str], cfg: AppConfig) -> list[str]:
    sc, errs = cfg.scoring, []
    if any(s.duration_s < sc.stitch.min_segment_s - 1e-6 for s in segs):
        errs.append("too_short_segment")
    total = sum(s.duration_s for s in segs)
    if not (sc.window.min_s - 1e-6 <= total <= sc.window.max_s + 1e-6):
        errs.append("total_duration")
    keys = [(s.ep, s.start_s) for s in segs]
    if any(a >= b for a, b in zip(keys, keys[1:])):
        errs.append("not_chronological")
    elif any(a.ep == b.ep and b.start_s < a.end_s - 1e-6 for a, b in zip(segs, segs[1:])):
        errs.append("overlap")
    if not roles or roles[0] != "hook" or roles[-1] != "cliff":
        errs.append("bad_roles")
    return errs


def stitch(episodes: dict[int, Episode], tags: dict[int, EpisodeTags], bible: Bible | None, client, cfg: AppConfig,
           meta: RunMeta) -> list[Candidate]:
    st, ww, w = cfg.scoring.stitch, cfg.scoring.window_weights, cfg.scoring.window
    beat_text = {b.beat_id: b.text for e in episodes.values() for b in e.beats}
    hooks = sorted(((h, t.ep) for t in tags.values() for h in t.hooks), key=lambda x: -x[0].hook_score)[:st.top_hooks]
    cliffs = sorted(((c, t.ep) for t in tags.values() for c in t.cliffhangers), key=lambda x: -x[0].cliff_score)[:st.top_cliffs]
    if not hooks or not cliffs:
        return []
    ctx = render_context(bible, 0).split("\n\nLOCAL")[0]
    prompt = load_prompt("stitcher").safe_substitute(
        context=ctx or "(no show context)", max=st.max_combos,
        hooks="\n".join(f"{h.beat_id} | ep {ep} | {beat_text.get(h.beat_id, '')}" for h, ep in hooks),
        cliffs="\n".join(f"{c.beat_id} | ep {ep} | {beat_text.get(c.beat_id, '')}" for c, ep in cliffs))
    try:
        out = client.complete_json("stitcher", [{"role": "user", "content": prompt}], StitchOut)
    except LLMStageError as e:
        meta.warn(f"stitcher failed ({e})")
        return []
    hook_s = {h.beat_id: h for t in tags.values() for h in t.hooks}
    cliff_s = {c.beat_id: c for t in tags.values() for c in t.cliffhangers}
    cands: list[Candidate] = []
    for k, combo in enumerate(out.combos[:st.max_combos], 1):
        if not combo.segments:
            meta.warn(f"stitch combo {k} dropped: empty")
            continue
        segs, roles = [], [r.role for r in combo.segments]
        for r in combo.segments:
            n = _ep_of(r.beat_id)
            seg = build_segment(r, episodes[n], tags.get(n), cfg) if n in episodes else None
            if seg is None:
                break
            segs.append(seg)
        errs = ["unknown_beat"] if len(segs) != len(combo.segments) else validate_combo(segs, roles, cfg)
        h, c = hook_s.get(combo.segments[0].beat_id), cliff_s.get(combo.segments[-1].beat_id)
        if not errs and (h is None or c is None):
            errs = ["not_a_tagged_hook_or_cliff"]
        if errs:
            meta.warn(f"stitch combo {k} dropped: {', '.join(errs)}")
            continue
        total = sum(s.duration_s for s in segs)
        lf = length_fit(total, w)
        first = next(b for b in episodes[segs[0].ep].beats if b.beat_id == h.beat_id)
        cands.append(Candidate(
            cand_id=f"M-{len(cands) + 1:03d}", kind="stitched", segments=segs, hook_beat_id=h.beat_id,
            cliff_beat_id=c.beat_id, hook_score=h.hook_score, cliff_score=c.cliff_score,
            story_importance=round((h.story_importance + c.story_importance) / 2, 3), rule_score=0.0,
            length_fit=round(lf, 3), penalties=0.0,
            window_score=round(ww.hook * h.hook_score + ww.cliff * c.cliff_score + ww.length_fit * lf, 4),
            scene_key=f"E{segs[0].ep:02d}-S{first.scene:02d}"))
    return cands
