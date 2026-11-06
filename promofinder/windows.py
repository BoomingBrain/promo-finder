"""Stage 5: turn tagged hook/cliff beats into clip windows. All timestamps come from beat boundaries."""
import re
from promofinder.config import ScoringConfig, WindowCfg
from promofinder.models import Beat, Candidate, Episode, EpisodeTags, Segment
from promofinder.signals import BeatSignals

EPS = 1e-6


def shot_at(ep: Episode, t: float) -> Beat | None:
    for b in ep.beats:
        if b.type == "CAMERA" and b.start_s <= t < b.end_s:
            return b
    return None


def snap_start(ep: Episode, hook: Beat, lead_max: float) -> float:
    shot = shot_at(ep, hook.start_s)
    if shot and hook.start_s - shot.start_s <= lead_max + EPS:
        return shot.start_s
    return hook.start_s


def snap_end(ep: Episode, cliff: Beat, resolved_start: float | None) -> float:
    shot = shot_at(ep, cliff.end_s - EPS)
    end = shot.end_s if shot else cliff.end_s
    later = [b.start_s for b in ep.beats if b.type == "DIALOGUE" and b.beat_id != cliff.beat_id
             and b.start_s >= cliff.end_s - EPS]
    limit = min([x for x in [resolved_start, min(later, default=None)] if x is not None], default=None)
    if limit is not None and end > limit + EPS:
        return cliff.end_s
    return end


def straddles_dialogue(ep: Episode, t: float) -> bool:
    return any(b.type == "DIALOGUE" and b.start_s + EPS < t < b.end_s - EPS for b in ep.beats)


def length_fit(d: float, w: WindowCfg) -> float:
    if d < w.min_s - EPS or d > w.max_s + EPS:
        return 0.0
    if w.sweet_min_s <= d <= w.sweet_max_s:
        return 10.0
    if d < w.sweet_min_s:
        return 10.0 - 5.0 * (w.sweet_min_s - d) / (w.sweet_min_s - w.min_s)
    return 10.0 - 5.0 * (d - w.sweet_max_s) / (w.max_s - w.sweet_max_s)


def check_clip_rules(ep: Episode, start: float, end: float, hook: Beat, cliff: Beat,
                     resolved_start: float | None, w: WindowCfg) -> list[str]:
    v = []
    d = end - start
    if d < w.min_s - EPS or d > w.max_s + EPS:
        v.append("duration")
    if hook.start_s < start - EPS or hook.start_s - start > w.hook_lead_max_s + EPS:
        v.append("hook_lead")
    if straddles_dialogue(ep, start) or straddles_dialogue(ep, end):
        v.append("mid_line_cut")
    if resolved_start is not None and start - EPS <= resolved_start < end - EPS:
        v.append("resolution_inside")
    if end < cliff.end_s - 0.05:
        v.append("cliff_cut")
    return v


def _location(heading: str) -> str:
    return re.split(r"\s+[–—-]\s+", heading.strip())[0].strip().upper()


def _uncovered_ratio(ep: Episode, start: float, end: float) -> float:
    iv = sorted((max(start, b.start_s), min(end, b.end_s)) for b in ep.beats
                if b.type in ("DIALOGUE", "ACTION") and b.end_s > start and b.start_s < end)
    covered, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    return 1.0 - covered / (end - start) if end > start else 0.0


def resolved_start_of(ep: Episode, resolved_by: str | None) -> float | None:
    b = next((x for x in ep.beats if x.beat_id == resolved_by), None) if resolved_by else None
    return b.start_s if b else None


def overlap_ratio(a: Segment, b: Segment) -> float:
    if a.ep != b.ep:
        return 0.0
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    shorter = min(a.duration_s, b.duration_s)
    return inter / shorter if shorter > 0 else 0.0


def build_candidates(ep: Episode, tags: EpisodeTags, sig: dict[str, BeatSignals], cfg: ScoringConfig) -> list[Candidate]:
    by_id = {b.beat_id: b for b in ep.beats}
    scenes = {s.num: s for s in ep.scenes}
    w, ww, pen = cfg.window, cfg.window_weights, cfg.penalties
    out: list[Candidate] = []
    for h in tags.hooks:
        hb = by_id.get(h.beat_id)
        if hb is None or h.hook_score < cfg.hook_min:
            continue
        for c in tags.cliffhangers:
            cb = by_id.get(c.beat_id)
            if cb is None or cb.end_s <= hb.start_s:
                continue
            rs = resolved_start_of(ep, c.resolved_by)
            start, end = snap_start(ep, hb, w.hook_lead_max_s), snap_end(ep, cb, rs)
            v = check_clip_rules(ep, start, end, hb, cb, rs, w)
            if {"duration", "resolution_inside", "hook_lead"} & set(v):
                continue
            p = 0.0
            if "mid_line_cut" in v:
                p += pen.mid_line_cut
            sh, sc = scenes.get(hb.scene), scenes.get(cb.scene)
            if hb.scene != cb.scene and sh and sc and _location(sh.heading) != _location(sc.heading):
                p += pen.scene_jump
            if _uncovered_ratio(ep, start, end) > pen.camera_heavy_ratio:
                p += pen.camera_heavy
            rule = 0.5 * sig.get(hb.beat_id, BeatSignals(0)).score + 0.5 * sig.get(cb.beat_id, BeatSignals(0)).score
            lf = length_fit(end - start, w)
            score = ww.hook * h.hook_score + ww.cliff * c.cliff_score + ww.rules * rule + ww.length_fit * lf - p
            out.append(Candidate(
                cand_id=f"S-{hb.beat_id}-{cb.beat_id}", kind="single", segments=[Segment(ep=ep.ep, start_s=start, end_s=end)],
                hook_beat_id=hb.beat_id, cliff_beat_id=cb.beat_id, hook_score=h.hook_score, cliff_score=c.cliff_score,
                story_importance=round((h.story_importance + c.story_importance) / 2, 3), rule_score=round(rule, 3),
                length_fit=round(lf, 3), penalties=round(p, 3), window_score=round(score, 4),
                scene_key=f"E{ep.ep:02d}-S{hb.scene:02d}"))
    out.sort(key=lambda x: -x.window_score)
    kept: list[Candidate] = []
    for cand in out:
        if all(overlap_ratio(cand.segments[0], k.segments[0]) <= cfg.nms_overlap for k in kept):
            kept.append(cand)
        if len(kept) >= cfg.k_per_episode:
            break
    return kept
