"""Automatic checks every promo must pass (reported in report.md)."""
from promofinder.config import ScoringConfig
from promofinder.models import Episode, EpisodeTags, Promo
from promofinder.windows import check_clip_rules, resolved_start_of


def _on_boundary(ep: Episode, t: float) -> bool:
    return any(abs(b.start_s - t) <= 0.01 or abs(b.end_s - t) <= 0.01 for b in ep.beats)


def check_promo(p: Promo, episodes: dict[int, Episode], tags: dict[int, EpisodeTags], cfg: ScoringConfig) -> list[str]:
    c, w = p.candidate, cfg.window
    v: list[str] = []
    for seg in c.segments:
        ep = episodes.get(seg.ep)
        if ep is None:
            return ["unknown_episode"]
        if not (_on_boundary(ep, seg.start_s) and _on_boundary(ep, seg.end_s)):
            v.append("not_on_boundary")
    if c.kind == "single":
        seg = c.segments[0]; ep = episodes[seg.ep]
        by_id = {b.beat_id: b for b in ep.beats}
        hook, cliff = by_id.get(c.hook_beat_id), by_id.get(c.cliff_beat_id)
        if hook is None or cliff is None:
            return v + ["unknown_beat"]
        t = tags.get(seg.ep)
        ct = next((x for x in t.cliffhangers if x.beat_id == cliff.beat_id), None) if t else None
        rs = resolved_start_of(ep, ct.resolved_by) if ct else None
        v += check_clip_rules(ep, seg.start_s, seg.end_s, hook, cliff, rs, w)
    else:
        if any(s.duration_s < cfg.stitch.min_segment_s - 1e-6 for s in c.segments):
            v.append("too_short_segment")
        if not (w.min_s - 1e-6 <= c.duration_s <= w.max_s + 1e-6):
            v.append("total_duration")
        keys = [(s.ep, s.start_s) for s in c.segments]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            v.append("not_chronological")
    return sorted(set(v), key=v.index)


def pass_rate(promos: list[Promo], episodes, tags, cfg: ScoringConfig) -> tuple[float, dict[str, list[str]]]:
    bad = {}
    for p in promos:
        v = check_promo(p, episodes, tags, cfg)
        if v:
            bad[p.candidate.cand_id] = v
    return (1.0 - len(bad) / len(promos)) if promos else 1.0, bad
