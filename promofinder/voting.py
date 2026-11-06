"""Majority-vote merge of several tagging runs of one episode."""
from collections import Counter
from promofinder.models import CliffTag, EpisodeTags, HookTag


def _mode(values):
    counts = Counter(values)
    best = max(counts.values())
    return next(v for v in values if counts[v] == best)   # ties -> earliest run


def _merge(lists: list[list], need: int, numeric: tuple[str, ...], categorical: tuple[str, ...], cls):
    order, groups = [], {}
    for tags in lists:
        for t in tags:
            if t.beat_id not in groups:
                groups[t.beat_id] = []; order.append(t.beat_id)
            groups[t.beat_id].append(t)
    kept = []
    for bid in order:
        g = groups[bid]
        if len(g) < need:
            continue
        data = g[0].model_dump()
        for f in numeric:
            data[f] = round(sum(getattr(t, f) for t in g) / len(g), 3)
        for f in categorical:
            data[f] = _mode([getattr(t, f) for t in g])
        kept.append(cls(**data))
    return kept, len(order)


def merge_votes(runs: list[EpisodeTags], votes: int) -> EpisodeTags:
    if votes <= 1 or len(runs) == 1:
        return runs[0].model_copy(update={"agreement": None, "votes": 1})
    need = votes // 2 + 1
    hooks, nh = _merge([r.hooks for r in runs], need, ("hook_score", "story_importance"), ("kind",), HookTag)
    cliffs, nc = _merge([r.cliffhangers for r in runs], need, ("cliff_score", "story_importance"),
                        ("spoiler_level", "resolved_by"), CliffTag)
    ctx_counts = Counter(c for r in runs for c in set(r.context_beats))
    ctx = [c for c in dict.fromkeys(c for r in runs for c in r.context_beats) if ctx_counts[c] >= need]
    union = nh + nc
    agreement = (len(hooks) + len(cliffs)) / union if union else 1.0
    return EpisodeTags(ep=runs[0].ep, hooks=hooks, cliffhangers=cliffs, context_beats=ctx, votes=votes, agreement=agreement)
