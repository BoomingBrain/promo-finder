"""'Where it may be picking badly': automatic per-promo risk notes (spec §8.5 categories) and run-level caveats."""
from promofinder.meta import RunMeta
from promofinder.models import Episode, Promo
from promofinder.windows import overlap_ratio

TALKY_RATIO = 0.8          # share of the clip covered by spoken lines
LOW_STANDALONE = 6.0


def _speech_ratio(p: Promo, episodes: dict[int, Episode]) -> float:
    total = spoken = 0.0
    for s in p.candidate.segments:
        total += s.duration_s
        ep = episodes.get(s.ep)
        if ep:
            spoken += sum(max(0.0, min(b.end_s, s.end_s) - max(b.start_s, s.start_s))
                          for b in ep.beats if b.type == "DIALOGUE")
    return spoken / total if total else 0.0


def risk_notes(p: Promo, above: list[Promo], episodes: dict[int, Episode]) -> list[str]:
    c, j, notes = p.candidate, p.judgement, []
    ep = episodes.get(c.segments[0].ep)
    hook = next((b for b in ep.beats if b.beat_id == c.hook_beat_id), None) if ep else None
    if hook is not None and hook.type != "DIALOGUE":
        notes.append(f"visual-only hook ({hook.type.lower()}): text may undersell or oversell it, check the footage")
    if j is None:
        notes.append("not judged: ranked on tagger scores only")
    else:
        if j.standalone < LOW_STANDALONE:
            bits = "; ".join(j.confusing_bits[:2])
            notes.append(f"may need context (stranger score {j.standalone:.0f}/10)" + (f": {bits}" if bits else ""))
        if j.resolution_leak:
            notes.append("may resolve itself before the cut")
        if j.spoiler_risk or "spoiler_major" in p.flags:
            notes.append("spoiler risk")
        if "hook3s_low" in p.flags:
            notes.append("weak first 3 seconds")
    if _speech_ratio(p, episodes) >= TALKY_RATIO:
        notes.append("too talky: almost all dialogue, little action")
    for q in above:
        if any(overlap_ratio(a, b) > 0.0 for a in c.segments for b in q.candidate.segments) or q.candidate.scene_key == c.scene_key:
            notes.append(f"near-duplicate of #{q.rank} (same scene or overlapping time)")
            break
    if c.kind == "stitched":
        notes.append("experimental multi-segment promo: check the joins")
    return notes


def run_notes(meta: RunMeta) -> list[str]:
    v, n = meta.values, meta.counters
    out = []
    cr = v.get("pairwise_consistent_rate")
    if cr is not None and cr < 0.5:
        out.append(f"Head-to-head ranking agreed with itself in only {cr:.0%} of pairs (both orders asked); "
                   "the top-15 order mostly follows the judge's scores.")
    if n.get("skipped_episodes"):
        out.append(f"{n['skipped_episodes']} episode(s) skipped: every model failed; rerun to retry them.")
    if n.get("quote_mismatches"):
        out.append(f"{n['quote_mismatches']} tagged lines were rejected because the quoted text did not match the script.")
    if n.get("fallback_switches"):
        out.append(f"{n['fallback_switches']} calls fell back to the second model (rate limits or errors).")
    out.append("Visual-only moments (a slap written only as CAMERA/ACTION) are judged from text and are a known weak spot.")
    return out
