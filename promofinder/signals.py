"""Stage 2: cheap rule-based per-beat signals (score boosts + prompt hints)."""
from dataclasses import dataclass, field
from promofinder.config import SignalsConfig
from promofinder.models import Beat, Episode


@dataclass
class BeatSignals:
    score: float
    hints: list[str] = field(default_factory=list)


def is_visual(beat: Beat, cfg: SignalsConfig) -> bool:
    if beat.type != "CAMERA":
        return True
    low = beat.text.lower()
    return any(term.lower() in low for term in cfg.visual_terms)


def compute_signals(ep: Episode, cfg: SignalsConfig) -> dict[str, BeatSignals]:
    pos, den = cfg.position, cfg.density
    last_in_scene = {}
    for b in ep.beats:
        last_in_scene[b.scene] = b.beat_id
    dialogue_starts = [b.start_s for b in ep.beats if b.type == "DIALOGUE"]
    out: dict[str, BeatSignals] = {}
    for b in ep.beats:
        score, hints = 0.0, []
        low = b.text.lower()
        for kw, w in cfg.keywords.items():
            if kw.lower() in low:
                score += w; hints.append(f"kw:{kw}")
        for tag in b.emotion_tags:
            for emo, w in cfg.emotion_intensity.items():
                if emo in tag.lower():
                    score += w; hints.append(f"emo:{emo}")
        if b.type == "SFX":
            for cue, w in cfg.sfx_cues.items():
                if cue in low:
                    score += w; hints.append(f"sfx:{cue}")
        if b.end_s >= ep.duration_s - pos["episode_end_window_s"]:
            score += pos["episode_end"]; hints.append("ep_end")
        if last_in_scene.get(b.scene) == b.beat_id:
            score += pos["scene_boundary"]; hints.append("scene_end")
        if b.type == "DIALOGUE":
            stripped = b.text.rstrip(" \"'”’")
            if stripped.endswith("?"):
                score += pos["question"]; hints.append("question")
            if stripped.endswith("!"):
                score += pos["exclamation"]; hints.append("exclamation")
        n = sum(1 for t in dialogue_starts if b.start_s <= t < b.start_s + den["window_s"])
        d = min(den["max_points"], den["max_points"] * n / den["lines_for_max"])
        if d >= den["max_points"]:
            hints.append("dense_dialogue")
        score += d
        out[b.beat_id] = BeatSignals(score=round(min(10.0, max(0.0, score)), 3), hints=hints)
    return out
