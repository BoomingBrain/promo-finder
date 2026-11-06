"""Render a clip as plain screenplay text for blind judging: no beat IDs, no timestamps, no show context."""
from promofinder.config import SignalsConfig
from promofinder.models import Candidate, Episode
from promofinder.signals import is_visual


def _line(b) -> str:
    if b.type == "DIALOGUE":
        tags = f" ({', '.join(b.emotion_tags)})" if b.emotion_tags else ""
        return f'{b.speaker}{tags}: "{b.text}"'
    return f"[{b.type}] {b.text}"


def clip_text(cand: Candidate, episodes: dict[int, Episode], sig_cfg: SignalsConfig,
              first_seconds: float | None = None) -> str:
    parts = []
    segs = cand.segments[:1] if first_seconds is not None else cand.segments
    for seg in segs:
        end = seg.start_s + first_seconds if first_seconds is not None else seg.end_s
        ep = episodes[seg.ep]
        lines = [_line(b) for b in ep.beats if seg.start_s - 1e-6 <= b.start_s < end - 1e-6 and is_visual(b, sig_cfg)]
        parts.append("\n".join(lines))
    return "\n--- CUT TO LATER ---\n".join(parts)
