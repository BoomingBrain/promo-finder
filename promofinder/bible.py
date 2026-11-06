"""Stage 3: one call that compresses all loglines + continuity notes into a show bible, and the two-layer context."""
import re
from pydantic import field_validator
from promofinder.config import AppConfig
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Bible, Episode
from promofinder.prompts import load_prompt


class BibleOut(Bible):
    @field_validator("episode_summaries", mode="before")
    @classmethod
    def _int_keys(cls, v):
        if not isinstance(v, dict):
            return v
        out = {}
        for k, s in v.items():
            m = re.search(r"\d+", str(k))
            if m:
                out[int(m.group())] = str(s)
        return out


def build_bible(episodes: list[Episode], client, cfg: AppConfig, meta: RunMeta) -> Bible | None:
    lines = "\n".join(f"Episode {e.ep}: {e.logline} / {e.continuity_notes}" for e in sorted(episodes, key=lambda e: e.ep))
    prompt = load_prompt("bible").safe_substitute(n=len(episodes), episodes=lines)
    try:
        out = client.complete_json("bible", [{"role": "user", "content": prompt}], BibleOut)
    except LLMStageError as e:
        meta.warn(f"bible failed ({e}); tagging continues without show context")
        return None
    return Bible.model_validate(out.model_dump())


def render_context(b: Bible | None, ep: int) -> str:
    if b is None:
        return ""
    chars = "\n".join(f"- {n} — {r}" for n, r in b.characters.items())
    arc = "\n".join(f"- {a.name} ({a.ep_start}–{a.ep_end})" + (f": {a.summary}" if a.summary else "") for a in b.arc)
    rev = "\n".join(f"- {r.desc} (ep {r.ep})" for r in b.major_reveals) or "- none listed"
    prev = b.episode_summaries.get(ep - 1, "none")
    nxt = b.episode_summaries.get(ep + 1, "none")
    return (f"SHOW CONTEXT (global)\nPremise: {b.premise}\nCharacters:\n{chars}\nArc:\n{arc}\nMajor reveals:\n{rev}\n\n"
            f"LOCAL: previous episode: {prev}; next episode: {nxt}")
