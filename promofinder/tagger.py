"""Stage 4: tag hook / cliffhanger beats of one episode, by beat ID only, with quote check + validation."""
import difflib
import re
from pydantic import BaseModel
from promofinder.config import AppConfig, SignalsConfig
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import CliffTag, Episode, EpisodeTags, HookTag
from promofinder.prompts import load_prompt
from promofinder.signals import BeatSignals, is_visual
from promofinder.timeutil import format_ts


class RawTags(BaseModel):
    hooks: list[HookTag] = []
    cliffhangers: list[CliffTag] = []
    context_beats: list[str] = []


def _speaker(b) -> str:
    if b.type != "DIALOGUE":
        return ""
    tags = f" ({', '.join(b.emotion_tags)})" if b.emotion_tags else ""
    return f" {b.speaker}{tags}"


def build_transcript(ep: Episode, sig: dict[str, BeatSignals], cfg: SignalsConfig) -> str:
    lines = []
    for b in ep.beats:
        if not is_visual(b, cfg):
            continue
        text = f'"{b.text}"' if b.type == "DIALOGUE" else b.text
        line = f"[{b.beat_id}] {format_ts(b.start_s)}–{format_ts(b.end_s)} {b.type}{_speaker(b)}: {text}"
        s = sig.get(b.beat_id)
        if s and s.hints:
            line += f"  {{sig: {', '.join(s.hints)}}}"
        lines.append(line)
    return "\n".join(lines)


_PUNCT = re.compile(r"[^\w\s]")


def normalize(s: str) -> str:
    s = (s or "").lower()
    for a, b in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("…", "..."), ("—", "-"), ("–", "-")):
        s = s.replace(a, b)
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


def quote_matches(quote: str, text: str, min_ratio: float) -> bool:
    q, t = normalize(quote), normalize(text)
    if not q:
        return False
    if q in t:
        return True
    prefix = " ".join(t.split()[:len(q.split())])
    return difflib.SequenceMatcher(None, q, prefix).ratio() >= min_ratio


_LATER = re.compile(r"^later_episode:\d+$")


def validate_tags(raw: RawTags, ep: Episode, min_ratio: float, meta: RunMeta) -> EpisodeTags:
    by_id = {b.beat_id: b for b in ep.beats}

    def ok(tag) -> bool:
        b = by_id.get(tag.beat_id)
        if b is None:
            meta.incr("unknown_ids")
            return False
        if not quote_matches(tag.quote, b.text, min_ratio):
            meta.incr("quote_mismatches")
            meta.warn(f"E{ep.ep:02d}: quote mismatch for {tag.beat_id}: {tag.quote[:40]!r}")
            return False
        return True

    hooks = [h for h in raw.hooks if ok(h)]
    cliffs = []
    for c in raw.cliffhangers:
        if not ok(c):
            continue
        rb = (c.resolved_by or "").strip()
        if rb in by_id or rb == "next_episode" or _LATER.match(rb):
            c.resolved_by = rb
        else:
            c.resolved_by = None
        cliffs.append(c)
    ctx = []
    for cid in raw.context_beats:
        if cid in by_id:
            ctx.append(cid)
        else:
            meta.incr("unknown_ids")
    return EpisodeTags(ep=ep.ep, hooks=hooks, cliffhangers=cliffs, context_beats=ctx)


def render_tagger_messages(ep: Episode, transcript: str, context: str, anchors: str, feedback_examples: str) -> list[dict]:
    system = load_prompt("tagger").safe_substitute(
        anchors=anchors or "(none)",
        feedback=f"EDITOR FEEDBACK FROM EARLIER RUNS\n{feedback_examples}" if feedback_examples else "")
    user = load_prompt("tagger_user").safe_substitute(
        context=context or "(none provided)", ep=ep.ep, logline=ep.logline, transcript=transcript)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def tag_episode(ep: Episode, sig: dict[str, BeatSignals], client, cfg: AppConfig, meta: RunMeta, context: str = "",
                anchors: str = "", feedback_examples: str = "", vote_index: int = 0,
                temperature: float | None = None) -> EpisodeTags | None:
    msgs = render_tagger_messages(ep, build_transcript(ep, sig, cfg.signals), context, anchors, feedback_examples)
    try:
        raw = client.complete_json("tagger", msgs, RawTags, vote_index=vote_index, temperature=temperature)
    except LLMStageError as e:
        meta.incr("skipped_episodes")
        meta.warn(f"E{ep.ep:02d}: tagging skipped ({e})")
        return None
    return validate_tags(raw, ep, cfg.scoring.quote_min_ratio, meta)
