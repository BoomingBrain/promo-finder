"""Pydantic data models shared by every stage."""
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, field_validator


def clamp_score(v) -> float:
    """Coerce '8', '8/10', 8.0 into a float clamped to [1, 10]."""
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            raise ValueError(f"not a score: {v!r}")
        v = float(m.group())
    return float(min(10.0, max(1.0, float(v))))


class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Beat(BaseModel):
    ep: int
    beat_id: str
    idx: int
    scene: int
    type: Literal["CAMERA", "ACTION", "DIALOGUE", "SFX", "OTHER"]
    speaker: str | None = None
    emotion_tags: list[str] = []
    text: str
    start_s: float
    end_s: float


class Scene(BaseModel):
    num: int
    heading: str
    start_s: float
    end_s: float


class Episode(BaseModel):
    ep: int
    logline: str
    continuity_notes: str
    duration_s: float
    scenes: list[Scene]
    beats: list[Beat]
    source_file: str


class ParseHealth(BaseModel):
    ep: int
    source_file: str
    n_blocks: int
    n_parsed: int
    n_missing_ts: int
    n_out_of_order: int
    fail_ratio: float


class HookTag(_LLMModel):
    beat_id: str
    hook_score: float
    story_importance: float
    kind: str = "other"
    why: str = ""
    quote: str = ""
    _c = field_validator("hook_score", "story_importance", mode="before")(lambda cls, v: clamp_score(v))


class CliffTag(_LLMModel):
    beat_id: str
    cliff_score: float
    story_importance: float
    open_question: str = ""
    resolved_by: str | None = None
    spoiler_level: Literal["none", "minor", "major"] = "none"
    quote: str = ""
    _c = field_validator("cliff_score", "story_importance", mode="before")(lambda cls, v: clamp_score(v))


class EpisodeTags(BaseModel):
    ep: int
    hooks: list[HookTag]
    cliffhangers: list[CliffTag]
    context_beats: list[str] = []
    votes: int = 1
    agreement: float | None = None


class Segment(BaseModel):
    ep: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


class Candidate(BaseModel):
    cand_id: str
    kind: Literal["single", "stitched"]
    segments: list[Segment]
    hook_beat_id: str
    cliff_beat_id: str
    hook_score: float
    cliff_score: float
    story_importance: float
    rule_score: float
    length_fit: float
    penalties: float
    window_score: float
    scene_key: str

    @property
    def duration_s(self) -> float:
        return round(sum(s.duration_s for s in self.segments), 3)


class Judgement(_LLMModel):
    cand_id: str
    standalone: float
    confusing_bits: list[str] = []
    hook_text: str = ""
    cliffhanger_text: str = ""
    why_it_works: str = ""
    spoiler_risk: bool = False
    resolution_leak: bool = False
    hook3s: float | None = None
    hook3s_why: str | None = None
    _c = field_validator("standalone", mode="before")(lambda cls, v: clamp_score(v))
    _h = field_validator("hook3s", mode="before")(lambda cls, v: None if v is None else clamp_score(v))


class Promo(BaseModel):
    rank: int
    candidate: Candidate
    judgement: Judgement | None = None
    final_score: float
    elo: float | None = None
    post_first_because: str | None = None
    flags: list[str] = []
    original_segments: list[Segment] | None = None


class Act(_LLMModel):
    name: str
    ep_start: int
    ep_end: int
    summary: str = ""


class Reveal(_LLMModel):
    desc: str
    ep: int
    setup_eps: list[int] = []


class Bible(_LLMModel):
    premise: str
    characters: dict[str, str]
    arc: list[Act]
    episode_summaries: dict[int, str]
    major_reveals: list[Reveal] = []
