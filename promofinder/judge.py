"""Stage 7: blind judge (sees only the clip text) + first-3-seconds hook test."""
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ConfigDict, field_validator
from promofinder.cliptext import clip_text
from promofinder.config import AppConfig
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Candidate, Episode, Judgement, clamp_score
from promofinder.prompts import load_prompt


class JudgeOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    standalone: float
    confusing_bits: list[str] = []
    hook_text: str = ""
    cliffhanger_text: str = ""
    why_it_works: str = ""
    spoiler_risk: bool = False
    resolution_leak: bool = False
    _c = field_validator("standalone", mode="before")(lambda cls, v: clamp_score(v))


class Hook3sOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    score: float
    why: str = ""
    _c = field_validator("score", mode="before")(lambda cls, v: clamp_score(v))


def judge_candidate(cand: Candidate, episodes: dict[int, Episode], client, cfg: AppConfig, meta: RunMeta,
                    feedback_examples: str = "") -> Judgement | None:
    prompt = load_prompt("judge").safe_substitute(
        clip=clip_text(cand, episodes, cfg.signals),
        feedback=f"EDITOR FEEDBACK FROM EARLIER RUNS\n{feedback_examples}" if feedback_examples else "")
    try:
        out = client.complete_json("judge", [{"role": "user", "content": prompt}], JudgeOut)
    except LLMStageError as e:
        meta.warn(f"{cand.cand_id}: judge failed ({e})")
        return None
    return Judgement(cand_id=cand.cand_id, **out.model_dump())


def hook3s_candidate(cand: Candidate, episodes: dict[int, Episode], client, cfg: AppConfig,
                     meta: RunMeta) -> tuple[float, str] | None:
    first3 = clip_text(cand, episodes, cfg.signals, first_seconds=cfg.scoring.window.hook_lead_max_s)
    prompt = load_prompt("hook3s").safe_substitute(first3=first3 or "(nothing happens)")
    try:
        out = client.complete_json("hook3s", [{"role": "user", "content": prompt}], Hook3sOut)
    except LLMStageError as e:
        meta.warn(f"{cand.cand_id}: hook3s failed ({e})")
        return None
    return out.score, out.why


def judge_all(cands: list[Candidate], episodes: dict[int, Episode], client, cfg: AppConfig, meta: RunMeta,
              feedback_examples: str = "") -> dict[str, Judgement]:
    def one(c: Candidate):
        j = judge_candidate(c, episodes, client, cfg, meta, feedback_examples)
        if j is None:
            return None
        h = hook3s_candidate(c, episodes, client, cfg, meta)
        if h is not None:
            j.hook3s, j.hook3s_why = h
        return j

    with ThreadPoolExecutor(max_workers=max(1, cfg.models.concurrency)) as ex:
        results = list(ex.map(one, cands))
    return {j.cand_id: j for j in results if j is not None}
