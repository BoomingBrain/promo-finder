"""Slate-level duplicate check: one LLM call flags finalists that repeat a better-ranked finalist's core moment
(e.g. the same cliffhanger staged again in the next episode). Code validates the indices and only ever drops
the LOWER-ranked clip. Word-overlap matching was tried first and could not see these: the script describes the
same beat in different words."""
from pydantic import BaseModel, ConfigDict
from promofinder.config import AppConfig
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Promo
from promofinder.prompts import load_prompt


class Dup(BaseModel):
    model_config = ConfigDict(extra="ignore")
    keep: int
    drop: int
    why: str = ""


class DedupeOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    duplicates: list[Dup] = []


def dedupe_finalists(finalists: list[Promo], client, cfg: AppConfig, meta: RunMeta) -> tuple[list[Promo], list[Promo]]:
    if len(finalists) < 2:
        return finalists, []
    lines = []
    for i, p in enumerate(finalists, 1):
        j = p.judgement
        lines.append(f"{i} | ep {p.candidate.segments[0].ep} | {j.hook_text if j else '?'} | {j.cliffhanger_text if j else '?'}")
    prompt = load_prompt("dedupe").safe_substitute(items="\n".join(lines))
    try:
        out = client.complete_json("pairwise", [{"role": "user", "content": prompt}], DedupeOut)
    except LLMStageError as e:
        meta.warn(f"dedupe failed ({e}); slate kept as is")
        return finalists, []
    drop: dict[int, int] = {}
    for d in out.duplicates:
        if 1 <= d.keep < d.drop <= len(finalists) and d.keep not in drop:
            drop.setdefault(d.drop, d.keep)
    kept, dropped = [], []
    for i, p in enumerate(finalists, 1):
        if i in drop:
            k = finalists[drop[i] - 1].candidate.cand_id
            dropped.append(p.model_copy(update={"flags": p.flags + [f"duplicate_of_{k}"]}))
            meta.incr("dedupe_dropped")
        else:
            kept.append(p)
    return kept, dropped
