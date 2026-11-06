"""Stage 8: code enumerates rule-valid cut shifts of ±N shots; the LLM only picks a letter."""
import string
from pydantic import BaseModel, ConfigDict
from promofinder.cliptext import clip_text
from promofinder.config import AppConfig, ScoringConfig
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Candidate, Episode, EpisodeTags, Promo, Segment
from promofinder.prompts import load_prompt
from promofinder.windows import check_clip_rules, resolved_start_of, shot_at


class RefineOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    choice: str
    why: str = ""


def options(c: Candidate, ep: Episode, tags: EpisodeTags | None, cfg: ScoringConfig) -> list[Segment]:
    if c.kind != "single":
        return []
    seg = c.segments[0]
    shots = [b for b in ep.beats if b.type == "CAMERA"]
    s_shot, e_shot = shot_at(ep, seg.start_s), shot_at(ep, seg.end_s - 1e-6)
    if s_shot is None or e_shot is None:
        return [seg]
    i, j = shots.index(s_shot), shots.index(e_shot)
    by_id = {b.beat_id: b for b in ep.beats}
    hook, cliff = by_id.get(c.hook_beat_id), by_id.get(c.cliff_beat_id)
    if hook is None or cliff is None:
        return [seg]
    ct = next((x for x in tags.cliffhangers if x.beat_id == cliff.beat_id), None) if tags else None
    rs = resolved_start_of(ep, ct.resolved_by) if ct else None
    n = cfg.refine.max_shift_shots
    found = []
    for ds in range(-n, n + 1):
        for de in range(-n, n + 1):
            if (ds, de) == (0, 0) or not (0 <= i + ds < len(shots) and 0 <= j + de < len(shots)):
                continue
            s, e = shots[i + ds].start_s, shots[j + de].end_s
            if not check_clip_rules(ep, s, e, hook, cliff, rs, cfg.window):
                found.append((abs(ds) + abs(de), ds, de, Segment(ep=seg.ep, start_s=s, end_s=e)))
    found.sort(key=lambda t: t[:3])
    out = [seg]
    for *_, sg in found:
        if sg not in out:
            out.append(sg)
    return out[:cfg.refine.max_options]


def refine_promo(p: Promo, episodes: dict[int, Episode], tags: dict[int, EpisodeTags], client, cfg: AppConfig,
                 meta: RunMeta) -> Promo:
    c = p.candidate
    if c.kind != "single":
        return p
    ops = options(c, episodes[c.segments[0].ep], tags.get(c.segments[0].ep), cfg.scoring)
    if len(ops) < 2:
        return p
    letters = string.ascii_uppercase[:len(ops)]
    blocks = [f"OPTION {L}:\n" + clip_text(c.model_copy(update={"segments": [o]}), episodes, cfg.signals)
              for L, o in zip(letters, ops)]
    prompt = load_prompt("refine").safe_substitute(options="\n\n".join(blocks))
    try:
        out = client.complete_json("refine", [{"role": "user", "content": prompt}], RefineOut)
    except LLMStageError as e:
        meta.warn(f"{c.cand_id}: refine failed ({e})")
        return p
    pick = out.choice.strip().upper()[:1]
    if pick not in letters:
        meta.incr("invalid_refine_picks")
        return p
    k = letters.index(pick)
    if k == 0:
        return p
    meta.incr("refined_cuts")
    return p.model_copy(update={"original_segments": [ops[0]],
                                "candidate": c.model_copy(update={"segments": [ops[k]]})})
