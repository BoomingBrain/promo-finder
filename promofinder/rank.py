"""Stage 10 scoring: final score with flags, diversity re-rank, finalist selection."""
from promofinder.config import ScoringConfig
from promofinder.models import Bible, Candidate, EpisodeTags, Judgement, Promo


def final_score(c: Candidate, j: Judgement | None, cliff_spoiler: str, cfg: ScoringConfig) -> tuple[float, list[str]]:
    fw, fl = cfg.final_weights, cfg.flags
    standalone = j.standalone if j else 5.0
    s = (fw.hook * c.hook_score + fw.cliff * c.cliff_score + fw.standalone * standalone
         + fw.story_importance * c.story_importance + fw.rules * c.rule_score)
    flags = []
    if cliff_spoiler == "major" or (j and j.spoiler_risk):
        s -= fl.spoiler_major; flags.append("spoiler_major")
    if j and j.resolution_leak:
        s -= fl.resolution_leak; flags.append("resolution_leak")
    if j and j.hook3s is not None and j.hook3s < cfg.hook3s_min:
        s -= fl.hook3s_low; flags.append("hook3s_low")
    if c.segments[0].ep <= cfg.early_episode.max_ep:
        s += cfg.early_episode.bonus; flags.append("early_episode")
    if j is None:
        flags.append("unjudged")
    return s, flags


def act_of(ep: int, bible: Bible | None) -> str | None:
    if bible is None:
        return None
    for a in bible.arc:
        if a.ep_start <= ep <= a.ep_end:
            return a.name
    return None


def diversify(promos: list[Promo], bible: Bible | None, cfg: ScoringConfig) -> list[Promo]:
    d = cfg.diversity
    placed, skipped, scene_n, act_n = [], [], {}, {}
    rest = list(promos)
    while rest and len(placed) < d.top_k:
        p = rest.pop(0)
        sk, act = p.candidate.scene_key, act_of(p.candidate.segments[0].ep, bible)
        if scene_n.get(sk, 0) >= d.max_per_scene or (act is not None and act_n.get(act, 0) >= d.max_per_act):
            skipped.append(p); continue
        placed.append(p)
        scene_n[sk] = scene_n.get(sk, 0) + 1
        if act is not None:
            act_n[act] = act_n.get(act, 0) + 1
    out = placed + skipped + rest
    for i, p in enumerate(out, 1):
        p.rank = i
    return out


def score_all(cands: list[Candidate], judgements: dict[str, Judgement], tags: dict[int, EpisodeTags],
              bible: Bible | None, cfg: ScoringConfig, episodes: dict | None = None) -> list[Promo]:
    from promofinder.quality import TALKY_RATIO, speech_ratio
    cliff_spoiler = {c.beat_id: c.spoiler_level for t in tags.values() for c in t.cliffhangers}
    promos = []
    for c in cands:
        j = judgements.get(c.cand_id)
        s, flags = final_score(c, j, cliff_spoiler.get(c.cliff_beat_id, "none"), cfg)
        if episodes is not None and speech_ratio(c, episodes) >= TALKY_RATIO:
            s -= cfg.flags.talky; flags.append("talky")
        promos.append(Promo(rank=0, candidate=c, judgement=j, final_score=round(s, 4), flags=flags))
    promos.sort(key=lambda p: -p.final_score)
    return diversify(promos, bible, cfg)


def finalists(promos: list[Promo], n: int) -> tuple[list[Promo], list[Promo]]:
    return promos[:n], promos[n:]
