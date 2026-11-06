"""Stage 9: Swiss-system pairwise comparisons of finalists with Elo; tracks first-position bias."""
import random
from typing import Literal
from pydantic import BaseModel, ConfigDict
from promofinder.cliptext import clip_text
from promofinder.config import AppConfig
from promofinder.llm import LLMStageError
from promofinder.meta import RunMeta
from promofinder.models import Episode, Promo
from promofinder.prompts import load_prompt


class PairOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    winner: Literal["A", "B"]
    why: str = ""


def elo_update(ra: float, rb: float, a_won: bool, k: float) -> tuple[float, float]:
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    sa = 1.0 if a_won else 0.0
    return ra + k * (sa - ea), rb + k * ((1 - sa) - (1 - ea))


def swiss_pairs(ids: list[str], ratings: dict[str, float]) -> tuple[list[tuple[str, str]], str | None]:
    order = sorted(ids, key=lambda i: (-ratings[i], ids.index(i)))
    pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
    return pairs, order[-1] if len(order) % 2 else None


def rank_pairwise(finalists: list[Promo], episodes: dict[int, Episode], client, cfg: AppConfig,
                  meta: RunMeta) -> list[Promo]:
    if len(finalists) < 2:
        return finalists
    pc = cfg.scoring.pairwise
    by_id = {p.candidate.cand_id: p for p in finalists}
    ids = list(by_id)
    ratings = {i: float(pc.start_rating) for i in ids}
    texts = {i: clip_text(by_id[i].candidate, episodes, cfg.signals) for i in ids}
    wins: dict[str, list[tuple[str, str]]] = {i: [] for i in ids}          # (beaten opponent, why)
    first_wins = total = 0
    for rnd in range(pc.rounds):
        rng = random.Random(pc.seed + rnd)
        pairs, _ = swiss_pairs(ids, ratings)
        for x, y in pairs:
            first, second = (x, y) if rng.random() < 0.5 else (y, x)
            prompt = load_prompt("pairwise").safe_substitute(a=texts[first], b=texts[second])
            try:
                out = client.complete_json("pairwise", [{"role": "user", "content": prompt}], PairOut)
            except LLMStageError as e:
                meta.warn(f"pairwise {first} vs {second} failed ({e})")
                continue
            total += 1
            winner, loser = (first, second) if out.winner == "A" else (second, first)
            if winner == first:
                first_wins += 1; meta.incr("pairwise_first_wins")
            wins[winner].append((loser, out.why))
            ratings[winner], ratings[loser] = elo_update(ratings[winner], ratings[loser], True, pc.k_factor)
    if total:
        meta.set("pairwise_first_position_rate", round(first_wins / total, 4))
    out = []
    for i in ids:
        best = max(wins[i], key=lambda w: ratings[w[0]], default=None)
        out.append(by_id[i].model_copy(update={"elo": round(ratings[i], 1), "post_first_because": best[1] if best else None}))
    out.sort(key=lambda p: (-p.elo, -p.final_score))
    return out
