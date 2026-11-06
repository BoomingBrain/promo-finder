"""Stage 9: Swiss-system pairwise comparisons of finalists with Elo; tracks first-position bias."""
from concurrent.futures import ThreadPoolExecutor
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


def _elo(ra: float, rb: float, sa: float, k: float) -> tuple[float, float]:
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    return ra + k * (sa - ea), rb + k * ((1 - sa) - (1 - ea))


def elo_update(ra: float, rb: float, a_won: bool, k: float) -> tuple[float, float]:
    return _elo(ra, rb, 1.0 if a_won else 0.0, k)


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
        pairs, _ = swiss_pairs(ids, ratings)
        # every pair is asked in BOTH orders; the first-run judge picked the first-shown clip 86% of the time
        asks = [o for x, y in pairs for o in ((x, y), (y, x))]

        def ask(fs):
            first, second = fs
            prompt = load_prompt("pairwise").safe_substitute(a=texts[first], b=texts[second])
            try:
                return fs, client.complete_json("pairwise", [{"role": "user", "content": prompt}], PairOut)
            except LLMStageError as e:
                meta.warn(f"pairwise {first} vs {second} failed ({e})")
                return fs, None

        with ThreadPoolExecutor(max_workers=max(1, cfg.models.concurrency)) as ex:
            results = dict(ex.map(ask, asks))            # pairs in a round are independent
        for x, y in pairs:
            picks = {}
            for first, second in ((x, y), (y, x)):
                out = results.get((first, second))
                if out is None:
                    continue
                total += 1
                winner = first if out.winner == "A" else second
                if winner == first:
                    first_wins += 1; meta.incr("pairwise_first_wins")
                picks[(first, second)] = (winner, out.why)
            if len(picks) < 2:
                continue                                   # need both orders to call it
            (w1, why1), (w2, why2) = picks[(x, y)], picks[(y, x)]
            if w1 == w2:
                loser = y if w1 == x else x
                wins[w1].append((loser, why1))
                ratings[w1], ratings[loser] = elo_update(ratings[w1], ratings[loser], True, pc.k_factor)
            else:
                meta.incr("pairwise_inconsistent")         # order decided it: a draw
                ratings[x], ratings[y] = _elo(ratings[x], ratings[y], 0.5, pc.k_factor)
    if total:
        meta.set("pairwise_first_position_rate", round(first_wins / total, 4))
        meta.set("pairwise_consistent_rate", round(1 - 2 * meta.counters.get("pairwise_inconsistent", 0) / total, 4))
    out = []
    for i in ids:
        best = max(wins[i], key=lambda w: ratings[w[0]], default=None)
        out.append(by_id[i].model_copy(update={"elo": round(ratings[i], 1), "post_first_because": best[1] if best else None}))
    out.sort(key=lambda p: (-p.elo, -p.final_score))
    return out
