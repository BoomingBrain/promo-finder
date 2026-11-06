"""Orchestrates the stages. Every stage writes its file under outputs/<show-slug>/ and is skipped on resume."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from promofinder.bible import build_bible, render_context
from promofinder.config import AppConfig, load_config
from promofinder.judge import judge_all
from promofinder.meta import RunMeta
from promofinder.models import Bible, Candidate, EpisodeTags, Judgement
from promofinder.parse import health_report, load_show, show_slug
from promofinder.rank import finalists, score_all
from promofinder.readers.base import ReadResult
from promofinder.report import write_outputs
from promofinder.signals import compute_signals
from promofinder.tagger import build_transcript, tag_episode
from promofinder.voting import merge_votes
from promofinder.windows import build_candidates


def parse_stages(s: str) -> set[int]:
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


@dataclass
class RunOptions:
    input: Path
    out_root: Path = Path("outputs")
    config_dir: Path = Path("config")
    cache_dir: Path = Path("cache")
    reader: str = "kuku_v1"
    stages: set[int] = field(default_factory=lambda: set(range(1, 12)))
    model_overrides: dict[str, str] = field(default_factory=dict)
    votes: int | None = None
    feedback: Path | None = None
    dry_run: bool = False
    fresh: bool = False
    force: bool = False


def prepare_config(opts: RunOptions) -> AppConfig:
    cfg = load_config(opts.config_dir)
    for stage, model in opts.model_overrides.items():
        if stage not in cfg.models.stages:
            raise ValueError(f"unknown stage for --model: {stage}")
        chain = cfg.models.stages[stage].models
        cfg.models.stages[stage].models = [model] + [m for m in chain if m != model]
    if opts.votes is not None:
        cfg.models.votes = opts.votes
    return cfg


def estimate(read: ReadResult, opts: RunOptions, cfg: AppConfig) -> list[tuple[str, int, int]]:
    s, sc, v = opts.stages, cfg.scoring, max(1, cfg.models.votes)
    rows = []
    if 3 in s:
        rows.append(("bible", 1, 6000))
    if 4 in s:
        tok = sum(int(len(build_transcript(e, compute_signals(e, cfg.signals), cfg.signals)) / 4 + 1500 + 800)
                  for e in read.episodes)
        rows.append(("tagger", len(read.episodes) * v, tok * v))
    if 6 in s:
        rows.append(("stitcher", 1, 8000))
    if 7 in s:
        rows.append(("judge", sc.judge_top_n, sc.judge_top_n * 1500))
        rows.append(("hook3s", sc.judge_top_n, sc.judge_top_n * 300))
    if 8 in s:
        rows.append(("refine", sc.finalists_n, sc.finalists_n * 2500))
    if 9 in s:
        rows.append(("pairwise", sc.pairwise.rounds * math.ceil(sc.finalists_n / 2), sc.pairwise.rounds * math.ceil(sc.finalists_n / 2) * 2000))
    return rows


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _default_client_factory(cfg: AppConfig, cache_dir: Path, meta: RunMeta):
    from promofinder.llm import LLMClient
    return LLMClient(cfg.models, cache_dir, meta)


def run(opts: RunOptions, client_factory=None) -> Path:
    cfg = prepare_config(opts)
    meta = RunMeta()
    read = load_show(opts.input, opts.reader, cfg.scoring.parse_fail_max_ratio, opts.force)
    out = Path(opts.out_root) / show_slug(opts.input)
    out.mkdir(parents=True, exist_ok=True)
    episodes = {e.ep: e for e in read.episodes}
    with meta.stage_timer("1_parse"):
        _write_json(out / "episodes.json", [e.model_dump(exclude={"beats"}) for e in read.episodes])
        with open(out / "beats.jsonl", "w", encoding="utf-8") as f:
            for e in read.episodes:
                for b in e.beats:
                    f.write(json.dumps(b.model_dump(), ensure_ascii=False) + "\n")
        (out / "parse_health.txt").write_text(health_report(read), encoding="utf-8")
        (out / "parse_warnings.txt").write_text("\n".join(read.warnings), encoding="utf-8")
    if opts.dry_run:
        rows = estimate(read, opts, cfg)
        print(f"{'stage':<10}{'calls':>8}{'est_tokens':>14}")
        for name, calls, tok in rows:
            print(f"{name:<10}{calls:>8}{tok:>14}")
        print(f"{'total':<10}{sum(r[1] for r in rows):>8}{sum(r[2] for r in rows):>14}")
        return out

    meta.set("stages", sorted(opts.stages))
    client = None

    def get_client():
        nonlocal client
        if client is None:
            client = (client_factory or _default_client_factory)(cfg, Path(opts.cache_dir), meta)
        return client

    with meta.stage_timer("2_signals"):
        signals = {e.ep: compute_signals(e, cfg.signals) for e in read.episodes}
        _write_json(out / "signals.json", {bid: {"score": s.score, "hints": s.hints}
                                           for sig in signals.values() for bid, s in sig.items()})

    bible: Bible | None = None
    bpath = out / "bible.json"
    if bpath.exists() and not (opts.fresh and 3 in opts.stages):
        bible = Bible.model_validate_json(bpath.read_text(encoding="utf-8"))
    elif 3 in opts.stages:
        with meta.stage_timer("3_bible"):
            bible = build_bible(read.episodes, get_client(), cfg, meta)
            if bible is not None:
                bpath.write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    contexts: dict[int, str] = {e: render_context(bible, e) for e in episodes}
    anchors = ""
    feedback_examples = ""

    tags: dict[int, EpisodeTags] = {}
    tag_dir = out / "tags"; tag_dir.mkdir(exist_ok=True)
    if 4 in opts.stages:
        with meta.stage_timer("4_tagging"):
            votes = max(1, cfg.models.votes)
            temp = cfg.models.vote_temperature if votes > 1 else None

            def tag_one(ep_num: int):
                path = tag_dir / f"ep_{ep_num:03d}.json"
                if path.exists() and not opts.fresh:
                    return ep_num, EpisodeTags.model_validate_json(path.read_text(encoding="utf-8"))
                runs = [tag_episode(episodes[ep_num], signals[ep_num], get_client(), cfg, meta,
                                    context=contexts.get(ep_num, ""), anchors=anchors,
                                    feedback_examples=feedback_examples, vote_index=v, temperature=temp)
                        for v in range(votes)]
                runs = [r for r in runs if r is not None]
                if not runs:
                    return ep_num, None
                merged = merge_votes(runs, votes)
                path.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
                return ep_num, merged

            with ThreadPoolExecutor(max_workers=max(1, cfg.models.concurrency)) as ex:
                for ep_num, t in ex.map(tag_one, sorted(episodes)):
                    if t is not None:
                        tags[ep_num] = t
            agr = [t.agreement for t in tags.values() if t.agreement is not None]
            if agr:
                meta.set("vote_agreement", round(sum(agr) / len(agr), 4))
    else:
        for p in sorted(tag_dir.glob("ep_*.json")):
            t = EpisodeTags.model_validate_json(p.read_text(encoding="utf-8")); tags[t.ep] = t

    cands: list[Candidate] = []
    if 5 in opts.stages:
        with meta.stage_timer("5_windows"):
            for ep_num in sorted(tags):
                cands += build_candidates(episodes[ep_num], tags[ep_num], signals[ep_num], cfg.scoring)
    # [Task 18 hook] stage 6: stitched candidates appended here
    _write_json(out / "candidates.json", [c.model_dump() for c in cands])

    judgements: dict[str, Judgement] = {}
    jpath = out / "judgements.json"
    if jpath.exists() and not opts.fresh:
        judgements = {k: Judgement.model_validate(v) for k, v in json.loads(jpath.read_text(encoding="utf-8")).items()}
    if 7 in opts.stages:
        with meta.stage_timer("7_judge"):
            top = sorted(cands, key=lambda c: -c.window_score)[:cfg.scoring.judge_top_n]
            todo = [c for c in top if c.cand_id not in judgements]
            if todo:
                judgements.update(judge_all(todo, episodes, get_client(), cfg, meta, feedback_examples))
            _write_json(jpath, {k: v.model_dump() for k, v in judgements.items()})

    with meta.stage_timer("10_rank"):
        promos = score_all(cands, judgements, tags, bible, cfg.scoring)
        finals, rest = finalists(promos, cfg.scoring.finalists_n)
    # [Task 16 hook] stage 8: refine finalists
    # [Task 17 hook] stage 9: pairwise-rank finalists
    promos = finals + rest
    for i, p in enumerate(promos, 1):
        p.rank = i
    meta.set("counts", {"episodes": len(episodes), "tagged_episodes": len(tags), "candidates": len(cands),
                        "judged": len(judgements), "promos": len(promos)})
    write_outputs(out, promos, episodes, tags, read, meta, cfg)
    return out
