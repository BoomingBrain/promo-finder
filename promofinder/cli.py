"""Command line: python -m promofinder run|eval ..."""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from promofinder.parse import ParseHealthError
from promofinder.pipeline import RunOptions, parse_stages, prepare_config, run
from promofinder.readers.base import ReaderError


REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_path(rel: str) -> Path:
    """Use ./rel when it exists (running from the repo), else the copy next to the package, so the CLI works from any folder."""
    return Path(rel) if Path(rel).exists() else REPO_ROOT / rel


def _models(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--model expects stage=model, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _experiment(args) -> int:
    import json
    from promofinder import experiments
    from promofinder.bible import build_bible
    from promofinder.llm import LLMClient
    from promofinder.meta import RunMeta
    from promofinder.models import Bible
    from promofinder.parse import load_show, show_slug
    cfg = prepare_config(RunOptions(input=args.input, config_dir=args.config))
    if not os.environ.get(cfg.models.api_key_env):
        print(f"Missing {cfg.models.api_key_env} (see .env.example)", file=sys.stderr)
        return 3
    read = load_show(args.input, args.reader, cfg.scoring.parse_fail_max_ratio, False)
    eps = {e.ep: e for e in read.episodes}
    nums = [int(n) for n in args.episodes.split(",") if n.strip()]
    out = Path(args.out) / show_slug(args.input); out.mkdir(parents=True, exist_ok=True)
    meta = RunMeta(); client = LLMClient(cfg.models, args.cache, meta)
    if args.which == "plan-a":
        result = experiments.plan_a(eps, nums, client, cfg, meta, out)
    else:
        bp = out / "bible.json"
        bible = Bible.model_validate_json(bp.read_text(encoding="utf-8")) if bp.exists() else build_bible(read.episodes, client, cfg, meta)
        experiments.hindsight(eps, nums, bible, client, cfg, meta, out)
        result = json.loads((out / "experiments" / "exp_b_summary.json").read_text(encoding="utf-8"))
    meta.save(out / "experiments" / f"meta_{args.which}.json")
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="promofinder", description="Find ranked promo clips in microdrama screenplays")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline on a show folder")
    r.add_argument("--input", required=True, type=Path, help="folder of episode .txt screenplays")
    r.add_argument("--out", default=Path("outputs"), type=Path, help="output root (default: outputs)")
    r.add_argument("--config", default=_repo_path("config"), type=Path)
    r.add_argument("--cache", default=Path("cache"), type=Path)
    r.add_argument("--reader", default="kuku_v1")
    r.add_argument("--stages", default="1-11", help="e.g. 1-11 or 1,2,4,5,7,10")
    r.add_argument("--model", action="append", help="stage=model, puts that model first in the stage's chain")
    r.add_argument("--votes", type=int)
    r.add_argument("--feedback", type=Path, help="editor-marked promos.csv from an earlier run")
    r.add_argument("--dry-run", action="store_true", help="parse and estimate calls/tokens; no model calls")
    r.add_argument("--fresh", action="store_true", help="ignore saved stage outputs and recompute")
    r.add_argument("--force", action="store_true", help="continue even if parse health check fails")
    e = sub.add_parser("eval", help="score a run against the gold set")
    e.add_argument("--run", required=True, type=Path, help="run output folder")
    e.add_argument("--gold", default=_repo_path("eval/gold.yaml"), type=Path)
    e.add_argument("--compare", type=Path, help="another run folder, for top-10 overlap")
    e.add_argument("--config", default=_repo_path("config"), type=Path)
    x = sub.add_parser("experiment", help="solution-doc experiments: plan-a (naive timestamps), hindsight (context bias)")
    x.add_argument("which", choices=["plan-a", "hindsight"])
    x.add_argument("--input", required=True, type=Path)
    x.add_argument("--episodes", required=True, help="comma-separated episode numbers")
    x.add_argument("--out", default=Path("outputs"), type=Path)
    x.add_argument("--config", default=_repo_path("config"), type=Path)
    x.add_argument("--cache", default=Path("cache"), type=Path)
    x.add_argument("--reader", default="kuku_v1")
    return ap


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        opts = RunOptions(input=args.input, out_root=args.out, config_dir=args.config, cache_dir=args.cache,
                          reader=args.reader, stages=parse_stages(args.stages), model_overrides=_models(args.model),
                          votes=args.votes, feedback=args.feedback, dry_run=args.dry_run, fresh=args.fresh, force=args.force)
        if not opts.dry_run:
            key_env = prepare_config(opts).models.api_key_env
            if not os.environ.get(key_env):
                print(f"Missing {key_env} (see .env.example)", file=sys.stderr)
                return 3
        try:
            out = run(opts)
        except ParseHealthError as e:
            print(e.report, file=sys.stderr)
            return 2
        except ReaderError as e:
            print(f"Reader error: {e}", file=sys.stderr)
            return 2
        print(f"Done. Outputs in {out}" if not opts.dry_run else f"(dry run; parse outputs in {out})")
        return 0
    if args.cmd == "experiment":
        return _experiment(args)
    if args.cmd == "eval":
        from promofinder.evaluate import evaluate
        r = evaluate(args.run, args.gold, args.compare, args.config)
        for k, v in r.items():
            if k != "gold_detail":
                print(f"{k:<28}{v}")
        print("gold picks (best rank of a promo covering >= 50%):")
        for gid, d in r["gold_detail"].items():
            print(f"  {gid:<6} rank {d['best_rank'] if d['best_rank'] is not None else 'missed':<8} overlap {d['overlap']:.0%}")
        return 0
    return 1
