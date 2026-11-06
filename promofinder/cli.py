"""Command line: python -m promofinder run|eval ..."""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from promofinder.parse import ParseHealthError
from promofinder.pipeline import RunOptions, parse_stages, prepare_config, run
from promofinder.readers.base import ReaderError


def _models(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--model expects stage=model, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="promofinder", description="Find ranked promo clips in microdrama screenplays")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline on a show folder")
    r.add_argument("--input", required=True, type=Path, help="folder of episode .txt screenplays")
    r.add_argument("--out", default=Path("outputs"), type=Path, help="output root (default: outputs)")
    r.add_argument("--config", default=Path("config"), type=Path)
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
    e.add_argument("--gold", default=Path("eval/gold.yaml"), type=Path)
    e.add_argument("--compare", type=Path, help="another run folder, for top-10 overlap")
    e.add_argument("--config", default=Path("config"), type=Path)
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
