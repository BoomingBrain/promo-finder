"""Stage 1: load a show through a reader, report parse health, fail loudly on bad data."""
import re
from pathlib import Path
from promofinder.readers import READERS
from promofinder.readers.base import ReaderError, ReadResult


class ParseHealthError(Exception):
    def __init__(self, report: str):
        super().__init__(report)
        self.report = report


def health_report(result: ReadResult) -> str:
    rows = ["ep   file                                      blocks parsed missing_ts out_of_order fail%"]
    for h in result.health:
        rows.append(f"{h.ep:<4} {h.source_file[:40]:<41} {h.n_blocks:>6} {h.n_parsed:>6} {h.n_missing_ts:>10} "
                    f"{h.n_out_of_order:>12} {h.fail_ratio * 100:>5.1f}")
    rows.append(f"episodes: {len(result.episodes)}  beats: {sum(len(e.beats) for e in result.episodes)}  "
                f"warnings: {len(result.warnings)}")
    return "\n".join(rows)


def load_show(path: Path, reader: str, fail_max_ratio: float, force: bool) -> ReadResult:
    if reader not in READERS:
        raise ReaderError(f"unknown reader {reader!r}; available: {', '.join(READERS)}")
    result = READERS[reader](Path(path))
    bad = [h for h in result.health if h.fail_ratio > fail_max_ratio]
    if bad and not force:
        lines = [f"{h.source_file}: {h.fail_ratio * 100:.1f}% of blocks failed to parse" for h in bad]
        raise ParseHealthError("Parse health check failed (use --force to continue anyway):\n"
                               + "\n".join(lines) + "\n\n" + health_report(result))
    return result


def show_slug(path: Path) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", Path(path).name.lower()).strip("-")
    return s[:60].strip("-")
