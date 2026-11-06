"""Timestamp parsing/formatting. Internally times are float seconds."""
import re

_TS = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def parse_ts(s: str) -> float:
    """Parse MM:SS.ss, M:SS or HH:MM:SS.ss (optionally wrapped in parentheses)."""
    t = s.strip().strip("()").strip()
    m = _TS.match(t)
    if not m:
        raise ValueError(f"unparseable timestamp: {s!r}")
    h, mnt, sec = m.groups()
    return round(int(h or 0) * 3600 + int(mnt) * 60 + float(sec), 3)


def format_ts(sec: float) -> str:
    """Format seconds as MM:SS.ss (minutes may exceed 59)."""
    cs = int(round(sec * 100))
    m, rem = divmod(cs, 6000)
    return f"{m:02d}:{rem // 100:02d}.{rem % 100:02d}"
