"""Reader for the Kuku screenplay text format (case-insensitive, whitespace-tolerant)."""
import re
from pathlib import Path
from promofinder.models import Beat, Episode, ParseHealth, Scene
from promofinder.readers.base import ReaderError, ReadResult
from promofinder.timeutil import parse_ts

I = re.IGNORECASE
RX_EPNUM = re.compile(r"^\s*EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", I)
RX_LOGLINE = re.compile(r"^\s*EPISODE LOGLINE:\s*(.*)$", I)
RX_CONT = re.compile(r"^\s*CONTINUITY NOTES:\s*(.*)$", I)
RX_SCENE = re.compile(r"^\s*SCENE\s+(\d+)\s*$", I)
RX_HEADING = re.compile(r"^\s*SCENE HEADING:\s*(.*)$", I)
RX_SSTART = re.compile(r"^\s*SCENE START:\s*(\S+)", I)
RX_SEND = re.compile(r"^\s*SCENE END:\s*(\S+)", I)
RX_BLOCK = re.compile(r"^\s*\*\*(.+?):\*\*\s*$")
RX_TIMING = re.compile(r"Start:\s*\(([^)]+)\)\s*\|\s*End:\s*\(([^)]+)\)", I)
RX_STOP = re.compile(r"^\s*(---\s*SCRIPT END|ACTIONS AND DIALOGUE:|SCENE\s+\d+\s*$|SCENE (HEADING|START|END):)", I)
QUOTES = "\"'“”‘’"
FIXED = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SFX", "SFX": "SFX"}


def _block_type(header: str) -> tuple[str, str | None, list[str]]:
    m = re.match(r"^(.*?)\s*(?:\((.*)\))?\s*$", header.strip())
    name, paren = m.group(1).strip(), m.group(2)
    tags = [t.strip() for t in paren.split(",") if t.strip()] if paren else []
    if name.upper() in FIXED:
        return FIXED[name.upper()], None, []
    if name and re.search(r"[A-Za-z]", name) and name == name.upper():
        return "DIALOGUE", name, tags
    return "OTHER", name or None, tags


def _episode_number(lines: list[str], path: Path) -> int:
    for ln in lines[:20]:
        m = RX_EPNUM.match(ln)
        if m:
            return int(m.group(1))
    m = re.search(r"\d+", path.stem)
    if m:
        return int(m.group())
    raise ReaderError(f"{path.name}: no episode number (no 'EPISODE NUMBER:' header and no number in filename)")


def read_episode(path: Path) -> tuple[Episode, ParseHealth, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    ep = _episode_number(lines, path)
    logline = cont = ""
    scenes: list[Scene] = []
    beats: list[Beat] = []
    warnings: list[str] = []
    n_blocks = n_missing = n_bad = n_ooo = 0
    cur_scene = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        if (m := RX_LOGLINE.match(ln)):
            logline = m.group(1).strip()
        elif (m := RX_CONT.match(ln)):
            cont = m.group(1).strip()
        elif (m := RX_SCENE.match(ln)):
            cur_scene = int(m.group(1))
            scenes.append(Scene(num=cur_scene, heading="", start_s=0.0, end_s=0.0))
        elif (m := RX_HEADING.match(ln)) and scenes:
            scenes[-1].heading = m.group(1).strip()
        elif (m := RX_SSTART.match(ln)) and scenes:
            scenes[-1].start_s = parse_ts(m.group(1))
        elif (m := RX_SEND.match(ln)) and scenes:
            scenes[-1].end_s = parse_ts(m.group(1))
        elif (m := RX_BLOCK.match(ln)):
            n_blocks += 1
            btype, speaker, tags = _block_type(m.group(1))
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tm = RX_TIMING.search(lines[j]) if j < len(lines) else None
            text_start = j + 1 if tm else j
            k = text_start
            text_lines = []
            while k < len(lines) and not RX_BLOCK.match(lines[k]) and not RX_STOP.match(lines[k]):
                if lines[k].strip():
                    text_lines.append(lines[k].strip())
                k += 1
            if not tm:
                n_missing += 1
                warnings.append(f"{path.name}: block '{m.group(1)}' near line {i + 1} has no timestamp; skipped")
                i = k
                continue
            try:
                start, end = parse_ts(tm.group(1)), parse_ts(tm.group(2))
            except ValueError as e:
                n_bad += 1
                warnings.append(f"{path.name}: line {j + 1}: {e}; skipped")
                i = k
                continue
            if start > end:
                n_bad += 1
                warnings.append(f"{path.name}: line {j + 1}: start after end; skipped")
                i = k
                continue
            if btype == "OTHER":
                warnings.append(f"{path.name}: unknown block header '{m.group(1)}' kept as OTHER")
            if beats and start < beats[-1].start_s:
                n_ooo += 1
            text = " ".join(text_lines).strip().strip(QUOTES).strip()
            idx = len(beats) + 1
            beats.append(Beat(ep=ep, beat_id=f"E{ep:02d}-B{idx:03d}", idx=idx, scene=cur_scene, type=btype,
                              speaker=speaker, emotion_tags=tags, text=text, start_s=start, end_s=end))
            i = k
            continue
        i += 1
    for b in beats:
        sc = next((s for s in scenes if s.num == b.scene), None)
        if sc and sc.end_s > 0 and (b.start_s < sc.start_s - 0.01 or b.end_s > sc.end_s + 0.01):
            warnings.append(f"{path.name}: {b.beat_id} ({b.start_s}-{b.end_s}) outside scene {b.scene} range")
    duration = scenes[-1].end_s if scenes and scenes[-1].end_s > 0 else max((b.end_s for b in beats), default=0.0)
    fail = (n_missing + n_bad) / n_blocks if n_blocks else 1.0
    health = ParseHealth(ep=ep, source_file=path.name, n_blocks=n_blocks, n_parsed=len(beats),
                         n_missing_ts=n_missing, n_out_of_order=n_ooo, fail_ratio=round(fail, 4))
    episode = Episode(ep=ep, logline=logline, continuity_notes=cont, duration_s=duration, scenes=scenes,
                      beats=beats, source_file=path.name)
    return episode, health, warnings


def read_show(path: Path) -> ReadResult:
    files = sorted(p for p in Path(path).iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    if not files:
        raise ReaderError(f"{path}: no .txt screenplay files found")
    eps, health, warns = [], [], []
    for f in files:
        e, h, w = read_episode(f)
        eps.append(e); health.append(h); warns.extend(w)
    seen = {}
    for e in eps:
        if e.ep in seen:
            raise ReaderError(f"{e.source_file}: duplicate episode number {e.ep} (also {seen[e.ep]})")
        seen[e.ep] = e.source_file
    order = sorted(range(len(eps)), key=lambda k: eps[k].ep)
    return ReadResult(episodes=[eps[k] for k in order], health=[health[k] for k in order], warnings=warns)
