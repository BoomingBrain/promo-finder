# Promo Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A re-runnable Python CLI that reads a microdrama show's timestamped screenplays and outputs ranked 20–60 s promo candidates. Each candidate has exact segment timestamps, a hook, a cliffhanger and why it works, plus evaluation tooling and an honest failure log.

**Architecture:** An 11-stage file-based pipeline. Code parses screenplays into numbered beats, builds and validates every clip window, and produces every timestamp. LLMs, via one OpenAI-compatible client with per-stage fallback chains, only label beats by ID, judge clips blind, pick among code-generated cut options and compare promos pairwise. Each stage writes its own file under `outputs/<show-slug>/`, so runs resume and every intermediate file is a raw output.

**Tech Stack:** Python 3.11, `openai` SDK (pointed at NVIDIA `https://integrate.api.nvidia.com/v1`), `pydantic` v2, `pyyaml`, `python-dotenv`, `pytest`. Standard library only for fuzzy matching (`difflib`), concurrency (`concurrent.futures`) and the CLI (`argparse`).

**Spec:** `docs/superpowers/specs/2026-11-05-promo-finder-design.md`. Read it alongside this plan. Section numbers below (§) refer to it.

## Global Constraints

- **Python and dependencies.** Python ≥ 3.11. Runtime deps are only `openai>=1.40`, `pydantic>=2.6`, `pyyaml>=6.0` and `python-dotenv>=1.0`. Dev dep: `pytest>=8`.
- **Confidentiality.** The input pack, `outputs/`, `cache/` and `.env` are **never committed**. `.gitignore` covers `.env`, `outputs/`, `cache/`, `data/` and `*.txt` outside `tests/fixtures/` and `prompts/`. Test fixtures are synthetic text written for this repo, never Kuku script.
- **Timestamps come only from code.** No LLM response is ever parsed for a timestamp, except in Experiment A, whose whole purpose is to measure that failure.
- **Time format.** Every timestamp written to output is `MM:SS.ss`, zero-padded with two decimals, e.g. `00:41.20`. Internally times are float seconds.
- **No hard-coded numbers.** Every tunable number lives in `config/scoring.yaml`, `config/signals.yaml` or `config/models.yaml` with the defaults given in this plan. Code reads them through `promofinder/config.py`.
- **Prompts.** Prompts live in `prompts/*.txt` and use `string.Template` `$placeholders`, because JSON braces in prompts must not collide with formatting. Each prompt file's SHA-1 is recorded in `run_meta.json` as its version.
- **Clip rules (§6 Stage 5).** A clip is 20.0–60.0 s long. The hook beat starts ≤ 3.0 s after the clip start. No DIALOGUE line straddles a cut. No `resolved_by` beat starts inside the clip.
- **Model config.** `config/models.yaml` holds `base_url`, `api_key_env: NVIDIA_API_KEY` and an ordered model list per stage (`bible`, `tagger`, `stitcher`, `judge`, `hook3s`, `refine`, `pairwise`).
- **Cache key.** It is `sha256(model, messages, temperature, vote_index)`, so the vote index is always part of the key (§13.1 entry 2).
- **Commits.** Commit after every task with a conventional-commit message, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A new show's folder contains non-screenplay files or oddly named episodes** (`notes.md`, `.DS_Store`, `ep7.txt` with no `EPISODE NUMBER` header). Non-`.txt` files are ignored. The episode number falls back to the first integer in the filename, and a file with neither header nor number raises `ReaderError` naming the file. Tests go in Task 2.
2. **Too few candidates.** An episode too short for any 20 s window, or a whole show yielding fewer than `finalists_n` candidates, produces fewer promos without crashing. Pairwise ranking with an odd count gives a bye. Tests go in Tasks 7 and 17.
3. **The LLM returns sloppy values:** scores as strings (`"8"`, `"8/10"`), out of range (`11`, `0`), extra fields, or a `<think>…</think>` preamble or ```` ```json ```` fences. Scores are coerced and clamped to 1–10, extra fields are ignored and the JSON is still extracted. Tests go in Tasks 4 and 5.
4. **Curly quotes, ellipses and em-dashes.** These are all over this pack (`“…”`, `—`), and the quote check must normalise them, so a correct quote of a line containing `…` and curly quotes passes. Test goes in Task 5.
5. **Missing API key.** `run` without `NVIDIA_API_KEY` exits with a clear message *before* parsing any work, but `run --dry-run` works with no key. Test goes in Task 11.

---

# Milestone M1: core, submittable (Tasks 1–11)

### Task 1: Repo scaffold, config loading, time utilities, data models

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `promofinder/__init__.py`, `promofinder/__main__.py` (calls `cli.main()`), `promofinder/timeutil.py`, `promofinder/config.py`, `promofinder/models.py`
- Create: `config/models.yaml`, `config/scoring.yaml`, `config/signals.yaml`
- Test: `tests/test_timeutil.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `timeutil.parse_ts(s: str) -> float` accepts `MM:SS.ss`, `M:SS`, `HH:MM:SS.ss` and surrounding parentheses or whitespace, and raises `ValueError` on anything else.
  - `timeutil.format_ts(sec: float) -> str` returns `MM:SS.ss`; minutes may exceed 59.
  - `config.load_config(config_dir: Path) -> AppConfig`, where `AppConfig` has `.models: ModelsConfig`, `.scoring: ScoringConfig` and `.signals: SignalsConfig`. These are pydantic models mirroring the YAML below, attribute names identical to the YAML keys. If `config/scoring.learned.yaml` exists, its keys override `scoring.yaml`.
  - `models.py`, pydantic v2, exact names:
    - `Beat(ep:int, beat_id:str, idx:int, scene:int, type:Literal["CAMERA","ACTION","DIALOGUE","SFX","OTHER"], speaker:str|None, emotion_tags:list[str], text:str, start_s:float, end_s:float)`
    - `Scene(num:int, heading:str, start_s:float, end_s:float)`
    - `Episode(ep:int, logline:str, continuity_notes:str, duration_s:float, scenes:list[Scene], beats:list[Beat], source_file:str)`
    - `ParseHealth(ep:int, source_file:str, n_blocks:int, n_parsed:int, n_missing_ts:int, n_out_of_order:int, fail_ratio:float)`
    - `HookTag(beat_id:str, hook_score:float, story_importance:float, kind:str, why:str, quote:str)`
    - `CliffTag(beat_id:str, cliff_score:float, story_importance:float, open_question:str, resolved_by:str|None, spoiler_level:Literal["none","minor","major"], quote:str)`
    - `EpisodeTags(ep:int, hooks:list[HookTag], cliffhangers:list[CliffTag], context_beats:list[str], votes:int=1, agreement:float|None=None)`
    - `Segment(ep:int, start_s:float, end_s:float)`, with a `duration_s` property
    - `Candidate(cand_id:str, kind:Literal["single","stitched"], segments:list[Segment], hook_beat_id:str, cliff_beat_id:str, hook_score:float, cliff_score:float, story_importance:float, rule_score:float, length_fit:float, penalties:float, window_score:float, scene_key:str)`, with a `duration_s` property (sum of segments). `scene_key` is `f"E{ep:02d}-S{scene:02d}"` of the first segment.
    - `Judgement(cand_id:str, standalone:float, confusing_bits:list[str], hook_text:str, cliffhanger_text:str, why_it_works:str, spoiler_risk:bool, resolution_leak:bool, hook3s:float|None=None, hook3s_why:str|None=None)`
    - `Promo(rank:int, candidate:Candidate, judgement:Judgement|None, final_score:float, elo:float|None=None, post_first_because:str|None=None, flags:list[str]=[], original_segments:list[Segment]|None=None)`
    - `Act(name:str, ep_start:int, ep_end:int, summary:str)`, `Reveal(desc:str, ep:int, setup_eps:list[int])`, `Bible(premise:str, characters:dict[str,str], arc:list[Act], episode_summaries:dict[int,str], major_reveals:list[Reveal])`
  - Score fields on `HookTag`, `CliffTag` and `Judgement` use a shared validator `clamp_score` that turns `"8"`, `"8/10"` and `8.0` into 8.0 and clamps to [1, 10]. pydantic `model_config = ConfigDict(extra="ignore")` is set on all LLM-facing models.

**Config defaults (write these exact values):**

`config/models.yaml`
```yaml
base_url: https://integrate.api.nvidia.com/v1
api_key_env: NVIDIA_API_KEY
rpm: 40
concurrency: 8
timeout_s: 120
max_retries: 3
backoff_base_s: 2.0
max_tokens: 2048
votes: 1
vote_temperature: 0.7
stages:
  bible:    {models: [nvidia/llama-3.3-nemotron-super-49b-v1, meta/llama-3.3-70b-instruct], temperature: 0.2}
  tagger:   {models: [nvidia/llama-3.3-nemotron-super-49b-v1, meta/llama-3.3-70b-instruct], temperature: 0.2}
  stitcher: {models: [nvidia/llama-3.3-nemotron-super-49b-v1, meta/llama-3.3-70b-instruct], temperature: 0.3}
  judge:    {models: [nvidia/llama-3.3-nemotron-super-49b-v1, meta/llama-3.3-70b-instruct], temperature: 0.0}
  hook3s:   {models: [meta/llama-3.3-70b-instruct, nvidia/llama-3.3-nemotron-super-49b-v1], temperature: 0.0}
  refine:   {models: [nvidia/llama-3.3-nemotron-super-49b-v1, meta/llama-3.3-70b-instruct], temperature: 0.0}
  pairwise: {models: [nvidia/llama-3.3-nemotron-super-49b-v1, meta/llama-3.3-70b-instruct], temperature: 0.0}
system_prefix: {"nvidia/llama-3.3-nemotron-super-49b-v1": "detailed thinking off"}
```
Model IDs must be verified against build.nvidia.com when the key is created (Task 11 Step 6).

`config/scoring.yaml`
```yaml
hook_min: 6
window: {min_s: 20.0, max_s: 60.0, sweet_min_s: 30.0, sweet_max_s: 45.0, hook_lead_max_s: 3.0}
window_weights: {hook: 0.4, cliff: 0.4, rules: 0.1, length_fit: 0.1}
penalties: {mid_line_cut: 2.0, scene_jump: 1.0, camera_heavy: 1.0, camera_heavy_ratio: 0.4}
nms_overlap: 0.5
k_per_episode: 3
judge_top_n: 30
finalists_n: 15
final_weights: {hook: 0.30, cliff: 0.30, standalone: 0.25, story_importance: 0.05, rules: 0.10}
flags: {spoiler_major: 1.5, resolution_leak: 3.0, hook3s_low: 1.0}
hook3s_min: 5
early_episode: {max_ep: 10, bonus: 0.3}
diversity: {top_k: 10, max_per_scene: 2, max_per_act: 3}
quote_min_ratio: 0.8
parse_fail_max_ratio: 0.05
pairwise: {rounds: 4, k_factor: 32, start_rating: 1000, seed: 7}
refine: {max_options: 6, max_shift_shots: 2}
stitch: {top_hooks: 20, top_cliffs: 20, max_combos: 8, min_segment_s: 5.0}
feedback: {min_marked: 10, max_nudge: 0.05, max_pos: 4, max_neg: 2}
```

`config/signals.yaml`
```yaml
keywords: {slap: 3, strike: 2, gasp: 1.5, reveal: 2, kneel: 1.5, kiss: 2, blood: 2, pregnant: 3, divorce: 3, dead: 2, "who are you": 2, cliffhanger: 3, "cut to black": 2, freeze: 1.5}
emotion_intensity: {furious: 2, shocked: 2, trembling: 1.5, cold: 1, commanding: 1, desperate: 1.5, screaming: 2, sobbing: 1.5, venomous: 2, unforgiving: 1}
sfx_cues: {swell: 2, sting: 2, silence: 1.5, drone: 1, peak: 1.5}
position: {episode_end_window_s: 10.0, episode_end: 2, scene_boundary: 1, question: 0.5, exclamation: 0.5}
density: {window_s: 10.0, lines_for_max: 5, max_points: 2}
visual_terms: [slap, strike, grab, collapse, kneel, kiss, tears, reveal, "close-up on", push-in, freeze, smash]
```

- [ ] **Step 1: Initialise the repo.** In `promo-finder/`, run `git init`, write `.gitignore` per Global Constraints, write `pyproject.toml` (package `promofinder`, deps above, `[tool.pytest.ini_options] testpaths=["tests"]`) and write `.env.example` containing `NVIDIA_API_KEY=`. Then `python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'`.
- [ ] **Step 2: Write failing tests**
```python
# tests/test_timeutil.py
from promofinder.timeutil import parse_ts, format_ts
import pytest
def test_parse_variants():
    assert parse_ts("00:18.00") == 18.0
    assert parse_ts("(02:13.80)") == 133.8
    assert parse_ts(" 1:05 ") == 65.0
    assert parse_ts("01:02:03.50") == 3723.5
def test_parse_rejects_garbage():
    with pytest.raises(ValueError): parse_ts("18 seconds")
def test_format_roundtrip():
    assert format_ts(41.2) == "00:41.20"
    assert format_ts(133.8) == "02:13.80"
    assert format_ts(3723.5) == "62:03.50"
# tests/test_config.py
from pathlib import Path
from promofinder.config import load_config
from promofinder.models import HookTag
def test_loads_defaults():
    cfg = load_config(Path("config"))
    assert cfg.scoring.window.max_s == 60.0
    assert cfg.models.stages["tagger"].models[0].startswith("nvidia/")
def test_learned_overrides(tmp_path):
    import shutil; shutil.copytree("config", tmp_path/"c")
    (tmp_path/"c"/"scoring.learned.yaml").write_text("final_weights: {hook: 0.35, cliff: 0.30, standalone: 0.20, story_importance: 0.05, rules: 0.10}\n")
    assert load_config(tmp_path/"c").scoring.final_weights.hook == 0.35
def test_score_coercion():
    t = HookTag(beat_id="E01-B003", hook_score="8/10", story_importance=11, kind="slap", why="x", quote="x", extra="ignored")
    assert (t.hook_score, t.story_importance) == (8.0, 10.0)
```
- [ ] **Step 3: Run** `pytest tests/test_timeutil.py tests/test_config.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 4: Implement** `timeutil.py`, `config.py` and `models.py` per the Interfaces, and write the three YAML files. `load_config` deep-merges `scoring.learned.yaml` over `scoring.yaml` before validation.
- [ ] **Step 5: Run** the same tests. Expected: PASS.
- [ ] **Step 6: Commit** (including the existing `docs/`): `feat: scaffold, config, time utils, data models`.

---

### Task 2: Reader interface, Kuku v1 reader, parse health and fail-loud

**Files:**
- Create: `promofinder/readers/__init__.py` (registry `READERS = {"kuku_v1": kuku_v1.read_show}`), `promofinder/readers/base.py`, `promofinder/readers/kuku_v1.py`, `promofinder/parse.py`
- Create fixtures: `tests/fixtures/show_a/episode-001-Episode 9001.txt`, `tests/fixtures/show_a/episode-002-Episode 9002.txt`, `tests/fixtures/show_a/notes.md`
- Test: `tests/test_parse.py`

**Interfaces:**
- Produces:
  - `readers.base.ReaderError(Exception)`
  - `readers.base.ReadResult(episodes: list[Episode], health: list[ParseHealth], warnings: list[str])`
  - `readers.kuku_v1.read_show(path: Path) -> ReadResult`
  - `parse.load_show(path: Path, reader: str, fail_max_ratio: float, force: bool) -> ReadResult` raises `ParseHealthError(report: str)` when any episode's `fail_ratio > fail_max_ratio` and `not force`
  - `parse.health_report(result: ReadResult) -> str` returns a plain-text table
  - `parse.show_slug(path: Path) -> str`: lowercase, non-alphanumerics become `-`, collapsed, max 60 chars

**Fixture `episode-001-Episode 9001.txt`.** Use the pack's exact layout from §4: header lines, `--- SCRIPT START ---`, `SCENE NN` and `Scene heading/start/end`, then `Actions and dialogue:` and blocks of `**HEADER:**` / `Start: (MM:SS.ss) | End: (MM:SS.ss)` / text. Blocks, in order (beat IDs follow):

| id | scene | header | start–end | text |
|---|---|---|---|---|
| B001 | 1 `INT. BALLROOM – NIGHT` 00:00–00:30 | CAMERA | 0–4 | Wide shot of the crowded ballroom. |
| B002 | 1 | CAMERA | 4–9 | Close-up on Mara as a glass shatters. |
| B003 | 1 | ACTION | 5–6 | Victor slaps Mara across the face. |
| B004 | 1 | VICTOR (Furious) | 6.5–8.5 | "You stole my mother’s ring!" |
| B005 | 1 | CAMERA | 9–20 | Medium shot of guests gasping. |
| B006 | 1 | MARA (Shocked, Trembling) | 10–12 | "I have never seen that ring…" |
| B007 | 1 | CAMERA | 20–30 | Slow push-in on Victor. |
| B008 | 1 | VICTOR (Cold) | 21–24 | "Then explain why it is in your bag." |
| B009 | 2 `INT. HALLWAY – CONTINUOUS` 00:30–00:55 | CAMERA | 30–42 | Tracking shot as Mara opens her bag. |
| B010 | 2 | MARA (Desperate) | 33–35 | "Someone put this here." |
| B011 | 2 | CAMERA | 42–52 | Close-up on the ring glinting in her palm. |
| B012 | 2 | SOUND FX | 48–52 | Ominous string swell rises to a cliffhanger peak. |
| B013 | 2 | CAMERA | 52–55 | Cut to black. |

Header: `EPISODE NUMBER: Episode 1`, `EPISODE LOGLINE: Mara is publicly accused of theft at her engagement party.`, `CONTINUITY NOTES: Synthetic test fixture.`

**Fixture `episode-002-Episode 9002.txt`.** It uses **upper-case** scene headers (`SCENE HEADING:`, `SCENE START:`, `SCENE END:`), with one scene, `SCENE HEADING: EXT. GARDEN – DAY`, from 00:00 to 00:15. Blocks: CAMERA 0–15 "Static shot of the garden."; MARA (Quiet) 2–4 "Who are you?"; ACTION 10–12 "Mara drops the ring." Header `EPISODE NUMBER: Episode 2`. `notes.md` contains any text.

- [ ] **Step 1: Write failing tests**
```python
# tests/test_parse.py
from pathlib import Path
import pytest
from promofinder.parse import load_show, show_slug, ParseHealthError
from promofinder.readers.base import ReaderError
A = Path("tests/fixtures/show_a")
def test_reads_both_formats_and_ignores_non_txt():
    r = load_show(A, "kuku_v1", 0.05, False)
    assert [e.ep for e in r.episodes] == [1, 2]
    e1 = r.episodes[0]
    assert e1.duration_s == 55.0 and len(e1.beats) == 13 and len(e1.scenes) == 2
    b4 = e1.beats[3]
    assert (b4.beat_id, b4.type, b4.speaker, b4.emotion_tags, b4.start_s) == ("E01-B004", "DIALOGUE", "VICTOR", ["Furious"], 6.5)
    assert e1.beats[11].type == "SFX" and e1.beats[11].scene == 2
    s = r.episodes[1].scenes[0]
    assert (s.heading, s.end_s, r.episodes[1].duration_s) == ("EXT. GARDEN – DAY", 15.0, 15.0)
def test_fail_loud_and_force(tmp_path):
    bad = tmp_path/"show"; bad.mkdir()
    txt = (A/"episode-002-Episode 9002.txt").read_text().replace("Start: (00:02.00) | End: (00:04.00)\n", "")
    (bad/"episode-002.txt").write_text(txt)
    with pytest.raises(ParseHealthError) as ei: load_show(bad, "kuku_v1", 0.05, False)
    assert "episode-002.txt" in ei.value.report
    r = load_show(bad, "kuku_v1", 0.05, True)
    assert r.health[0].n_missing_ts == 1
def test_episode_number_fallback_and_error(tmp_path):
    d = tmp_path/"s"; d.mkdir()
    body = (A/"episode-002-Episode 9002.txt").read_text().replace("EPISODE NUMBER: Episode 2\n", "")
    (d/"ep7.txt").write_text(body)
    assert load_show(d, "kuku_v1", 0.05, False).episodes[0].ep == 7
    (d/"intro.txt").write_text(body)
    with pytest.raises(ReaderError, match="intro.txt"): load_show(d, "kuku_v1", 0.05, False)
def test_slug():
    assert show_slug(Path("I Married a Broken Soldier. He was a Duke")) == "i-married-a-broken-soldier-he-was-a-duke"
```
- [ ] **Step 2: Run** `pytest tests/test_parse.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** the reader. Parsing is line-based with case-insensitive regexes:
  - block header: `^\*\*(.+?):\*\*\s*$`, where the speaker is the text before `(` and emotion tags are the comma-split contents of the parentheses
  - timing: `Start:\s*\(([^)]+)\)\s*\|\s*End:\s*\(([^)]+)\)`
  - block text: the following non-empty lines up to the next header, `SCENE` line or `--- SCRIPT END ---`, joined, with surrounding straight or curly quotes stripped

  Type mapping: `CAMERA`, `ACTION` and `SOUND FX`→`SFX` are fixed; any other all-caps header with a speaker is `DIALOGUE`; anything else is `OTHER` plus a warning. A block whose timing line is missing counts in `n_missing_ts` and is skipped. `duration_s` is the last scene's end, or else the max beat end. Episode number comes from `EPISODE NUMBER: Episode N`, else the first integer in the filename, else `ReaderError(f"{file}: no episode number")`. Episodes are sorted by `ep`. Beat IDs are `f"E{ep:02d}-B{idx:03d}"` with `idx` starting at 1.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Smoke-check on the real pack** (read-only, nothing written): `python -c "from pathlib import Path; from promofinder.parse import load_show, health_report; r=load_show(Path('../I Married a Broken Soldier. He was a Duke'),'kuku_v1',0.05,False); print(len(r.episodes)); print(health_report(r))"`. Expected: `50`, and Episode 6 parsed with a fail ratio of 0.
- [ ] **Step 6: Commit** `feat: kuku_v1 reader with parse health and fail-loud`.

---

### Task 3: Rule signals

**Files:**
- Create: `promofinder/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: `Episode`, `Beat`, `SignalsConfig`
- Produces:
  - `signals.BeatSignals(score: float, hints: list[str])`, where `score` is clamped to [0, 10] and `hints` are short tags like `"kw:slap"`, `"emo:furious"`, `"sfx:swell"`, `"ep_end"`, `"question"`
  - `signals.compute_signals(ep: Episode, cfg: SignalsConfig) -> dict[str, BeatSignals]` keyed by `beat_id`
  - `signals.is_visual(beat: Beat, cfg: SignalsConfig) -> bool`: True for non-CAMERA beats, and for CAMERA beats whose text contains any `visual_terms` (case-insensitive)

Score per beat is the sum of:
- each matched keyword weight, matched on lowercase text
- emotion weights for each tag, lowercased
- SFX cue weights (SFX beats only)
- position: `episode_end` if `beat.end_s >= duration - episode_end_window_s`; `scene_boundary` if it is the last beat of its scene; `question` / `exclamation` for DIALOGUE ending in `?` / `!`
- density: `min(max_points, max_points * n_dialogue_starting_in[start, start+window_s) / lines_for_max)`

- [ ] **Step 1: Write failing tests**
```python
from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.signals import compute_signals, is_visual
def _ep1():
    return load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes[0]
def test_signal_hits():
    s = compute_signals(_ep1(), load_config(Path("config")).signals)
    assert "kw:slap" in s["E01-B003"].hints
    assert "emo:furious" in s["E01-B004"].hints and "exclamation" in s["E01-B004"].hints
    assert {"sfx:swell", "kw:cliffhanger", "ep_end"} <= set(s["E01-B012"].hints)
    assert 0 <= s["E01-B012"].score <= 10 and s["E01-B012"].score > s["E01-B001"].score
def test_visual_filter():
    cfg = load_config(Path("config")).signals; b = _ep1().beats
    assert is_visual(b[1], cfg)        # "Close-up on Mara..."
    assert not is_visual(b[0], cfg)    # "Wide shot..."
    assert is_visual(b[3], cfg)        # dialogue always kept
```
- [ ] **Step 2: Run** `pytest tests/test_signals.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: rule-based beat signals`.

---

### Task 4: LLM client: fallback chain, retries, rate limit, cache, JSON extraction, run metadata

**Files:**
- Create: `promofinder/llm.py`, `promofinder/meta.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `meta.RunMeta`: thread-safe, with a lock.
    - `record_call(stage:str, model:str, prompt_tokens:int, completion_tokens:int, latency_s:float, cached:bool, fallback:bool)`
    - `incr(counter:str, n:int=1)`; known counters are `quote_mismatches`, `unknown_ids`, `json_retries`, `fallback_switches`, `skipped_episodes`, `invalid_refine_picks` and `pairwise_first_wins`
    - `warn(msg:str)`, `set(key:str, value)`, `stage_timer(stage:str)` (a context manager) and `save(path:Path)`
    - `.calls: list[dict]`
  - `llm.Completion(text:str, prompt_tokens:int, completion_tokens:int)`
  - `llm.TransientError(Exception)` for 429, 5xx and timeouts; `llm.ModelNotFound(Exception)`; `llm.LLMStageError(Exception)` when every model in a chain fails
  - `llm.Transport = Callable[[str, list[dict], float, int], Completion]`, taking (model, messages, temperature, max_tokens)
  - `llm.openai_transport(base_url:str, api_key:str, timeout_s:float) -> Transport` maps `openai.RateLimitError`, `APITimeoutError` and `InternalServerError` to `TransientError`, and `NotFoundError` to `ModelNotFound`
  - `llm.extract_json(text:str) -> dict|list` strips `<think>…</think>` blocks and ```` ``` ```` fences, then returns the first balanced `{…}` or `[…]` that parses. Raises `ValueError` otherwise.
  - `llm.RateLimiter(rpm:int, clock=time.monotonic, sleep=time.sleep)` with `.acquire()` ensuring at least `60/rpm` s between acquisitions; thread-safe
  - `llm.LLMClient(cfg: ModelsConfig, cache_dir: Path, meta: RunMeta, transport: Transport|None=None, sleep=time.sleep)`, where `transport=None` builds `openai_transport` from the env var named by `cfg.api_key_env`
    - `.complete_json(stage:str, messages:list[dict], schema:type[BaseModel], vote_index:int=0, temperature:float|None=None) -> BaseModel`
    - `.complete_text(stage, messages, vote_index=0, temperature=None) -> str`

**`complete_json` algorithm (pins behaviour the tests check):**
```
for i, model in enumerate(cfg.stages[stage].models):
    if i > 0: meta.incr("fallback_switches")
    msgs = [system_prefix[model] as system msg if configured] + messages
    for attempt in (1, 2):                       # schema/JSON attempts
        text = cached(model, msgs, temp, vote_index) or call with retries:
                 up to max_retries on TransientError, sleep(backoff_base_s * 2**k)
               (TransientError after retries or ModelNotFound -> next model)
        try: return schema.model_validate(extract_json(text))
        except (ValueError, ValidationError) as e:
            meta.incr("json_retries"); msgs = msgs + [assistant text, user "Your reply was invalid: {e}. Reply with only valid JSON matching the schema."]
    # two invalid replies -> next model
raise LLMStageError(stage)
```
Cache file: `cache_dir/<sha256 hex>.json` storing `{text, prompt_tokens, completion_tokens}`. It is written **only after the reply passes schema validation**, so a bad reply is never replayed on resume. The `RateLimiter` uses the client's injected `sleep` and sleeps once for the computed wait, with no loop. The key is `sha256(json.dumps([model, msgs, temperature, vote_index]))`. Every response, cached or not, is recorded via `meta.record_call`, with `fallback=(i>0)`.

- [ ] **Step 1: Write failing tests**, using a fake transport scripted by a list of responses or exceptions.
```python
import pytest, json
from pathlib import Path
from pydantic import BaseModel
from promofinder.config import load_config
from promofinder.llm import LLMClient, Completion, TransientError, ModelNotFound, LLMStageError, extract_json, RateLimiter
from promofinder.meta import RunMeta
class Out(BaseModel): x: int
def fake(script):
    calls = []
    def t(model, messages, temperature, max_tokens):
        calls.append(model); item = script.pop(0)
        if isinstance(item, Exception): raise item
        return Completion(item, 10, 5)
    t.calls = calls; return t
def client(tmp_path, script):
    m = RunMeta(); t = fake(script)
    return LLMClient(load_config(Path("config")).models, tmp_path, m, transport=t, sleep=lambda s: None), m, t
def test_extract_json_variants():
    assert extract_json('<think>hmm</think>```json\n{"x": 1}\n```') == {"x": 1}
    assert extract_json('Sure! [{"a":2}] done') == [{"a": 2}]
def test_retry_then_success(tmp_path):
    c, m, t = client(tmp_path, [TransientError(), '{"x": 3}'])
    assert c.complete_json("judge", [{"role":"user","content":"q"}], Out).x == 3
def test_invalid_json_twice_falls_back(tmp_path):
    c, m, t = client(tmp_path, ["nope", "still nope", '{"x": 4}'])
    assert c.complete_json("judge", [{"role":"user","content":"q"}], Out).x == 4
    assert t.calls[2] != t.calls[0] and m.counters["fallback_switches"] == 1 and m.counters["json_retries"] == 2
def test_model_not_found_falls_back(tmp_path):
    c, m, t = client(tmp_path, [ModelNotFound(), '{"x": 5}'])
    assert c.complete_json("judge", [{"role":"user","content":"q"}], Out).x == 5
def test_all_fail_raises(tmp_path):
    c, m, t = client(tmp_path, [ModelNotFound(), ModelNotFound()])
    with pytest.raises(LLMStageError): c.complete_json("judge", [{"role":"user","content":"q"}], Out)
def test_cache_hit_and_vote_index_in_key(tmp_path):
    c, m, t = client(tmp_path, ['{"x": 1}', '{"x": 2}'])
    msgs = [{"role":"user","content":"q"}]
    assert c.complete_json("judge", msgs, Out).x == 1
    assert c.complete_json("judge", msgs, Out).x == 1          # cached, no transport call
    assert c.complete_json("judge", msgs, Out, vote_index=1).x == 2
    assert len(t.calls) == 2 and sum(1 for k in m.calls if k["cached"]) == 1
def test_rate_limiter_spacing():
    now = [0.0]; slept = []
    rl = RateLimiter(60, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0]+s)))
    rl.acquire(); rl.acquire()
    assert slept and abs(slept[0] - 1.0) < 1e-6
```
- [ ] **Step 2: Run** `pytest tests/test_llm.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `meta.py` and `llm.py` per the Interfaces and algorithm. `counters` is a `defaultdict(int)` exposed as `.counters`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: LLM client with fallback chain, retries, rate limit, cache`.

---

### Task 5: Compact transcript and beat tagger with quote check and validation

**Files:**
- Create: `promofinder/tagger.py`, `promofinder/prompts.py`, `prompts/tagger.txt`
- Test: `tests/test_tagger.py`, `tests/fakes.py` (shared `FakeLLM`)

**Interfaces:**
- Consumes: `LLMClient.complete_json`, `compute_signals`, `is_visual`, `EpisodeTags`/`HookTag`/`CliffTag`
- Produces:
  - `prompts.load_prompt(name:str) -> string.Template` reads `prompts/<name>.txt`
  - `prompts.prompt_versions() -> dict[str,str]` gives the SHA-1 of every prompt file
  - `tagger.RawTags(hooks:list[HookTag], cliffhangers:list[CliffTag], context_beats:list[str]=[])` is the schema passed to `complete_json`
  - `tagger.build_transcript(ep:Episode, sig:dict[str,BeatSignals], cfg:SignalsConfig) -> str` emits one line per beat where `is_visual`, in the format `[E01-B004] 00:06.50–00:08.50 DIALOGUE VICTOR (Furious): "You stole my mother’s ring!"  {sig: kw:…, emo:furious}`. The `{sig: …}` part is omitted when there are no hints; ACTION, SFX and CAMERA lines have no speaker.
  - `tagger.normalize(s:str) -> str` lowercases, maps curly quotes to straight, maps `…` to `...` and `—`/`–` to `-`, strips punctuation and collapses whitespace
  - `tagger.quote_matches(quote:str, text:str, min_ratio:float) -> bool` is True if `normalize(quote)` is a substring of `normalize(text)`, or if `difflib.SequenceMatcher` ratio between `normalize(quote)` and the same-word-count prefix of `normalize(text)` is ≥ `min_ratio`
  - `tagger.validate_tags(raw:RawTags, ep:Episode, min_ratio:float, meta:RunMeta) -> EpisodeTags` drops tags whose `beat_id` is not in `ep` (`meta.incr("unknown_ids")`) and tags failing `quote_matches` (`meta.incr("quote_mismatches")`). `resolved_by` is normalised so a beat ID not in `ep` becomes `None` unless it is `"next_episode"` or matches `later_episode:\d+`. Unknown `context_beats` are dropped and also counted in `unknown_ids`.
  - `tagger.render_tagger_messages(ep:Episode, transcript:str, context:str, anchors:str, feedback_examples:str) -> list[dict]`, where `context`, `anchors` and `feedback_examples` may be `""` (filled in by Tasks 12, 13 and 19)
  - `tagger.tag_episode(ep, sig, client, cfg:AppConfig, meta, context:str="", anchors:str="", feedback_examples:str="", vote_index:int=0, temperature:float|None=None) -> EpisodeTags|None` returns `None` on `LLMStageError`, after `meta.incr("skipped_episodes")` and a `meta.warn`

**`prompts/tagger.txt` must state, in this order:**
1. role ("promo editor for vertical microdramas")
2. the hook definition (stops a scroller in its first 3 s with **no context**)
3. the cliffhanger definition
4. the rule "Use the SHOW CONTEXT only to fill `story_importance`, `resolved_by` and `spoiler_level`. Score `hook_score` and `cliff_score` as a stranger who has never seen the show."
5. "Refer to lines only by their [ID]; never write timestamps."
6. "For each tag copy the first ~6 words of that line into `quote`."
7. the JSON schema with field names exactly as `HookTag`/`CliffTag`
8. placeholders `$context`, `$anchors`, `$feedback`, `$logline` and `$transcript`

- [ ] **Step 1: Write failing tests**
```python
# tests/fakes.py
import json
from promofinder.llm import LLMStageError
class FakeLLM:
    """Returns queued python objects validated through the requested schema; records messages."""
    def __init__(self, replies): self.replies = list(replies); self.seen = []
    def complete_json(self, stage, messages, schema, vote_index=0, temperature=None):
        self.seen.append((stage, messages, vote_index, temperature))
        r = self.replies.pop(0)
        if isinstance(r, Exception): raise r
        return schema.model_validate(r)
# tests/test_tagger.py
from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.signals import compute_signals
from promofinder.meta import RunMeta
from promofinder.llm import LLMStageError
from promofinder.tagger import build_transcript, quote_matches, tag_episode
from tests.fakes import FakeLLM
CFG = load_config(Path("config")); EP = load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes[0]
SIG = compute_signals(EP, CFG.signals)
def test_transcript_drops_plain_camera_keeps_visual():
    t = build_transcript(EP, SIG, CFG.signals)
    assert "[E01-B001]" not in t and "[E01-B002]" in t
    assert '[E01-B004] 00:06.50–00:08.50 DIALOGUE VICTOR (Furious): "You stole my mother’s ring!"' in t
def test_quote_normalisation():
    assert quote_matches('"I have never seen that ring..."', "I have never seen that ring…", 0.8)
    assert quote_matches("You stole my mother's", "You stole my mother’s ring!", 0.8)
    assert not quote_matches("Someone put this here", "You stole my mother’s ring!", 0.8)
def test_tag_episode_validates():
    reply = {"hooks": [
        {"beat_id": "E01-B003", "hook_score": "9", "story_importance": 5, "kind": "slap", "why": "slap", "quote": "Victor slaps Mara across"},
        {"beat_id": "E01-B099", "hook_score": 8, "story_importance": 5, "kind": "other", "why": "x", "quote": "x"},
        {"beat_id": "E01-B006", "hook_score": 7, "story_importance": 5, "kind": "other", "why": "x", "quote": "Someone put this here"}],
      "cliffhangers": [{"beat_id": "E01-B012", "cliff_score": 9, "story_importance": 6, "open_question": "who framed her?",
                        "resolved_by": "next_episode", "spoiler_level": "none", "quote": "Ominous string swell rises"}],
      "context_beats": ["E01-B004", "E01-B777"]}
    m = RunMeta(); tags = tag_episode(EP, SIG, FakeLLM([reply]), CFG, m)
    assert [h.beat_id for h in tags.hooks] == ["E01-B003"] and tags.hooks[0].hook_score == 9.0
    assert tags.cliffhangers[0].resolved_by == "next_episode" and tags.context_beats == ["E01-B004"]
    assert m.counters["unknown_ids"] == 2 and m.counters["quote_mismatches"] == 1
def test_stage_error_skips_episode():
    m = RunMeta(); assert tag_episode(EP, SIG, FakeLLM([LLMStageError("tagger")]), CFG, m) is None
    assert m.counters["skipped_episodes"] == 1
```
- [ ] **Step 2: Run** `pytest tests/test_tagger.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `prompts.py`, `prompts/tagger.txt` and `tagger.py` per the Interfaces.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: beat tagger with transcript, quote check, validation`.

---

### Task 6: Majority-vote merge

**Files:**
- Create: `promofinder/voting.py`
- Test: `tests/test_voting.py`

**Interfaces:**
- Consumes: `EpisodeTags`
- Produces: `voting.merge_votes(runs: list[EpisodeTags], votes: int) -> EpisodeTags`
  - `None` runs are removed by the caller first; `votes` is the requested count
  - A beat ID is kept if it appears in ≥ `votes // 2 + 1` runs (separately for hooks and cliffhangers)
  - Numeric fields are averaged. `kind`, `spoiler_level` and `resolved_by` take the most common value, with ties going to the earliest run. Text fields come from the first run containing the beat.
  - `context_beats` is the union of IDs in ≥ majority runs
  - `agreement` is kept IDs ÷ union of IDs (hooks plus cliffs), or 1.0 when the union is empty. `votes` is set.
  - With `votes == 1`, it returns `runs[0]` with `agreement=None`.

- [ ] **Step 1: Write failing test**
```python
from promofinder.models import EpisodeTags, HookTag, CliffTag
from promofinder.voting import merge_votes
def H(b, s): return HookTag(beat_id=b, hook_score=s, story_importance=5, kind="slap", why="w", quote="q")
def test_majority_and_average():
    runs = [EpisodeTags(ep=1, hooks=[H("E01-B003", 9), H("E01-B004", 6)], cliffhangers=[], context_beats=[]),
            EpisodeTags(ep=1, hooks=[H("E01-B003", 7)], cliffhangers=[], context_beats=[]),
            EpisodeTags(ep=1, hooks=[H("E01-B003", 8), H("E01-B010", 5)], cliffhangers=[], context_beats=[])]
    m = merge_votes(runs, 3)
    assert [h.beat_id for h in m.hooks] == ["E01-B003"] and m.hooks[0].hook_score == 8.0
    assert m.votes == 3 and abs(m.agreement - 1/3) < 1e-9
def test_single_vote_passthrough():
    r = EpisodeTags(ep=1, hooks=[H("E01-B003", 9)], cliffhangers=[], context_beats=[])
    assert merge_votes([r], 1).agreement is None
```
- [ ] **Step 2: Run** `pytest tests/test_voting.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: majority-vote merge for tagging runs`.

---

### Task 7: Window builder (snap, rules, score, NMS)

**Files:**
- Create: `promofinder/windows.py`
- Test: `tests/test_windows.py`

**Interfaces:**
- Consumes: `Episode`, `EpisodeTags`, `BeatSignals`, `ScoringConfig`
- Produces:
  - `windows.shot_at(ep:Episode, t:float) -> Beat|None` returns the CAMERA beat with `start_s <= t < end_s`
  - `windows.snap_start(ep, hook:Beat, lead_max:float) -> float` returns the containing shot's start if `hook.start_s - shot.start_s <= lead_max`, else `hook.start_s`
  - `windows.snap_end(ep, cliff:Beat, resolved_start:float|None) -> float` returns the end of `shot_at(ep, cliff.end_s - 1e-6)`. If that end exceeds `resolved_start` or the start of the first DIALOGUE beat starting at or after `cliff.end_s`, it returns `cliff.end_s` instead.
  - `windows.straddles_dialogue(ep, t:float) -> bool` is True if some DIALOGUE beat has `start_s < t < end_s`
  - `windows.length_fit(d:float, w) -> float` is 10 inside `[sweet_min_s, sweet_max_s]`, falls linearly to 5 at `min_s` and at `max_s`, and is 0 outside `[min_s, max_s]`
  - `windows.check_clip_rules(ep, start:float, end:float, hook:Beat, cliff:Beat, resolved_start:float|None, w) -> list[str]` returns rule violations from `{"duration", "hook_lead", "mid_line_cut", "resolution_inside", "cliff_cut"}`. `cliff_cut` means `end < cliff.end_s - 0.05`. Task 10 (invariants) and Task 16 (refine) reuse it.
  - `windows.build_candidates(ep:Episode, tags:EpisodeTags, sig:dict[str,BeatSignals], cfg:ScoringConfig) -> list[Candidate]`

**`build_candidates` rules:**
- **Pairs considered:** hooks with `hook_score >= hook_min` × cliffhangers whose beat has `end_s > hook.start_s`.
- **Resolution beat:** `resolved_start` is the start of the `resolved_by` beat when it is a beat ID in this episode, else `None`.
- **Window:** `start = snap_start`, `end = snap_end`. Keep the pair only if `check_clip_rules` contains neither `duration` nor `resolution_inside` nor `hook_lead`.
- **Penalties:**
  - `mid_line_cut` if either cut straddles dialogue
  - `scene_jump` if the hook's and cliff's scenes differ and their headings' location parts differ. The location part is the heading with any `– DAY/NIGHT/CONTINUOUS` suffix removed. A cut to `CONTINUOUS` in the same location is not a jump.
  - `camera_heavy` if the fraction of window time not covered by DIALOGUE or ACTION beats exceeds `camera_heavy_ratio`
- **Rule score:** `0.5*sig[hook].score + 0.5*sig[cliff].score`.
- **Window score:** `ww.hook*hook_score + ww.cliff*cliff_score + ww.rules*rule_score + ww.length_fit*length_fit - penalties`.
- **Story importance:** the mean of the hook and cliff tags' values.
- **IDs:** `cand_id = f"S-{hook.beat_id}-{cliff.beat_id}"`, `kind="single"`, `scene_key` from the hook beat.
- **NMS:** sort by `window_score` descending. Drop any candidate whose overlap with an already-kept one, measured as intersection ÷ shorter duration, is greater than `nms_overlap`. Keep at most `k_per_episode`.

- [ ] **Step 1: Write failing tests** (fixture times from Task 2).
```python
from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.signals import compute_signals
from promofinder.models import EpisodeTags, HookTag, CliffTag
from promofinder.windows import snap_start, snap_end, length_fit, build_candidates
CFG = load_config(Path("config")); SHOW = load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False)
E1, E2 = SHOW.episodes; B = {b.beat_id: b for b in E1.beats}
def H(b, s=9): return HookTag(beat_id=b, hook_score=s, story_importance=5, kind="slap", why="w", quote="q")
def C(b, s=9, res=None): return CliffTag(beat_id=b, cliff_score=s, story_importance=5, open_question="?", resolved_by=res, spoiler_level="none", quote="q")
def test_snapping():
    assert snap_start(E1, B["E01-B003"], 3.0) == 4.0
    assert snap_end(E1, B["E01-B012"], None) == 52.0
    assert snap_end(E1, B["E01-B008"], None) == 30.0
def test_length_fit():
    w = CFG.scoring.window
    assert length_fit(35, w) == 10 and length_fit(20, w) == 5 and length_fit(60, w) == 5 and length_fit(61, w) == 0
def test_build_keeps_valid_rejects_short_and_dedupes():
    tags = EpisodeTags(ep=1, hooks=[H("E01-B003")], cliffhangers=[C("E01-B006"), C("E01-B008"), C("E01-B012")], context_beats=[])
    cands = build_candidates(E1, tags, compute_signals(E1, CFG.signals), CFG.scoring)
    ids = {c.cand_id for c in cands}
    assert "S-E01-B003-E01-B006" not in ids                       # 16 s, too short
    assert ids <= {"S-E01-B003-E01-B008", "S-E01-B003-E01-B012"} and ids
    for c in cands:
        assert 20 <= c.duration_s <= 60 and c.segments[0].start_s == 4.0
def test_snap_end_stops_before_resolution():
    # shot containing B010's end (35.0) runs to 42.0, past a resolution starting at 40.0 -> cut at the cliff line's end
    assert snap_end(E1, B["E01-B010"], resolved_start=40.0) == 35.0
    assert snap_end(E1, B["E01-B010"], resolved_start=None) == 42.0
def test_short_episode_yields_nothing():
    tags = EpisodeTags(ep=2, hooks=[H("E02-B002")], cliffhangers=[C("E02-B003")], context_beats=[])
    assert build_candidates(E2, tags, compute_signals(E2, CFG.signals), CFG.scoring) == []
```
- [ ] **Step 2: Run** `pytest tests/test_windows.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces and rules.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: window builder with shot snapping, rules, scoring, NMS`.

---

### Task 8: Blind judge and first-3-seconds test

**Files:**
- Create: `promofinder/judge.py`, `promofinder/cliptext.py`, `prompts/judge.txt`, `prompts/hook3s.txt`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `Candidate`, `Episode`, `LLMClient`, `is_visual`
- Produces:
  - `cliptext.clip_text(cand:Candidate, episodes:dict[int,Episode], sig_cfg:SignalsConfig, first_seconds:float|None=None) -> str` gives the visual beats whose `start_s` lies in each segment `[start, end)`, formatted `SPEAKER (Emotion): "line"` or `[ACTION] text` / `[SFX] text` / `[CAMERA] text`, with **no beat IDs and no timestamps**. Segments are joined by `--- CUT TO LATER ---`. With `first_seconds`, it includes only beats starting in `[seg0.start, seg0.start+first_seconds)`.
  - `judge.JudgeOut(standalone, confusing_bits, hook_text, cliffhanger_text, why_it_works, spoiler_risk, resolution_leak)` is the schema; fields as `Judgement` minus `cand_id`/`hook3s*`
  - `judge.Hook3sOut(score: float, why: str)` (score clamped 1–10)
  - `judge.judge_candidate(cand, episodes, client, cfg:AppConfig, meta, feedback_examples:str="") -> Judgement|None`
  - `judge.hook3s_candidate(cand, episodes, client, cfg, meta) -> tuple[float, str]|None`
  - `judge.judge_all(cands:list[Candidate], episodes, client, cfg, meta, feedback_examples:str="") -> dict[str, Judgement]` runs both calls for each candidate in a `ThreadPoolExecutor(cfg.models.concurrency)` and merges the hook3s result into `Judgement.hook3s` / `hook3s_why`. Failed candidates are omitted with `meta.warn`.

**Prompt requirements:**
- `judge.txt` says the viewer "has never seen this show", contains **no show context placeholder**, asks the three stranger-test questions from §6 Stage 7, and requests the `JudgeOut` JSON. Placeholders: `$clip`, `$feedback`.
- `hook3s.txt` asks "Scrolling past, would this make you stop? 1–10, and what grabbed you." Placeholder: `$first3`.

- [ ] **Step 1: Write failing tests**
```python
from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.meta import RunMeta
from promofinder.models import Candidate, Segment
from promofinder.cliptext import clip_text
from promofinder.judge import judge_all
from tests.fakes import FakeLLM
CFG = load_config(Path("config")); EPS = {e.ep: e for e in load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False).episodes}
CAND = Candidate(cand_id="S-x", kind="single", segments=[Segment(ep=1, start_s=4.0, end_s=30.0)], hook_beat_id="E01-B003",
                 cliff_beat_id="E01-B008", hook_score=9, cliff_score=8, story_importance=5, rule_score=5, length_fit=10,
                 penalties=0, window_score=8, scene_key="E01-S01")
def test_clip_text_no_ids_or_times_and_first3():
    t = clip_text(CAND, EPS, CFG.signals)
    assert "E01-B" not in t and "00:" not in t and 'VICTOR (Furious): "You stole my mother’s ring!"' in t
    f3 = clip_text(CAND, EPS, CFG.signals, first_seconds=3.0)
    assert "slaps" in f3 and "never seen that ring" not in f3   # window [4.0, 7.0): B003 (5.0) in, B006 (10.0) out
def test_judge_all_merges_hook3s_and_is_blind():
    jo = {"standalone": 8, "confusing_bits": [], "hook_text": "slap", "cliffhanger_text": "ring in bag?", "why_it_works": "w",
          "spoiler_risk": False, "resolution_leak": False}
    llm = FakeLLM([jo, {"score": 9, "why": "slap"}])
    CFG.models.concurrency = 1
    out = judge_all([CAND], EPS, llm, CFG, RunMeta())
    assert out["S-x"].standalone == 8 and out["S-x"].hook3s == 9
    judge_msgs = [m for s, m, *_ in llm.seen if s == "judge"][0]
    assert "Mara is publicly accused" not in str(judge_msgs)   # no logline/show context leaks to the judge
```
- [ ] **Step 2: Run** `pytest tests/test_judge.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `cliptext.py`, `judge.py` and both prompts.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: blind judge and first-3-seconds test`.

---

### Task 9: Final scoring, diversity and finalist selection

**Files:**
- Create: `promofinder/rank.py`
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: `Candidate`, `Judgement`, `EpisodeTags`, `Bible|None`, `ScoringConfig`
- Produces:
  - `rank.final_score(c:Candidate, j:Judgement|None, cliff_spoiler:str, cfg:ScoringConfig) -> tuple[float, list[str]]` returns the score and its flags. The formula is `fw.hook*c.hook_score + fw.cliff*c.cliff_score + fw.standalone*(j.standalone if j else 5.0) + fw.story_importance*c.story_importance + fw.rules*c.rule_score`, then:
    - minus `flags.spoiler_major` with flag `"spoiler_major"` if `cliff_spoiler=="major"` or `j.spoiler_risk`
    - minus `flags.resolution_leak` with flag `"resolution_leak"` if `j.resolution_leak`
    - minus `flags.hook3s_low` with flag `"hook3s_low"` if `j.hook3s is not None and j.hook3s < hook3s_min`
    - plus `early_episode.bonus` with flag `"early_episode"` if `c.segments[0].ep <= early_episode.max_ep`
    - plus 0 and flag `"unjudged"` when `j is None`
  - `rank.act_of(ep:int, bible:Bible|None) -> str|None` returns the name of the act whose range contains `ep`
  - `rank.diversify(promos:list[Promo], bible:Bible|None, cfg) -> list[Promo]` takes a list already sorted by `final_score` descending. It greedily fills the first `diversity.top_k` slots, skipping a promo if its `scene_key` already has `max_per_scene` placed promos or its act already has `max_per_act`. Skipped promos follow in original order, then the rest, and `rank` is reassigned 1..n.
  - `rank.score_all(cands:list[Candidate], judgements:dict[str,Judgement], tags:dict[int,EpisodeTags], bible:Bible|None, cfg) -> list[Promo]` scores every candidate, sorts and diversifies. `cliff_spoiler` is looked up from the cliff tag of `c.cliff_beat_id`.
  - `rank.finalists(promos:list[Promo], n:int) -> tuple[list[Promo], list[Promo]]` returns (first n, rest)

- [ ] **Step 1: Write failing tests**
```python
from pathlib import Path
from promofinder.config import load_config
from promofinder.models import Candidate, Segment, Judgement, Promo, Bible, Act
from promofinder.rank import final_score, diversify
CFG = load_config(Path("config")).scoring
def cand(i, ep=1, scene="E01-S01", h=8, c=8):
    return Candidate(cand_id=f"S-{i}", kind="single", segments=[Segment(ep=ep, start_s=0, end_s=30)], hook_beat_id="h", cliff_beat_id="c",
                     hook_score=h, cliff_score=c, story_importance=5, rule_score=5, length_fit=10, penalties=0, window_score=8, scene_key=scene)
J = Judgement(cand_id="x", standalone=8, confusing_bits=[], hook_text="", cliffhanger_text="", why_it_works="", spoiler_risk=False, resolution_leak=True, hook3s=3)
def test_final_score_flags():
    s, flags = final_score(cand(1, ep=20), J, "major", CFG)
    base = .30*8 + .30*8 + .25*8 + .05*5 + .10*5
    assert abs(s - (base - 1.5 - 3.0 - 1.0)) < 1e-9 and set(flags) == {"spoiler_major", "resolution_leak", "hook3s_low"}
    s2, f2 = final_score(cand(2, ep=3), None, "none", CFG)
    assert "early_episode" in f2 and "unjudged" in f2
def test_diversify_caps_per_scene():
    ps = [Promo(rank=0, candidate=cand(i), judgement=None, final_score=10 - i) for i in range(4)]
    ps.append(Promo(rank=0, candidate=cand(9, scene="E02-S01", ep=2), judgement=None, final_score=1))
    out = diversify(ps, None, CFG)
    assert [p.candidate.cand_id for p in out][:3] == ["S-0", "S-1", "S-9"] and [p.rank for p in out] == [1, 2, 3, 4, 5]
```
- [ ] **Step 2: Run** `pytest tests/test_rank.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: final scoring, diversity re-rank, finalist selection`.

---

### Task 10: Invariant checks and report outputs

**Files:**
- Create: `promofinder/invariants.py`, `promofinder/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `check_clip_rules` (Task 7), `Promo`, `Episode`, `ReadResult`, `RunMeta`, `format_ts`
- Produces:
  - `invariants.check_promo(p:Promo, episodes:dict[int,Episode], tags:dict[int,EpisodeTags], cfg:ScoringConfig) -> list[str]`
    - **Single promos:** the `check_clip_rules` violations, plus `"not_on_boundary"` if a cut isn't within 0.01 s of some beat's `start_s`/`end_s` in that episode.
    - **Stitched promos:** each segment ≥ `stitch.min_segment_s`, total in [min_s, max_s], segments chronological, every cut on a boundary.
  - `invariants.pass_rate(promos, episodes, tags, cfg) -> tuple[float, dict[str,list[str]]]` returns the fraction of promos with no violations, and the violations keyed by `cand_id`
  - `report.segments_str(c:Candidate) -> str` gives `"E01 00:04.00–00:30.00"`, with multiple segments joined by `"; "`
  - `report.write_outputs(out_dir:Path, promos:list[Promo], episodes, tags, read:ReadResult, meta:RunMeta, cfg:AppConfig) -> None` writes:
    - `promos.json`: a list of `Promo.model_dump()`, each with a `segments_ts` list of `{ep, start, end}` in `MM:SS.ss`
    - `promos.csv` (UTF-8): columns exactly `rank,cand_id,kind,segments,duration_s,hook,cliffhanger,why_it_works,post_first_because,final_score,elo,hook_score,cliff_score,standalone,story_importance,rule_score,hook3s,flags,editor_verdict,editor_note`; the last two are empty and `flags` are joined with `|`
    - `report.md`: title, run summary (models used, calls, tokens), the line `Invariant pass rate: NN%` (integer percent) followed by violations, parse health (`health_report`), then each promo as `### #rank — segments (duration s)` with hook, cliffhanger, why it works, scores and flags, plus `**Post first because:**` for ranks 1–5 when present
    - `run_meta.json` via `meta.save`, after `meta.set("prompt_versions", prompt_versions())`

- [ ] **Step 1: Write failing tests**
```python
import csv, json
from pathlib import Path
from promofinder.parse import load_show
from promofinder.config import load_config
from promofinder.meta import RunMeta
from promofinder.models import Candidate, Segment, Promo
from promofinder.invariants import check_promo
from promofinder.report import write_outputs, segments_str
CFG = load_config(Path("config")); READ = load_show(Path("tests/fixtures/show_a"), "kuku_v1", 0.05, False)
EPS = {e.ep: e for e in READ.episodes}
def P(start, end, hook="E01-B003", cliff="E01-B008"):
    c = Candidate(cand_id=f"S-{start}", kind="single", segments=[Segment(ep=1, start_s=start, end_s=end)], hook_beat_id=hook, cliff_beat_id=cliff,
                  hook_score=9, cliff_score=8, story_importance=5, rule_score=5, length_fit=10, penalties=0, window_score=8, scene_key="E01-S01")
    return Promo(rank=1, candidate=c, judgement=None, final_score=7.0)
def test_invariants():
    assert check_promo(P(4.0, 30.0), EPS, {}, CFG.scoring) == []
    v = check_promo(P(4.0, 22.5), EPS, {}, CFG.scoring)
    assert "mid_line_cut" in v and "not_on_boundary" in v and "cliff_cut" in v
def test_outputs(tmp_path):
    write_outputs(tmp_path, [P(4.0, 30.0)], EPS, {}, READ, RunMeta(), CFG)
    row = next(csv.DictReader(open(tmp_path/"promos.csv", encoding="utf-8")))
    assert row["segments"] == "E01 00:04.00–00:30.00" and row["editor_verdict"] == ""
    assert json.loads((tmp_path/"promos.json").read_text())[0]["segments_ts"][0]["start"] == "00:04.00"
    md = (tmp_path/"report.md").read_text(); assert "Invariant pass rate: 100%" in md
    assert "prompt_versions" in json.loads((tmp_path/"run_meta.json").read_text())
```
- [ ] **Step 2: Run** `pytest tests/test_report.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: invariant checks and report outputs`.

---

### Task 11: Pipeline orchestration, CLI, resume, dry run, first real run (M1 complete)

**Files:**
- Create: `promofinder/pipeline.py`, `promofinder/cli.py`, `README.md`, `eval/gold.yaml` (template, see Step 1), `docs/ai-log.md`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above. Optional stages call into Tasks 12–19 behind `if stage in stages` guards. Until those exist, the stage's module simply isn't imported. Each later task adds its own call at the marked hook.
- Produces:
  - `pipeline.parse_stages(s:str) -> set[int]` handles `"1-11"`, `"1,2,4,5,7,10"` and mixed forms
  - `pipeline.RunOptions(input:Path, out_root:Path, config_dir:Path, reader:str="kuku_v1", stages:set[int]=set(range(1,12)), model_overrides:dict[str,str]={}, votes:int|None=None, feedback:Path|None=None, dry_run:bool=False, fresh:bool=False, force:bool=False)`
  - `pipeline.estimate(read:ReadResult, opts, cfg) -> list[tuple[str,int,int]]` returns (stage name, calls, est_tokens). Estimates:
    - tagger tokens = `len(transcript)/4 + 1500` per call × `n_eps × votes`
    - bible = 1 call
    - stitcher = 1 call if stage 6 is on
    - judge and hook3s = `judge_top_n` each
    - refine = `finalists_n`
    - pairwise = `rounds × ceil(finalists_n/2)`
  - `pipeline.run(opts:RunOptions, client_factory=None) -> Path` returns the output dir. `client_factory(cfg, cache_dir, meta) -> LLMClient` defaults to building a real client.
    - `--model stage=model` prepends that model to the stage's chain.
    - `--votes` overrides `cfg.models.votes`.
  - `cli.main(argv:list[str]|None=None) -> int` provides subcommands `run` (flags mirror `RunOptions`: `--input --out --config --reader --stages --model --votes --feedback --dry-run --fresh --force`) and `eval` (added in Task 14). In `run`, a `ParseHealthError` prints the report and returns 2. If the API key env var is missing and the run isn't `--dry-run`, it prints `Missing NVIDIA_API_KEY (see .env.example)` and returns 3. `.env` is loaded via `python-dotenv`.

**`run` order, with files in `out_dir = out_root/show_slug(input)`:**
1. **Parse.** Write `episodes.json`, `beats.jsonl`, `parse_health.txt` and `parse_warnings.txt`. If `dry_run`, print the `estimate` table and return.
2. **Signals:** `signals.json`.
3. **[Task 12 hook]** `bible.json`.
4. **Tagging.** For each episode, parallel up to `concurrency`, write `tags/ep_NNN.json`. If the file exists and `not fresh`, load it and skip the call. Votes run with `vote_index` 0..v-1 and are merged with `merge_votes`; vote temperature is used when votes > 1. `meta.set("vote_agreement", mean)`.
5. **Windows:** `candidates.json`.
6. **[Task 18 hook]** stitched candidates are appended to `candidates.json`.
7. **Judging.** Take the top `judge_top_n` candidates by `window_score` and write `judgements.json`, resuming like tagging, keyed by `cand_id`.
8. **Ranking.** Run `score_all`, then `finalists`.
9. **[Task 16 hook]** refine the finalists (`refine.json`).
10. **[Task 17 hook]** pairwise-rank the finalists (`pairwise.json`). The final list is the finalists in their new order, then the rest, with ranks reassigned.
11. **Report.** Write all outputs and save `run_meta.json`. Every stage is wrapped in `meta.stage_timer`.

`eval/gold.yaml` template:
```yaml
show: i-married-a-broken-soldier-he-was-a-duke
picked_by: "<your name>, before seeing tool output"
picks: []   # - {id: G01, segments: [{ep: 1, start: "00:11.00", end: "00:45.00"}], note: "short reason, no script quotes"}
```

`docs/ai-log.md` starts with the table header from §13.1 and the two already-logged entries, copied from the spec.

- [ ] **Step 1: Write failing tests** (an end-to-end run on the fixture with a fake client factory).
```python
import json, os
from pathlib import Path
from promofinder.pipeline import parse_stages, run, RunOptions
from promofinder.cli import main
from tests.fakes import FakeLLM
TAG = {"hooks": [{"beat_id": "E01-B003", "hook_score": 9, "story_importance": 5, "kind": "slap", "why": "w", "quote": "Victor slaps Mara across"}],
       "cliffhangers": [{"beat_id": "E01-B012", "cliff_score": 9, "story_importance": 6, "open_question": "?", "resolved_by": "next_episode",
                         "spoiler_level": "none", "quote": "Ominous string swell rises"}], "context_beats": []}
EMPTY = {"hooks": [], "cliffhangers": [], "context_beats": []}
JO = {"standalone": 8, "confusing_bits": [], "hook_text": "slap", "cliffhanger_text": "?", "why_it_works": "w", "spoiler_risk": False, "resolution_leak": False}
class Routed(FakeLLM):
    def complete_json(self, stage, messages, schema, vote_index=0, temperature=None):
        self.seen.append((stage, messages, vote_index, temperature))
        if stage == "tagger": r = TAG if "Mara is publicly accused" in str(messages) else EMPTY   # ep 1 logline
        elif stage == "judge": r = JO
        else: r = {"score": 8, "why": "w"}
        return schema.model_validate(r)
def opts(tmp_path, **kw):
    return RunOptions(input=Path("tests/fixtures/show_a"), out_root=tmp_path, config_dir=Path("config"), stages=parse_stages("1,2,4,5,7,10"), **kw)
def test_parse_stages():
    assert parse_stages("1-3,7") == {1, 2, 3, 7}
def test_end_to_end_and_resume(tmp_path):
    llm = Routed([]); out = run(opts(tmp_path), client_factory=lambda *a: llm)
    promos = json.loads((out/"promos.json").read_text())
    assert promos[0]["candidate"]["cand_id"] == "S-E01-B003-E01-B012" and (out/"tags"/"ep_001.json").exists()
    n = len(llm.seen); run(opts(tmp_path), client_factory=lambda *a: llm)
    assert len(llm.seen) == n                                   # resumed: no new LLM calls
def test_dry_run_needs_no_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert main(["run", "--input", "tests/fixtures/show_a", "--out", str(tmp_path), "--dry-run"]) == 0
    assert "tagger" in capsys.readouterr().out
def test_missing_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr("promofinder.cli.load_dotenv", lambda *a, **k: None)
    assert main(["run", "--input", "tests/fixtures/show_a", "--out", str(tmp_path)]) == 3
```
- [ ] **Step 2: Run** `pytest tests/test_pipeline.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `pipeline.py` and `cli.py` per the Interfaces and run order. Write `README.md` covering setup (venv, `.env`), `run`, `--dry-run`, `--stages`, `--model`, `--votes`, the outputs explained, the confidentiality note and how to swap to any OpenAI-compatible endpoint (edit `base_url` / `api_key_env`).
- [ ] **Step 4: Run** `pytest -v` (full suite). Expected: all PASS.
- [ ] **Step 5: Commit** `feat: pipeline orchestration, CLI, resume, dry run (M1)`.
- [ ] **Step 6: Human checkpoint, before any real model output exists.**
  - Ask the user to (a) create an NVIDIA API key at build.nvidia.com and put it in `.env`, (b) confirm the model IDs in `config/models.yaml` exist in the catalogue, correcting them if not, and note their free credit balance, and (c) **fill `eval/gold.yaml` with ~12 hand-picked promos from ~8 episodes spread across the show, before seeing any tool output** (§12).
  - Do not run Step 7 until the user confirms (c) is done.
- [ ] **Step 7: First real run.** `python -m promofinder run --input "../I Married a Broken Soldier. He was a Duke" --stages 1,2,4,5,7,10 --votes 1`. Expected: it exits 0, `outputs/i-married-a-broken-soldier-he-was-a-duke/report.md` exists, the invariant pass rate is printed, and `run_meta.json` has the call counts. Then read the top 10 in `report.md` and log any AI mistake you see in `docs/ai-log.md`.
- [ ] **Step 8: Commit** `docs: ai-log after first real run` (logs only; `outputs/` is git-ignored).

---

# Milestone M2: better judgement (Tasks 12–15)

### Task 12: Show bible and two-layer context

**Files:**
- Create: `promofinder/bible.py`, `prompts/bible.txt`
- Modify: `promofinder/pipeline.py` (stage 3 hook: build or load `bible.json`, pass `context=render_context(bible, ep.ep)` to `tag_episode`, pass `bible` to `score_all`)
- Test: `tests/test_bible.py`

**Interfaces:**
- Produces:
  - `bible.build_bible(episodes:list[Episode], client, cfg, meta) -> Bible|None` makes one `bible`-stage call whose input is every episode's `Episode N: logline / continuity notes`. It returns `None` (with `meta.warn`) on `LLMStageError`. Numeric-string keys in `episode_summaries` are converted to int.
  - `bible.render_context(b:Bible|None, ep:int) -> str` returns `""` for `None`. Otherwise it returns `SHOW CONTEXT (global): premise, characters as "Name — role" lines, arc acts with ranges, major reveals with episodes` followed by `LOCAL: previous episode: <summary of ep-1 or "none">; next episode: <summary of ep+1 or "none">`.

- [ ] **Step 1: Write failing test**
```python
from promofinder.models import Bible, Act, Reveal
from promofinder.bible import render_context
B = Bible(premise="A jilted bride marries a wounded soldier.", characters={"Mara": "jilted bride, the heroine"},
          arc=[Act(name="Betrayal", ep_start=1, ep_end=5, summary="s")], episode_summaries={1: "accused", 2: "meets stranger"},
          major_reveals=[Reveal(desc="soldier is a duke", ep=30, setup_eps=[2, 9])])
def test_context_layers():
    c = render_context(B, 1)
    assert "Mara — jilted bride" in c and "Betrayal (1–5)" in c and "soldier is a duke (ep 30)" in c
    assert "previous episode: none" in c and "next episode: meets stranger" in c
    assert render_context(None, 1) == ""
```
- [ ] **Step 2: Run** `pytest tests/test_bible.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `bible.py`, `prompts/bible.txt` and the pipeline hook.
- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: show bible with two-layer tagging context`.

### Task 13: Scoring anchors

**Files:**
- Create: `prompts/anchors.yaml`
- Modify: `promofinder/prompts.py`, `promofinder/pipeline.py` (pass `anchors=render_anchors()` to `tag_episode`)
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces: `prompts.render_anchors() -> str`. It renders `anchors.yaml`, which holds `hooks:` and `cliffhangers:` lists of `{score:int, example:str, why:str}`, at least 3 each with scores 3, 6 and 9, as `EXAMPLE (score N/10): example — why` lines. The examples are **original text written for this repo** (e.g. "A woman quietly folds laundry while talking about the weather" scored 3).

- [ ] **Step 1: Write failing test:** `render_anchors()` contains `"(score 3/10)"`, `"(score 9/10)"` and at least 6 `EXAMPLE` lines.
- [ ] **Step 2: Run** it. Expected: FAIL.
- [ ] **Step 3: Implement** and wire into the pipeline.
- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: calibration anchors in tagger prompt`.

### Task 14: Eval command

**Files:**
- Create: `promofinder/evaluate.py`
- Modify: `promofinder/cli.py` (add `eval --run DIR --gold PATH [--compare DIR]`)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `evaluate.overlap(a:Segment, b:Segment) -> float` is the intersection ÷ duration of `b` (the gold segment), and 0 for different episodes
  - `evaluate.evaluate(run_dir:Path, gold_path:Path, compare_dir:Path|None=None) -> dict` returns these keys:
    - `recall_at_20`: the fraction of gold picks with any segment overlapping ≥ 0.5 with any segment of a top-20 promo
    - `precision_at_10`: ✓ count ÷ marked count among the top 10, from `promos.csv` `editor_verdict` (`✓`/`y`/`yes` count as positive, `✗`/`n`/`no` as negative); `None` if none are marked
    - `invariant_pass_rate`, recomputed from `promos.json` + `episodes.json` + `tags/`
    - `quote_mismatch_rate`: `quote_mismatches` ÷ (`quote_mismatches` + total kept tags)
    - `vote_agreement`
    - `top10_overlap_with_compare`: Jaccard of top-10 `cand_id`s, or `None`
    - `n_gold`
  - It writes `run_dir/eval_<YYYYmmdd-HHMMSS>.json` and the CLI prints a table.

- [ ] **Step 1: Write failing tests.** Using a temp run dir holding a hand-written `promos.json`, `promos.csv`, `episodes.json`, `run_meta.json` and a 2-pick gold file, assert `recall_at_20 == 0.5` when one pick overlaps 60% and the other is in a different episode. Assert `precision_at_10 == 1.0` with one `✓` row, and `None` with no marks. Assert `overlap(Segment(ep=1,start_s=0,end_s=30), Segment(ep=1,start_s=10,end_s=40)) == 20/30`.
- [ ] **Step 2: Run** `pytest tests/test_evaluate.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: eval command (recall@20, precision@10, invariants, stability)`.

### Task 15: Experiments A and B, second real run

**Files:**
- Create: `promofinder/experiments.py`, `prompts/plan_a.txt`
- Modify: `promofinder/cli.py` (add `experiment plan-a|hindsight --input DIR --episodes 1,5,...`)
- Test: `tests/test_experiments.py`

**Interfaces:**
- Produces:
  - `experiments.check_naive_clip(ep:Episode, start:str, end:str, w) -> list[str]` uses `parse_ts` and returns errors from `{"unparseable", "not_on_boundary", "mid_line_cut", "duration", "beyond_episode"}`. This is the only place LLM-written timestamps are parsed.
  - `experiments.plan_a(episodes, ep_nums:list[int], client, cfg, meta, out_dir) -> dict` sends one `judge`-stage call per episode. The prompt is `plan_a.txt`: full raw episode text, then "return promo clips as JSON `[{start:"MM:SS.ss", end:"MM:SS.ss", why}]`", using `complete_json` with schema `NaiveClips(clips:list[NaiveClip])`. It saves raw replies to `out_dir/experiments/exp_a_raw/ep_NNN.json` and the counts per error type plus totals to `exp_a_summary.json`.
  - `experiments.hindsight(episodes, ep_nums, bible, client, cfg, meta, out_dir) -> dict` tags each episode twice, with `context=render_context(bible, ep)` and with `context=""`, using a separate `vote_index` (100 and 101) so the cache can't conflate them. For beats whose hook or cliff score rises by ≥ 2 with context, it builds windows from the *with-context* tags, judges them blind with `judge_all`, and writes `exp_b_hindsight.json` listing `{beat_id, score_with, score_without, cand_id, standalone}`.

- [ ] **Step 1: Write failing tests** for `check_naive_clip` on the fixture:
  - `("00:04.00","00:30.00")` returns `[]`
  - `("00:04.30","00:22.50")` includes `not_on_boundary` and `mid_line_cut`
  - `("00:50.00","01:30.00")` includes `beyond_episode`
  - `("abc","00:30.00")` returns `["unparseable"]`
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: experiments A (naive timestamps) and B (hindsight bias)`.
- [ ] **Step 6: Real runs.**
  - Run `python -m promofinder experiment plan-a --input "../I Married…" --episodes 1,10,20,30,40` and `experiment hindsight --episodes 1,7,13,19,25,31,37,43`.
  - Then run the full M2 pipeline `run --stages 1-5,7,10 --votes 1 --fresh`, then `eval --run outputs/<slug> --gold eval/gold.yaml`.
  - Record the real numbers and the clearest real example from each experiment in `docs/ai-log.md`. Write "no failure observed" if that's what happened (§13.2).
- [ ] **Step 7: Commit** `docs: experiment results in ai-log (M2)`.

---

# Milestone M3: better ranking and cuts (Tasks 16–17)

### Task 16: Cut-point refinement

**Files:**
- Create: `promofinder/refine.py`, `prompts/refine.txt`
- Modify: `promofinder/pipeline.py` (stage 8 hook, writes `refine.json`)
- Test: `tests/test_refine.py`

**Interfaces:**
- Consumes: `check_clip_rules`, `shot_at`, `clip_text`
- Produces:
  - `refine.options(c:Candidate, ep:Episode, tags:EpisodeTags, cfg:ScoringConfig) -> list[Segment]` applies only to `kind=="single"`.
    - Shots are the CAMERA beats in order. The start index is the shot containing the start; the end index is the shot containing `end-1e-6`.
    - It enumerates (ds, de) in `[-max_shift_shots..+max_shift_shots]²`, with new start = start of shot[i+ds] and new end = end of shot[j+de].
    - It keeps those with no `check_clip_rules` violations, using the candidate's hook and cliff beats and their `resolved_by` start.
    - It sorts by `abs(ds)+abs(de)` then `(ds,de)` and truncates to `max_options`.
    - The original `(0,0)` is always first, as option A.
  - `refine.RefineOut(choice:str, why:str)`
  - `refine.refine_promo(p:Promo, episodes, tags, client, cfg, meta) -> Promo` returns `p` unchanged if there are fewer than 2 options. Otherwise it asks the `refine` stage to pick a letter, given each option's `clip_text`. An invalid letter keeps the original and runs `meta.incr("invalid_refine_picks")`. When a different option is chosen, it sets `original_segments` and replaces `candidate.segments`; `duration_s` follows automatically.

- [ ] **Step 1: Write failing tests** on the fixture candidate `S-E01-B003-E01-B008` (4.0–30.0):
  - `options()[0] == Segment(ep=1,start_s=4.0,end_s=30.0)`
  - every option passes `check_clip_rules == []` and there are ≤ 6 options
  - a fake reply `{"choice":"B","why":"w"}` sets `original_segments` and changes the segments to `options()[1]`
  - `{"choice":"Z"}` leaves the segments unchanged and increments `invalid_refine_picks`
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** `refine.py`, the prompt and the pipeline hook.
- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: cut-point refinement (code options, LLM picks)`.

### Task 17: Pairwise Swiss ranking with Elo

**Files:**
- Create: `promofinder/pairwise.py`, `prompts/pairwise.txt`
- Modify: `promofinder/pipeline.py` (stage 9 hook, writes `pairwise.json`), `promofinder/report.py` (shows `elo`; `post_first_because` already exists)
- Test: `tests/test_pairwise.py`

**Interfaces:**
- Produces:
  - `pairwise.elo_update(ra:float, rb:float, a_won:bool, k:float) -> tuple[float,float]` is the standard Elo update: expected `1/(1+10**((rb-ra)/400))`
  - `pairwise.swiss_pairs(ids:list[str], ratings:dict[str,float]) -> tuple[list[tuple[str,str]], str|None]` sorts by rating descending (ties broken by input order), pairs adjacent IDs and returns the leftover as a bye
  - `pairwise.PairOut(winner: Literal["A","B"], why: str)`
  - `pairwise.rank_pairwise(finalists:list[Promo], episodes, client, cfg, meta) -> list[Promo]`
    - Ratings start at `start_rating` and run for `rounds` rounds.
    - In each pair, presentation order is shuffled with `random.Random(seed + round)`, and the prompt shows option A and B `clip_text`.
    - Whenever the winner is the first-shown option, it runs `meta.incr("pairwise_first_wins")`. It sets `meta.set("pairwise_first_position_rate", first_wins/total)`.
    - A failed call skips that pair.
    - Each promo's `post_first_because` is the `why` of the win against the highest-rated opponent it beat.
    - It returns the finalists sorted by final Elo descending, ties broken by `final_score`, with `elo` set.
    - With fewer than 2 finalists it returns the input unchanged.

- [ ] **Step 1: Write failing tests:**
  - `elo_update(1000,1000,True,32) == (1016.0, 984.0)`
  - `swiss_pairs(["a","b","c"], {...equal})` returns `([("a","b")], "c")`
  - with 3 finalists and a fake that always answers the option whose text contains the strongest candidate's hook, that candidate ends rank 1, has non-empty `post_first_because`, and `pairwise_first_position_rate` is between 0 and 1
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** `pairwise.py`, the prompt and the hooks.
- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: pairwise Swiss ranking with Elo and position-bias tracking`.
- [ ] **Step 6: Real run** with `--stages 1-5,7-10` (resuming M2's tags and judgements), then `eval`. Log the pairwise first-position rate and any odd picks in `docs/ai-log.md`, and commit the log.

---

# Milestone M4: extras (Tasks 18–20)

### Task 18: Stitcher (multi-segment promos)

**Files:**
- Create: `promofinder/stitcher.py`, `prompts/stitcher.txt`
- Modify: `promofinder/pipeline.py` (stage 6 hook)
- Test: `tests/test_stitcher.py`

**Interfaces:**
- Produces:
  - `stitcher.SegReq(beat_id:str, role:Literal["hook","bridge","cliff"], before_s:float=0, after_s:float=0)`, `stitcher.Combo(segments:list[SegReq], why:str)`, `stitcher.StitchOut(combos:list[Combo])`
  - `stitcher.build_segment(req:SegReq, ep:Episode, tags:EpisodeTags, cfg) -> Segment|None`:
    - **hook:** start = `snap_start`; end = end of `shot_at(start + max(after_s, min_segment_s))`
    - **cliff:** end = `snap_end`; start = start of `shot_at(beat.start_s - max(before_s, 0))`
    - **bridge:** start = start of `shot_at(beat.start_s - before_s)`; end = end of `shot_at(beat.end_s + after_s - 1e-6)`
    - returns `None` if the shot lookup fails
  - `stitcher.validate_combo(segs:list[Segment], roles:list[str], cfg) -> list[str]` returns errors from:
    - `"too_short_segment"` (< `min_segment_s`)
    - `"total_duration"` (outside [min_s, max_s])
    - `"not_chronological"` ((ep, start) not strictly increasing)
    - `"overlap"`
    - `"bad_roles"` (first role not hook, or last not cliff)
  - `stitcher.stitch(episodes, tags:dict[int,EpisodeTags], bible, client, cfg, meta) -> list[Candidate]`
    - **Shortlist:** the top `top_hooks` hooks and top `top_cliffs` cliffs with their text and episode.
    - **Call:** one `stitcher`-stage call with `render_context(bible, ep)` global part, truncated to the first `max_combos` combos.
    - **Validation:** invalid combos are dropped and logged with `meta.warn`.
    - **Valid combos** become `Candidate(kind="stitched", cand_id=f"M-{n:03d}", ...)`, with hook and cliff scores and the window-score formula as in Task 7 (`length_fit` on the total duration, `penalties=0`) and `scene_key` of the first segment.

- [ ] **Step 1: Write failing tests** for `validate_combo`:
  - `[E01 4–12, E01 44–52]` with roles `[hook, cliff]` (16 s total) returns `["total_duration"]`
  - out-of-order segments yield `not_chronological`
  - a 3 s segment yields `too_short_segment`
  - roles `[cliff, hook]` yield `bad_roles`
  - `[E01 4–20, E01 42–52]` with `[hook, cliff]` returns `[]`

  Add one `stitch()` test with a fake reply containing one valid and one invalid combo, asserting exactly one `M-001` candidate.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** `stitcher.py`, the prompt and the hook.
- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: experimental multi-segment stitcher`.

### Task 19: Editor feedback loop

**Files:**
- Create: `promofinder/feedback.py`
- Modify: `promofinder/pipeline.py` (when `opts.feedback` is set, load before tagging and pass `feedback_examples` to `tag_episode` and `judge_all`), `promofinder/config.py` (no change if `scoring.learned.yaml` override already works)
- Test: `tests/test_feedback.py`

**Interfaces:**
- Produces:
  - `feedback.Marked(cand_id:str, verdict:bool, note:str, row:dict)`
  - `feedback.read_marked(csv_path:Path) -> list[Marked]` keeps only rows whose `editor_verdict` is set (`✓/y/yes` true, `✗/n/no` false)
  - `feedback.examples_text(marked, cfg) -> str` uses up to `max_pos` ✓ and `max_neg` ✗ rows, rendered `EDITOR LIKED: hook / cliffhanger — note` and `EDITOR REJECTED: … — note`, with no timestamps
  - `feedback.nudge_weights(marked, cfg:ScoringConfig) -> dict|None` returns `None` if `len(marked) < min_marked`. For each of `hook, cliff, standalone, story_importance, rules`, it computes `d = mean(✓) − mean(✗)` of that score from the CSV columns. CSV columns are `hook_score`, `cliff_score`, `standalone`, `story_importance` and `rule_score`. It sets `w' = w + clamp(d/10, -max_nudge, +max_nudge)`, renormalises so the weights sum to 1.0, and returns `{"final_weights": {...}}`.
  - `feedback.append_gold(marked, promos_json:Path, gold_path:Path) -> int` appends ✓ promos as gold picks (`id: F<n>`, segments as `{ep,start,end}`, note) unless one with identical segments exists, and returns the count added
  - `feedback.apply(csv_path, run_dir, cfg, config_dir, gold_path) -> str` writes `config_dir/scoring.learned.yaml` when `nudge_weights` returns a dict, calls `append_gold`, and returns `examples_text`

- [ ] **Step 1: Write failing tests:**
  - With 3 marked rows, `nudge_weights` returns `None` and `examples_text` contains `EDITOR LIKED` and no `00:`.
  - With 10 rows where ✓ rows have higher `standalone`, the new `standalone` weight rises by ≤ 0.05 and the weights sum to 1.0 (±1e-9).
  - `append_gold` adds ✓ promos once and returns 0 on a second call.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement**, including the pipeline wiring.
- [ ] **Step 4: Run** `pytest -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: editor feedback loop (examples, weight nudges, gold growth)`.

### Task 20: Final full run, packaging, submission material

**Files:**
- Modify: `README.md` (final usage, all commands, timings from the real run), `docs/ai-log.md`
- Create: `docs/solution-doc-outline.md` (the §13.5 outline filled with real numbers, for the user to turn into the 2-page PDF in their own words)

- [ ] **Step 1: Dry run.** `python -m promofinder run --input "../I Married…" --votes 3 --dry-run`. Confirm the estimated calls fit the user's remaining NVIDIA credits, and ask the user before spending them.
- [ ] **Step 2: Final run.** `python -m promofinder run --input "../I Married…" --votes 3 --fresh`, then `python -m promofinder eval --run outputs/<slug> --gold eval/gold.yaml`. Expected: exits 0, every output file exists and the eval JSON is written. Record wall time, calls, tokens, invariant pass rate, recall@20, vote agreement and pairwise position rate. Optional, if credits remain (§8.4): rerun stages 4 and 7 with `--model tagger=<other free model> --model judge=<other free model> --out outputs-alt`, then `eval --compare` against the main run, and note the differences.
- [ ] **Step 3: Human review.** Ask the user to mark ✓/✗ on the top 10 in `promos.csv`, rerun `eval` for precision@10, and categorise the bad picks using the §8.5 categories in `docs/ai-log.md`.
- [ ] **Step 4: Fill in** `docs/solution-doc-outline.md` with the real numbers and the single most convincing real failure (§13.4). Do not invent results.
- [ ] **Step 5: Packaging check.** Run `git status --ignored` and `git ls-files`. Confirm that no `.txt` outside `tests/fixtures/` and `prompts/`, no `outputs/`, no `cache/` and no `.env` is tracked. Tell the user to zip `outputs/<slug>/` (excluding nothing) into the anyone-with-link Google Drive folder.
- [ ] **Step 6: Commit** `docs: final README, ai-log and solution outline`.
