# Promo Finder

Reads a microdrama show's timestamped screenplays and outputs a **ranked shortlist of 20–60 s promo clips**: the episode(s) and exact start/end of every segment, the hook, the cliffhanger, why it works, and an honest list of where it is likely picking badly.

**How it splits the work:** code parses the screenplay into numbered lines, builds every clip and produces every timestamp. LLMs only label lines *by ID*, judge clips *blind* (as a stranger would see them), pick between code-generated cut options and compare promos head-to-head. An editor makes the final call. Design: [`docs/superpowers/specs/2026-11-05-promo-finder-design.md`](docs/superpowers/specs/2026-11-05-promo-finder-design.md).

## Setup

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp .env.example .env        # then put your key in it: NVIDIA_API_KEY=nvapi-...
```

Free key: build.nvidia.com. Models per stage are in `config/models.yaml` (ordered fallback chains).

## Run

```bash
# estimate calls/tokens first (no key needed, no credits spent)
.venv/bin/python -m promofinder run --input "/path/to/show folder" --dry-run

# full run
.venv/bin/python -m promofinder run --input "/path/to/show folder"

# only some stages, cheaper single-vote tagging
.venv/bin/python -m promofinder run --input "/path/to/show" --stages 1,2,4,5,7,10 --votes 1

# put a different model first for one stage
.venv/bin/python -m promofinder run --input "/path/to/show" --model judge=some/other-model
```

Runs **resume**: every stage saves its output under `outputs/<show-slug>/` and is skipped next time (`--fresh` recomputes). Responses are cached in `cache/`. A run stops with a parse-health report if more than 5% of any episode's lines fail to parse (`--force` overrides).

**Swap the model provider:** any OpenAI-compatible endpoint works — edit `base_url` and `api_key_env` in `config/models.yaml` (e.g. Kuku's own API, Groq, Ollama).

## Stages

| # | Stage | Who | Output |
|---|---|---|---|
| 1 | Parse | code | `episodes.json`, `beats.jsonl`, `parse_health.txt` |
| 2 | Rule signals | code | `signals.json` |
| 3 | Show bible (full-story context) | LLM, 1 call | `bible.json` |
| 4 | Tag hooks/cliffhangers by line ID (+ quote check, optional majority vote) | LLM, 1 call/episode | `tags/ep_NNN.json` |
| 5 | Build clips (snap to camera shots, 20–60 s, hook in first 3 s, no mid-line cuts) | code | `candidates.json` |
| 6 | Multi-segment stitcher | LLM proposes, code validates | appended to `candidates.json` |
| 7 | Blind judge + first-3-seconds test | LLM | `judgements.json` |
| 8 | Cut refinement | code offers options, LLM picks | `refine.json` |
| 9 | Pairwise "which would you post first?" ranking | LLM | `pairwise.json` |
| 10 | Rank + report | code | `promos.json`, `promos.csv`, `report.md`, `run_meta.json` |
| 11 | Editor feedback (`--feedback promos.csv`) | code | prompt examples, `config/scoring.learned.yaml` |

## Outputs

- `report.md` — ranked promos with hook, cliffhanger, why it works, scores, flags, invariant pass rate, parse health, warnings.
- `promos.csv` — one row per promo; fill `editor_verdict` (✓/✗) and `editor_note` to feed back into the next run.
- `promos.json` — full detail, timestamps as `MM:SS.ss`.
- `run_meta.json` — which model answered every call (incl. fallbacks), tokens, timings, quote mismatches, vote agreement.

## Confidentiality

The input screenplays, `outputs/` and `cache/` are **never committed** (`.gitignore`). Outputs quote the script; share them privately (the submission's raw-outputs Drive folder). Test fixtures are synthetic.

## Tests

```bash
.venv/bin/pytest -q
```
