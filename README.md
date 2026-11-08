# Promo Finder

Reads a microdrama show's timestamped screenplays and outputs a **ranked shortlist of 20–60 s promo clips**: the episode(s) and exact start/end of every segment, the hook, the cliffhanger, why it works, and an honest list of where it is likely picking badly.

**How it splits the work:** code parses the screenplay into numbered lines, builds every clip and produces every timestamp. LLMs only label lines *by ID*, judge clips *blind* (as a stranger would see them), pick between code-generated cut options and compare promos head-to-head. An editor makes the final call. Design: [`docs/superpowers/specs/2026-11-05-promo-finder-design.md`](docs/superpowers/specs/2026-11-05-promo-finder-design.md).

## Navigating the repo

**Reading order for a first look (about 15 minutes):**
1. This README: what it does and how to run it.
2. [`promofinder/pipeline.py`](promofinder/pipeline.py): the whole pipeline in one file. Each `# stage N` block calls one module.
3. [`docs/ai-log.md`](docs/ai-log.md): what went wrong during the build and what changed because of it.
4. [`docs/solution-doc-outline.md`](docs/solution-doc-outline.md): results, numbers and known failure modes.
5. [`docs/superpowers/specs/…-design.md`](docs/superpowers/specs/2026-11-05-promo-finder-design.md): the full design. The [`plans/`](docs/superpowers/plans/2026-11-06-promo-finder.md) file next to it is the task-by-task build plan.

```
promo-finder/
├── promofinder/              the package (python -m promofinder …)
│   ├── cli.py                commands: run · eval · experiment
│   ├── pipeline.py           runs the stages in order, saves and resumes each one
│   │
│   ├── readers/kuku_v1.py    stage 1: screenplay .txt → numbered beats (add a new reader here for another format)
│   ├── parse.py              stage 1: loads a show folder, parse-health check
│   ├── signals.py            stage 2: keyword and cue hints (gasp, slap, "cliffhanger" swell…)
│   ├── bible.py              stage 3: one-call show summary + per-episode context
│   ├── tagger.py, voting.py  stage 4: AI tags hook and cliffhanger beats by ID; quote check; majority vote
│   ├── windows.py            stage 5: code builds every clip (shot snapping, 20–60 s, editing rules)
│   ├── stitcher.py           stage 6: experimental multi-segment promos
│   ├── judge.py, cliptext.py stage 7: blind "stranger" judge + first-3-seconds test (sees clip text only)
│   ├── refine.py             stage 8: code offers alternative cuts, AI picks one
│   ├── dedupe.py             stage 9: drops finalists that repeat a better clip's moment
│   ├── pairwise.py           stage 9: head-to-head "which would you post first?" with Elo, both orders asked
│   ├── rank.py               stage 10: final score, diversity rules
│   ├── report.py, quality.py stage 10: report.md / promos.csv / promos.json + "where it may be picking badly"
│   ├── invariants.py         automatic checks every promo must pass
│   ├── feedback.py           stage 11: editor ✓/✗ → prompt examples, weight nudges
│   ├── evaluate.py           `eval`: recall@20 vs gold set, precision@10 from editor marks
│   ├── experiments.py        `experiment`: A (AI-written timestamps) and B (hindsight bias)
│   └── llm.py, models.py, config.py, prompts.py, meta.py, timeutil.py
│                             shared plumbing: model client (fallbacks, rate-limit waits, cache),
│                             data models, config loading, prompt loading, run metadata, MM:SS.ss helpers
├── prompts/                  every prompt sent to a model (+ anchors.yaml calibration examples)
├── config/
│   ├── models.yaml           provider, model per stage, rate limits  ← swap models here
│   ├── scoring.yaml          weights, clip rules, pool sizes, penalties ← tune ranking here
│   └── signals.yaml          keyword lists for stage 2
├── eval/gold.yaml            12 hand-reviewed gold picks (episode + timestamps only, never shown to a model)
├── tests/                    pytest suite; fixtures are synthetic mini-screenplays
└── docs/                     AI log, solution-doc outline, design spec, build plan
```

**Where to change things:**

| I want to… | Edit |
|---|---|
| use a different model or provider | `config/models.yaml` (any OpenAI-compatible endpoint) |
| change how promos are scored or how many candidates are kept | `config/scoring.yaml` |
| change what a model is asked | the matching file in `prompts/` (each run records prompt hashes in `run_meta.json`) |
| support a different screenplay format | add a reader in `promofinder/readers/` and register it in `readers/__init__.py`, then run with `--reader <name>` |
| add or change an editing rule | `check_clip_rules` in `promofinder/windows.py` (also enforced by `invariants.py`) |

**Not in the repo, on purpose:** the input screenplays, `outputs/`, `cache/` and `.env`. See [Confidentiality](#confidentiality).

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

### Evaluate, experiment, feed back

```bash
# score a run against the hand-picked gold set (recall@20, precision@10 from editor marks, invariants, stability)
.venv/bin/python -m promofinder eval --run outputs/<show-slug> --gold eval/gold.yaml [--compare outputs/<other-run>]

# Experiment A: let the LLM write timestamps itself, then check them
.venv/bin/python -m promofinder experiment plan-a --input "/path/to/show" --episodes 1,10,20,30,40
# Experiment B: does full-story context make the tagger pick moments strangers can't follow?
.venv/bin/python -m promofinder experiment hindsight --input "/path/to/show" --episodes 1,7,13,19,25,31,37,43

# editor loop: mark editor_verdict ✓/✗ (+ editor_note) in promos.csv, then rerun with it
.venv/bin/python -m promofinder run --input "/path/to/show" --feedback outputs/<show-slug>/promos.csv
```

`eval` scores only the human-picked `G` entries for `recall_at_20`; promos an editor ✓'d on earlier tool output are added to the gold file as `F` entries and reported separately (`recall_at_20_feedback`), so the tool never grades itself on its own picks. Gold picks are never shown to any model.

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
| 6 | Multi-segment stitcher (experimental) | LLM proposes beat IDs, code builds + validates | `stitched.json`, appended to candidates |
| 7 | Blind judge + first-3-seconds test | LLM | `judgements.json` |
| 8 | Cut refinement | code offers options, LLM picks | `refine.json` |
| 9 | Duplicate check, then pairwise "which would you post first?" ranking (both orders) | LLM | `dedupe.json`, `pairwise.json` |
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
