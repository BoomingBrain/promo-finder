# AI tools log — where they got it wrong

Tools used: Claude Code (design + implementation), NVIDIA API catalogue models (tagging, judging, ranking — exact models per run are recorded in each `run_meta.json`).

Raw outputs that quote the screenplay live in the Google Drive raw-outputs folder, never in this repo.

| # | Tool | What I asked | What it did wrong | How I caught it | What I changed |
|---|---|---|---|---|---|
| 1 | Claude Code | Scan the 50 screenplays for structure stats | Data-scan regex matched only `Scene end:`; reported 49 episodes because Episode 6 uses `SCENE END:` | Count didn't match the 50 files | Case-insensitive parser, parse-health summary + fail-loud threshold, parser test for the upper-case variant |
| 2 | Claude Code | Design the LLM response cache | First cache key was `model + prompt`, so 3 majority-vote runs would have returned one cached answer three times and faked "agreement" | Design review | Vote index added to the cache key; test `test_cache_hit_and_vote_index_in_key` |
| 3 | Claude Code + NVIDIA catalogue | Pick default models | The planned default model IDs (`nvidia/llama-3.3-nemotron-super-49b-v1`, `meta/llama-3.3-70b-instruct`) no longer exist; two other models are *listed* by `/v1/models` but return 404 when called; a reasoning model returned an empty `content` with all 200 tokens spent on reasoning | Live probe calls before the first run | Fallback chains per stage, 404 → next model, empty content falls back to the reasoning field, larger `max_tokens`, models re-chosen from a live probe |
