# Solution doc outline: Problem 1, "Find the Promos"

This is raw material for the 2-page PDF. Rewrite it in your own words.
Every number below comes from a real run. `⟨PENDING⟩` marks a number that doesn't exist yet. Don't fill those in by hand.

## 1. Problem picked and why (2–3 lines)
Problem 1 is about finding the best 20–60 s promos in a 50-episode microdrama screenplay. Each promo must open on a hook in its first 3 s, end on a cliffhanger, and make sense to a stranger. It is a ranking problem with hard edit rules, and I can measure it against a hand-picked gold set.

## 2. Assumptions
- The screenplay timestamps match the final video timeline.
- A hook in the first 3 s can be judged from text. Visual-only moments are a known weak spot.
- Most real promos are one continuous stretch. Multi-segment promos are an experimental extra.
- Late-show spoilers are allowed but flagged.
- Early-episode promos get a small bonus, because promos send viewers to Episode 1.

## 3. How it works: code vs AI vs human
- **Code** does all the timing work:
  - parses each screenplay into numbered lines
  - builds every clip and writes every timestamp
  - snaps cuts to camera shots
  - never cuts through a spoken line
  - keeps each clip at 20–60 s
  - never includes the moment the cliffhanger is answered
- **AI** works with line IDs only, never timestamps. It:
  1. reads a one-call show summary
  2. tags hook and cliffhanger lines by ID; a quote check rejects any line it invents
  3. judges each clip *blind*, from its text only, as a stranger would see it
  4. scores how strong the first 3 seconds are
  5. picks between code-built alternative cuts
  6. compares finalists head-to-head ("which would you post first?") with Elo
- **Human:** the editor marks ✓/✗ in `promos.csv`. Those marks become prompt examples and small weight nudges, and they are scored separately from the gold set.
- **Models:** any OpenAI-compatible endpoint works through `config/models.yaml`, with a fallback chain per stage. This build used NVIDIA's free `nemotron-3-ultra-550b-a55b`, plus `nemotron-3.5-lightning-30b-a3b` for the 3-second test.

## 4. Scale and cost
- **First full run** (no show summary, no head-to-head ranking, judge capped at 30):
  - 112 calls, about 373k tokens
  - about 14 min wall time: tagging 672 s, judge 145 s
- **M2 run:** ⟨PENDING⟩
- **Final 3-vote run:** ⟨PENDING⟩
- **Per new show:** the same command runs unchanged, it resumes after a crash, and responses are cached.

## 5. How I know it's good, and where it fails
- **Gold set:** 12 picks from 9 episodes. I drafted them from the screenplays and the user reviewed them (5 changed, 1 swapped), all before any full tool run. No model ever sees them.

| Run | recall@20 | in candidate pool | invariants | notes |
|---|---|---|---|---|
| Baseline (stages 1,2,4,5,7,10) | **1/12** | 9/12 | 100% | tagger scores inflated, judge cut at 30 |
| M2 (+ show summary, calibration examples, judge all) | ⟨PENDING⟩ | ⟨PENDING⟩ | ⟨PENDING⟩ | |
| M3 (+ stitch, refine, pairwise) | ⟨PENDING⟩ | ⟨PENDING⟩ | ⟨PENDING⟩ | pairwise first-position rate ⟨PENDING⟩ |
| Final (3 votes) | ⟨PENDING⟩ | | | precision@10 from editor ✓/✗: ⟨PENDING⟩ |

- **Where it fails:** fill this in from the misses `eval` lists and the editor's ✗ notes. Known so far:
  - **Pool size:** with only 3 clips per episode, some strong moments never become candidates.
  - **Sound-only hooks:** a hook carried only by sound, such as pounding at a door, is hard to judge from text.

## 6. AI tools and where they got it wrong (~5 lines)
- **Tools:** Claude Code (design and code) and NVIDIA API models (tagging, judging, ranking). Models per call are recorded in `run_meta.json`.
- **The one failure:** pick one of the two candidates below.
  - **Candidate A, Experiment A:** when the model wrote its own clip timestamps, only **6 of 19 (32%)** were usable. 12 were off a shot boundary, 3 cut through a line and 1 was the wrong length. That is why the AI only tags line IDs and code makes every cut.
  - **Candidate B, Experiment B:** given the whole story, the tagger picked 16 moments it never picked without it. Those clips scored **5.59/10** on the blind stranger test, against **7.75** for moments it picked either way. That is why the judge stays blind.
- **Also real:** the tagger's scores were inflated (median 9/10), which drove the 1/12 baseline. Calibration examples and head-to-head ranking were added as the fix.

## 7. Links
- Repo: ⟨your GitHub URL⟩
- Raw outputs: ⟨anyone-with-link Google Drive folder⟩, a zip of `outputs/<show-slug>/` (it quotes the script, so it is never committed)
- Optional video: ⟨link⟩
