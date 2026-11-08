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
- **Models:** any OpenAI-compatible endpoint works through `config/models.yaml`, with a fallback chain per stage.
  - The final run uses NVIDIA's free `nemotron-3-ultra-550b-a55b` for the show summary and tagging.
  - It uses `z-ai/glm-5.3-flash` for judging, the 3-second test, cut choice and head-to-head. GLM was chosen by a both-orders consistency test against DeepSeek and Mistral Large 4.
  - A rate limit waits and retries the same model; it never downgrades.

## 4. Scale and cost
- **First full run** (no show summary, no head-to-head ranking, judge capped at 30):
  - 112 calls, about 373k tokens
  - about 14 min wall time: tagging 672 s, judge 145 s
- **M2 run** (+ show summary, calibration examples, judge every candidate): 235 calls, about 569k tokens, tagging 604 s + judging 351 s.
- **3-vote run** (all 10 stages): 427 calls (324 live), about 1.86M tokens, 41 min 42 s wall time.
  - The free tier struggled at 3× volume: 36 of 150 tagging calls failed outright, 84 calls fell back to the second model and 95 replies needed a JSON retry.
  - Token use was about 2× the dry-run estimate because of the retries.
- **Final run** (1 vote, all 10 stages): 307 calls (255 live), about 401k tokens, 48 min 51 s wall time.
  - 0 failed calls and 1 rate-limit wait.
  - Judging is the slow part: GLM took about 37 min for 94 clips.
- **Per new show:** the same command runs unchanged, it resumes after a crash, and responses are cached.

## 5. How I know it's good, and where it fails
- **Gold set:** 12 picks from 9 episodes. I drafted them from the screenplays and the user reviewed them (5 changed, 1 swapped), all before any full tool run. No model ever sees them.

| Run | recall@20 | in candidate pool | invariants | notes |
|---|---|---|---|---|
| Baseline (stages 1,2,4,5,7,10) | **1/12** | 9/12 | 100% | tagger scores inflated, judge cut at 30 |
| M2 (+ show summary, calibration examples, judge all) | **3/12** | 8/12 | 98.9% | 1 mid-line cut found → bug fixed (ai-log #10) |
| M3 (+ stitch, refine, debiased pairwise, cut fix) | **3/12** | 8/12 | 100% | branding-iron pick reached #1; pairwise agreed with itself 21% |
| 3 votes, all stages (mixed models under rate limits) | **3/12** (#1, #5, #9) | 6/12 | 100% | vote agreement 50%; pairwise agreed with itself 14–25%; the small fallback model answered 37–50% of calls |
| **Final: 1 vote, Ultra tags, GLM-5.3-flash judges, no downgrade on 429** | **4/12** (#1, #5, #8, #20) | 8/12 | 100% | pairwise agreed with itself **79%**, first-shown won 39%; 0 failed calls; precision@10 **5/10** (editor marks AI-drafted, pending user review) |

- **Where it fails:** this comes from real evidence, not guesses. `report.md` now flags these automatically for the top 20.
  - **Ranking, not finding, was the first bottleneck.** In the baseline, 9/12 gold moments were candidates but only 1 made the top 20. Calibration examples and judging every candidate took it to 3/12.
  - **Tagging is noisy.** The 3 votes agreed on only 50% of tagged moments, partly because a third of the votes came from the small fallback model. Majority voting then *shrank* the pool (6/12 gold moments vs 8/12 with 1 vote), so 3 votes did not beat 1 vote on recall, and the final run uses 1.
  - **Model choice matters more than model size.** In earlier runs the head-to-head stage looked useless: it picked the first-shown clip 86–93% of the time.
    - The real cause was rate limits quietly pushing half the calls onto a small backup model.
    - With the client fixed to wait instead of downgrading, and GLM-5.3-flash judging, it agreed with itself 79% of the time with no position bias.
    - In a 10-pair test the 1T-parameter Mistral Large 4 was *worse* (5/10) than the 'flash' GLM (9/10).
  - **Cuts that are close but too long.** G02 and G10 are covered at 45–48%: the tool's clip starts or ends a few shots away from the editor's, so it falls just short of the 50% rule.
  - **Editor review of the final top 10 (5 ✓ / 5 ✗).** The misses fall into the spec's categories:
    - **Rule gaps the code could catch:**
      - #10 opens on the second half of a sentence: the previous beat ends with "...", a split line.
      - #10 also ends on a triumphant exit, which resolves the scene instead of leaving a cliffhanger.
      - #8 drops the setup that tells a stranger she is condemned, so its cliffhanger reads as ambiguous.
    - **Near-duplicate across episodes:** #6 repeats #5's exact cliffhanger, the branding iron at her eye, one episode later. The duplicate check only works within a scene.
    - **Too talky:** in #3 (legal exposition) and #9 (coded negotiation), most of the clip is dialogue a stranger can't decode.
    - **Even the ✓ picks often need a small trim:** #2's middle sags, and #5 should hold 5 s longer on the eye shot.
 7 of the final top 20 open on an action or camera beat, so text-only judging may misread them. They are flagged for the editor.
  - **Missed scenes.** Some strong moments never became candidates at all (G03, G04, G06). With only 3 clips per episode, a talky set-up beat can crowd them out.

## 6. AI tools and where they got it wrong (~5 lines)
- **Tools:** Claude Code (design and code) and NVIDIA API models (tagging, judging, ranking). Models per call are recorded in `run_meta.json`.
- **The one failure:** pick one of the two candidates below.
  - **Candidate A, Experiment A:** when the model wrote its own clip timestamps, only **6 of 19 (32%)** were usable. 12 were off a shot boundary, 3 cut through a line and 1 was the wrong length. That is why the AI only tags line IDs and code makes every cut.
  - **Candidate B, Experiment B:** given the whole story, the tagger picked 16 moments it never picked without it. Those clips scored **5.59/10** on the blind stranger test, against **7.75** for moments it picked either way. That is why the judge stays blind.
- **Also real:**
  - The tagger's scores were inflated (median 9/10), which drove the 1/12 baseline.
  - The head-to-head judge picked the first-shown clip 86% of the time (ai-log #11).
  - Claude Code itself wrote a clip builder that let a mid-line cut through (ai-log #10), and a hindsight metric that counted "not picked" as a score of 0 (ai-log #9).

## 7. Links
- Repo: https://github.com/BoomingBrain/promo-finder (currently private; make it public or invite the reviewers before submitting)
- Raw outputs: ⟨anyone-with-link Google Drive folder⟩, a zip of `outputs/<show-slug>/` (it quotes the script, so it is never committed)
- Optional video: ⟨link⟩
