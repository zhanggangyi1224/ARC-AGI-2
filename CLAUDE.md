# CLAUDE.md

Project context for the **ARC Prize 2026 — ARC-AGI-2** effort. Drop this at the repo root. Read it before doing anything in this repo; it captures the goal, the hard constraints, the plan, and the conventions an assistant or teammate must follow.

---

## 1. Project snapshot

- **What:** Build an AI that solves ARC-AGI-2 static grid puzzles (infer a transformation rule from a few demo input→output pairs, produce outputs for new test inputs).
- **Track:** ARC Prize 2026 — ARC-AGI-2 (the static-reasoning track). $700K pool.
- **Team:** Two people. Strong software engineering, lighter ML.
  - **A** — method / model (TTT loop, training, model side).
  - **B** — infra / eval (eval harness, augmentation, submission pipeline, compute budget).
- **Goals:** Compete seriously for a prize **and** produce a competition paper (the paper track is automatic-eligible off the same Kaggle submission, so document as we go).
- **Realistic target:** A **Progress Prize** (relative leaderboard rank), not the 85% bonus. For reference, the 2025 top score was ~24% and 4th place was ~6.7% — a careful fork of a prior winner placed in the money.

---

## 2. Strategic decision (READ THIS)

We are running a **one-month AGI-2 trial (Jun 1–30, 2026)** with a hard checkpoint, then deciding whether to continue or switch to ARC-AGI-3.

**Switch criteria at Jun 30** (see `ARC-AGI-2-One-Month-Plan.xlsx`, **Gates** tab):
- Reproduced a baseline offline? (table stakes)
- Did our own changes beat baseline by a meaningful margin?
- Is there a non-empty queue of promising ideas?
- Is the team energized, not grinding?

**Decision rule:**
- 3–4 green → **continue AGI-2**, we're on a real slope.
- Reproduced-but-stuck + empty idea queue → **switch to AGI-3**.
- Couldn't reproduce baseline offline → fit/infra problem, lean toward switching.

> "Nothing got" is the wrong test: on AGI-2 you'll always reproduce *some* baseline. The real signal is **trajectory** — slope of improvement and length of the idea queue, not the absolute score.

If we switch, the Kaggle pipeline, eval harness, team workflow, and results log all carry over to AGI-3. The TTT *method* work does not.

---

## 3. Hard constraints (design against these from day one)

- **Eval envelope:** Final scored submission runs **offline in a Kaggle notebook, single GPU (historically P100-class), ≤12 hours, NO internet.** Confirm the exact 2026 spec on the Kaggle competition page. Every method must fit this — no big multi-GPU ensembles, no API calls.
- **Scoring:** Submit **exactly 2 predictions per test input**. If *either* exactly matches ground truth, that task scores 1, else 0. Final = average over all test outputs. **Exploit both attempts** — the 2nd guess should be genuinely diverse, not a near-duplicate.
- **Open source is mandatory and BEFORE scoring:** Prize eligibility requires open-sourcing the solution before receiving official private scores. Use a permissive license (**CC0** or **MIT-0**). Keep repo hygiene from commit one.
- **Submission format:** Must be a Kaggle notebook, one-click runnable, reproducible inside the time budget.

---

## 4. Hardware & environment

**Local dev machine:** Mac mini, Apple Silicon, 64GB unified memory, 2TB storage.

- **The Mac is the cockpit, not the engine.** Use it for: the eval harness, augmentation/data pipeline, results log, light local inference (MLX or llama.cpp; quantized models fit easily in 64GB).
- **Do NOT** run the CUDA TTT stack on Metal — `unsloth`, `vLLM`, `bitsandbytes` 4-bit, `flash-attention` are CUDA-only or poorly supported on MPS. Trying to make them work locally is a time sink. Don't.
- **TRM (~7M params)** is small at *inference* — MPS can run it (slowly: rough estimate ~20–60h for full public_eval × 1000-augment voting; use for small-subset smoke tests, not full repros). **Training TRM from scratch at ARC-AGI-2 scale is NOT Mac-feasible** — the paper used 4× H100 × ~3 days; the published [verification checkpoint](https://huggingface.co/arcprize/trm_arc_prize_verification) used 1× 8-H100 node × 20–30h. We reproduce via that checkpoint, not by retraining.
- **Dev-time training:** Use Kaggle's free GPU notebooks (separate from the submission) or a cheap rented GPU (Colab / RunPod / Lambda). **Pin versions** so "works on my Mac" == "works in eval."
- **Storage:** 2TB is a non-issue; ARC data is tiny and checkpoints are small.
- **Method steer:** Hardware + the 12h/1-GPU envelope both point to the **lean end** — TRM-style or a single small-model TTT, not a heavy ensemble we couldn't submit anyway.

---

## 5. Approach

Two related-but-distinct winning patterns in 2025:

- **Test-time training (TTT)** (NVARC, MindsAI): start from a pretrained small LLM, fine-tune it *at inference* on the task's demo pairs and augmented variants, sample many candidates, vote.
- **Recursive reasoning at inference** (TRM): conventionally pretrain a small (~7M) network on ARC training+concept data; at inference, the model recursively refines its predicted output over K steps (ACT-halted) against a large bank of augmented variants of the test input, then content-hash-votes the top-2. **Not** TTT — no per-task gradient updates.

Most leverage in either is pipeline engineering (augmentation quality, selection/voting, staying in budget), which suits us; pushing the training loop past a fork needs the ML half.

**Start simplest:** fork ONE credible baseline and reproduce its score before changing anything. Candidates:
- **TRM** (2025 paper winner) — pretrained recursive-reasoning model, ~7M params. We use the published MIT-licensed verification checkpoint (`huggingface.co/arcprize/trm_arc_prize_verification`) rather than retraining from scratch. **Recommended starting point** — smallest blast radius, cleanest reproduction path.
- **NVARC** (2025 1st) — TTT on top of a TRM-class model; open code + paper.
- **MindsAI** (2025 3rd) — engineered TTT pipeline on top of a small LLM; open code + paper.

**Reproduce before you innovate.** This is the single biggest separator between teams that climb and teams that thrash.

---

## 6. Repository layout (proposed)

```
.
├── CLAUDE.md                          # this file
├── ARC-AGI-2-One-Month-Plan.xlsx      # tracker: Plan / Gates / Results Log / Setup tabs
├── LICENSE                            # CC0 or MIT-0
├── README.md
├── data/                    # ARC-AGI-2 public training + eval (gitignored if large)
├── src/
│   ├── eval/                # local eval harness (owner B)
│   │   ├── runner.py        # load tasks, run solver, score with 2-attempt metric
│   │   └── report.py        # per-task pass/fail + failure-cluster analysis
│   ├── augment/             # rotations, reflections, color permutations
│   ├── solver/              # the method (owner A): TTT loop / TRM
│   ├── select/              # candidate generation + voting + 2nd-attempt diversity
│   └── submit/              # Kaggle notebook entry point, budget guards
├── notebooks/               # Kaggle dev + submission notebooks
├── experiments/             # configs per run, named by date+id
└── tests/
```

---

## 7. Dev workflow & commands

> These are the team's conventions. Wire them to real scripts/Makefile targets as the repo fills in. An assistant should use these rather than inventing ad-hoc commands.

- **Eval locally (primary loop):** `python -m src.eval.runner --split public_eval --solver <name>` → prints score + writes a per-task report. This is the workhorse; **never** burn a Kaggle submission to test something the local harness can check.
- **Failure analysis:** `python -m src.eval.report --run <id>` → clusters failures by task type.
- **Train (dev):** run on Kaggle GPU notebook or rented GPU, **not** the Mac (except TRM). Save checkpoints to `experiments/<date>-<id>/`.
- **Submit:** the Kaggle notebook in `notebooks/` is the entry point; it must run offline within the 12h/1-GPU budget and emit 2 predictions per test input.

**Golden rules of the loop:**
1. Trustworthy **local eval** first → Kaggle leaderboard is *confirmation*, not exploration.
2. Land a real submission **early** (week 2) to prove the offline pipeline before piling on features.
3. Every change gets logged with its score delta (see §8).

---

## 8. Results log discipline (non-negotiable)

Log **every** experiment in the **Results Log** tab of `ARC-AGI-2-One-Month-Plan.xlsx` from day one: date, who, change, baseline score, new score, delta, keep/drop, why / next idea. It is both the debugging trail and the **skeleton of the paper**. Do not skip it when busy — that's exactly when it matters.

---

## 9. One-month plan + gates (condensed)

| Week | Dates | Focus | Gate |
|------|-------|-------|------|
| 1 | Jun 1–7 | Repo + log setup; solve tasks by hand; load data; fork ONE baseline; build local eval harness | **G1:** baseline reproduced on Kaggle AND local harness matches |
| 2 | Jun 8–14 | Land a real leaderboard submission; augmentation pipeline; candidate+voting scaffold; failure clustering | **G2:** scored submission inside budget + ranked failure clusters |
| 3 | Jun 15–21 | Attack 2–3 high-leverage fixes; push TTT loop (A); own the 2-attempt strategy (B); log everything | **G3 (trajectory):** own changes beat baseline above noise AND idea queue grew |
| 4 | Jun 22–30 | Consolidate best direction; harden submission; write up month; (optional) one AGI-3 preview agent | **FINAL:** run the §2 go/no-go |

Full tracker with owners and status: `ARC-AGI-2-One-Month-Plan.xlsx` (**Plan / Gates / Results Log / Setup** tabs).

---

## 10. Conventions

- **Language/stack:** Python. Keep dependencies minimal and **version-pinned** (mirror the Kaggle environment).
- **Configs over hardcoding:** experiments are driven by config files in `experiments/`, named `YYYY-MM-DD-<short-id>`.
- **Determinism:** seed everything; record seeds in the run config so scores are reproducible.
- **Git:** small, frequent commits; descriptive messages; never commit large data or checkpoints (gitignore them).
- **Licensing:** repo carries CC0 or MIT-0; keep it prize-eligible at all times.
- **No secrets / no network in submission code.**

---

## 11. Rules for the assistant working in this repo

**Do:**
- Default to the **lean** approaches (TRM / single-small-model TTT) that fit the 12h/1-GPU envelope.
- Check changes against the **local eval harness** before suggesting a Kaggle submission.
- Keep every method runnable **offline** and within the time budget.
- Update the **Results Log** tab of `ARC-AGI-2-One-Month-Plan.xlsx` (or remind the team to) after any experiment.
- Prefer reproducing/understanding the chosen baseline over adding novelty early.

**Don't:**
- Don't propose running the CUDA TTT stack on the Mac, or any solution needing internet/APIs at eval time.
- Don't add a method that can't fit a single GPU in 12 hours.
- Don't introduce dependencies that aren't available/pinnable in the Kaggle eval environment.
- Don't let the 2nd of the two predictions be a trivial near-duplicate of the first.
- Don't break the open-source-before-scoring rule or the permissive license.

---

## 12. Resources

- ARC Prize 2026 overview — arcprize.org/competitions/2026
- ARC-AGI-2 competition — arcprize.org/competitions/2026/arc-agi-2
- ARC-AGI-3 competition (fallback track) — arcprize.org/competitions/2026/arc-agi-3
- Paper track — arcprize.org/competitions/2026/paper
- ARC-AGI-1 & 2 technical guide — arcprize.org/guide/1
- 2025 results & analysis — arcprize.org/blog/arc-prize-2025-results-analysis
- Kaggle (AGI-2) — kaggle.com/competitions/arc-prize-2026-arc-agi-2
- ARC Prize GitHub — github.com/arcprize

---

*Figures and rules reflect the official ARC Prize pages and 2025 results analysis as of 1 June 2026. Confirm final deadlines, the exact compute/runtime limit, and any rule updates on the live Kaggle competition page before committing.*
