# Vendored: Tiny Recursive Model (TRM)

This directory is a vendored copy of the official TRM reference implementation.
We treat it as a read-only upstream, with our own wrappers and adapters living
outside (in `src/solver/trm_solver.py` and the Kaggle notebook).

## Provenance

- **Upstream:** https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **Commit:** `c01103738605ba39d1430519b1ee0c62f4c707f8`
- **Paper:** Alexia Jolicoeur-Martineau, "Less is More: Recursive Reasoning with
  Tiny Networks" (arXiv:2510.04871, 2025).
- **License:** MIT (see `LICENSE` in this directory; compatible with our
  MIT-0 project license).

The upstream repo is archived (read-only) as of late 2025; we cannot send PRs.

## What's vendored

- All Python sources (`models/`, `dataset/`, `evaluators/`, `utils/`,
  `pretrain.py`, `puzzle_dataset.py`).
- The pinned configs (`config/`).
- The `kaggle/combined/` ARC task JSONs — these are the **exact** task files
  the verification checkpoint's `puzzle_emb` table was built from. Do not
  swap them for arcprize/ARC-AGI-2's `data/training/` JSONs unless you also
  re-train the puzzle embeddings.

## What's omitted

- `assets/` — upstream figures (PNGs), not needed for code.
- `.git/` — upstream history.

## Verification checkpoint

We use the ARC Prize Foundation's verification checkpoint, MIT-licensed, at
https://huggingface.co/arcprize/trm_arc_prize_verification — specifically
`arc_v2_public/step_723914`. See `scripts/fetch_trm_checkpoint.sh`. The
checkpoint is downloaded to `experiments/trm_arc_v2_public/` (gitignored).
