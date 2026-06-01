# ARC-AGI-2 (ARC Prize 2026)

Team entry for the ARC Prize 2026 **ARC-AGI-2** static-reasoning track.

See [`CLAUDE.md`](./CLAUDE.md) for the full project brief — goal, hard constraints
(offline / single GPU / 12h / no internet at eval), hardware notes, conventions,
and the one-month plan. The plan, gates, and per-experiment results log live in
[`ARC-AGI-2-One-Month-Plan.xlsx`](./ARC-AGI-2-One-Month-Plan.xlsx).

## Quick start

```bash
# (one-time) fetch the ARC-AGI-2 public dataset into ./data/arc-agi-2/
make data         # or see scripts/fetch_data.sh

# run the local eval harness against the public eval split
python -m src.eval.runner --split public_eval --solver identity --limit 20
```

The harness loads tasks, runs the named solver, scores with the official
2-attempt metric, and writes a per-task report under `runs/<timestamp>/`.

## Layout

```
src/eval/       local eval harness (data loader, metric, runner, report)
src/solver/     solver implementations (identity, TRM, ...)
src/augment/    data augmentation (Week 2)
src/select/     candidate generation + voting (Week 2)
src/submit/     Kaggle notebook entry point (Week 2)
data/           ARC-AGI-2 public data (gitignored; fetched on demand)
experiments/    per-run configs, named YYYY-MM-DD-<id>
runs/           harness output, one folder per run (gitignored)
tests/          unit tests (start with the 2-attempt metric)
```

## License

MIT-0 (MIT No Attribution). Required for ARC Prize eligibility — solution must
be open-sourced under a permissive license before official private scores are
released.
