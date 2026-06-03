"""TRM inference driver.

A thin, device-portable wrapper around TRM's eval loop. Loads the published
verification checkpoint, runs ACT inference over a preprocessed dataset,
voting-aggregates over augmentations, and writes a Kaggle-format
`submission.json` plus a pass@K summary.

Usage (after running `scripts/fetch_trm_checkpoint.sh` and the
`build_arc_dataset` preprocessing):

    python -m src.solver.trm_inference \\
        --checkpoint experiments/trm_arc_v2_public/step_723914 \\
        --config     experiments/trm_arc_v2_public/all_config.yaml \\
        --data       data/arc2concept-aug-1000 \\
        --device     cuda \\
        --out        runs/<id>

This module avoids hydra/omegaconf gymnastics: it builds the model directly
from the saved YAML and the dataset metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Vendored TRM uses unqualified imports (`from models.common import ...`,
# `from dataset.common import ...`, ...). Make those resolvable here without
# rewriting the upstream source.
_TRM_ROOT = Path(__file__).resolve().parent / "trm"
if str(_TRM_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRM_ROOT))

# TRM imports (resolve via the path insert above)
import torch  # noqa: E402
from torch import nn  # noqa: E402

from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from evaluators.arc import ARC  # noqa: E402
from models.losses import ACTLossHead  # noqa: E402
from models.recursive_reasoning.trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1,
)
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Model construction


def _build_model(
    arch_cfg: dict,
    train_metadata: PuzzleDatasetMetadata,
    device: torch.device,
) -> nn.Module:
    """Instantiate TRM + ACTLossHead matching the saved checkpoint's config."""
    model_cfg = dict(
        batch_size=arch_cfg.get("batch_size", 1),  # unused at inference
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
        # arch-specific fields copied through verbatim
        H_cycles=arch_cfg["H_cycles"],
        L_cycles=arch_cfg["L_cycles"],
        H_layers=arch_cfg["H_layers"],
        L_layers=arch_cfg["L_layers"],
        hidden_size=arch_cfg["hidden_size"],
        expansion=arch_cfg["expansion"],
        num_heads=arch_cfg["num_heads"],
        pos_encodings=arch_cfg["pos_encodings"],
        halt_max_steps=arch_cfg["halt_max_steps"],
        halt_exploration_prob=arch_cfg["halt_exploration_prob"],
        puzzle_emb_ndim=arch_cfg["puzzle_emb_ndim"],
        puzzle_emb_len=arch_cfg["puzzle_emb_len"],
        forward_dtype=arch_cfg["forward_dtype"],
        mlp_t=arch_cfg.get("mlp_t", False),
        no_ACT_continue=arch_cfg.get("no_ACT_continue", True),
    )
    with torch.device(device):
        inner = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        loss_kwargs = {"loss_type": arch_cfg["loss"]["loss_type"]}
        model = ACTLossHead(inner, **loss_kwargs)
    model.eval()
    return model


def _load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Load a TRM training state_dict, stripping the torch.compile prefix."""
    raw = torch.load(str(ckpt_path), map_location=device)
    if not isinstance(raw, dict):
        raise ValueError(f"unexpected checkpoint type: {type(raw)}")

    # The training run used torch.compile(loss_head_model), so saved keys are
    # `_orig_mod.<...>`. We didn't compile here, so strip the prefix.
    PREFIX = "_orig_mod."
    state_dict = {
        (k[len(PREFIX):] if k.startswith(PREFIX) else k): v for k, v in raw.items()
    }

    # Honour the upstream resize-on-mismatch heuristic for puzzle_emb
    # (defensive: should be a no-op when our identifier_map matches training).
    puzzle_emb_key = "model.inner.puzzle_emb.weights"
    if puzzle_emb_key in state_dict:
        ckpt_w = state_dict[puzzle_emb_key]
        expected_shape = model.model.inner.puzzle_emb.weights.shape  # type: ignore[union-attr]
        if ckpt_w.shape != expected_shape:
            print(
                f"[warn] puzzle_emb shape mismatch (ckpt {tuple(ckpt_w.shape)} vs "
                f"model {tuple(expected_shape)}); resetting to mean — this means "
                f"the test puzzle IDs don't match training and the eval will be junk."
            )
            state_dict[puzzle_emb_key] = (
                torch.mean(ckpt_w, dim=0, keepdim=True).expand(expected_shape).contiguous()
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        print(f"[warn] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys (first 5): {unexpected[:5]}")


# ---------------------------------------------------------------------------
# Inference loop


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _run_inference(
    model: nn.Module,  # the ACTLossHead-wrapped model
    data_path: Path,
    eval_metadata: PuzzleDatasetMetadata,
    device: torch.device,
    batch_size: int,
    submission_K: int,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    ds = PuzzleDataset(
        config=PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[str(data_path)],
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="test",
    )

    evaluator = ARC(
        data_path=str(data_path),
        eval_metadata=eval_metadata,
        submission_K=submission_K,
        pass_Ks=(1, 2, 5, 10, 100, 1000),
        aggregated_voting=True,
    )
    evaluator.begin_eval()

    # Skip the ACTLossHead — at inference we want neither loss computation
    # nor the fp64 upcast in stablemax_cross_entropy (that's what blew the
    # T4 OOM at batch=768). The verification checkpoint was saved with the
    # loss-wrapped model, so weights are under model.model.<...>; we still
    # call through model so those keys resolve, but we explicitly only run
    # the inner ACT module.
    inner_act = model.model  # TinyRecursiveReasoningModel_ACTV1

    # Estimate total batches so the user sees a denominator, not a black box.
    # PuzzleDataset iterates over *examples* (each puzzle in the test split
    # contributes ~mean_puzzle_examples rows: demo pairs + test pairs all
    # get model-forwarded). Round UP so ETA never goes negative.
    total_examples = int(round(eval_metadata.mean_puzzle_examples * eval_metadata.total_puzzles))
    estimated_batches = max(1, (total_examples + batch_size - 1) // batch_size)
    print(
        f"[trm] starting inference: {total_examples:,} rows "
        f"({eval_metadata.total_puzzles:,} puzzles × ~{eval_metadata.mean_puzzle_examples:.1f} ex/puzzle), "
        f"batch_size={batch_size} → ~{estimated_batches:,} batches",
        flush=True,
    )

    started = time.monotonic()
    n_batches = 0
    # Print on batches 1, 2, 5, 10, 25, then every 50. Front-loaded prints
    # surface OOM-vs-progress quickly; back-loaded keeps the log readable.
    progress_milestones = {1, 2, 5, 10, 25}
    with torch.inference_mode():
        for set_name, batch, _ in iter(ds):
            n_batches += 1
            batch = _to_device(batch, device)
            with torch.device(device):
                carry = inner_act.initial_carry(batch)
            while True:
                carry, outputs = inner_act(carry=carry, batch=batch)
                if carry.halted.all():
                    break
            # Move q_halt_logits to CPU *before* the evaluator — it does
            # v.to(torch.float64).sigmoid() on this tensor, and MPS has no
            # float64 support. Doing the .cpu() here is a no-op on CUDA
            # (the evaluator would .cpu() it anyway) and unblocks MPS.
            preds = {
                "preds": torch.argmax(outputs["logits"], dim=-1),
                "q_halt_logits": outputs["q_halt_logits"].cpu(),
            }
            evaluator.update_batch(carry.current_data, preds)
            if n_batches in progress_milestones or n_batches % 50 == 0:
                elapsed = time.monotonic() - started
                rate = n_batches / elapsed if elapsed > 0 else 0
                eta_s = (estimated_batches - n_batches) / rate if rate > 0 else 0
                print(
                    f"  ...batch {n_batches:,}/{estimated_batches:,} "
                    f"in {elapsed:.1f}s ({rate:.2f} batch/s, ETA {eta_s/60:.1f} min)",
                    flush=True,
                )
            if limit_batches is not None and n_batches >= limit_batches:
                print(f"[trm] stopping early at --limit-batches={limit_batches}", flush=True)
                break

    # ARC.result() uses dist.gather_object; we're single-process, so call the
    # post-gather aggregation directly.
    return _aggregate_and_score(evaluator)


def _aggregate_and_score(evaluator: ARC) -> dict[str, Any]:
    """Replay evaluator.result()'s post-gather aggregation, single-process."""
    from dataset.build_arc_dataset import arc_grid_to_np, grid_hash  # local import after sys.path edit

    submission: dict[str, list[dict[str, list[list[int]]]]] = {}
    pass_Ks = evaluator.pass_Ks
    correct = [0.0 for _ in pass_Ks]

    # Single rank: replace dist.gather with a singleton list.
    global_hmap_preds = [(evaluator._local_hmap, evaluator._local_preds)]

    for name, puzzle in evaluator.test_puzzles.items():
        submission[name] = []
        num_test_correct = [0 for _ in pass_Ks]
        for pair in puzzle["test"]:
            input_hash = grid_hash(arc_grid_to_np(pair["input"]))
            label_hash = grid_hash(arc_grid_to_np(pair["output"]))

            p_map: dict[str, list[float]] = {}
            for hmap, preds in global_hmap_preds:
                for h, q in preds.get(name, {}).get(input_hash, []):
                    p_map.setdefault(h, [0.0, 0.0])
                    p_map[h][0] += 1
                    p_map[h][1] += q

            if not p_map:
                print(f"[warn] puzzle {name} input {input_hash[:8]} has no predictions")
                continue

            for h, stats in p_map.items():
                stats[1] /= stats[0]

            sorted_preds = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
            for i, k in enumerate(pass_Ks):
                num_test_correct[i] += int(any(h == label_hash for h, _ in sorted_preds[:k]))

            pred_grids = []
            for h, _ in sorted_preds[: evaluator.submission_K]:
                for hmap, _ in global_hmap_preds:
                    if h in hmap:
                        pred_grids.append(hmap[h])
                        break
            while len(pred_grids) < evaluator.submission_K:
                pred_grids.append(pred_grids[0])

            submission[name].append(
                {f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)}
            )

        for i in range(len(pass_Ks)):
            correct[i] += num_test_correct[i] / len(puzzle["test"])

    n = len(evaluator.test_puzzles)
    pass_at_k = {f"pass@{k}": correct[i] / n for i, k in enumerate(pass_Ks)}
    return {"submission": submission, "pass_at_k": pass_at_k, "num_tasks": n}


# ---------------------------------------------------------------------------
# CLI


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _read_metadata(data_path: Path) -> PuzzleDatasetMetadata:
    with open(data_path / "test" / "dataset.json") as f:
        return PuzzleDatasetMetadata(**json.load(f))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TRM inference driver")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="all_config.yaml shipped with the checkpoint")
    p.add_argument("--data", type=Path, required=True, help="preprocessed arc2concept-aug-1000 dir")
    p.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="default 64 is safe on T4 16GB. P100/V100/A100 can push 128–256+.",
    )
    p.add_argument("--submission-K", type=int, default=2)
    p.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="stop after this many batches (for local smoke tests; pass@K will be partial)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--disable-compile", action="store_true", default=True, help="kept as no-op; we never compile here")
    args = p.parse_args(argv)

    os.environ.setdefault("DISABLE_COMPILE", "1")
    # Reduce CUDA allocator fragmentation; harmless on MPS/CPU.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    device = torch.device(args.device)
    print(f"[trm] device={device} checkpoint={args.checkpoint.name}")

    cfg = _read_yaml(args.config)
    metadata = _read_metadata(args.data)
    print(f"[trm] dataset: seq_len={metadata.seq_len} vocab={metadata.vocab_size} "
          f"puzzle_ids={metadata.num_puzzle_identifiers}")

    model = _build_model(cfg["arch"], metadata, device)
    _load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    result = _run_inference(
        model=model,
        data_path=args.data,
        eval_metadata=metadata,
        device=device,
        batch_size=args.batch_size,
        submission_K=args.submission_K,
        limit_batches=args.limit_batches,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "submission.json", "w") as f:
        json.dump(result["submission"], f)
    summary = {
        "num_tasks": result["num_tasks"],
        "pass_at_k": result["pass_at_k"],
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "batch_size": args.batch_size,
        "submission_K": args.submission_K,
    }
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TRM inference complete ===")
    for k, v in result["pass_at_k"].items():
        print(f"  {k:>10s}: {v:.4f}")
    print(f"  submission: {args.out / 'submission.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
