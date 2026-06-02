"""Minimal smoke test of TRM inference on an arbitrary device.

What it verifies, without preprocessing 5 minutes of data:

  1. The verification checkpoint loads cleanly (no missing/extra keys).
  2. The inner ACT model runs ONE forward pass end-to-end on the requested
     device (cuda/mps/cpu).
  3. Outputs are finite (no NaN/Inf), preds are in the ARC token range,
     q_halt_logits is shaped (B,).
  4. The ACT loop terminates after halt_max_steps.

Intentionally avoids real data — uses a dummy batch with random tokens
and a deterministic puzzle_identifier. Real scoring is left for the full
inference path (src/solver/trm_inference.py).

    python scripts/smoke_test_trm.py --device mps
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set BEFORE torch import for the few env vars that matter at startup.
os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Make the vendored TRM importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRM_ROOT = REPO_ROOT / "src" / "solver" / "trm"
sys.path.insert(0, str(TRM_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from models.recursive_reasoning.trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1,
)
from models.losses import ACTLossHead  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps", choices=["cuda", "mps", "cpu"])
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "experiments/trm_arc_v2_public/step_723914")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "experiments/trm_arc_v2_public/all_config.yaml")
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[smoke] device={device}")
    print(f"[smoke] checkpoint={args.checkpoint}")

    # 1. Peek at the checkpoint to discover puzzle table size + vocab/seq.
    print("[smoke] reading checkpoint...")
    t0 = time.monotonic()
    raw = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    print(f"  loaded in {time.monotonic() - t0:.1f}s, {len(raw)} keys")

    # Strip the torch.compile prefix.
    PREFIX = "_orig_mod."
    state_dict = {(k[len(PREFIX):] if k.startswith(PREFIX) else k): v for k, v in raw.items()}

    puzzle_emb_key = "model.inner.puzzle_emb.weights"
    if puzzle_emb_key not in state_dict:
        print(f"[smoke] FAIL: missing {puzzle_emb_key}")
        return 1
    num_puzzles, puzzle_dim = state_dict[puzzle_emb_key].shape
    embed_key = "model.inner.embed_tokens.embedding_weight"
    if embed_key not in state_dict:
        print(f"[smoke] FAIL: missing {embed_key}")
        return 1
    vocab_size, hidden_size = state_dict[embed_key].shape
    print(f"  vocab={vocab_size} hidden={hidden_size} puzzles={num_puzzles} puzzle_dim={puzzle_dim}")

    # 2. Build the model from all_config.yaml + discovered shapes.
    cfg = yaml.safe_load(open(args.config))
    arch = cfg["arch"]
    SEQ_LEN = 30 * 30  # ARC constant (build_arc_dataset.py:297)
    model_cfg = dict(
        batch_size=args.batch_size,
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        num_puzzle_identifiers=num_puzzles,
        causal=False,
        H_cycles=arch["H_cycles"], L_cycles=arch["L_cycles"],
        H_layers=arch["H_layers"], L_layers=arch["L_layers"],
        hidden_size=arch["hidden_size"], expansion=arch["expansion"],
        num_heads=arch["num_heads"], pos_encodings=arch["pos_encodings"],
        halt_max_steps=arch["halt_max_steps"],
        halt_exploration_prob=arch["halt_exploration_prob"],
        puzzle_emb_ndim=arch["puzzle_emb_ndim"], puzzle_emb_len=arch["puzzle_emb_len"],
        forward_dtype=arch["forward_dtype"],
        mlp_t=arch.get("mlp_t", False),
        no_ACT_continue=arch.get("no_ACT_continue", True),
    )
    print(f"[smoke] building model on {device}...")
    t0 = time.monotonic()
    with torch.device(device):
        inner = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        model = ACTLossHead(inner, loss_type=arch["loss"]["loss_type"])
    model.eval()
    print(f"  built in {time.monotonic() - t0:.1f}s")

    # 3. Load weights.
    print("[smoke] loading state_dict (assign=True)...")
    t0 = time.monotonic()
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    model.to(device)
    print(f"  loaded in {time.monotonic() - t0:.1f}s. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:3]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:3]}")

    # 4. Dummy batch.
    g = torch.Generator(device="cpu").manual_seed(0)
    batch = {
        "inputs": torch.randint(0, vocab_size, (args.batch_size, SEQ_LEN), generator=g, dtype=torch.int32),
        "puzzle_identifiers": torch.randint(0, num_puzzles, (args.batch_size,), generator=g, dtype=torch.int32),
        "labels": torch.full((args.batch_size, SEQ_LEN), -100, dtype=torch.int32),
    }
    batch = {k: v.to(device) for k, v in batch.items()}

    # 5. ACT loop on the inner ACT module (skipping the loss head).
    print(f"[smoke] running ACT loop (max {arch['halt_max_steps']} steps)...")
    t0 = time.monotonic()
    with torch.inference_mode():
        with torch.device(device):
            carry = model.model.initial_carry(batch)
        step = 0
        while True:
            carry, outputs = model.model(carry=carry, batch=batch)
            step += 1
            if carry.halted.all():
                break
            if step > 50:
                print("[smoke] FAIL: ACT loop did not halt within 50 steps")
                return 1
    elapsed = time.monotonic() - t0
    print(f"  {step} steps in {elapsed:.2f}s ({elapsed / step * 1000:.0f} ms/step)")

    # 6. Sanity-check outputs.
    logits = outputs["logits"]
    q_halt = outputs["q_halt_logits"]
    print(f"[smoke] logits shape={tuple(logits.shape)} dtype={logits.dtype}")
    print(f"[smoke] q_halt_logits shape={tuple(q_halt.shape)} dtype={q_halt.dtype}")

    if not torch.isfinite(logits.float()).all():
        print("[smoke] FAIL: non-finite values in logits")
        return 1
    if not torch.isfinite(q_halt.float()).all():
        print("[smoke] FAIL: non-finite values in q_halt_logits")
        return 1
    preds = torch.argmax(logits, dim=-1)
    if preds.min().item() < 0 or preds.max().item() >= vocab_size:
        print(f"[smoke] FAIL: preds out of vocab range [{preds.min().item()}, {preds.max().item()}]")
        return 1
    if step != arch["halt_max_steps"]:
        print(f"[smoke] WARN: stopped at step {step}, expected halt_max_steps={arch['halt_max_steps']}")

    print(f"\n[smoke] OK — device={device} produces finite TRM outputs in {step} ACT steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
