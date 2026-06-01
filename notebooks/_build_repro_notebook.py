"""One-shot generator for notebooks/01_trm_repro_kaggle.ipynb.

The Kaggle notebook is short but tedious to keep clean as raw JSON. Author
cells in this file as Python literals and run this script to (re)generate the
.ipynb. Diffable, no surprises.

    python notebooks/_build_repro_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# TRM ARC-AGI-2 reproduction (Kaggle GPU)

Loads the ARC Prize Foundation's MIT-licensed TRM verification checkpoint and
runs inference on the ARC-AGI-2 public eval set. This is **G1**: confirm that
our infra reproduces the paper's reported ~8% public-eval score before we
change anything about the method.

**Runs on:** Kaggle exploration notebook, GPU (T4 or P100), internet ON.

**Steps:**
1. env check
2. clone repo
3. install missing pip deps (most preinstalled on Kaggle)
4. fetch the 2.47 GB TRM checkpoint from Hugging Face
5. preprocess ARC-AGI-2 into TRM's `arc2concept-aug-1000` format
6. run inference (~1–3 h)
7. score with our local 2-attempt metric to cross-check""",
    ),
    (
        "markdown",
        "## 1. Env check",
    ),
    (
        "code",
        """import torch, sys, platform
print('python    :', platform.python_version())
print('torch     :', torch.__version__)
print('cuda avail:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu       :', torch.cuda.get_device_name(0))
    print('vram GB   :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
!df -h /kaggle/working | tail -1
!free -g | head -2""",
    ),
    (
        "markdown",
        """## 2. Clone the repo

Replace `REPO_URL` with our repo's URL once it's pushed to GitHub (prize rules
require open source anyway). For now you can also upload the repo as a Kaggle
Dataset and copy it in from `/kaggle/input/<dataset-name>/`.""",
    ),
    (
        "code",
        """REPO_URL = 'https://github.com/zhanggangyi1224/ARC-AGI-2.git'
WORK = '/kaggle/working'
import os, subprocess
os.chdir(WORK)
if not os.path.isdir(f'{WORK}/repo'):
    subprocess.check_call(['git', 'clone', '--depth=1', REPO_URL, 'repo'])
os.chdir(f'{WORK}/repo')
!pwd && git log -1 --oneline""",
    ),
    (
        "markdown",
        """## 3. Install pip deps not already on Kaggle

We only need *inference* deps. Skip `adam-atan2` (CUDA-built optimizer; only
needed for training), `wandb`, `hydra-core` (we don't use the hydra entry
point), and `triton` (preinstalled with torch on Kaggle).""",
    ),
    (
        "code",
        """!pip install -q einops 'pydantic>=2' pyyaml numba""",
    ),
    (
        "markdown",
        "## 4. Fetch the verification checkpoint (~2.47 GB)",
    ),
    (
        "code",
        """!bash scripts/fetch_trm_checkpoint.sh""",
    ),
    (
        "markdown",
        """## 5. Preprocess ARC-AGI-2 into TRM's expected format

This builds `data/arc2concept-aug-1000/{train,test}/` with the **same**
identifier_map the verification checkpoint was trained against. We call the
preprocessor directly (no CLI) so we don't need `argdantic`.""",
    ),
    (
        "code",
        """import sys
TRM_ROOT = '/kaggle/working/repo/src/solver/trm'
if TRM_ROOT not in sys.path:
    sys.path.insert(0, TRM_ROOT)
import os
os.chdir(TRM_ROOT)  # build_arc_dataset reads kaggle/combined/* relative to cwd
from dataset.build_arc_dataset import DataProcessConfig, convert_dataset
cfg = DataProcessConfig(
    input_file_prefix='kaggle/combined/arc-agi',
    output_dir='data/arc2concept-aug-1000',
    subsets=['training2', 'evaluation2', 'concept'],
    test_set_name='evaluation2',
    num_aug=1000,
    seed=42,
)
convert_dataset(cfg)
os.chdir('/kaggle/working/repo')
!ls src/solver/trm/data/arc2concept-aug-1000/test/ | head -5""",
    ),
    (
        "markdown",
        """## 6. Run TRM inference

Runs ACT-halted inference over all augmented test rows, votes content-hashed
predictions, writes `submission.json` (Kaggle format) and `summary.json`
(pass@K table).""",
    ),
    (
        "code",
        """RUN_OUT = '/kaggle/working/runs/trm_repro_kaggle'
!python -m src.solver.trm_inference \\
    --checkpoint experiments/trm_arc_v2_public/step_723914 \\
    --config     experiments/trm_arc_v2_public/all_config.yaml \\
    --data       src/solver/trm/data/arc2concept-aug-1000 \\
    --device     cuda \\
    --batch-size 768 \\
    --out        {RUN_OUT}""",
    ),
    (
        "markdown",
        "## 7. Results",
    ),
    (
        "code",
        """import json
RUN_OUT = '/kaggle/working/runs/trm_repro_kaggle'
summary = json.load(open(f'{RUN_OUT}/summary.json'))
print(json.dumps(summary, indent=2))""",
    ),
    (
        "markdown",
        """## 8. (Optional) re-score with our own harness

Sanity check: our local-eval harness should agree with TRM's evaluator on the
overall pass@2 number to within numerical precision. The harness reads the
submission.json directly, so it doesn't need a GPU.""",
    ),
    (
        "code",
        """!python -m src.solver.from_submission \\
    --submission /kaggle/working/runs/trm_repro_kaggle/submission.json \\
    --split public_eval""",
    ),
]


def build() -> dict:
    nb_cells = []
    for cell_type, source in CELLS:
        lines = source.split("\n")
        # nbformat wants source as a list of strings, each ending with \n except possibly the last
        src_lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": cell_type, "metadata": {}, "source": src_lines}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "01_trm_repro_kaggle.ipynb"
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {out}")
