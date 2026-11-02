"""Command-line entry point for isolated, reproducible surrogate experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/default")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "reference",
            "models",
            "plan",
            "mlp-select",
            "mlp-curves",
            "kriging",
            "evaluate",
            "figures",
            "predict",
            "export",
            "smoke",
            "verify",
            "explorer",
            "browser-models",
        ],
    )
    args, remaining = parser.parse_known_args()
    root = args.run_dir.resolve()
    os.environ["AIRFOIL_RUN_DIR"] = str(root)
    os.environ["AIRFOIL_DEVICE"] = args.device
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    sys.path.insert(0, str(PROJECT / "src"))
    if args.command in ("mlp-select", "mlp-curves", "kriging", "evaluate"):
        if (root / "REFERENCE_ONLY").exists():
            parser.error(
                "This directory contains published reference evidence. Select a fresh --run-dir for training/evaluation."
            )
        if not (root / "data/development.npz").is_file():
            parser.error("Run prepare first in this --run-dir.")
    table = {
        "prepare": ("download_assets.py", ["--group", "inputs"]),
        "reference": ("download_assets.py", ["--group", "reference"]),
        "models": ("download_assets.py", ["--group", "models"]),
        "plan": ("run_kriging_study.py", ["plan", "--device", args.device]),
        "mlp-select": ("run_mlp_study.py", ["select"]),
        "mlp-curves": ("run_mlp_study.py", ["curves"]),
        "kriging": ("run_kriging_study.py", ["curves", "--device", args.device]),
        "evaluate": ("evaluate_study.py", []),
        "figures": ("make_figures.py", []),
        "predict": ("predict.py", []),
        "export": ("export_results.py", []),
        "smoke": ("tests/smoke.py", []),
        "verify": ("verify_pretrained.py", []),
        "explorer": ("build_explorer.py", []),
        "browser-models": ("export_browser_models.py", []),
    }
    script, defaults = table[args.command]
    sys.argv = [str(PROJECT / script), *defaults, *remaining]
    runpy.run_path(str(PROJECT / script), run_name="__main__")


if __name__ == "__main__":
    main()
