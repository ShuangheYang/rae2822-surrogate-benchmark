"""Predict CL, CD and CL/CD from 20 CST coefficients, with no CFD solver."""

from pathlib import Path
import argparse
import json
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from common import ROOT, PROJECT_ROOT, device_name, read_json, with_ratio, model_path
from mlp_engine import reload_model, predict
from kriging_engine import ExactIsotropicKriging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["mlp", "kriging"], default="mlp")
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional freshly trained NPZ; relative paths use --run-dir",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", type=Path, help='JSON: 20 numbers, N-by-20 numbers, or {"cst": ...}'
    )
    source.add_argument("--cst", nargs=20, type=float)
    args = parser.parse_args()
    payload = read_json(args.input) if args.input else args.cst
    x = np.asarray(
        payload["cst"] if isinstance(payload, dict) else payload, dtype=np.float64
    )
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 20 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Input must be finite N-by-20 CST in U0..U9,L0..L9 order.")
    bounds = read_json(PROJECT_ROOT / "configs/cst_bounds.json")
    outside = (
        (x < np.array(bounds["lower"]) - 1e-12)
        | (x > np.array(bounds["upper"]) + 1e-12)
    ).any(1)
    if outside.any():
        warnings.warn(
            "Inputs outside the studied design domain; predictions are extrapolations."
        )
    path = (
        model_path(args.model)
        if args.model
        else ROOT / "pretrained" / args.family / "model.npz"
    )
    if not path.exists():
        parser.error("Model not found. Run the models command or specify --model.")
    if args.family == "mlp":
        net, meta = reload_model(path, device_name())
        y = predict(net, x, meta)
    else:
        y = ExactIsotropicKriging.load(path, device=device_name()).predict(x)
    print(
        json.dumps(
            {"columns": ["CL", "CD", "L_D"], "prediction": with_ratio(y).tolist()},
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
