"""Verify released representative model predictions against all 1,500 saved test predictions; no fitting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from common import ROOT, device_name, write_json
from mlp_engine import reload_model, predict
from kriging_engine import ExactIsotropicKriging


def main():
    with np.load(ROOT / "results/main_predictions.npz", allow_pickle=False) as d:
        x = d["X"].copy()
        truth = {k: d[k].copy() for k in ["MLP", "Kriging"]}
    mlp, meta = reload_model(ROOT / "pretrained/mlp/model.npz", device_name())
    gp = ExactIsotropicKriging.load(
        ROOT / "pretrained/kriging/model.npz", device=device_name()
    )
    actual = {"MLP": predict(mlp, x, meta), "Kriging": gp.predict(x)}
    evidence = {}
    for name, value in actual.items():
        np.testing.assert_allclose(value, truth[name][:, :2], rtol=1e-8, atol=1e-10)
        evidence[name] = {
            "n": len(x),
            "max_absolute_difference": float(np.abs(value - truth[name][:, :2]).max()),
        }
    write_json(
        ROOT / "pretrained_verification.json",
        {"device": device_name(), "results": evidence},
    )
    print(evidence)


if __name__ == "__main__":
    main()
