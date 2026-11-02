"""Synthetic verification without reading CFD data or final test labels.
Run python src/test_kriging_engine.py in an environment with PyTorch and scikit-learn.
Check mean, noisy variance, and likelihood against scikit-learn; analytic gradients
against centered differences; CPU/GPU agreement; and save/load prediction identity."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

from kriging_engine import (
    ExactIsotropicKriging,
    KrigingConfig,
    _TorchObjective,
    _squared_distance,
)


def run():
    rng = np.random.default_rng(20261013)
    x = rng.normal(size=(80, 20))
    y = np.column_stack(
        (
            0.8 + 0.1 * np.sin(x[:, 0]) + 0.01 * x[:, 1],
            0.02 + 0.003 * np.cos(x[:, 2]) + 0.0001 * x[:, 3],
        )
    )
    query = rng.normal(size=(11, 20))
    hyper = [
        {"signal": 2.3, "length_scale": 3.4, "noise": 0.007},
        {"signal": 1.7, "length_scale": 4.1, "noise": 0.005},
    ]
    results = []
    for kernel in ("matern52", "rbf"):
        model = ExactIsotropicKriging(
            {"kernel": kernel, "device": "cpu", "optimize": False, "verbose": False}
        )
        model.fit(x, y, initial_hyperparameters=hyper)
        pred, std = model.predict(query, return_std=True)
        s = model.scalers
        xs = (x - s["x_mean"]) / s["x_scale"]
        qs = (query - s["x_mean"]) / s["x_scale"]
        for j in range(2):
            p = hyper[j]
            spatial = (
                Matern(p["length_scale"], nu=2.5)
                if kernel == "matern52"
                else RBF(p["length_scale"])
            )
            reference = GaussianProcessRegressor(
                kernel=ConstantKernel(p["signal"]) * spatial + WhiteKernel(p["noise"]),
                optimizer=None,
                alpha=model.config.jitter,
                normalize_y=False,
            )
            reference.fit(xs, (y[:, j] - s["y_mean"][j]) / s["y_scale"][j])
            rp, rs = reference.predict(qs, return_std=True)
            rp, rs = rp * s["y_scale"][j] + s["y_mean"][j], rs * s["y_scale"][j]
            errors = {
                "kernel": kernel,
                "output": j,
                "max_mean_error": float(np.max(abs(pred[:, j] - rp))),
                "max_std_error": float(np.max(abs(std[:, j] - rs))),
                "nll_error": float(
                    abs(
                        model.models[j]["negative_log_likelihood"]
                        + reference.log_marginal_likelihood()
                    )
                ),
            }
            assert errors["max_mean_error"] < 1e-10
            assert errors["max_std_error"] < 1e-10
            assert errors["nll_error"] < 1e-8
            results.append(errors)
        xt = torch.tensor(xs, dtype=torch.float64)
        yt = torch.tensor(
            (y[:, 0] - s["y_mean"][0]) / s["y_scale"][0], dtype=torch.float64
        )
        objective = _TorchObjective(_squared_distance(xt, xt), yt, model.config)
        theta = np.log(
            [hyper[0]["signal"], hyper[0]["length_scale"], hyper[0]["noise"]]
        )
        _, analytical = objective(theta)
        numeric = []
        for j in range(3):
            step = np.zeros(3)
            step[j] = 1e-5
            numeric.append(
                (objective(theta + step)[0] - objective(theta - step)[0]) / 2e-5
            )
        np.testing.assert_allclose(analytical, numeric, rtol=1e-5, atol=1e-6)
        results.append(
            {
                "kernel": kernel,
                "gradient_max_abs_error": float(np.max(abs(analytical - numeric))),
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            model.save(path)
            loaded = ExactIsotropicKriging.load(path, device="cpu")
            np.testing.assert_allclose(
                model.predict(query), loaded.predict(query), rtol=1e-13, atol=1e-13
            )
        if torch.cuda.is_available():
            gpu = ExactIsotropicKriging(
                {
                    "kernel": kernel,
                    "device": "cuda:0",
                    "optimize": False,
                    "verbose": False,
                }
            )
            gpu.fit(x, y, initial_hyperparameters=hyper)
            gp, gs = gpu.predict(query, return_std=True)
            np.testing.assert_allclose(gp, pred, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(gs, std, rtol=1e-10, atol=1e-12)
            results.append(
                {
                    "kernel": kernel,
                    "gpu_cpu_mean_max_error": float(np.max(abs(gp - pred))),
                }
            )
    # Optimization smoke test: final likelihood must improve on initialization; record budget and termination.
    trained = ExactIsotropicKriging(
        {
            "device": "cpu",
            "verbose": False,
            "n_restarts": 1,
            "max_iterations": 12,
            "max_function_evaluations": 25,
        }
    )
    trained.fit(x, y)
    for target in trained.models:
        assert (
            target["negative_log_likelihood"]
            <= target["evaluations"][0]["negative_log_likelihood"] + 1e-7
        )
        assert len(target["optimizer_records"]) == 2
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": results,
                "optimization": [
                    {
                        "target": m["target"],
                        "selected_restart_success": m["selected_restart_success"],
                        "nll": m["negative_log_likelihood"],
                    }
                    for m in trained.models
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
