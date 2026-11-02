"""Evaluate frozen, verified models on the designated test partition and retain traceable evidence.
Do not use test metrics to choose models or omit valid difficult cases. E2 uses the same
test normalization for all models. Record model reuse and uncertainty sources explicitly.
"""

import gc
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from common import (
    ROOT,
    device_name,
    model_path as resolve_model_path,
    read_json,
    write_json,
    file_hash,
    metrics,
    with_ratio,
)
from mlp_engine import reload_model, predict
from kriging_engine import ExactIsotropicKriging, _squared_distance, _kernel_block

DEVICE = device_name()


class ResidentKrigingMean:
    """Cache training tensors for fair timing; equations, kernels, and parameters remain unchanged."""

    def __init__(self, gp):
        self.gp = gp
        self.x = torch.as_tensor(gp.x_train, dtype=torch.float64, device=DEVICE)
        self.alpha = [
            torch.as_tensor(m["alpha"], dtype=torch.float64, device=DEVICE)
            for m in gp.models
        ]

    @torch.no_grad()
    def __call__(self, x):
        gp = self.gp
        query = torch.as_tensor(
            (x - gp.scalers["x_mean"]) / gp.scalers["x_scale"],
            dtype=torch.float64,
            device=DEVICE,
        )
        cols = []
        for j, m in enumerate(gp.models):
            pieces = []
            for start in range(0, len(x), gp.config.prediction_batch_size):
                k = _kernel_block(
                    _squared_distance(
                        query[start : start + gp.config.prediction_batch_size], self.x
                    ),
                    m["signal"],
                    m["length_scale"],
                    gp.config.kernel,
                )
                pieces.append((k @ self.alpha[j]).cpu().numpy())
            cols.append(
                np.concatenate(pieces) * gp.scalers["y_scale"][j]
                + gp.scalers["y_mean"][j]
            )
        return np.column_stack(cols)


def bootstrap_comparison(y, p, q, repeats=2000):
    """Paired test-sample bootstrap; excludes training randomness and CFD label error."""
    rng = np.random.default_rng(20261013)
    y, p, q = map(with_ratio, (y, p, q))
    ratios = []
    differences = []
    scale = y[:, :2].std(0)
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        a = np.sqrt(np.mean((p[idx] - y[idx]) ** 2, axis=0))
        b = np.sqrt(np.mean((q[idx] - y[idx]) ** 2, axis=0))
        ratios.append(100 * (1 - a / b))
        differences.append(50 * np.sum((a[:2] - b[:2]) / scale))
    return {
        "replicates": repeats,
        "resampling_unit": "paired test airfoil; training fixed",
        "MLP_RMSE_reduction_percent_CI95": np.quantile(
            ratios, [0.025, 0.975], axis=0
        ).T.tolist(),
        "E2_MLP_minus_Kriging_percentage_points_CI95": np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
    }


def inference_time(fn, x, device, repeats=15):
    # Same device, inputs, and batch; time inference with models already loaded.
    for _ in range(3):
        fn(x)
    values = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return {
        "n_predictions": len(x),
        "repeats": repeats,
        "median_seconds": float(np.median(values)),
        "minimum_seconds": min(values),
        "microseconds_per_sample": float(1e6 * np.median(values) / len(x)),
        "q25_seconds": float(np.quantile(values, 0.25)),
        "q75_seconds": float(np.quantile(values, 0.75)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "device": device,
        "includes_model_loading": False,
        "includes_input_scaling_and_output_copy": True,
    }


def main():
    assert (
        ROOT / "results/mlp_learning_complete.json"
    ).exists(), "MLP runs are incomplete"
    assert (
        ROOT / "results/kriging_learning_complete.json"
    ).exists(), "Kriging runs are incomplete"
    protocol = read_json(ROOT / "protocol.json")
    records = []
    for family, folder in [("MLP", "learning"), ("Kriging", "kriging")]:
        rr = [
            read_json(p)
            for p in sorted((ROOT / "models" / folder).glob("*/result.json"))
        ]
        assert len(rr) == len(protocol["budgets"]) * len(protocol["seeds"]), (
            family,
            len(rr),
        )
        assert {(r["n_train"], r["seed"]) for r in rr} == {
            (n, s) for n in protocol["budgets"] for s in protocol["seeds"]
        }
        for r in rr:
            assert file_hash(resolve_model_path(r["model_path"])) == r["model_sha256"]
            records.append(dict(r, family=family))
    freeze = {
        "time": datetime.now().astimezone().isoformat(),
        "protocol_sha256": file_hash(ROOT / "protocol.json"),
        "models": [
            {
                "family": r["family"],
                "n_train": r["n_train"],
                "seed": r["seed"],
                "model_path": r["model_path"],
                "sha256": r["model_sha256"],
            }
            for r in records
        ],
        "test_opened_after_model_list_frozen": True,
    }
    freeze_path = ROOT / "results/final_model_freeze.json"
    if freeze_path.exists():
        assert (
            read_json(freeze_path)["models"] == freeze["models"]
        ), "Do not replace frozen models and repeat testing"
    else:
        write_json(freeze_path, freeze)
    data = np.load(ROOT / "data/sealed_test.npz")
    X, Y, ids = data["X"], data["Y"], data["ids"]
    train = np.load(ROOT / "data/development.npz")
    result = []
    predictions = {}
    cache = {}
    loaded_main = {}
    for r in records:
        key = f"{r['family']}_n{r['n_train']:05d}_s{r['seed']}"
        sha = r["model_sha256"]
        if sha in cache:
            p = cache[sha].copy()
        else:
            if r["family"] == "MLP":
                model, meta = reload_model(r["model_path"], device=DEVICE)
                p = predict(model, X, meta)
            else:
                model = ExactIsotropicKriging.load(
                    resolve_model_path(r["model_path"]), device=DEVICE
                )
                p = model.predict(X)
            cache[sha] = p.copy()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        predictions[key] = p
        result.append(
            {
                "family": r["family"],
                "n_train": r["n_train"],
                "seed": r["seed"],
                "model_path": r["model_path"],
                "model_sha256": sha,
                "fit_seconds": r["fit_seconds"],
                "test_metrics": metrics(Y, p),
                "validation_metrics": r["validation_metrics"],
                "validation_score": r.get("validation_score"),
                "best_step": r.get("best_step"),
                "parameter_count": r.get("parameter_count"),
                "reused_deterministic_fit": r.get(
                    "reused_deterministic_fit", r.get("reused_fit", False)
                ),
            }
        )
        print(key, result[-1]["test_metrics"]["E2_percent"], flush=True)
    np.savez_compressed(
        ROOT / "results/test_predictions.npz", ids=ids, truth=Y, X=X, **predictions
    )
    p = predictions["MLP_n10000_s11"]
    q = predictions["Kriging_n10000_s11"]
    ensemble = np.mean(
        [predictions[f"MLP_n10000_s{s}"] for s in protocol["seeds"]], axis=0
    )
    np.savez_compressed(
        ROOT / "results/main_predictions.npz",
        ids=ids,
        truth=Y,
        MLP=p,
        Kriging=q,
        MLP_ensemble=ensemble,
        X=X,
    )
    aggregates = []
    for family in ["MLP", "Kriging"]:
        for n in protocol["budgets"]:
            rr = [r for r in result if r["family"] == family and r["n_train"] == n]
            values = np.array([r["test_metrics"]["E2_percent"] for r in rr])
            agg = {
                "family": family,
                "n_train": n,
                "E2_mean_percent": float(values.mean()),
                "E2_std_percent": float(values.std(ddof=1)),
                "fit_seconds_mean": float(np.mean([r["fit_seconds"] for r in rr])),
            }
            for target in ["CL", "CD", "L_D"]:
                for metric in ["RMSE", "MAPE_percent", "P95_APE_percent"]:
                    v = np.array([r["test_metrics"][target][metric] for r in rr])
                    agg[f"{target}_{metric}_mean"] = float(v.mean())
                    agg[f"{target}_{metric}_std"] = float(v.std(ddof=1))
            aggregates.append(agg)
    crossing = []
    diff = []
    for n in protocol["budgets"]:
        a = next(r for r in aggregates if r["n_train"] == n and r["family"] == "MLP")
        b = next(
            r for r in aggregates if r["n_train"] == n and r["family"] == "Kriging"
        )
        diff.append(a["E2_mean_percent"] - b["E2_mean_percent"])
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            lo, hi = protocol["budgets"][i : i + 2]
            interpolated = np.exp(
                np.log(lo)
                + (np.log(hi) - np.log(lo)) * (-diff[i]) / (diff[i + 1] - diff[i])
            )
            crossing.append(
                {
                    "bracket": [lo, hi],
                    "log_n_linear_interpolation": float(interpolated),
                    "is_measured_exact_threshold": False,
                }
            )
    thresholds = []
    for family in ["MLP", "Kriging"]:
        for target in ["CL", "CD", "L_D"]:
            for threshold in [0.1, 0.25, 0.5, 1.0]:
                passed = [
                    r["n_train"]
                    for r in aggregates
                    if r["family"] == family
                    and r[f"{target}_MAPE_percent_mean"] <= threshold
                ]
                thresholds.append(
                    {
                        "family": family,
                        "target": target,
                        "MAPE_threshold_percent": threshold,
                        "minimum_observed_training_n": min(passed) if passed else None,
                        "extra_validation_labels": protocol["validation_n"],
                        "criterion": "mean across three recorded seeds; fixed full-budget-selected configuration",
                    }
                )
    # Error associations with geometry and label quality are observational, not causal.
    quality = np.load(ROOT / "data/test_geometry_quality.npz")
    lookup = {s: i for i, s in enumerate(quality["ids"])}
    ix = np.array([lookup[s] for s in ids])
    width = np.maximum(0.2 * abs(train["baseline_cst"]), 1e-12)
    distance = cdist(X / width, train["X"] / width).min(axis=1)
    features = {"nearest_training_distance": distance}
    for name in quality.files:
        if (
            name != "ids"
            and np.issubdtype(quality[name].dtype, np.number)
            and quality[name].ndim == 1
        ):
            features[name] = quality[name][ix]
    correlations = []
    for family, pp in [("MLP", p), ("Kriging", q)]:
        error = abs((with_ratio(pp) - with_ratio(Y)) / with_ratio(Y)) * 100
        for key, values in features.items():
            if np.ptp(values) == 0:
                continue
            for j, target in enumerate(["CL", "CD", "L_D"]):
                rho, pvalue = spearmanr(values, error[:, j])
                correlations.append(
                    {
                        "model": family,
                        "feature": key,
                        "target": target,
                        "spearman_rho": float(rho),
                        "p_value_unadjusted": float(pvalue),
                        "exploratory": True,
                    }
                )
    np.savez_compressed(ROOT / "results/test_geometry_quality.npz", ids=ids, **features)
    # Warm inference on the same GPU, excluding model loading and CFD solution.
    torch.set_num_threads(2)
    # Record whether another Python workload is using the GPU.
    try:
        if not DEVICE.startswith("cuda"):
            raise RuntimeError("CPU timing; GPU workload is not applicable")
        gpu_processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        other_python = [
            line.strip()
            for line in gpu_processes.splitlines()
            if "python" in line.lower()
            and line.split(",")[0].strip() != str(os.getpid())
        ]
        timing_context = {
            "dedicated_machine": False,
            "concurrent_gpu_python_workload": bool(other_python),
            "interpretation": "Warm inference latency on a shared workstation with the recorded GPU workload.",
        }
    except Exception as exc:
        timing_context = {
            "dedicated_machine": False,
            "load_snapshot_error": str(exc),
            "interpretation": "warm engineering latency; exclusive hardware not established",
        }
    main_m = next(
        r
        for r in result
        if r["family"] == "MLP" and r["n_train"] == 10000 and r["seed"] == 11
    )
    main_k = next(
        r
        for r in result
        if r["family"] == "Kriging" and r["n_train"] == 10000 and r["seed"] == 11
    )
    model, meta = reload_model(main_m["model_path"], device=DEVICE)
    timing = {
        "MLP_batch": inference_time(lambda x: predict(model, x, meta), X, DEVICE),
        "MLP_single": inference_time(lambda x: predict(model, x, meta), X[:1], DEVICE),
    }
    assert np.allclose(
        predict(model, X, meta), p, rtol=1e-10, atol=1e-12
    ), "Predictions differ after save/load"
    # Compute learned directional sensitivity on training inputs, not physical CFD gradients.
    xx = torch.tensor(
        (train["X"][:500] - np.array(meta["x_mean"])) / np.array(meta["x_scale"]),
        dtype=next(model.parameters()).dtype,
        device=DEVICE,
        requires_grad=True,
    )
    zz = model(xx)
    jac = []
    for j in range(2):
        val = zz[:, j]
        if j == 1 and meta["config"].get("log_cd"):
            val = (val * meta["y_scale"][j] + meta["y_mean"][j]).exp() / train["Y"][
                :, j
            ].std()
        jac.append(
            torch.autograd.grad(val.sum(), xx, retain_graph=True)[0]
            .detach()
            .cpu()
            .numpy()
        )
    G = np.concatenate(jac, axis=0)
    eig = np.linalg.eigvalsh(G.T @ G / len(G))[::-1]
    sensitivity = {
        "definition": "gradient of standardized model output with respect to standardized CST; first500 training points",
        "eigenvalue_fraction": (eig / eig.sum()).tolist(),
        "axis_energy_fraction": (
            np.mean(G**2, axis=0) / np.mean(G**2, axis=0).sum()
        ).tolist(),
        "CFD_derivative_validation": False,
    }
    # Compute the discrete CST geometry metric directly without CFD gradients or extra fitting.
    gx = (1 - np.cos(np.linspace(0, np.pi, 400))) / 2
    basis = np.array(
        [math.comb(9, i) * gx**i * (1 - gx) ** (9 - i) for i in range(10)]
    ).T
    basis *= np.sqrt(gx[:, None]) * (1 - gx[:, None])
    cst_matrix = np.zeros((800, 20))
    cst_matrix[:400, :10] = basis
    cst_matrix[400:, 10:] = basis
    cst_matrix *= np.array(meta["x_scale"])[None, :]
    geig = np.linalg.eigvalsh(cst_matrix.T @ cst_matrix)[::-1]
    sensitivity["geometry_metric_eigenvalue_fraction"] = (geig / geig.sum()).tolist()
    sensitivity["geometry_sampling"] = (
        "400 cosine-spaced points per surface; discrete unweighted ordinate norm"
    )
    del model, xx, zz
    gc.collect()
    torch.cuda.empty_cache()
    gp = ExactIsotropicKriging.load(
        resolve_model_path(main_k["model_path"]), device=DEVICE
    )
    resident = ResidentKrigingMean(gp)
    assert np.allclose(
        resident(X), q, rtol=1e-10, atol=1e-12
    ), "Resident-tensor timing wrapper differs from original API predictions"
    timing.update(
        Kriging_batch=inference_time(resident, X, DEVICE),
        Kriging_single=inference_time(resident, X[:1], DEVICE),
        Kriging_API_batch=inference_time(gp.predict, X, DEVICE),
        Kriging_API_single=inference_time(gp.predict, X[:1], DEVICE),
    )
    for key, value in timing.items():
        value["training_reference_tensors_resident"] = not key.startswith("Kriging_API")
    summary = {
        "status": "final",
        "protocol": protocol,
        "main_seed": 11,
        "records": result,
        "aggregates": aggregates,
        "main": {
            "MLP": metrics(Y, p),
            "Kriging": metrics(Y, q),
            "MLP_ensemble_supplement": metrics(Y, ensemble),
        },
        "bootstrap": bootstrap_comparison(Y, p, q),
        "crossings": crossing,
        "correlations": correlations,
        "data_efficiency_thresholds": thresholds,
        "timing": timing,
        "learned_sensitivity": sensitivity,
        "timing_context": timing_context,
        "prediction_sha256": file_hash(ROOT / "results/test_predictions.npz"),
        "freeze_sha256": file_hash(freeze_path),
        "test_ids": ids.tolist(),
        "cfd_seconds_median_test": float(np.median(features["cfd_seconds"])),
        "test_evaluated_at": datetime.now().astimezone().isoformat(),
    }
    write_json(ROOT / "results/evaluation.json", summary)
    print("Frozen evaluation complete; do not tune against the test set.", flush=True)


if __name__ == "__main__":
    main()
