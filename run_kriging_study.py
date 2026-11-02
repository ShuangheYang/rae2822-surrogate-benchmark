"""Resumable exact isotropic Kriging learning curves under a frozen protocol.
Commands: python run_kriging_study.py plan; smoke; curves --device cuda:0.
Reads development.npz, split.json, and protocol.json, excluding final test labels.
Each budget uses all designated training rows. The 500 validation labels are evaluated
but do not control kernels, starts, bounds, or termination. Exact dense inference is
retained; kernel parameters are fitted by training marginal likelihood only."""

from __future__ import annotations

import os

# Use sequential MKL in this process to avoid MKL/libomp conflicts.
# This affects only this process, not installed environments or system settings.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import torch

from common import (
    ROOT,
    PROJECT_ROOT,
    model_path as resolve_model_path,
    file_hash,
    metrics,
    read_json,
    with_ratio,
    write_json,
)
from kriging_engine import ExactIsotropicKriging, KrigingConfig

MODEL_ID = "Kriging"
DEFAULT_CONFIG = {
    "kernel": "matern52",
    "device": "cuda:0",
    "cpu_threads": 2,
    "max_iterations": 100,
    "max_function_evaluations": 180,
    "n_restarts": 1,
    "jitter": 1e-10,
    "max_jitter": 1e-4,
    "length_scale_bounds": (0.1, 100.0),
    "signal_bounds": (1e-3, 1e3),
    "noise_bounds": (1e-8, 0.1),
    "verbose": True,
}
# Allow one tenfold expansion for an outward likelihood gradient, within the fixed hard limits below.
OUTER_BOUNDS = {
    "signal_bounds": (1e-5, 1e5),
    "length_scale_bounds": (0.01, 1000.0),
    "noise_bounds": (1e-12, 1.0),
}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def object_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_stop():
    for name in ("STOP_KRIGING", "STOP_RESEARCH"):
        if (ROOT / name).exists():
            raise InterruptedError(
                f"Detected {name}; stopping current fit while retaining completed results"
            )


def progress(_record):
    check_stop()


def load_development():
    """Read development copies only and verify all IDs against the frozen split."""
    protocol = read_json(ROOT / "protocol.json")
    split = read_json(ROOT / "data/split.json")
    with np.load(ROOT / "data/development.npz", allow_pickle=False) as archive:
        arrays = {
            key: archive[key].copy()
            for key in ("X", "Y", "ids", "X_val", "Y_val", "ids_val")
        }
    if arrays["X"].shape != (protocol["train_n"], 20) or arrays["Y"].shape != (
        protocol["train_n"],
        2,
    ):
        raise ValueError("Training-copy shape differs from the frozen protocol")
    if arrays["X_val"].shape != (protocol["validation_n"], 20):
        raise ValueError("Validation-copy shape differs from the frozen protocol")
    if (
        arrays["ids"].tolist() != split["train_ids"]
        or arrays["ids_val"].tolist() != split["validation_ids"]
    ):
        raise ValueError(
            "Training or validation ID ordering differs from the frozen split"
        )
    for seed in protocol["seeds"]:
        order = np.asarray(split["nested_training_orders"][str(seed)], dtype=int)
        if not np.array_equal(np.sort(order), np.arange(protocol["train_n"])):
            raise ValueError(
                f"seed={seed} nested ordering is not a permutation of all training rows"
            )
    for key in ("X", "Y", "X_val", "Y_val"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains NaN/Inf")
    return protocol, split, arrays


def make_frozen_spec(device):
    config = asdict(KrigingConfig(**{**DEFAULT_CONFIG, "device": device}))
    config.pop("device")
    return {
        "schema_version": 1,
        "model": MODEL_ID,
        "study_protocol_sha256": file_hash(ROOT / "protocol.json"),
        "split_sha256": file_hash(ROOT / "data/split.json"),
        "development_sha256": file_hash(ROOT / "data/development.npz"),
        "engine_sha256": file_hash(PROJECT_ROOT / "src/kriging_engine.py"),
        "driver_sha256": file_hash(__file__),
        "base_config": config,
        "device_policy": "CPU/GPU may differ across runs; actual device recorded per fit. This is not a common-hardware training-time benchmark.",
        "hyperparameters": "CL/CD independent signal variance, shared isotropic length scale, noise variance",
        "starts": "Two starts: preceding-budget parameters plus default; at first budget default plus fixed-seed random start.",
        "objective": "Exact training negative log marginal likelihood; no validation/test-driven kernel changes",
        "boundary_policy": {
            "expansions_per_fit": 1,
            "factor": 10.0,
            "log_boundary_tolerance": 1e-4,
            "outward_gradient_threshold": 1e-5,
            "outer_bounds": OUTER_BOUNDS,
            "decision": "Only training-NLL outward gradients; keep lower summed training NLL.",
        },
        "full_budget_reuse": "All 10000 rows in canonical original order fitted once using the first protocol seed; other seeds alias the same model, not independent GP replicates.",
        "warm_start_scaling": "Kernel warm starts are starting values only; each current budget refits its own training-only scalers and all three hyperparameters.",
        "no_final_test_loaded": True,
    }


def freeze_spec(device):
    path = ROOT / "results/kriging_protocol.json"
    spec = make_frozen_spec(device)
    if path.exists():
        old = read_json(path)
        if object_hash(old) != object_hash(spec):
            raise RuntimeError(
                "Kriging protocol, code, or data changed; use a new experiment directory."
            )
    else:
        write_json(path, spec)
    return spec, file_hash(path)


def boundary_diagnostic(model):
    """Decide one bound expansion using only the best training-likelihood boundary positions and gradients."""
    config = asdict(model.config)
    proposed = {key: list(config[key]) for key in OUTER_BOUNDS}
    evidence = []
    for output in model.models:
        best = min(
            output["evaluations"], key=lambda record: record["negative_log_likelihood"]
        )
        for j, (parameter, key) in enumerate(
            (
                ("signal", "signal_bounds"),
                ("length_scale", "length_scale_bounds"),
                ("noise", "noise_bounds"),
            )
        ):
            lo, hi = config[key]
            value, gradient = float(best["log_parameters"][j]), float(
                best["gradient"][j]
            )
            side = (
                "lower"
                if abs(value - np.log(lo)) < 1e-4
                else "upper" if abs(value - np.log(hi)) < 1e-4 else None
            )
            if side is None:
                continue
            outward = gradient > 1e-5 if side == "lower" else gradient < -1e-5
            old = proposed[key].copy()
            if outward:
                if side == "lower":
                    proposed[key][0] = min(
                        proposed[key][0], max(lo / 10.0, OUTER_BOUNDS[key][0])
                    )
                else:
                    proposed[key][1] = max(
                        proposed[key][1], min(hi * 10.0, OUTER_BOUNDS[key][1])
                    )
            evidence.append(
                {
                    "target": output["target"],
                    "parameter": parameter,
                    "side": side,
                    "log_nll_gradient": gradient,
                    "outward_gradient": outward,
                    "old_bounds": old,
                    "proposed_bounds": proposed[key].copy(),
                }
            )
    changed = {
        key: bounds for key, bounds in proposed.items() if bounds != list(config[key])
    }
    return changed, evidence


def fit_with_diagnostic(x, y, config, initial, folder):
    """Fit two starts with at most one likelihood-driven bound expansion; retain every fit record."""
    started = time.perf_counter()
    initial_model = ExactIsotropicKriging(config).fit(
        x, y, initial_params=initial, progress_callback=progress
    )
    write_json(folder / "initial_fit.json", initial_model.summary())
    changes, evidence = boundary_diagnostic(initial_model)
    selected = initial_model
    decision = {
        "boundary_evidence": evidence,
        "expanded_once": bool(changes),
        "changed_bounds": changes,
        "selected": "initial",
        "initial_training_nll_sum": sum(
            m["negative_log_likelihood"] for m in initial_model.models
        ),
    }
    if changes:
        print(
            f"  Outward likelihood gradient; applying one predefined bound expansion: {changes}",
            flush=True,
        )
        config2 = {**asdict(initial_model.config), **changes}
        expanded = ExactIsotropicKriging(config2).fit(
            x,
            y,
            initial_params=initial_model.hyperparameters_,
            progress_callback=progress,
        )
        write_json(folder / "expanded_fit.json", expanded.summary())
        decision["expanded_training_nll_sum"] = sum(
            m["negative_log_likelihood"] for m in expanded.models
        )
        if (
            decision["expanded_training_nll_sum"]
            <= decision["initial_training_nll_sum"] + 1e-8
        ):
            selected = expanded
            decision["selected"] = "expanded"
        decision["remaining_boundary_evidence"] = boundary_diagnostic(selected)[1]
    decision["selection_used_validation"] = False
    decision["total_fit_seconds"] = time.perf_counter() - started
    write_json(folder / "boundary_diagnostic.json", decision)
    return selected, decision


def run_folder(n, seed):
    return ROOT / "models/kriging" / f"n{n:05d}_s{seed}"


def signature_for(n, seed, indices, arrays, spec_hash, canonical=False):
    return {
        "n_train": n,
        "requested_seed": seed,
        "kriging_protocol_sha256": spec_hash,
        "training_ids_sha256": object_hash(arrays["ids"][indices].tolist()),
        "validation_ids_sha256": object_hash(arrays["ids_val"].tolist()),
        "canonical_full_fit": canonical,
    }


def resume_result(folder, signature):
    path = folder / "result.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result["signature"] != signature:
        raise RuntimeError(f"Resume result sample/protocol mismatch: {path}")
    if result["model_sha256"] != file_hash(resolve_model_path(result["model_path"])):
        raise RuntimeError(f"Resume model verification failed: {result['model_path']}")
    print(
        f"RESUME Kriging n={signature['n_train']} seed={signature['requested_seed']} complete; skipped",
        flush=True,
    )
    return result


def train_one(
    n, seed, indices, arrays, spec_hash, previous=None, canonical=False, device="cuda:0"
):
    folder = run_folder(n, seed)
    folder.mkdir(parents=True, exist_ok=True)
    signature = signature_for(n, seed, indices, arrays, spec_hash, canonical)
    cached = resume_result(folder, signature)
    if cached:
        return cached
    config = read_json(ROOT / "results/kriging_protocol.json")["base_config"]
    config.update(seed=seed, stop_path=str(ROOT / "STOP_KRIGING"), device=device)
    initial, warm = None, None
    if previous is not None:
        prev_model = ExactIsotropicKriging.load(
            resolve_model_path(previous["model_path"]), device="cpu"
        )
        initial = prev_model.hyperparameters_
        # Retain likelihood-requested bounds instead of clipping warm starts to the old bounds.
        for key in OUTER_BOUNDS:
            config[key] = getattr(prev_model.config, key)
        warm = {
            "model_path": previous["model_path"],
            "model_sha256": previous["model_sha256"],
            "n_train": previous["n_train"],
            "initial_hyperparameters": initial,
        }
        del prev_model
    print(f"START Kriging n={n} seed={seed}, device={config['device']}", flush=True)
    started = now()
    write_json(
        folder / "status.json",
        {"status": "running", "started_at": started, "signature": signature},
    )
    try:
        check_stop()
        model, decision = fit_with_diagnostic(
            arrays["X"][indices], arrays["Y"][indices], config, initial, folder
        )
        model_path = folder / "model.npz"
        model.save(model_path)
        pv = model.predict(arrays["X_val"])
        scores = metrics(arrays["Y_val"], pv)
        std3 = with_ratio(arrays["Y"][indices]).std(0)
        val_rmse = np.sqrt(
            np.mean((with_ratio(pv) - with_ratio(arrays["Y_val"])) ** 2, axis=0)
        )
        result = {
            "model": MODEL_ID,
            "model_path": model_path.relative_to(ROOT).as_posix(),
            "model_sha256": file_hash(model_path),
            "stage": "learning",
            "n_train": n,
            "seed": seed,
            "signature": signature,
            "training_ids": arrays["ids"][indices].tolist(),
            "training_indices": indices.tolist(),
            "validation_ids": arrays["ids_val"].tolist(),
            "validation_predictions": pv.tolist(),
            "validation_metrics": scores,
            "validation_score": float(np.mean(val_rmse / std3)),
            "validation_score_definition": "Mean CL/CD/L_D validation RMSE divided by current TRAIN std; recorded only, never used for kernel fitting.",
            "config": asdict(model.config),
            "fit_summary": model.summary(),
            "fit_seconds": decision["total_fit_seconds"],
            "boundary_diagnostic": decision,
            "warm_start": warm,
            "started_at": started,
            "completed_at": now(),
            "reused_fit": False,
            "independent_replicate": not canonical,
            "exact_inference": True,
            "test_accessed": False,
        }
        write_json(folder / "result.json", result)
        write_json(
            folder / "status.json",
            {
                "status": "complete",
                "completed_at": now(),
                "result_sha256": file_hash(folder / "result.json"),
            },
        )
        print(
            f"DONE Kriging n={n} seed={seed}, val={result['validation_score']:.6g}, "
            f"time={result['fit_seconds']:.1f}s",
            flush=True,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except BaseException as exc:
        write_json(
            folder / "status.json",
            {
                "status": (
                    "interrupted"
                    if isinstance(exc, (KeyboardInterrupt, InterruptedError))
                    else "failed"
                ),
                "recorded_at": now(),
                "signature": signature,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def reuse_full_fit(source, n, seed, arrays, requested_indices, spec_hash):
    canonical_indices = np.arange(n)
    folder = run_folder(n, seed)
    folder.mkdir(parents=True, exist_ok=True)
    signature = signature_for(
        n, seed, canonical_indices, arrays, spec_hash, canonical=True
    )
    cached = resume_result(folder, signature)
    if cached:
        return cached
    record = dict(source)
    record.update(
        seed=seed,
        signature=signature,
        reused_fit=True,
        independent_replicate=False,
        reused_from_result=(run_folder(n, source["seed"]) / "result.json")
        .relative_to(ROOT)
        .as_posix(),
        optimization_seed=source["seed"],
        incremental_fit_seconds=0.0,
        requested_nested_training_ids=arrays["ids"][requested_indices].tolist(),
        reuse_reason="Identical complete training set; canonical row order and identical exact GP reused. This is not an independent GP repeat.",
        completed_at=now(),
    )
    write_json(folder / "result.json", record)
    write_json(
        folder / "status.json",
        {
            "status": "complete_reused",
            "completed_at": now(),
            "result_sha256": file_hash(folder / "result.json"),
        },
    )
    print(
        f"REUSE Kriging n={n} seed={seed}: same full-data GP, source seed={source['seed']}",
        flush=True,
    )
    return record


def curves(device, max_budget=None, selected_seeds=None):
    protocol, split, arrays = load_development()
    check_stop()
    _, spec_hash = freeze_spec(device)
    seeds = selected_seeds or protocol["seeds"]
    if any(seed not in protocol["seeds"] for seed in seeds) or len(set(seeds)) != len(
        seeds
    ):
        raise ValueError("seeds must be a distinct subset of the frozen protocol seeds")
    seeds = [s for s in protocol["seeds"] if s in seeds]
    if max_budget is not None and max_budget not in protocol["budgets"]:
        raise ValueError("max-budget must match a budget in the frozen protocol")
    budgets = [n for n in protocol["budgets"] if max_budget is None or n <= max_budget]
    primary = protocol["seeds"][0]
    previous, records = {}, []
    for n in budgets:
        for seed in seeds:
            requested = np.asarray(
                split["nested_training_orders"][str(seed)][:n], dtype=int
            )
            if n == protocol["train_n"] and seed != primary:
                canonical_path = run_folder(n, primary) / "result.json"
                if not canonical_path.exists():
                    raise RuntimeError(
                        f"Full budget requires source seed={primary} first, then reuse for other seeds"
                    )
                source = read_json(canonical_path)
                if source["model_sha256"] != file_hash(
                    resolve_model_path(source["model_path"])
                ):
                    raise RuntimeError("Full-budget shared model verification failed")
                record = reuse_full_fit(source, n, seed, arrays, requested, spec_hash)
            else:
                indices = np.arange(n) if n == protocol["train_n"] else requested
                record = train_one(
                    n,
                    seed,
                    indices,
                    arrays,
                    spec_hash,
                    previous.get(seed),
                    canonical=n == protocol["train_n"],
                    device=device,
                )
            previous[seed] = record
            records.append(record)
            # Update the index after each model; resumed runs verify and skip saved models.
            write_json(
                ROOT / "results/kriging_progress.json",
                {
                    "updated_at": now(),
                    "kriging_protocol_sha256": spec_hash,
                    "requested_budgets": budgets,
                    "requested_seeds": seeds,
                    "runs": [
                        {
                            "n_train": r["n_train"],
                            "seed": r["seed"],
                            "model_path": r["model_path"],
                            "result_path": str(
                                run_folder(r["n_train"], r["seed"]) / "result.json"
                            ),
                            "result_sha256": file_hash(
                                run_folder(r["n_train"], r["seed"]) / "result.json"
                            ),
                            "model_sha256": r["model_sha256"],
                            "reused_fit": r["reused_fit"],
                        }
                        for r in records
                    ],
                },
            )
    complete = budgets == protocol["budgets"] and seeds == protocol["seeds"]
    index = read_json(ROOT / "results/kriging_progress.json")
    index["all_protocol_runs_complete"] = complete
    if complete:
        write_json(ROOT / "results/kriging_learning_complete.json", index)
        write_json(
            ROOT / "results/kriging_full_model.json",
            {
                "model_path": previous[primary]["model_path"],
                "model_sha256": previous[primary]["model_sha256"],
                "result_path": str(
                    run_folder(protocol["train_n"], primary) / "result.json"
                ),
                "optimization_seed": primary,
                "kernel": "matern52",
                "selection_uses_test": False,
                "kriging_protocol_sha256": spec_hash,
            },
        )
    print(
        f"Schedule complete: {len(records)} records; independent fits: {sum(not r['reused_fit'] for r in records)}.",
        flush=True,
    )


def smoke():
    """Use 40 synthetic rows on two CPU threads without occupying the study GPU."""
    rng = np.random.default_rng(20261013)
    x = rng.normal(size=(40, 20))
    y = np.column_stack((0.7 + 0.05 * np.sin(x[:, 0]), 0.018 + 0.002 * np.cos(x[:, 1])))
    config = {
        **DEFAULT_CONFIG,
        "device": "cpu",
        "max_iterations": 4,
        "max_function_evaluations": 15,
        "verbose": False,
        "seed": 11,
    }
    with tempfile.TemporaryDirectory(prefix="kriging_driver_smoke_") as temporary:
        folder = Path(temporary)
        model, decision = fit_with_diagnostic(x, y, config, None, folder)
        model.save(folder / "smoke.npz")
        loaded = ExactIsotropicKriging.load(folder / "smoke.npz", device="cpu")
        np.testing.assert_allclose(
            model.predict(x[:3]), loaded.predict(x[:3]), rtol=1e-12, atol=1e-12
        )
        # Test boundary diagnostics with explicit cases; inward gradients must not trigger expansion.
        saved = model.models[0]["evaluations"]
        fake = {
            "negative_log_likelihood": -1e10,
            "log_parameters": np.log([1.0, 100.0, 1e-4]).tolist(),
            "gradient": [0.0, -1.0, 0.0],
        }
        model.models[0]["evaluations"] = [fake]
        changed, evidence = boundary_diagnostic(model)
        assert changed["length_scale_bounds"][1] == 1000.0
        fake["gradient"][1] = 1.0
        assert "length_scale_bounds" not in boundary_diagnostic(model)[0]
        model.models[0]["evaluations"] = saved
        print(
            json.dumps(
                {
                    "status": "passed",
                    "n_synthetic": 40,
                    "device": "cpu",
                    "save_load_equal": True,
                    "boundary_gradient_direction_verified": True,
                    "fit_seconds": decision["total_fit_seconds"],
                    "test_data_loaded": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "smoke", "curves"))
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Exact-compute device; may change on resume, with actual devices recorded per model",
    )
    parser.add_argument("--max-budget", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "plan":
        protocol, split, arrays = load_development()
        print(
            json.dumps(
                {
                    "budgets": protocol["budgets"],
                    "seeds": protocol["seeds"],
                    "train_shape": list(arrays["X"].shape),
                    "validation_shape": list(arrays["X_val"].shape),
                    "actual_fit_count": (len(protocol["budgets"]) - 1)
                    * len(protocol["seeds"])
                    + 1,
                    "full_fit_reused": True,
                    "final_test_loaded": False,
                    "requested_device": args.device,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        curves(args.device, args.max_budget, args.seeds)


if __name__ == "__main__":
    main()
