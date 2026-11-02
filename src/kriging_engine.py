"""Exact Gaussian-process regression with isotropic Matern-5/2 or RBF covariance.
Use all training centers with float64 dense Cholesky factorization. Distances may be
blocked for memory management; no sparse or inducing-point approximation is used.
Each output has its own signal variance, shared input length scale, and noise variance,
optimized in log coordinates against training marginal likelihood. Numerical jitter
is distinct from noise. Scaling is estimated from training data only. Optimizer
termination, bound activity, and numerical diagnostics are preserved.
Set optimize=False and initial_hyperparameters only for explicitly fixed-parameter
experiments or verification. The main study uses optimize=True."""

from __future__ import annotations

import ctypes
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.optimize import minimize


@dataclass
class KrigingConfig:
    """signal and noise denote variances; specify optimization bounds on their original scales."""

    kernel: str = "matern52"
    device: str = "auto"
    cpu_threads: int = 2
    block_size: int = 256
    prediction_batch_size: int = 256
    max_iterations: int = 150
    max_function_evaluations: int = 240
    n_restarts: int = 2
    seed: int = 42
    optimize: bool = True
    length_scale_initial: float = math.sqrt(20.0)
    signal_initial: float = 1.0
    noise_initial: float = 1.0e-4
    length_scale_bounds: tuple = (0.1, 100.0)
    signal_bounds: tuple = (1.0e-3, 1.0e3)
    noise_bounds: tuple = (1.0e-8, 0.1)
    jitter: float = 1.0e-10
    max_jitter: float = 1.0e-4
    ftol: float = 1.0e-9
    gtol: float = 1.0e-5
    max_line_search: int = 30
    verbose: bool = True
    stop_path: str | None = None

    def validate(self):
        if self.kernel not in ("matern52", "rbf"):
            raise ValueError("kernel must be matern52 or rbf")
        for name in ("length_scale_bounds", "signal_bounds", "noise_bounds"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
                raise ValueError(f"{name} must satisfy 0 < lower < upper")
        for name in (
            "cpu_threads",
            "block_size",
            "prediction_batch_size",
            "max_iterations",
            "max_function_evaluations",
            "max_line_search",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.n_restarts) is not int or self.n_restarts < 0:
            raise ValueError("n_restarts must be a nonnegative integer")
        if not 0 < self.jitter <= self.max_jitter:
            raise ValueError("Require 0 < jitter <= max_jitter")
        for value, bounds in (
            (self.signal_initial, self.signal_bounds),
            (self.length_scale_initial, self.length_scale_bounds),
            (self.noise_initial, self.noise_bounds),
        ):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError("Initial kernel parameters exceed the bounds")


def _available_ram():
    """Query available memory without requiring psutil."""
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("available_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.available_phys)
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _device_for(n, config):
    # Conservative estimate for distances, covariance/factorization, inverse, and linear-algebra workspaces.
    estimated = int(5.0 * n * n * 8 + max(n * config.block_size * 8 * 8, 512 * 2**20))
    requested = config.device
    if requested == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            if estimated < free * 0.88:
                device = "cuda:0"
    else:
        device = requested
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable in the current PyTorch environment"
            )
        index = torch.device(device).index or 0
        free, _ = torch.cuda.mem_get_info(index)
        if estimated > free * 0.95:
            raise MemoryError(
                f"Exact GP requires approximately {estimated / 2**30:.2f} GiB，"
                f"Available CUDA memory: {free / 2**30:.2f} GiB; free GPU memory or select CPU"
            )
    else:
        available = _available_ram()
        if available is not None and estimated > available * 0.9:
            raise MemoryError(
                f"Exact GP requires approximately {estimated / 2**30:.2f} GiB，"
                f"Available system memory: {available / 2**30:.2f} GiB; no sparse approximation has been substituted"
            )
    return torch.device(device), estimated


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_stop(config):
    if config.stop_path and Path(config.stop_path).exists():
        raise InterruptedError("Stop requested; Kriging optimization terminated")


def _squared_distance(x, y):
    """Squared Euclidean distance in standardized inputs; clamp roundoff-negative values to zero."""
    result = x @ y.T
    result.mul_(-2.0)
    result.add_(x.square().sum(1)[:, None])
    result.add_(y.square().sum(1)[None, :])
    return result.clamp_min_(0.0)


def _kernel_block(d2, signal, length, kind, derivative=False):
    if kind == "matern52":
        u = d2.sqrt().mul_(math.sqrt(5.0) / length)
        e = torch.exp(-u)
        spatial = (1.0 + u + u.square() / 3.0) * e * signal
        if derivative:
            d_length = (u.square() + u.pow(3)) * e * (signal / 3.0)
            return spatial, d_length
    else:
        z = d2 / (length * length)
        spatial = torch.exp(-0.5 * z).mul_(signal)
        if derivative:
            return spatial, spatial * z
    return spatial


class _TorchObjective:
    """Exact negative log marginal likelihood and analytic gradients in log(signal,length,noise)."""

    def __init__(self, d2, y, config, target="", callback=None):
        self.d2, self.y, self.config = d2, y, config
        self.target, self.callback = target, callback
        self.n, self.device = len(y), y.device
        self.evaluations = []
        self.best = None
        self.started = time.perf_counter()

    @torch.no_grad()
    def factor(self, theta):
        signal, length, noise = np.exp(theta)
        c = torch.empty_like(self.d2)
        for start in range(0, self.n, self.config.block_size):
            end = min(start + self.config.block_size, self.n)
            c[start:end] = _kernel_block(
                self.d2[start:end], signal, length, self.config.kernel
            )
        # Enforce numerical symmetry; blocked self-distances are analytically symmetric.
        c.diagonal().add_(noise)
        jitter, previous = self.config.jitter, 0.0
        while True:
            _check_stop(self.config)
            c.diagonal().add_(jitter - previous)
            l, info = torch.linalg.cholesky_ex(c, check_errors=False)
            if int(info.item()) == 0:
                break
            del l
            if jitter >= self.config.max_jitter:
                raise np.linalg.LinAlgError(
                    "Kriging covariance remains nonpositive definite at maximum jitter"
                )
            previous, jitter = jitter, min(jitter * 10, self.config.max_jitter)
        del c
        alpha = torch.cholesky_solve(self.y[:, None], l)[:, 0]
        nll = 0.5 * torch.dot(self.y, alpha) + torch.log(l.diagonal()).sum()
        nll += 0.5 * self.n * math.log(2.0 * math.pi)
        return l, alpha, float(nll.item()), jitter

    @torch.no_grad()
    def __call__(self, theta):
        _check_stop(self.config)
        l, alpha, nll, jitter = self.factor(theta)
        inv = torch.cholesky_inverse(l)
        del l
        signal, length, noise = np.exp(theta)
        gradient = torch.zeros(3, dtype=torch.float64, device=self.device)
        gradient[2] = 0.5 * noise * (inv.diagonal().sum() - torch.dot(alpha, alpha))
        for start in range(0, self.n, self.config.block_size):
            end = min(start + self.config.block_size, self.n)
            weight = inv[start:end] - alpha[start:end, None] * alpha[None, :]
            k, dk = _kernel_block(
                self.d2[start:end], signal, length, self.config.kernel, derivative=True
            )
            gradient[0] += 0.5 * torch.sum(weight * k)
            gradient[1] += 0.5 * torch.sum(weight * dk)
        g = gradient.cpu().numpy()
        if not np.isfinite(nll) or not np.isfinite(g).all():
            raise FloatingPointError("Nonfinite Kriging likelihood or gradients")
        row = {
            "evaluation": len(self.evaluations) + 1,
            "negative_log_likelihood": nll,
            "log_parameters": np.asarray(theta).tolist(),
            "gradient": g.tolist(),
            "jitter": jitter,
            "elapsed_seconds": time.perf_counter() - self.started,
        }
        self.evaluations.append(row)
        if self.best is None or nll < self.best[0]:
            self.best = (nll, np.asarray(theta).copy())
        if self.callback:
            self.callback({"target": self.target, **row})
        if self.config.verbose and (
            len(self.evaluations) == 1 or len(self.evaluations) % 10 == 0
        ):
            print(
                f"  Kriging/{self.target}: likelihood evaluation {len(self.evaluations)}, "
                f"NLL={nll:.6f}, {row['elapsed_seconds']:.1f}s",
                flush=True,
            )
        return nll, g


class ExactIsotropicKriging:
    """Training-only interface; excludes validation/test labels from fitting and scaling."""

    def __init__(self, config: KrigingConfig | dict | None = None):
        self.config = (
            config
            if isinstance(config, KrigingConfig)
            else KrigingConfig(**(config or {}))
        )
        self.config.validate()
        self.models = []
        self.diagnostics = {}
        self.scalers = {}
        self.x_train = None

    def fit(
        self,
        x,
        y,
        *,
        initial_hyperparameters=None,
        initial_params=None,
        progress_callback: Callable | None = None,
    ):
        """Fit the full exact GP. Provide separate CL/CD warm-start dictionaries through
        initial_hyperparameters (alias initial_params). They remain optimizable when
        optimize=True. n_restarts counts additional starts: the standard start is
        retained after a warm start, followed by seeded random starts.
        """
        if initial_params is not None:
            if initial_hyperparameters is not None:
                raise ValueError(
                    "Specify only one of initial_params and initial_hyperparameters"
                )
            initial_hyperparameters = initial_params
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 20 or len(x) < 2:
            raise ValueError(
                "Training inputs must be N-by-20 CST coefficients with at least two rows"
            )
        if (
            y.shape != (len(x), 2)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("Targets must be finite N-by-2 (CL,CD) values")
        if initial_hyperparameters is not None and len(initial_hyperparameters) != 2:
            raise ValueError(
                "initial_hyperparameters must include separate CL and CD entries"
            )
        _check_stop(self.config)
        self.models = []
        xs, ys = x.std(axis=0), y.std(axis=0)
        self.scalers = {
            "x_mean": x.mean(0),
            "x_scale": np.where(xs > 1e-14, xs, 1.0),
            "y_mean": y.mean(0),
            "y_scale": np.where(ys > 1e-14, ys, 1.0),
        }
        self.x_train = (x - self.scalers["x_mean"]) / self.scalers["x_scale"]
        y_scaled = (y - self.scalers["y_mean"]) / self.scalers["y_scale"]
        device, estimated = _device_for(len(x), self.config)
        torch.set_num_threads(self.config.cpu_threads)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        xt = torch.as_tensor(self.x_train, device=device, dtype=torch.float64)
        d2 = torch.empty((len(x), len(x)), dtype=torch.float64, device=device)
        for first in range(0, len(x), self.config.block_size):
            d2[first : first + self.config.block_size] = _squared_distance(
                xt[first : first + self.config.block_size], xt
            )
        d2.diagonal().zero_()
        distance_seconds = time.perf_counter() - started
        bounds = np.log(
            np.asarray(
                [
                    self.config.signal_bounds,
                    self.config.length_scale_bounds,
                    self.config.noise_bounds,
                ],
                dtype=np.float64,
            )
        )
        rng = np.random.default_rng(self.config.seed)
        for j, target in enumerate(("CL", "CD")):
            target_started = time.perf_counter()
            objective = _TorchObjective(
                d2,
                torch.as_tensor(y_scaled[:, j], device=device),
                self.config,
                target,
                progress_callback,
            )
            if initial_hyperparameters is None:
                initial = np.log(
                    [
                        self.config.signal_initial,
                        self.config.length_scale_initial,
                        self.config.noise_initial,
                    ]
                )
            else:
                p = initial_hyperparameters[j]
                initial = np.log([p["signal"], p["length_scale"], p["noise"]])
                if (
                    not np.isfinite(initial).all()
                    or (initial < bounds[:, 0]).any()
                    or (initial > bounds[:, 1]).any()
                ):
                    raise ValueError(
                        "Initial kernel parameters must be finite, positive, and within bounds"
                    )
            starts = [initial]
            if self.config.optimize:
                remaining_restarts = self.config.n_restarts
                if initial_hyperparameters is not None and remaining_restarts:
                    # Include a standard start alongside warm starts to avoid locking the budget chain into a local optimum.
                    starts.append(
                        np.log(
                            [
                                self.config.signal_initial,
                                self.config.length_scale_initial,
                                self.config.noise_initial,
                            ]
                        )
                    )
                    remaining_restarts -= 1
                starts += [
                    rng.uniform(bounds[:, 0], bounds[:, 1])
                    for _ in range(remaining_restarts)
                ]
            optimizer_records = []
            for restart, theta in enumerate(starts):
                before = len(objective.evaluations)
                start_time = time.perf_counter()
                if self.config.optimize:
                    result = minimize(
                        objective,
                        theta,
                        jac=True,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={
                            "maxiter": self.config.max_iterations,
                            "maxfun": self.config.max_function_evaluations,
                            "ftol": self.config.ftol,
                            "gtol": self.config.gtol,
                            "maxls": self.config.max_line_search,
                        },
                    )
                    record = {
                        "restart": restart,
                        "success": bool(result.success),
                        "message": str(result.message),
                        "iterations": int(result.nit),
                        "function_evaluations": int(result.nfev),
                        "final_nll": float(result.fun),
                        "final_log_parameters": result.x.tolist(),
                        "initial_log_parameters": theta.tolist(),
                    }
                else:
                    value, _ = objective(theta)
                    record = {
                        "restart": 0,
                        "success": None,
                        "message": "fixed hyperparameters; no optimization",
                        "iterations": 0,
                        "function_evaluations": 1,
                        "final_nll": value,
                        "final_log_parameters": theta.tolist(),
                        "initial_log_parameters": theta.tolist(),
                    }
                record.update(
                    seconds=time.perf_counter() - start_time,
                    evaluation_first=before + 1,
                    evaluation_last=len(objective.evaluations),
                )
                optimizer_records.append(record)
            best_nll, best_theta = objective.best
            l, alpha, checked_nll, used_jitter = objective.factor(best_theta)
            signal, length, noise = np.exp(best_theta)
            boundary = [
                name
                for k, name in enumerate(("signal", "length_scale", "noise"))
                if abs(best_theta[k] - bounds[k, 0]) < 1e-4
                or abs(best_theta[k] - bounds[k, 1]) < 1e-4
            ]
            selected_evaluation = min(
                objective.evaluations, key=lambda r: r["negative_log_likelihood"]
            )
            source_restart = next(
                r
                for r in optimizer_records
                if r["evaluation_first"]
                <= selected_evaluation["evaluation"]
                <= r["evaluation_last"]
            )
            self.models.append(
                {
                    "target": target,
                    "alpha": alpha.cpu().numpy(),
                    "signal": float(signal),
                    "length_scale": float(length),
                    "noise": float(noise),
                    "jitter": used_jitter,
                    "negative_log_likelihood": checked_nll,
                    "optimizer_records": optimizer_records,
                    "selected_restart": source_restart["restart"],
                    "selected_restart_success": source_restart["success"],
                    "parameters_at_bounds": boundary,
                    "fit_seconds": time.perf_counter() - target_started,
                    "evaluations": objective.evaluations,
                }
            )
            del l, alpha, objective
            gc.collect()
        _synchronize(device)
        self.diagnostics = {
            "model": "exact_isotropic_kriging",
            "approximation": "none",
            "dtype": "float64",
            "kernel": self.config.kernel,
            "n_train": len(x),
            "n_features": 20,
            "n_outputs": 2,
            "device": str(device),
            "estimated_peak_bytes": estimated,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "distance_seconds": distance_seconds,
            "fit_seconds": time.perf_counter() - started,
            "standardization": "training-only population mean/std",
            "mean_function": "training target mean, fixed",
            "jitter_policy": "absolute diagonal increment, escalated only when Cholesky fails",
        }
        del d2, xt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def predict(self, x, *, return_std=False, include_noise=True, with_ratio=False):
        """Predict means and optionally CL/CD standard deviations. The ratio is not
        assumed Gaussian. include_noise=True matches sklearn WhiteKernel predictive
        variance; False returns latent-function uncertainty. Jitter is excluded
        from observation noise. Reconstruct Cholesky only when uncertainty is requested.
        """
        if len(self.models) != 2:
            raise RuntimeError("Model has not been fitted or loaded")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 20 or not len(x) or not np.isfinite(x).all():
            raise ValueError(
                "Prediction inputs must be a finite nonempty N-by-20 array"
            )
        device, _ = _device_for(len(self.x_train) if return_std else 1, self.config)
        xt = torch.as_tensor(self.x_train, dtype=torch.float64, device=device)
        xq = torch.as_tensor(
            (x - self.scalers["x_mean"]) / self.scalers["x_scale"],
            dtype=torch.float64,
            device=device,
        )
        columns, deviations = [], []
        for j, model in enumerate(self.models):
            alpha = torch.as_tensor(model["alpha"], dtype=torch.float64, device=device)
            l = None
            if return_std:
                c = torch.empty((len(xt), len(xt)), dtype=torch.float64, device=device)
                for start in range(0, len(xt), self.config.block_size):
                    c[start : start + self.config.block_size] = _kernel_block(
                        _squared_distance(
                            xt[start : start + self.config.block_size], xt
                        ),
                        model["signal"],
                        model["length_scale"],
                        self.config.kernel,
                    )
                c.diagonal().fill_(model["signal"] + model["noise"] + model["jitter"])
                l = torch.linalg.cholesky(c)
                del c
            prediction, std = [], []
            for start in range(0, len(xq), self.config.prediction_batch_size):
                cross = _kernel_block(
                    _squared_distance(
                        xq[start : start + self.config.prediction_batch_size], xt
                    ),
                    model["signal"],
                    model["length_scale"],
                    self.config.kernel,
                )
                prediction.append((cross @ alpha).cpu().numpy())
                if return_std:
                    v = torch.linalg.solve_triangular(l, cross.T, upper=False)
                    var = (
                        model["signal"]
                        + (model["noise"] if include_noise else 0.0)
                        - v.square().sum(0)
                    )
                    # Allow roundoff-scale negative variance only; larger negative values indicate numerical failure.
                    if float(var.min().item()) < -1e-7 * max(model["signal"], 1):
                        raise FloatingPointError(
                            "Predictive variance is negative beyond roundoff tolerance"
                        )
                    std.append(var.clamp_min_(0).sqrt().cpu().numpy())
            columns.append(
                np.concatenate(prediction) * self.scalers["y_scale"][j]
                + self.scalers["y_mean"][j]
            )
            if return_std:
                deviations.append(np.concatenate(std) * self.scalers["y_scale"][j])
            del l
        mean = np.column_stack(columns)
        if with_ratio:
            if np.any(np.abs(mean[:, 1]) < 1e-15):
                raise FloatingPointError(
                    "Predicted CD is near zero; a finite lift-to-drag ratio cannot be computed"
                )
            mean = np.column_stack((mean, mean[:, 0] / mean[:, 1]))
        if not np.isfinite(mean).all():
            raise FloatingPointError("Predictions contain NaN/Inf")
        return (mean, np.column_stack(deviations)) if return_std else mean

    def save(self, path):
        """Save the NumPy model atomically without pickle; retain all optimizer diagnostics in model metadata."""
        if len(self.models) != 2:
            raise RuntimeError("Cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "config": asdict(self.config),
            "diagnostics": self.diagnostics,
            "models": [
                {k: v for k, v in m.items() if k != "alpha"} for m in self.models
            ],
        }
        arrays = {
            **self.scalers,
            "x_train": self.x_train,
            "alpha_CL": self.models[0]["alpha"],
            "alpha_CD": self.models[1]["alpha"],
            "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        }
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)

    @property
    def hyperparameters_(self):
        """Return two kernel-parameter dictionaries usable as initial_params at the next budget."""
        return [
            {key: model[key] for key in ("signal", "length_scale", "noise")}
            for model in self.models
        ]

    def summary(self):
        """Return JSON-serializable fit records without large arrays or test data."""
        return {
            "config": asdict(self.config),
            "diagnostics": self.diagnostics,
            "outputs": [
                {key: value for key, value in model.items() if key != "alpha"}
                for model in self.models
            ],
        }

    @classmethod
    def load(cls, path, *, device="auto"):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["schema_version"] != 1:
                raise ValueError("Unsupported model format")
            config = {**metadata["config"], "device": device}
            model = cls(config)
            model.scalers = {
                k: saved[k].copy() for k in ("x_mean", "x_scale", "y_mean", "y_scale")
            }
            model.x_train = saved["x_train"].copy()
            model.models = metadata["models"]
            for j, target in enumerate(("CL", "CD")):
                model.models[j]["alpha"] = saved["alpha_" + target].copy()
            model.diagnostics = metadata["diagnostics"]
        return model
