"""Two-output MLP with CST scaling, optional linear branch, two-stage optimization,
and validation checkpoint selection. The linear branch belongs to the same
differentiable network. All scales use the current training subset; no final test
labels enter training."""

import json
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch import nn
from common import with_ratio, device_name, model_path


class AirfoilMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config["hidden"]
        layers = []
        widths = [20] + hidden
        for i in range(len(hidden)):
            layer = nn.Linear(widths[i], widths[i + 1])
            nn.init.normal_(layer.weight, std=math.sqrt(1 / widths[i]))
            nn.init.zeros_(layer.bias)
            layers += [
                layer,
                nn.Tanh() if config["activation"] == "tanh" else nn.SiLU(),
            ]
            if config.get("dropout", 0):
                layers.append(nn.Dropout(config["dropout"]))
        last = nn.Linear(hidden[-1], 2)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.network = nn.Sequential(*layers)
        self.skip = nn.Linear(20, 2, bias=False) if config.get("skip") else None
        if self.skip is not None:
            nn.init.zeros_(self.skip.weight)

    def forward(self, x):
        out = self.network(x)
        return out + self.skip(x) if self.skip is not None else out


def _physical_np(z, center, scale, log_cd):
    out = np.asarray(z, dtype=np.float64) * scale + center
    if log_cd:
        out[:, 1] = np.exp(out[:, 1])
    return out


def fit(X, Y, Xv, Yv, config, seed, progress=None):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(seed)
    dev = torch.device(device_name())
    xm = X.mean(0)
    xs = np.maximum(X.std(0), 1e-12)
    target = Y.copy()
    vtarget = Yv.copy()
    if config.get("log_cd"):
        target[:, 1] = np.log(target[:, 1])
        vtarget[:, 1] = np.log(vtarget[:, 1])
    ym = target.mean(0)
    ys = np.maximum(target.std(0), 1e-12)
    std3 = with_ratio(Y).std(0)
    xn = (X - xm) / xs
    yn = (target - ym) / ys
    tensors = [
        torch.tensor(v, dtype=torch.float32, device=dev)
        for v in [xn, yn, (Xv - xm) / xs, (vtarget - ym) / ys]
    ]
    x, y, xv, yv = tensors
    net = AirfoilMLP(config).to(dev)
    if net.skip is not None:
        beta = np.linalg.solve(xn.T @ xn + 0.01 * np.eye(20), xn.T @ yn)
        with torch.no_grad():
            net.skip.weight.copy_(torch.tensor(beta.T, dtype=torch.float32, device=dev))
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    trace = []
    best = float("inf")
    best_state = None
    best_step = None
    best_epoch = 0
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    start = time.perf_counter()

    def loss(pred, truth):
        total = (pred - truth).square().mean()
        if config.get("ratio_weight", 0):
            mean = torch.as_tensor(ym, device=dev, dtype=pred.dtype)
            scale = torch.as_tensor(ys, device=dev, dtype=pred.dtype)
            pp, tt = pred * scale + mean, truth * scale + mean
            pd = pp[:, 1].exp() if config.get("log_cd") else pp[:, 1].clamp_min(1e-5)
            td = tt[:, 1].exp() if config.get("log_cd") else tt[:, 1]
            total = (
                total
                + config["ratio_weight"]
                * ((pp[:, 0] / pd - tt[:, 0] / td) / std3[2]).square().mean()
            )
        return total

    def observe(stage, step):
        nonlocal best, best_state, best_step, best_epoch
        net.eval()
        with torch.no_grad():
            p = net(xv)
            v_loss = float(loss(p, yv))
            tr_loss = float(loss(net(x), y))
        phys = _physical_np(p.cpu().numpy(), ym, ys, config.get("log_cd"))
        if not np.isfinite(phys).all() or np.any(phys[:, 1] <= 0):
            score = 1e10
            rmse = np.array([1e10] * 3)
        else:
            rmse = np.sqrt(np.mean((with_ratio(phys) - with_ratio(Yv)) ** 2, axis=0))
            score = float(np.mean(rmse / std3))
        if score < best:
            best = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
            best_step = {"stage": stage, "step": step}
            if stage == "AdamW":
                best_epoch = step
        trace.append(
            {
                "stage": stage,
                "step": step,
                "train_loss": tr_loss,
                "validation_loss": v_loss,
                "validation_score": score,
                "validation_RMSE": rmse.tolist(),
            }
        )
        return score

    for epoch in range(config["epochs"] + 1):
        if (
            epoch % config.get("validation_interval", 20) == 0
            or epoch == config["epochs"]
        ):
            observe("AdamW", epoch)
            if progress and epoch % 1000 == 0:
                progress(f"AdamW {epoch}: val={best:.6g}")
            if (
                epoch >= config["minimum_epochs"]
                and epoch - best_epoch >= config["patience"]
            ):
                break
        if epoch == config["epochs"]:
            break
        net.train()
        lr = 1e-6 + 0.5 * (config["learning_rate"] - 1e-6) * (
            1 + math.cos(math.pi * epoch / config["epochs"])
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        bs = config.get("batch_size", 0)
        batches = (
            [None]
            if not bs or bs >= len(x)
            else torch.randperm(len(x), device=dev).split(bs)
        )
        for idx in batches:
            xb, yb = (x, y) if idx is None else (x[idx], y[idx])
            optimizer.zero_grad(set_to_none=True)
            objective = loss(net(xb), yb)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite training loss")
            objective.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
    adam_end = epoch
    net.load_state_dict(best_state)
    if config.get("lbfgs_steps", 0):
        net = net.double()
        x, y, xv, yv = [t.double() for t in [x, y, xv, yv]]
        net.eval()  # Deterministic full-batch curvature optimization with dropout disabled.
        lb = torch.optim.LBFGS(
            net.parameters(),
            lr=0.8,
            max_iter=5,
            max_eval=8,
            history_size=30,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-13,
        )
        stale = 0
        previous = best
        for step in range(1, config["lbfgs_steps"] + 1):

            def closure():
                lb.zero_grad(set_to_none=True)
                value = loss(net(x), y)
                if config.get("lbfgs_l2", 0):
                    value = value + config["lbfgs_l2"] * sum(
                        p.square().sum() for p in net.parameters()
                    )
                value.backward()
                return value

            lb.step(closure)
            if step % 5 == 0 or step == config["lbfgs_steps"]:
                observe("LBFGS", step)
                stale = 0 if best < previous - 1e-8 else stale + 5
                previous = best
                if progress and step % 100 == 0:
                    progress(f"LBFGS {step}: val={best:.6g}")
                if stale >= 80:
                    break
    net.load_state_dict(best_state)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    metadata = {
        "config": config,
        "seed": seed,
        "n_train": len(X),
        "n_validation": len(Xv),
        "x_mean": xm.tolist(),
        "x_scale": xs.tolist(),
        "y_mean": ym.tolist(),
        "y_scale": ys.tolist(),
        "parameter_count": sum(p.numel() for p in net.parameters()),
        "best_step": best_step,
        "adam_end_epoch": adam_end,
        "validation_score": best,
        "train_target_std": std3.tolist(),
        "fit_seconds": time.perf_counter() - start,
        "dtype": str(next(net.parameters()).dtype),
        "trace": trace,
        "device": str(dev),
        "state_dict": best_state,
    }
    return net, metadata


def predict(net, X, metadata, batch_size=4096):
    parameter = next(net.parameters())
    net.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = (
                X[start : start + batch_size] - np.array(metadata["x_mean"])
            ) / np.array(metadata["x_scale"])
            t = torch.tensor(x, dtype=parameter.dtype, device=parameter.device)
            parts.append(net(t).cpu().numpy())
    return _physical_np(
        np.concatenate(parts),
        np.array(metadata["y_mean"]),
        np.array(metadata["y_scale"]),
        metadata["config"].get("log_cd"),
    )


def reload_model(path, device="cpu"):
    with np.load(model_path(path), allow_pickle=False) as archive:
        saved = json.loads(str(archive["metadata"]))
        for key in ("x_mean", "x_scale", "y_mean", "y_scale"):
            if key not in saved:
                saved[key] = archive[key].tolist()
        saved["state_dict"] = {
            k.removeprefix("weight::"): torch.from_numpy(archive[k].copy())
            for k in archive.files
            if k.startswith("weight::")
        }
    model = AirfoilMLP(saved["config"])
    if "float64" in saved.get("dtype", ""):
        model = model.double()
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    model.eval()
    return model, saved


def save_model(metadata, path):
    """Atomic, pickle-free serialization of weights, scales and optimization records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "weight::" + k: v.detach().cpu().numpy()
        for k, v in metadata["state_dict"].items()
    }
    meta = {k: v for k, v in metadata.items() if k != "state_dict"}
    arrays["metadata"] = np.asarray(json.dumps(meta, allow_nan=False))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
