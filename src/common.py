"""Common paths, atomic records, and evaluation metrics. Source CFD files remain read-only."""

from pathlib import Path
import os
import hashlib
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("AIRFOIL_RUN_DIR", PROJECT_ROOT / "runs/default")).resolve()
TARGETS = ["CL", "CD", "L_D"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def with_ratio(y):
    y = np.asarray(y, dtype=np.float64)
    if y.shape[1] == 3:
        return y
    if not np.isfinite(y).all() or np.any(y[:, 1] <= 0):
        raise ValueError(
            "Predictions are nonfinite or have nonpositive drag; do not hide or clip them."
        )
    return np.column_stack([y, y[:, 0] / y[:, 1]])


def metrics(truth, prediction):
    truth, prediction = with_ratio(truth), with_ratio(prediction)
    errors = prediction - truth
    output = {}
    for j, q in enumerate(TARGETS):
        e, t = errors[:, j], truth[:, j]
        output[q] = {
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE": float(np.mean(abs(e))),
            "R2": float(1 - np.sum(e**2) / np.sum((t - t.mean()) ** 2)),
            "MAPE_percent": float(100 * np.mean(abs(e / t))),
            "P95_APE_percent": float(np.quantile(100 * abs(e / t), 0.95)),
            "max_APE_percent": float(np.max(100 * abs(e / t))),
        }
    output["E2_percent"] = float(
        50 * np.sum(np.sqrt(np.mean(errors[:, :2] ** 2, axis=0)) / truth[:, :2].std(0))
    )
    return output


def development_data():
    """Training and selection use this interface without loading final test labels."""
    return np.load(ROOT / "data" / "development.npz")


def model_path(value):
    """Resolve portable model references under the selected experiment directory."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def device_name():
    import torch

    name = os.environ.get("AIRFOIL_DEVICE", "auto")
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
    return name
