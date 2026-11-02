"""Export the pinned, already-trained models for float64 browser inference. No fitting."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import PROJECT_ROOT, ROOT, file_hash, read_json


def main():
    destination = PROJECT_ROOT / "docs/assets/inference"
    destination.mkdir(parents=True, exist_ok=True)
    lock = read_json(PROJECT_ROOT / "configs/assets.lock.json")
    manifest = {
        "schema": 1,
        "precision": "IEEE 754 float64; no quantization",
        "features": [f"{side}{i}" for side in "UL" for i in range(10)],
        "outputs": ["CL", "CD", "CL/CD"],
        "condition": {"Mach": 0.73, "Reynolds": 6500000, "AoA_deg": 2.79},
        "bounds": read_json(PROJECT_ROOT / "configs/cst_bounds.json"),
        "models": {},
    }
    for family in ("mlp", "kriging"):
        entry = next(
            e
            for e in lock["artifacts"]
            if e["destination"] == f"pretrained/{family}/model.npz"
        )
        path = ROOT / entry["destination"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(
                f"{family}: input must match the pinned released checkpoint"
            )
        with np.load(path, allow_pickle=False) as source:
            meta = json.loads(str(source["metadata"]))
            arrays = {k: source[k].copy() for k in source.files if k != "metadata"}
        if family == "mlp":
            if meta["config"]["activation"] != "tanh" or "float64" not in meta["dtype"]:
                raise ValueError("Browser export expects the selected float64 Tanh MLP")
            indices = sorted(
                int(k.split(".")[1])
                for k in arrays
                if k.startswith("weight::network.") and k.endswith(".weight")
            )
            definition = {
                "activation": "tanh",
                "log_cd": meta["config"].get("log_cd", False),
                "layers": [
                    {
                        "weight": f"weight::network.{i}.weight",
                        "bias": f"weight::network.{i}.bias",
                    }
                    for i in indices
                ],
                "skip": "weight::skip.weight" if meta["config"].get("skip") else None,
            }
        else:
            if meta["config"]["kernel"] != "matern52":
                raise ValueError(
                    "Browser export expects the released Matérn-5/2 Kriging"
                )
            definition = {
                "kernel": "matern52",
                "parameters": [
                    {k: m[k] for k in ("signal", "length_scale")}
                    for m in meta["models"]
                ],
                "training_points": len(arrays["x_train"]),
            }
        offset, layouts, pieces = 0, {}, []
        for key, value in sorted(arrays.items()):
            value = np.asarray(value, dtype="<f8", order="C")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite model array: {family}/{key}")
            layouts[key] = {
                "offset": offset,
                "length": int(value.size),
                "shape": list(value.shape),
            }
            pieces.append(value.tobytes())
            offset += value.size
        payload = b"".join(pieces)
        digest = hashlib.sha256(payload).hexdigest()
        filename = f"{family}-{digest[:12]}.f64.bin"
        (destination / filename).write_bytes(payload)
        manifest["models"][family] = {
            "file": filename,
            "sha256": digest,
            "bytes": len(payload),
            "source_url": entry["url"],
            "source_npz_sha256": entry["sha256"],
            "arrays": layouts,
            "definition": definition,
        }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        {
            k: {"bytes": v["bytes"], "file": v["file"]}
            for k, v in manifest["models"].items()
        }
    )


if __name__ == "__main__":
    main()
