"""Screen, confirm, and fit MLP learning curves under the frozen resumable protocol."""

import argparse
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import torch
from common import (
    ROOT,
    PROJECT_ROOT,
    read_json,
    write_json,
    development_data,
    file_hash,
    metrics,
)
from mlp_engine import fit, predict, reload_model, save_model


def one(config, seed, indices, stage):
    path = ROOT / "models" / stage / f"{config['id']}_n{len(indices):05d}_s{seed}"
    path.mkdir(parents=True, exist_ok=True)
    signature = {
        "development": file_hash(ROOT / "data/development.npz"),
        "protocol": file_hash(ROOT / "protocol.json"),
        "engine": file_hash(PROJECT_ROOT / "src/mlp_engine.py"),
        "driver": file_hash(__file__),
        "config": config,
        "indices": np.asarray(indices).tolist(),
        "seed": int(seed),
    }
    if (path / "result.json").exists():
        result = read_json(path / "result.json")
        if result.get("signature") != signature:
            raise RuntimeError("Resume mismatch: use a new run directory.")
        assert result["model_sha256"] == file_hash(path / "model.npz")
        return result
    a = development_data()
    print(f"START {stage} {config['id']} n={len(indices)} seed={seed}", flush=True)
    net, meta = fit(
        a["X"][indices],
        a["Y"][indices],
        a["X_val"],
        a["Y_val"],
        config,
        seed,
        progress=lambda s: print(" ", s, flush=True),
    )
    pv = predict(net, a["X_val"], meta)
    save_model(meta, path / "model.npz")
    result = {k: v for k, v in meta.items() if k != "state_dict"}
    result.update(
        stage=stage,
        signature=signature,
        model_path=(path / "model.npz").relative_to(ROOT).as_posix(),
        model_sha256=file_hash(path / "model.npz"),
        validation_metrics=metrics(a["Y_val"], pv),
        training_ids=a["ids"][indices].tolist(),
        validation_predictions=pv.tolist(),
    )
    write_json(path / "result.json", result)
    print(
        f"DONE {config['id']} score={meta['validation_score']:.6g} seconds={meta['fit_seconds']:.1f}",
        flush=True,
    )
    del net, meta
    gc.collect()
    torch.cuda.empty_cache()
    return result


def select():
    protocol = read_json(ROOT / "protocol.json")
    allidx = np.arange(protocol["train_n"])
    screened = [one(c, 11, allidx, "screen") for c in protocol["candidates"]]
    top = sorted(screened, key=lambda r: r["validation_score"])[:3]
    confirmation = []
    for r in top:
        runs = [r] + [one(r["config"], s, allidx, "confirm") for s in [22, 33]]
        confirmation.append(
            {
                "config": r["config"],
                "mean_validation_score": float(
                    np.mean([v["validation_score"] for v in runs])
                ),
                "runs": [v["model_path"] for v in runs],
                "scores": [v["validation_score"] for v in runs],
            }
        )
    winner = min(confirmation, key=lambda x: x["mean_validation_score"])
    write_json(
        ROOT / "results/mlp_selection.json",
        {
            "selected": winner,
            "confirmation": confirmation,
            "selection_uses_test": False,
            "protocol_sha256": file_hash(ROOT / "protocol.json"),
        },
    )
    print(
        "SELECTED", winner["config"]["id"], winner["mean_validation_score"], flush=True
    )


def curves():
    protocol = read_json(ROOT / "protocol.json")
    split = read_json(ROOT / "data/split.json")
    c = read_json(ROOT / "results/mlp_selection.json")["selected"]["config"]
    records = []
    for n in protocol["budgets"]:
        for seed in protocol["seeds"]:
            indices = np.asarray(split["nested_training_orders"][str(seed)][:n])
            # Kriging uses the same ordering, retaining full-budget order for numerical reproducibility.
            records.append(one(c, seed, indices, "learning"))
    write_json(
        ROOT / "results/mlp_learning_complete.json",
        {"runs": [r["model_path"] for r in records]},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "curves"])
    args = p.parse_args()
    select() if args.stage == "select" else curves()
