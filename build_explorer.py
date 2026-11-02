"""Export a compact, static browser payload from evaluated scientific records."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from common import ROOT, PROJECT_ROOT, read_json, write_json, file_hash, with_ratio


def main():
    ev = read_json(ROOT / "results/evaluation.json")
    with np.load(ROOT / "results/main_predictions.npz", allow_pickle=False) as d:
        truth = with_ratio(d["truth"])
        mlp = with_ratio(d["MLP"])
        gp = with_ratio(d["Kriging"])
        cases = [
            {
                "id": str(d["ids"][i]),
                "cst": d["X"][i].tolist(),
                "truth": truth[i].tolist(),
                "MLP": mlp[i].tolist(),
                "Kriging": gp[i].tolist(),
            }
            for i in range(len(truth))
        ]
    with np.load(ROOT / "data/development.npz", allow_pickle=False) as d:
        baseline = d["baseline_cst"].tolist()
    write_json(
        PROJECT_ROOT / "docs/assets/benchmark.json",
        {
            "schema": 1,
            "source_sha256": file_hash(ROOT / "results/main_predictions.npz"),
            "model_seed": ev["main_seed"],
            "training_n": ev["protocol"]["train_n"],
            "validation_n": ev["protocol"]["validation_n"],
            "test_n": len(cases),
            "baseline": baseline,
            "cases": cases,
            "learning": ev["aggregates"],
            "main_metrics": ev["main"],
            "crossings": ev["crossings"],
            "notice": "Saved representative-model predictions, not live model inference.",
        },
    )
    print("Static explorer data exported: all", len(cases), "test cases.")


if __name__ == "__main__":
    main()
