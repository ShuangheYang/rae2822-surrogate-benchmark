"""Export evaluated experiments as an Excel workbook. Requires no CFD software."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from common import ROOT, read_json, with_ratio


def sheet(book, name, columns, rows):
    ws = book.create_sheet(name)
    ws.append(columns)
    for row in rows:
        ws.append([v.item() if isinstance(v, np.generic) else v for v in row])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="73588C")
    for col in ws.columns:
        letter = col[0].column_letter
        ws.column_dimensions[letter].width = min(
            45, max(16, len(str(col[0].value)) + 2)
        )
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.000000"
            if isinstance(cell.value, str) and cell.value.isdigit():
                cell.number_format = "@"
    return ws


def main():
    ev = read_json(ROOT / "results/evaluation.json")
    book = Workbook()
    book.remove(book.active)
    cols = [
        "model",
        "target",
        "RMSE",
        "MAE",
        "R2",
        "MAPE_percent",
        "P95_APE_percent",
        "max_APE_percent",
    ]
    sheet(
        book,
        "Main metrics",
        cols,
        [
            [f, q, *[ev["main"][f][q][k] for k in cols[2:]]]
            for f in ["MLP", "Kriging"]
            for q in ["CL", "CD", "L_D"]
        ],
    )
    keys = list(ev["aggregates"][0])
    sheet(
        book, "Learning curves", keys, [[r[k] for k in keys] for r in ev["aggregates"]]
    )
    prediction_rows = []
    with np.load(ROOT / "results/main_predictions.npz", allow_pickle=False) as d:
        y = with_ratio(d["truth"])
        for f in ["MLP", "Kriging"]:
            p = with_ratio(d[f])
            ape = 100 * np.abs((p - y) / y)
            prediction_rows.extend(
                [
                    [
                        f'LHS_2000:{str(d["ids"][i])}',
                        str(d["ids"][i]),
                        f,
                        *y[i],
                        *p[i],
                        *ape[i],
                        *d["X"][i],
                    ]
                    for i in range(len(y))
                ]
            )
    sheet(
        book,
        "Test predictions",
        [
            "sample_key",
            "id",
            "model",
            "true_CL",
            "true_CD",
            "true_L_D",
            "pred_CL",
            "pred_CD",
            "pred_L_D",
            "APE_CL_percent",
            "APE_CD_percent",
            "APE_L_D_percent",
            *ev["protocol"]["features"],
        ],
        prediction_rows,
    )
    runs = ev["records"]
    rows = []
    for r in runs:
        for q in ["CL", "CD", "L_D"]:
            rows.append(
                [
                    r["family"],
                    r["n_train"],
                    r["seed"],
                    q,
                    r["model_sha256"],
                    r["fit_seconds"],
                    r["reused_deterministic_fit"],
                    *[r["test_metrics"][q][k] for k in cols[2:]],
                ]
            )
    sheet(
        book,
        "All evaluated runs",
        [
            "model",
            "n_train",
            "seed",
            "target",
            "model_sha256",
            "fit_seconds",
            "reused_fit",
            *cols[2:],
        ],
        rows,
    )
    candidates = [
        read_json(p) for p in sorted((ROOT / "models/screen").glob("*/result.json"))
    ]
    sheet(
        book,
        "MLP screening",
        ["id", "seed", "validation_score", "configuration"],
        [
            [
                r["config"]["id"],
                r["seed"],
                r["validation_score"],
                json.dumps(r["config"]),
            ]
            for r in candidates
        ],
    )
    ci = ev["bootstrap"]["MLP_RMSE_reduction_percent_CI95"]
    sheet(
        book,
        "Paired bootstrap",
        ["target", "RMSE_reduction_percent", "CI95_low", "CI95_high"],
        [
            [
                q,
                100
                * (1 - ev["main"]["MLP"][q]["RMSE"] / ev["main"]["Kriging"][q]["RMSE"]),
                *ci[i],
            ]
            for i, q in enumerate(["CL", "CD", "L_D"])
        ],
    )
    sheet(
        book,
        "Timing",
        [
            "measurement",
            "device",
            "n_predictions",
            "median_seconds",
            "q25_seconds",
            "q75_seconds",
            "p95_seconds",
        ],
        [
            [
                k,
                *[
                    v[field]
                    for field in [
                        "device",
                        "n_predictions",
                        "median_seconds",
                        "q25_seconds",
                        "q75_seconds",
                        "p95_seconds",
                    ]
                ],
            ]
            for k, v in ev["timing"].items()
        ],
    )
    sheet(
        book,
        "Definitions",
        ["item", "definition"],
        [
            [
                "source",
                "results/evaluation.json and results/main_predictions.npz in the selected experiment",
            ],
            [
                "E2_percent",
                "50 * (RMSE_CL/std_test_CL + RMSE_CD/std_test_CD); population standard deviations",
            ],
            ["L_D", "CL/CD; derived from predicted coefficients"],
            [
                "CI95",
                "Paired test-sample percentile bootstrap with fixed trained models",
            ],
            [
                "test scope",
                "Independent LHS pool, same design domain and flow condition; prior research exposure",
            ],
            [
                "timing",
                "Warm inference on the recorded device; excludes model loading and CFD",
            ],
        ],
    )
    target = ROOT / "results/benchmark_results.xlsx"
    temporary = target.with_name("benchmark_results.tmp.xlsx")
    book.save(temporary)
    temporary.replace(target)
    print(target)


if __name__ == "__main__":
    main()
