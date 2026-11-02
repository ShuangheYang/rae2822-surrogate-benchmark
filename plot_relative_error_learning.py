"""Plot separate CL and CD MAPE learning curves from frozen evaluation records.
Only results/evaluation.json is read. No model is loaded or trained.
MAPE fields are already percentages and must not be multiplied by 100 again."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import ROOT

BUDGETS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
SEEDS = (11, 22, 33)
COLORS = {"MLP": "#9779B5", "Kriging": "#5C9EA8"}
OUTPUTS = {"CL": "14_learning_CL_MAPE", "CD": "15_learning_CD_MAPE"}


def verified_curves(evaluation: dict, target: str) -> dict:
    """Check every aggregate mean and sample standard deviation against all three seed records."""
    curves = {}
    for family in COLORS:
        means, deviations = [], []
        for budget in BUDGETS:
            records = [
                record
                for record in evaluation["records"]
                if record["family"] == family and record["n_train"] == budget
            ]
            seeds = sorted(record["seed"] for record in records)
            if seeds != list(SEEDS):
                raise ValueError(
                    f"{family}, n={budget}: Missing or duplicate seed records: {seeds}"
                )
            values = np.asarray(
                [record["test_metrics"][target]["MAPE_percent"] for record in records],
                dtype=float,
            )
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(
                    f"{family}, n={budget}: MAPE is invalid for the logarithmic axes"
                )

            summaries = [
                item
                for item in evaluation["aggregates"]
                if item["family"] == family and item["n_train"] == budget
            ]
            if len(summaries) != 1:
                raise ValueError(
                    f"{family}, n={budget}: Missing or duplicate aggregate record"
                )
            summary = summaries[0]
            mean = float(np.mean(values))
            deviation = float(np.std(values, ddof=1))
            if not np.isclose(
                mean, summary[f"{target}_MAPE_percent_mean"], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(f"{family}, n={budget}: MAPE mean mismatch")
            if not np.isclose(
                deviation, summary[f"{target}_MAPE_percent_std"], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f"{family}, n={budget}: MAPE standard deviation mismatch"
                )
            if mean - deviation <= 0:
                raise ValueError(
                    "Mean minus standard deviation is nonpositive; revise the axes instead of clipping the band"
                )
            means.append(mean)
            deviations.append(deviation)
        curves[family] = {"mean": np.asarray(means), "std": np.asarray(deviations)}
    return curves


def draw_relative_error_learning(root: Path = ROOT) -> list[Path]:
    """Plot separate lift and drag curves, connecting measured budgets without smoothing or estimating thresholds."""
    evaluation = json.loads(
        (root / "results/evaluation.json").read_text(encoding="utf-8")
    )
    output_dir = root / "report/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "legend.frameon": False,
    }
    generated = []
    with plt.rc_context(style):
        for target, title, symbol, ticks in [
            ("CL", "Lift coefficient", r"C_L", [0.02, 0.05, 0.1, 0.2, 0.5, 1]),
            ("CD", "Drag coefficient", r"C_D", [0.05, 0.1, 0.2, 0.5, 1, 2, 5]),
        ]:
            curves = verified_curves(evaluation, target)
            fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")

            # Mark a crossing bracket only where adjacent measured mean differences change sign.
            difference = curves["MLP"]["mean"] - curves["Kriging"]["mean"]
            crossing_indices = np.flatnonzero(difference[:-1] * difference[1:] < 0)
            for index in crossing_indices:
                ax.axvspan(
                    BUDGETS[index], BUDGETS[index + 1], color="#F7EBDD", zorder=0
                )

            for family, marker in [("MLP", "o"), ("Kriging", "s")]:
                mean, std = curves[family]["mean"], curves[family]["std"]
                ax.fill_between(
                    BUDGETS,
                    mean - std,
                    mean + std,
                    color=COLORS[family],
                    alpha=0.12,
                    linewidth=0,
                    zorder=2,
                )
                ax.plot(
                    BUDGETS,
                    mean,
                    marker=marker,
                    markersize=5.5,
                    linewidth=2.0,
                    color=COLORS[family],
                    label=family,
                    zorder=3,
                )
                # Endpoint labels average three seed records; they differ from the representative seed-11 model.
                ax.annotate(
                    f"{mean[-1]:.4f}%",
                    xy=(BUDGETS[-1], mean[-1]),
                    xytext=(-10, -20 if family == "MLP" else 12),
                    textcoords="offset points",
                    ha="right",
                    color=COLORS[family],
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set(
                xscale="log",
                yscale="log",
                xlim=(40, 14500),
                xlabel=r"Training geometries $n$ (+500 validation samples)",
                ylabel="Mean absolute percentage error (%)",
                title=f"{title} ${symbol}$: relative-error learning curve",
            )
            lower = min(np.min(c["mean"] - c["std"]) for c in curves.values())
            upper = max(np.max(c["mean"] + c["std"]) for c in curves.values())
            ax.set_ylim(lower * 0.62, upper * 1.48)
            # Use ordinary numeric labels on logarithmic axes; values remain percentages.
            ax.xaxis.set_major_locator(FixedLocator(BUDGETS))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(which="major", color="#E1E5EC", linewidth=0.7, zorder=1)

            handles, labels = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor="#B6A9C4", alpha=0.28, edgecolor="none"))
            labels.append("Mean +/- one sample standard deviation")
            if len(crossing_indices):
                handles.append(Patch(facecolor="#F7EBDD", edgecolor="none"))
                labels.append("Measured sign-change interval")
            ax.legend(handles, labels, loc="upper right", fontsize=9)
            for extension in ("pdf", "png"):
                path = output_dir / f"{OUTPUTS[target]}.{extension}"
                fig.savefig(path, bbox_inches="tight", facecolor="white")
                generated.append(path)
            plt.close(fig)
    return generated


if __name__ == "__main__":
    for output_path in draw_relative_error_learning():
        print(output_path)
    print(
        "Generated separate CL and CD MAPE curves without training or changing frozen results."
    )
