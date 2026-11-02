"""Render publication figures from frozen results; source CFD files remain read-only."""

from pathlib import Path
import sys, math

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import ROOT, read_json, with_ratio

# Experiment directory is selected by common.ROOT.
OUT = ROOT / "report/figures"
PURPLE = "#9779B5"
TEAL = "#5C9EA8"
ORANGE = "#CF7B17"
GRAY = "#64748B"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.color": "#D9DEE7",
        "grid.alpha": 0.55,
        "grid.linewidth": 0.55,
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight")
    plt.close(fig)


def geometry(c):
    x = (1 - np.cos(np.linspace(0, np.pi, 400))) / 2
    B = np.array([math.comb(9, i) * x**i * (1 - x) ** (9 - i) for i in range(10)]).T
    C = np.sqrt(x) * (1 - x)
    return x, C * (B @ c[:10]), C * (B @ c[10:])


def diagram():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    boxes = [
        (0.1, 2.7, 2.4, 0.8, "LHS_10000\n10,000 training geometries"),
        (0.1, 0.3, 2.4, 0.8, "Independent LHS_2000\n500 validation + 1,500 test"),
        (
            3.3,
            2.7,
            2.8,
            0.8,
            "12 MLP candidates; confirm top 3\nSelected configuration; 3 seeds",
        ),
        (3.3, 1.3, 2.8, 0.8, "Exact Kriging\nIsotropic Matern 5/2"),
        (
            7,
            2.1,
            2.8,
            1.1,
            "Paired learning curves\n50 to 10,000 training samples\nMatched subsets and metrics",
        ),
        (
            7,
            0.25,
            2.8,
            1.05,
            "Evaluate frozen models\nAccuracy, robustness, cost\nGeometry and sensitivity",
        ),
    ]
    for i, (x, y, w, h, t) in enumerate(boxes):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.08",
                fc="#F4EDF7" if i != 3 else "#ECF7F8",
                ec=PURPLE if i != 3 else TEAL,
                lw=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    for start, end in [
        ((2.6, 3.1), (3.15, 3.1)),
        ((2.55, 2.8), (3.15, 1.7)),
        ((6.2, 3.05), (6.85, 2.9)),
        ((6.2, 1.7), (6.85, 2.35)),
        ((8.4, 2), (8.4, 1.45)),
        ((2.6, 0.7), (6.85, 0.7)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=15, color=GRAY, lw=1.2
            )
        )
    ax.text(
        5,
        0.03,
        "Validation selects models; test results do not tune them",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    save(fig, "01_protocol")


def architecture():
    c = read_json(ROOT / "results/mlp_selection.json")["selected"]["config"]
    hidden = c["hidden"]
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ax.axis("off")
    ax.set(xlim=(-0.25, 12.4), ylim=(-0.45, 3))
    xs = np.linspace(0, 6.4, len(hidden) + 2)
    texts = (
        ["20 CST inputs\nStandardized"]
        + [f'{v} neurons\n{c["activation"]}' for v in hidden]
        + ["Nonlinear\n2 outputs"]
    )
    for i, (x, t) in enumerate(zip(xs, texts)):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.65),
                1.1,
                0.95,
                boxstyle="round,pad=.08",
                fc="#F4EDF7",
                ec=PURPLE,
                lw=1.3,
            )
        )
        ax.text(x + 0.55, 1.125, t, ha="center", va="center", fontsize=9.5)
        if i:
            ax.add_patch(
                FancyArrowPatch(
                    (xs[i - 1] + 1.2, 1.125),
                    (x - 0.1, 1.125),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    color=GRAY,
                )
            )
    ax.text(
        8.15,
        1.13,
        "+",
        ha="center",
        va="center",
        fontsize=21,
        bbox=dict(boxstyle="circle,pad=.15", fc="white", ec=PURPLE),
    )
    ax.add_patch(
        FancyArrowPatch(
            (7.6, 1.125), (7.9, 1.125), arrowstyle="-|>", mutation_scale=13, color=GRAY
        )
    )
    ax.plot([0.55, 0.55, 8.15, 8.15], [1.73, 2.4, 2.4, 1.55], color=PURPLE, lw=1.2)
    ax.annotate(
        "",
        xy=(8.15, 1.45),
        xytext=(8.15, 1.8),
        arrowprops=dict(arrowstyle="-|>", color=PURPLE),
    )
    ax.text(
        4.3,
        2.48,
        "Trainable linear branch $Bz$ (ridge initialization on training data)",
        ha="center",
        va="bottom",
        fontsize=10,
        color=PURPLE,
    )
    ax.add_patch(
        FancyArrowPatch(
            (8.4, 1.125), (8.9, 1.125), arrowstyle="-|>", mutation_scale=13, color=GRAY
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (9, 0.65),
            1.2,
            0.95,
            boxstyle="round,pad=.08",
            fc="#ECF7F8",
            ec=TEAL,
            lw=1.3,
        )
    )
    ax.text(
        9.6,
        1.125,
        "Physical scale\n$C_L$, $C_D$",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.add_patch(
        FancyArrowPatch(
            (10.3, 1.125),
            (10.8, 1.125),
            arrowstyle="-|>",
            mutation_scale=13,
            color=GRAY,
        )
    )
    ax.text(
        11.5,
        1.125,
        "Derived ratio\n$K=C_L/C_D$",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=.35", fc="white", ec=GRAY),
    )
    ax.text(
        4.2,
        0.1,
        "Shared features predict lift and drag; lift-to-drag ratio is derived",
        ha="center",
        fontsize=10,
        color=GRAY,
    )
    save(fig, "13_architecture")


def main():
    ev = read_json(ROOT / "results/evaluation.json")
    d = np.load(ROOT / "results/main_predictions.npz")
    y = with_ratio(d["truth"])
    pm = with_ratio(d["MLP"])
    pk = with_ratio(d["Kriging"])
    train = np.load(ROOT / "data/development.npz")
    agg = ev["aggregates"]
    labels = [r"$C_L$", r"$C_D$", r"$C_L/C_D$"]
    diagram()
    architecture()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.15), layout="constrained")
    for i in range(0, 10000, 500):
        x, u, l = geometry(train["X"][i])
        axes[0].plot(x, u, color=PURPLE, alpha=0.16, lw=0.65)
        axes[0].plot(x, l, color=PURPLE, alpha=0.16, lw=0.65)
    x, u, l = geometry(train["baseline_cst"])
    axes[0].plot(x, u, color="#111827", lw=1.1)
    axes[0].plot(x, l, color="#111827", lw=1.1, label="Baseline RAE2822")
    axes[0].set(xlabel="$x/c$", ylabel="$y/c$", title="Airfoil design space")
    axes[0].legend(fontsize=8)
    for j in range(2):
        axes[j + 1].hist(
            train["Y"][:, j],
            bins=40,
            density=True,
            color=PURPLE,
            alpha=0.55,
            label="Training pool: 10,000",
        )
        axes[j + 1].hist(
            d["truth"][:, j],
            bins=30,
            density=True,
            histtype="step",
            color=TEAL,
            lw=1.4,
            label="Test pool: 1,500",
        )
        axes[j + 1].set(
            xlabel=labels[j], ylabel="Density", title="Independent samples, same domain"
        )
    axes[2].legend(fontsize=8)
    save(fig, "02_dataset")
    candidates = [
        read_json(p) for p in sorted((ROOT / "models/screen").glob("*/result.json"))
    ]
    fig, ax = plt.subplots(figsize=(9.8, 4))
    names = [r["config"]["id"].split("_")[0] for r in candidates]
    vals = [100 * r["validation_score"] for r in candidates]
    colors = [
        (
            PURPLE
            if r["config"]["id"] == ev["protocol"].get("selected_id", "")
            else "#BFA3CD"
        )
        for r in candidates
    ]
    winner = read_json(ROOT / "results/mlp_selection.json")["selected"]["config"]["id"]
    colors = [PURPLE if r["config"]["id"] == winner else "#BFA3CD" for r in candidates]
    ax.bar(names, vals, color=colors, width=0.64)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set(
        ylabel="Validation score $S$ (%)",
        xlabel="Configuration ID",
        title="MLP screening on 10,000 samples (seed 11)",
    )
    ax.set_ylim(0, max(vals) * 1.18)
    save(fig, "03_mlp_screen")
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    for family, color, marker in [("MLP", PURPLE, "o"), ("Kriging", TEAL, "s")]:
        rows = [r for r in agg if r["family"] == family]
        n = np.array([r["n_train"] for r in rows])
        mu = np.array([r["E2_mean_percent"] for r in rows])
        sd = np.array([r["E2_std_percent"] for r in rows])
        ax.plot(n, mu, color=color, marker=marker, lw=2, label=family)
        ax.fill_between(n, np.maximum(mu - sd, 1e-8), mu + sd, color=color, alpha=0.14)
    for crossing in ev["crossings"]:
        lo, hi = crossing["bracket"]
        ax.axvspan(lo, hi, color=ORANGE, alpha=0.12, label="Observed crossing interval")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Training geometries $n$ (+500 validation labels)",
        ylabel="Normalized two-output error $E_2$ (%)",
        title="Sample efficiency: three paired nested subsets",
    )
    ax.set_xticks(
        ev["protocol"]["budgets"], [str(n) for n in ev["protocol"]["budgets"]]
    )
    ax.legend()
    save(fig, "04_learning_E2")
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45), layout="constrained")
    for j, target in enumerate(["CL", "CD", "L_D"]):
        for family, color, marker in [("MLP", PURPLE, "o"), ("Kriging", TEAL, "s")]:
            rows = [r for r in agg if r["family"] == family]
            n = np.array([r["n_train"] for r in rows])
            v = np.array([r[f"{target}_MAPE_percent_mean"] for r in rows])
            sd = np.array([r[f"{target}_MAPE_percent_std"] for r in rows])
            axes[j].plot(n, v, marker=marker, color=color, label=family, ms=4, lw=1.7)
            axes[j].fill_between(
                n, np.maximum(v - sd, 1e-9), v + sd, color=color, alpha=0.12
            )
        axes[j].set(
            xscale="log",
            yscale="log",
            title=labels[j],
            xlabel="Training geometries $n$",
            ylabel="MAPE (%)",
        )
    axes[0].legend()
    save(fig, "05_learning_targets")
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 6.2), layout="constrained")
    for i, (family, p, color) in enumerate(
        [("MLP", pm, PURPLE), ("Kriging", pk, TEAL)]
    ):
        for j, target in enumerate(["CL", "CD", "L_D"]):
            ax = axes[i, j]
            ax.scatter(y[:, j], p[:, j], s=5, color=color, alpha=0.25, rasterized=True)
            low = min(y[:, j].min(), p[:, j].min())
            hi = max(y[:, j].max(), p[:, j].max())
            ax.plot([low, hi], [low, hi], color="#475569", lw=0.8)
            met = ev["main"][family][target]
            ax.set(
                xlabel="CFD " + labels[j],
                ylabel="Predicted " + labels[j],
                title=f'{family} | MAPE={met["MAPE_percent"]:.3f}%',
            )
    save(fig, "06_parity")
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.35), layout="constrained")
    for j in range(3):
        for family, p, color in [("MLP", pm, PURPLE), ("Kriging", pk, TEAL)]:
            ape = np.sort(abs((p[:, j] - y[:, j]) / y[:, j]) * 100)
            axes[j].plot(
                ape,
                np.arange(1, len(ape) + 1) / len(ape),
                color=color,
                lw=1.7,
                label=family,
            )
        axes[j].set(
            xscale="log",
            xlabel="Absolute percentage error (%)",
            ylabel="Cumulative fraction",
            title=labels[j],
            ylim=(0, 1.01),
        )
    axes[0].legend()
    save(fig, "07_error_cdf")
    q = np.load(ROOT / "results/test_geometry_quality.npz")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained")
    for ax, key, label in [
        (axes[0], "maximum_thickness", "Maximum relative thickness"),
        (
            axes[1],
            "nearest_training_distance",
            "Nearest training distance (domain normalized)",
        ),
    ]:
        bins = np.quantile(q[key], np.linspace(0, 1, 6))
        group = np.clip(np.digitize(q[key], bins) - 1, 0, 4)
        for family, p, color in [("MLP", pm, PURPLE), ("Kriging", pk, TEAL)]:
            ape = 100 * abs((p[:, 1] - y[:, 1]) / y[:, 1])
            yy = [np.mean(ape[group == b]) for b in range(5)]
            xx = [np.mean(q[key][group == b]) for b in range(5)]
            ax.plot(xx, yy, "o-", color=color, label=family)
        ax.set(
            xlabel=label,
            ylabel="$C_D$ MAPE (%)",
            title="Test error in five equal-count bins",
        )
    axes[0].legend()
    save(fig, "08_geometry_error")
    worst = np.argsort(abs((pm[:, 1] - y[:, 1]) / y[:, 1]))[-3:][::-1]
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 6.6), layout="constrained")
    for ax, ix in zip(axes, worst):
        x, u, l = geometry(d["X"][ix])
        xb, ub, lb = geometry(train["baseline_cst"])
        ax.plot(xb, ub, color=GRAY, ls="--", lw=1)
        ax.plot(xb, lb, color=GRAY, ls="--", lw=1, label="Baseline")
        ax.plot(x, u, color=PURPLE, lw=1.5)
        ax.plot(x, l, color=PURPLE, lw=1.5, label="Test airfoil")
        em = 100 * abs((pm[ix, 1] - y[ix, 1]) / y[ix, 1])
        ek = 100 * abs((pk[ix, 1] - y[ix, 1]) / y[ix, 1])
        ax.set(
            xlabel="$x/c$",
            ylabel="$y/c$",
            ylim=(-0.075, 0.095),
            title=f'LHS_2000:{d["ids"][ix]} | Drag APE: MLP {em:.3f}% / Kriging {ek:.3f}%',
        )
    axes[0].legend(loc="lower right", ncol=2)
    save(fig, "09_hard_airfoils")
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.5), layout="constrained")
    sen = ev["learned_sensitivity"]
    fraction = np.array(sen["eigenvalue_fraction"])
    axes[0].plot(
        range(1, 21),
        100 * np.cumsum(fraction),
        "o-",
        color=PURPLE,
        ms=4,
        label="MLP response gradients",
    )
    axes[0].plot(
        range(1, 21),
        100 * np.cumsum(sen["geometry_metric_eigenvalue_fraction"]),
        "s-",
        color=TEAL,
        ms=3,
        label="Discrete CST geometry metric",
    )
    axes[0].plot(
        range(1, 21),
        np.arange(1, 21) * 5,
        "--",
        color=GRAY,
        lw=1,
        label="Euclidean identity metric",
    )
    axes[0].axhline(95, color=GRAY, ls=":", lw=0.8)
    axes[0].set(
        xlabel="Number of directions",
        ylabel="Cumulative spectral energy (%)",
        title="Directional concentration",
        xticks=[1, 5, 10, 15, 20],
        ylim=(0, 102),
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].bar(
        range(20),
        100 * np.array(sen["axis_energy_fraction"]),
        color=[PURPLE] * 10 + [TEAL] * 10,
    )
    axes[1].set(
        xticks=range(20),
        xticklabels=ev["protocol"]["features"],
        ylabel="Coordinate gradient energy (%)",
        title="Standardized CST sensitivity",
    )
    axes[1].tick_params(axis="x", rotation=60, labelsize=8)
    save(fig, "10_sensitivity")
    main = next(
        r
        for r in ev["records"]
        if r["family"] == "MLP" and r["n_train"] == 10000 and r["seed"] == 11
    )
    tr = read_json(ROOT / f"models/learning/{winner}_n10000_s11/result.json")["trace"]
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.5), layout="constrained")
    for ax, stage, label in [
        (axes[0], "AdamW", "AdamW epochs"),
        (axes[1], "LBFGS", "L-BFGS outer calls (up to 5 steps each)"),
    ]:
        rr = [r for r in tr if r["stage"] == stage]
        if rr:
            ax.plot(
                [r["step"] for r in rr],
                [100 * r["validation_score"] for r in rr],
                color=PURPLE,
                lw=1.8,
            )
        ax.set(
            xlabel=label, ylabel="Validation score $S$ (%)", title=stage, yscale="log"
        )
    save(fig, "11_optimization")
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), layout="constrained")
    for ax, kind, label in [
        (axes[0], "single", "Single-airfoil latency (ms)"),
        (axes[1], "batch", "1,500-airfoil batch time (ms)"),
    ]:
        vals = [
            1000 * ev["timing"][f"{f}_{kind}"]["median_seconds"]
            for f in ["MLP", "Kriging"]
        ]
        bars = ax.bar(["MLP", "Kriging"], vals, color=[PURPLE, TEAL], width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.3f}", ha="center"
            )
        ax.set(
            ylabel=label,
            yscale="log",
            title=f"{ev['timing']['MLP_batch']['device']}; loaded models; warm median",
        )
        ax.set_ylim(min(vals) / 2, max(vals) * 2.4)
    save(fig, "12_inference_cost")
    from plot_relative_error_learning import draw_relative_error_learning

    draw_relative_error_learning(ROOT)
    print("Generated 15 figure sets in PDF and PNG.")


if __name__ == "__main__":
    main()
