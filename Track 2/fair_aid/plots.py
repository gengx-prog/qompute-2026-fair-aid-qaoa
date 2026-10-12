"""Editable, source-backed Matplotlib figures for the six-project experiment."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

BLUE = "#245C91"
ORANGE = "#C76C22"
GREY = "#9CA5AF"
INK = "#263445"


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 15, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GREY, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "grid.color": "#E2E7EC", "grid.linewidth": 0.7,
        "savefig.facecolor": "white", "savefig.dpi": 180,
    })


def plot_tradeoff(exact_rows: list[dict]):
    """Show all feasible allocations once, and label every exact optimum."""
    set_style()
    unique = {row["selected"]: row for row in exact_rows}
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    ax.scatter([float(r["equity_points"]) for r in unique.values()],
               [float(r["net_benefit"]) for r in unique.values()],
               s=62, color="#C5CDD5", edgecolors="white", linewidths=0.8,
               label=f"All {len(unique)} feasible allocations", zorder=2)
    optima: dict[str, list[dict]] = {}
    for row in exact_rows:
        if is_true(row["is_optimal"]):
            optima.setdefault(row["selected"], []).append(row)
    ordered = sorted(optima.items(), key=lambda item: float(item[1][0]["equity_points"]))
    for idx, (selected, rows) in enumerate(ordered):
        row = rows[0]
        x, y = float(row["equity_points"]), float(row["net_benefit"])
        weights = sorted({float(r["weight"]) for r in rows})
        ax.scatter(x, y, marker="*", s=240, color=BLUE, edgecolors="white",
                   linewidths=0.9, zorder=3,
                   label="Exact optimum for a tested weight" if idx == 0 else None)
        label = f"{selected.replace(',', '+')}\nw = {', '.join(f'{w:g}' for w in weights)}"
        dx, dy = (12, 12) if idx % 2 == 0 else (-12, -30)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="bottom", fontsize=10,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9})
    ax.set(title="The objective weight changes which projects are selected",
           xlabel="Equity proxy (synthetic points)",
           ylabel="Net benefit after pair overlap (synthetic points)")
    ax.grid(alpha=0.8, zorder=0)
    ax.margins(x=0.16, y=0.2)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(0.08, 0.035,
             "Exactly 3 of 6 equal-cost projects. Utility = net benefit + w × equity proxy.\n"
             "Source: exact_solutions.csv; synthetic teaching data. Equity points are not a fairness guarantee.",
             fontsize=9, color="#556270")
    fig.subplots_adjust(left=0.1, right=0.96, top=0.88, bottom=0.2)
    return fig


def plot_probabilities(summary_rows: list[dict], exact_rows: list[dict]):
    """Compare exact final-state probabilities and seed spread to random baselines."""
    set_style()
    weights = sorted({float(r["weight"]) for r in summary_rows})
    depths = sorted({int(r["depth"]) for r in summary_rows})
    seed_counts = {len({int(r["seed"]) for r in summary_rows
                        if float(r["weight"]) == w and int(r["depth"]) == p})
                   for w in weights for p in depths}
    colors = [BLUE, ORANGE, "#667C36", "#AE527A"]
    markers = ["o", "s", "^", "D"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.7))
    x = np.array(weights, dtype=float)
    minimum_step = float(np.min(np.diff(x))) if len(x) > 1 else 1.0
    for ax, field, title in zip(axes, ["feasible_probability", "optimal_probability"],
                                ["A sampled allocation meets the budget", "A sampled allocation is globally optimal"]):
        for idx, depth in enumerate(depths):
            arrays = [np.array([float(r[field]) for r in summary_rows
                                if float(r["weight"]) == weight and int(r["depth"]) == depth])
                      for weight in weights]
            means = np.array([values.mean() for values in arrays])
            lower = means - np.array([values.min() for values in arrays])
            upper = np.array([values.max() for values in arrays]) - means
            shift = (idx - (len(depths) - 1) / 2) * minimum_step * 0.12
            ax.errorbar(x + shift, means, yerr=np.stack([lower, upper]),
                        color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
                        capsize=4, linewidth=1.8, markersize=6,
                        label=f"QAOA p = {depth}", zorder=3)
        if field == "feasible_probability":
            # The experiment fixes six decisions and a cardinality of three.
            n_feasible = len([r for r in exact_rows if float(r["weight"]) == weights[0]])
            ax.axhline(n_feasible / 64, color=INK, linestyle=":", linewidth=1.4,
                       label=f"Uniform over 64: {n_feasible}/64")
            ax.axhline(1.0, color="#66717D", linestyle="--", linewidth=1.1,
                       label="Uniform feasible: 100%")
        else:
            exact_by_weight = [[r for r in exact_rows if float(r["weight"]) == w]
                               for w in weights]
            numbers = np.array([sum(is_true(r["is_optimal"]) for r in rows)
                                for rows in exact_by_weight])
            feasible_numbers = np.array([len(rows) for rows in exact_by_weight])
            ax.plot(x, numbers / 64, color=INK, linestyle=":", linewidth=1.4,
                    label="Uniform over 64: optimum probability")
            ax.plot(x, numbers / feasible_numbers, color="#66717D", linestyle="--",
                    linewidth=1.3, label="Uniform feasible: optimum probability")
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xticks(x, [f"{w:g}" for w in weights])
        ax.set_xlabel("Equity weight w")
        if field == "feasible_probability":
            ax.set_ylim(0, 1.06)
        else:
            largest = max(max(float(row[field]) for row in summary_rows),
                          float(np.max(numbers / feasible_numbers)))
            ax.set_ylim(0, min(1.06, max(0.10, largest * 1.2)))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.85)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.21), frameon=False, fontsize=9)
    axes[0].set_ylabel("Feasible probability per draw")
    axes[1].set_ylabel("Optimal probability per draw")
    figure.suptitle("QAOA performance across objective weights", fontsize=16, y=0.98)
    seeds_label = str(next(iter(seed_counts))) if len(seed_counts) == 1 else "varying"
    figure.text(0.08, 0.015,
                f"Points: mean over {seeds_label} optimization seeds; whiskers: minimum–maximum, not confidence intervals.\n"
                "Exact statevector probabilities; vertical scales differ. Source: summary.csv and exact_solutions.csv.",
                fontsize=9, color="#556270")
    figure.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.35, wspace=0.25)
    return figure


def plot_penalty_sweep(rows: list[dict]):
    """Show feasibility of the exact penalized ground state, not QAOA samples."""
    set_style()
    weights = sorted({float(row["weight"]) for row in rows})
    penalties = sorted({float(row["penalty"]) for row in rows})
    values = np.full((len(weights), len(penalties)), np.nan)
    bounds = {}
    for row in rows:
        i, j = weights.index(float(row["weight"])), penalties.index(float(row["penalty"]))
        values[i, j] = float(is_true(row["ground_feasible"]))
        bounds[float(row["weight"])] = float(row["bound"])
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ax.imshow(values, vmin=0, vmax=1, cmap=ListedColormap(["#E9EDF1", "#3D6D99"]), aspect="auto")
    for i in range(len(weights)):
        for j in range(len(penalties)):
            if not np.isnan(values[i, j]):
                ax.text(j, i, "Feasible" if values[i, j] else "Infeasible", ha="center", va="center",
                        color="white" if values[i, j] else INK, fontsize=10)
    ax.set_xticks(np.arange(len(penalties)), [f"{p:g}" for p in penalties])
    ax.set_yticks(np.arange(len(weights)), [f"w = {w:g}  (bound {bounds[w]:g})" for w in weights])
    ax.set(title="A penalty must make the exact ground state feasible",
           xlabel="Budget-constraint penalty A (synthetic points)")
    ax.tick_params(length=0, pad=8)
    fig.text(0.05, 0.035,
             "Exact enumeration of all 64 states; feasibility requires exactly 3 projects.\n"
             "A strictly above the stated bound is sufficient; it need not be necessary. Source: penalty_sweep.csv.",
             fontsize=9, color="#556270")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.82, bottom=0.25)
    return fig


def generate_figures(results_dir: Path) -> list[Path]:
    """Create PNG and SVG versions, return the three PNG paths in reading order."""
    results_dir = Path(results_dir)
    exact_rows = read_rows(results_dir / "exact_solutions.csv")
    summary_rows = read_rows(results_dir / "summary.csv")
    penalty_rows = read_rows(results_dir / "penalty_sweep.csv")
    chart_dir = results_dir / "figures"
    chart_dir.mkdir(parents=True, exist_ok=True)
    makers = [
        ("allocation_tradeoff", lambda: plot_tradeoff(exact_rows)),
        ("qaoa_probabilities", lambda: plot_probabilities(summary_rows, exact_rows)),
        ("penalty_feasibility", lambda: plot_penalty_sweep(penalty_rows)),
    ]
    paths = []
    for name, make in makers:
        figure = make()
        for suffix in ("png", "svg"):
            figure.savefig(chart_dir / f"{name}.{suffix}")
        plt.close(figure)
        paths.append(chart_dir / f"{name}.png")
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    arguments = parser.parse_args()
    for path in generate_figures(arguments.results_dir):
        print(path)
