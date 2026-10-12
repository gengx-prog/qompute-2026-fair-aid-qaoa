"""Editable benchmark and policy-stress figures for the Fair Aid notebook.

Run ``python plot_comparison.py`` after ``benchmark.py``. All plotted values
come from the comparison CSV files; no example observations are inserted.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from plots import BLUE, GREY, INK, ORANGE, read_rows, set_style


METHODS = (
    ("x", 1, "X mixer, p = 1", BLUE, "o", False),
    ("x", 2, "X mixer, p = 2", BLUE, "s", True),
    ("xy", 1, "XY mixer, p = 1", ORANGE, "^", False),
    ("xy", 2, "XY mixer, p = 2", ORANGE, "D", True),
)


def _method_name(method: str) -> str:
    """Keep the recorded baseline identity, with readable separators."""
    return str(method).replace("_", " ").replace("+", " + ")


def _finite(value, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Non-finite {field} cannot be plotted: {value!r}")
    return number


def plot_mixer_comparison(summary_rows: list[dict], classical_rows: list[dict]):
    """Compare four QAOA settings; whiskers show seed min/max, not a CI.

    Probability panels show the exact final-state probability per measurement,
    rather than best-of-many-shot success. Classical deterministic optimality
    is stated in the figure note so that it cannot force a misleading common
    vertical scale on the probability panels. Its regret is plotted directly.
    """
    if not summary_rows:
        raise ValueError("summary.csv contains no benchmark runs")
    set_style()
    weights = sorted({float(row["weight"]) for row in summary_rows})
    x = np.array(weights, dtype=float)
    step = float(np.min(np.diff(x))) if len(x) > 1 else 1.0
    grouped = {}
    seen = set()
    for row in summary_rows:
        key = (float(row["weight"]), str(row["mixer"]).lower(), int(row["depth"]))
        run_key = (*key, int(row["seed"]))
        if run_key in seen:
            raise ValueError(f"Duplicate benchmark initialization: {run_key}")
        seen.add(run_key)
        grouped.setdefault(key, []).append(row)
    for weight in weights:
        for mixer, depth, *_ in METHODS:
            if (weight, mixer, depth) not in grouped:
                raise ValueError(f"Missing benchmark group: {(weight, mixer, depth)}")

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.5))
    specifications = (
        ("optimal_probability", "Probability of the exact optimum", "Optimal probability per draw"),
        ("feasible_probability", "Probability of meeting the budget", "Feasible probability per draw"),
        ("conditional_regret", "Utility loss among feasible measurements", "Conditional regret (utility points)"),
        ("two_qubit_gates", "Circuit cost including initial-state preparation", "Two-qubit gates (compiled count)"),
    )
    method_handles = []
    for idx, (mixer, depth, label, color, marker, filled) in enumerate(METHODS):
        method_handles.append(Line2D(
            [], [], color=color, marker=marker, linestyle="none", markersize=7,
            markerfacecolor=color if filled else "white", markeredgewidth=1.5,
            label=label,
        ))
        positions = x + (idx - 1.5) * step * 0.14
        for ax, (field, title, ylabel) in zip(axes.flat, specifications):
            arrays = [np.array([_finite(row[field], field) for row in grouped[(w, mixer, depth)]])
                      for w in weights]
            means = np.array([values.mean() for values in arrays])
            lower = np.maximum(0, means - np.array([values.min() for values in arrays]))
            upper = np.maximum(0, np.array([values.max() for values in arrays]) - means)
            ax.errorbar(
                positions, means, yerr=np.stack([lower, upper]),
                color=color, marker=marker, linestyle="none", markersize=6.5,
                markerfacecolor=color if filled else "white", markeredgewidth=1.4,
                capsize=3, elinewidth=1.4, zorder=3,
            )
            ax.set_title(title, fontsize=12, pad=12)
            ax.set_ylabel(ylabel)

    optimum_ax, feasible_ax, regret_ax, gates_ax = axes.flat
    random_optimum = [
        _finite(grouped[(w, "x", 1)][0].get("uniform_feasible_optimal_probability", 0.05),
                "uniform_feasible_optimal_probability")
        for w in weights
    ]
    optimum_ax.plot(x, random_optimum, color=INK, linewidth=1.4, linestyle="--",
                    label="Uniform feasible sampling", zorder=2)
    feasible_ax.axhline(1.0, color=INK, linewidth=1.4, linestyle="--",
                       label="Uniform feasible sampling: 100%", zorder=2)
    optimum_ax.legend(loc="upper left", fontsize=9, frameon=False)
    feasible_ax.legend(loc="lower left", fontsize=9, frameon=False)
    optimum_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    feasible_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    largest_optimum = max(max(_finite(row["optimal_probability"], "optimal_probability")
                              for row in summary_rows), max(random_optimum))
    optimum_ax.set_ylim(0, min(1.08, max(0.10, largest_optimum * 1.22)))
    feasible_ax.set_ylim(0, 1.09)

    classical_methods = sorted({str(row["method"]) for row in classical_rows})
    classical_summaries = {}
    for index, method in enumerate(classical_methods):
        rows = [row for row in classical_rows if str(row["method"]) == method]
        by_weight = {float(row["weight"]): row for row in rows}
        if len(by_weight) != len(rows):
            raise ValueError(f"Duplicate classical baseline weight for {method}")
        if set(weights) - set(by_weight):
            raise ValueError(f"Classical baseline {method} is missing weights")
        linestyle = ("--", ":", "-.")[index % 3]
        regret_ax.plot(x, [_finite(by_weight[w]["regret"], "regret") for w in weights],
                       color=INK, linestyle=linestyle, marker="x", markersize=5,
                       linewidth=1.2, label=_method_name(method), zorder=2)
        optimal = [_finite(by_weight[w]["optimal_probability"], "optimal_probability")
                   for w in weights]
        feasible = [_finite(by_weight[w]["feasible_probability"], "feasible_probability")
                    for w in weights]
        optimal_text = (f"{optimal[0]:.0%}" if len(set(optimal)) == 1
                        else "/".join(f"{value:.0%}" for value in optimal))
        feasible_text = (f"{feasible[0]:.0%}" if len(set(feasible)) == 1
                         else "/".join(f"{value:.0%}" for value in feasible))
        # Merge genuinely identical probability summaries, never average the
        # performance of distinct classical methods into a new baseline.
        classical_summaries.setdefault((optimal_text, feasible_text), []).append(_method_name(method))
    if classical_methods:
        regret_ax.legend(loc="upper left", fontsize=9, frameon=False)
    regret_ax.set_ylim(bottom=-0.025 * max(1.0, regret_ax.get_ylim()[1]))
    gates_ax.set_ylim(bottom=0)
    gates_ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    for ax in axes.flat:
        ax.set_xticks(x, [f"{weight:g}" for weight in weights])
        ax.set_xlabel("Equity weight w")
        ax.set_xlim(x[0] - step * 0.45, x[-1] + step * 0.45)
        ax.grid(axis="y", alpha=0.85, zorder=0)

    seed_counts = {len(rows) for rows in grouped.values()}
    seed_text = (str(next(iter(seed_counts))) if len(seed_counts) == 1
                 else f"{min(seed_counts)}–{max(seed_counts)}")
    figure.suptitle("Budget-preserving XY mixer and standard X mixer", fontsize=17, y=0.98)
    figure.legend(handles=method_handles, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, 0.94), frameon=False, fontsize=10)
    notes = [
        f"Points: mean of {seed_text} optimization seeds; whiskers: minimum–maximum, not confidence intervals. Panel scales differ.",
        "Exact ideal statevector probabilities; gate counts include state preparation. No quantum speedup is established.",
    ]
    if classical_summaries:
        notes.append("Classical probabilities (values follow ascending w when not constant):")
        for (optimal_text, feasible_text), names in classical_summaries.items():
            notes.append(f"{' and '.join(names)}: optimum {optimal_text}; feasible {feasible_text}.")
    notes.append("Sources: comparison/summary.csv and classical_baselines.csv. Six synthetic projects; select exactly three.")
    wrapped_notes = [line for note in notes for line in textwrap.wrap(note, width=110)]
    figure.text(0.075, 0.024, "\n".join(wrapped_notes), fontsize=8.7,
                color="#556270", linespacing=1.5)
    note_height = len(wrapped_notes) * 8.7 * 1.5 / (72 * figure.get_figheight())
    figure.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=max(0.20, 0.11 + note_height),
                           hspace=0.47, wspace=0.25)
    return figure


def plot_score_stability(stability_rows: list[dict], selection_rows: list[dict]):
    """Visualize exact classical re-optimization under chosen score scenarios."""
    if not stability_rows or not selection_rows:
        raise ValueError("Score-stability CSV files must contain observations")
    set_style()
    ordered = sorted(stability_rows, key=lambda row: float(row["weight"]))
    weights = [float(row["weight"]) for row in ordered]
    if len(set(weights)) != len(weights):
        raise ValueError("policy_stability.csv must have one row per weight")
    projects = sorted({str(row["project"]) for row in selection_rows})
    rates = np.array([_finite(row["baseline_still_optimal_rate"], "baseline_still_optimal_rate")
                      for row in ordered])
    matrix = np.full((len(weights), len(projects)), np.nan)
    for row in selection_rows:
        weight, project = float(row["weight"]), str(row["project"])
        if weight not in weights:
            raise ValueError(f"Selection stability has an unknown weight: {weight}")
        i, j = weights.index(weight), projects.index(project)
        if not np.isnan(matrix[i, j]):
            raise ValueError(f"Duplicate selection frequency: {(weight, project)}")
        matrix[i, j] = _finite(row["inclusion_frequency"], "inclusion_frequency")
    if np.isnan(matrix).any():
        raise ValueError("selection_stability.csv is missing weight/project combinations")
    if np.any((matrix < 0) | (matrix > 1)) or np.any((rates < 0) | (rates > 1)):
        raise ValueError("Stability frequencies must be between zero and one")

    figure, (bar_ax, heat_ax) = plt.subplots(1, 2, figsize=(13.2, 6.6),
                                          gridspec_kw={"width_ratios": [1, 1.28]})
    positions = np.arange(len(weights))
    bars = bar_ax.bar(positions, rates, width=0.6, color=BLUE, zorder=3)
    bar_ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=5, fontsize=11)
    selections = [str(row["baseline_selected"]).replace(",", "+") for row in ordered]
    bar_ax.set_xticks(positions, [f"w = {w:g}\n{selected}"
                                 for w, selected in zip(weights, selections)])
    bar_ax.set_ylim(0, 1.14)
    bar_ax.set_yticks(np.arange(0, 1.01, 0.2))
    bar_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    bar_ax.set_title("Original allocation remains optimal", fontsize=12, pad=15)
    bar_ax.set_ylabel("Fraction of stress scenarios")
    bar_ax.set_xlabel("Weight and original optimal project set", labelpad=12)
    bar_ax.grid(axis="y", alpha=0.85, zorder=0)

    heat = heat_ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(len(weights)):
        for j in range(len(projects)):
            heat_ax.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center",
                         color="white" if matrix[i, j] >= 0.70 else INK, fontsize=10)
    heat_ax.set_xticks(np.arange(len(projects)), projects)
    heat_ax.set_yticks(np.arange(len(weights)), [f"w = {w:g}" for w in weights])
    heat_ax.set_xlabel("Project selected after re-optimization", labelpad=12)
    heat_ax.set_title("Project inclusion frequency", fontsize=12, pad=15)
    heat_ax.tick_params(length=0, pad=8)
    colorbar = figure.colorbar(heat, ax=heat_ax, fraction=0.05, pad=0.04,
                              ticks=[0, 0.25, 0.5, 0.75, 1.0])
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label("Fraction of stress scenarios", labelpad=10)

    scenarios = sorted({int(row["scenarios"]) for row in ordered})
    perturbations = sorted({float(row["relative_perturbation"]) for row in ordered})
    scenario_text = "/".join(str(value) for value in scenarios)
    perturbation_text = "/".join(f"{value:.0%}" for value in perturbations)
    figure.suptitle("How sensitive is the allocation to its synthetic scores?", fontsize=17, y=0.97)
    figure.text(0.075, 0.035,
                f"{scenario_text} chosen stress scenarios per weight; relative score-perturbation setting: {perturbation_text}.\n"
                "Each scenario is solved by exact classical enumeration. These are scenario frequencies, not confidence intervals or real-world probabilities.\n"
                "Source: comparison/policy_stability.csv and selection_stability.csv. Inclusion frequency does not measure beneficiary coverage.",
                fontsize=8.8, color="#556270", linespacing=1.7)
    figure.subplots_adjust(left=0.075, right=0.94, top=0.83, bottom=0.27, wspace=0.38)
    return figure


def generate_comparison_figures(results_dir: Path) -> list[Path]:
    """Export both comparison figures as PNG/SVG and return their PNG paths.

    ``results_dir`` is the benchmark output directory containing summary.csv,
    classical_baselines.csv, policy_stability.csv, and selection_stability.csv.
    """
    results_dir = Path(results_dir)
    summary = read_rows(results_dir / "summary.csv")
    classical = read_rows(results_dir / "classical_baselines.csv")
    stability = read_rows(results_dir / "policy_stability.csv")
    selections = read_rows(results_dir / "selection_stability.csv")
    chart_dir = results_dir / "figures"
    chart_dir.mkdir(parents=True, exist_ok=True)
    makers = (
        ("mixer_comparison", lambda: plot_mixer_comparison(summary, classical)),
        ("score_stability", lambda: plot_score_stability(stability, selections)),
    )
    paths = []
    for name, make in makers:
        figure = make()
        for extension in ("png", "svg"):
            figure.savefig(chart_dir / f"{name}.{extension}")
        plt.close(figure)
        paths.append(chart_dir / f"{name}.png")
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "comparison")
    arguments = parser.parse_args()
    for path in generate_comparison_figures(arguments.results_dir):
        print(path)
