"""Compare QAOA variants and audit the policy on the same synthetic instance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from allocation import AllocationProblem, int_to_bits, load_problem
from run_experiments import HERE, run_suite, write_csv

WEIGHTS = (0.0, 0.5, 1.0, 2.0)
SEEDS = (11, 29, 47, 71, 97)


def greedy_solution(problem: AllocationProblem, weight: float, *, improve: bool = False) -> dict:
    """Marginal-utility construction, optionally followed by improving 1-swaps.

    Does not consult the enumerated optimum. Ties use the input project order.
    This is a heuristic, without a global-optimality guarantee.
    """
    problem._validate_parameters(weight, 0)
    started = time.perf_counter()
    evaluations = 0

    def utility(bits):
        nonlocal evaluations
        evaluations += 1
        metrics = problem.metrics(bits)
        return metrics["net_benefit"] + weight * metrics["equity_points"]

    chosen = np.zeros(problem.n, dtype=int)
    value = 0.0
    for _ in range(problem.k):
        candidates = []
        for i in np.flatnonzero(chosen == 0):
            candidate = chosen.copy()
            candidate[i] = 1
            candidates.append((utility(candidate), int(i), candidate))
        value, _, chosen = max(candidates, key=lambda c: (c[0], -c[1]))
    swaps = 0
    while improve:
        best_value, best_bits = value, None
        for i in np.flatnonzero(chosen):
            for j in np.flatnonzero(chosen == 0):
                candidate = chosen.copy()
                candidate[i], candidate[j] = 0, 1
                candidate_value = utility(candidate)
                if candidate_value > best_value + 1e-10:
                    best_value, best_bits = candidate_value, candidate
        if best_bits is None:
            break
        chosen, value = best_bits, best_value
        swaps += 1
    return {**problem.metrics(chosen), "utility": float(value), "swaps": swaps,
            "objective_evaluations": evaluations, "elapsed_seconds": time.perf_counter() - started}


def classical_baselines(problem: AllocationProblem, weights=WEIGHTS) -> list[dict]:
    rows = []
    for weight in weights:
        started = time.perf_counter()
        all_rows = problem.enumerate(weight, 20)
        feasible = [row for row in all_rows if row["feasible"]]
        optimum = max(row["utility"] for row in feasible)
        winners = [row for row in feasible if abs(row["utility"] - optimum) < 1e-9]
        exact_seconds = time.perf_counter() - started
        for method in ("greedy", "greedy_swap", "exact", "uniform_feasible"):
            if method.startswith("greedy"):
                result = greedy_solution(problem, weight, improve=method == "greedy_swap")
                probability = float(abs(result["utility"] - optimum) < 1e-9)
            elif method == "exact":
                result = {"selected": ";".join(row["selected"] for row in winners), "utility": optimum,
                          "objective_evaluations": len(all_rows), "elapsed_seconds": exact_seconds}
                probability = 1.0
            else:
                # Exact distribution reference, not a finite-shot simulation.
                result = {"selected": "uniform over all feasible sets",
                          "utility": float(np.mean([row["utility"] for row in feasible])),
                          "objective_evaluations": len(feasible), "elapsed_seconds": None}
                probability = len(winners) / len(feasible)
            rows.append({"weight": weight, "method": method, "selected": result["selected"],
                         "utility": result["utility"], "exact_utility": optimum,
                         "regret": max(0.0, optimum - result["utility"]),
                         "objective_evaluations": result["objective_evaluations"],
                         "elapsed_seconds": result["elapsed_seconds"],
                         "optimal_probability": probability, "feasible_probability": 1.0})
    return rows


def policy_audit(problem: AllocationProblem, weights=WEIGHTS, *, scenarios=500,
                 relative_perturbation=0.1, seed=2026) -> dict:
    """Exact score-perturbation audit; these are chosen stress tests, not data CIs.

    Draw one set of independent Uniform[1-epsilon,1+epsilon] score multipliers
    per scenario for benefits, priorities, and upper-triangle overlaps. Reuse
    the same scenarios at each policy weight. Zero priorities remain zero.
    In a tied perturbed optimum, inclusion credit is split equally across ties.
    """
    if isinstance(scenarios, bool) or not isinstance(scenarios, int) or scenarios <= 0 or not 0 <= relative_perturbation < 1:
        raise ValueError("Use positive integer scenarios and 0 <= perturbation < 1.")
    weights = tuple(weights)
    if not weights:
        raise ValueError("Provide policy weights.")
    for weight in weights:
        problem._validate_parameters(weight, 0)
    all_rows = problem.enumerate(0, 20)
    feasible = [row for row in all_rows if row["feasible"]]
    states = [row["state"] for row in feasible]
    decisions = np.array([int_to_bits(state, problem.n) for state in states])
    rng = np.random.default_rng(seed)
    perturbed_inputs = []
    for scenario in range(scenarios):
        benefit = problem.benefit * rng.uniform(1 - relative_perturbation, 1 + relative_perturbation, problem.n)
        equity = problem.equity * rng.uniform(1 - relative_perturbation, 1 + relative_perturbation, problem.n)
        upper = np.triu(problem.overlap * rng.uniform(
            1 - relative_perturbation, 1 + relative_perturbation, (problem.n, problem.n)), 1)
        perturbed_inputs.append({"scenario": scenario, "benefit": benefit.tolist(),
                                 "equity": equity.tolist(), "overlap": (upper + upper.T).tolist()})
    summary, selections, details, quotas = [], [], [], []
    for weight in weights:
        utility = np.array([row["net_benefit"] + weight * row["equity_points"] for row in feasible])
        baseline_index = int(np.argmax(utility))  # Declared deterministic tie rule.
        baseline_selected = feasible[baseline_index]["selected"]
        inclusion = np.zeros(problem.n)
        retained, regrets = [], []
        for inputs in perturbed_inputs:
            perturbed_benefit = np.array(inputs["benefit"])
            perturbed_equity = np.array(inputs["equity"])
            overlap = np.array(inputs["overlap"])
            scenario_utility = (decisions @ (perturbed_benefit + weight * perturbed_equity)
                                - np.einsum("bi,ij,bj->b", decisions, overlap, decisions) / 2)
            best = float(scenario_utility.max())
            tied = np.isclose(scenario_utility, best, atol=1e-9, rtol=0)
            inclusion += decisions[tied].mean(axis=0)
            regret = max(0.0, best - float(scenario_utility[baseline_index]))
            remains = bool(tied[baseline_index])
            retained.append(remains)
            regrets.append(regret)
            details.append({"weight": weight, "scenario": inputs["scenario"],
                            "baseline_selected": baseline_selected, "baseline_still_optimal": remains,
                            "optimal_selected": ";".join(feasible[i]["selected"] for i in np.flatnonzero(tied)),
                            "baseline_regret": regret})
        summary.append({"weight": weight, "scenarios": scenarios, "relative_perturbation": relative_perturbation,
                        "baseline_selected": baseline_selected,
                        "baseline_still_optimal_rate": float(np.mean(retained)),
                        "mean_baseline_regret": float(np.mean(regrets)),
                        "max_baseline_regret": float(max(regrets))})
        for i, project in enumerate(problem.ids):
            selections.append({"weight": weight, "project": project,
                               "inclusion_frequency": float(inclusion[i] / scenarios)})
        for quota in range(min(problem.k, int(problem.underserved.sum())) + 1):
            allowed = [i for i, row in enumerate(feasible) if row["underserved_projects"] >= quota]
            best = float(utility[allowed].max())
            tied = [i for i in allowed if abs(utility[i] - best) < 1e-9]
            for i in tied:
                row = feasible[i]
                quotas.append({"weight": weight, "quota": quota, "selected": row["selected"],
                               "utility": best, "utility_loss": max(0.0, float(utility.max() - best)),
                               "net_benefit": row["net_benefit"], "equity_points": row["equity_points"],
                               "underserved_projects": row["underserved_projects"]})
    return {"summary": summary, "selections": selections, "details": details, "quotas": quotas,
            "inputs": perturbed_inputs, "seed": seed}


def run_benchmark(output_dir: Path = HERE / "results" / "comparison", *, seeds=SEEDS,
                  maxiter=300, shots=2048, scenarios=500) -> dict:
    output_dir = Path(output_dir)
    manifest = run_suite(output_dir, weights=WEIGHTS, depths=(1, 2), seeds=seeds,
                         maxiter=maxiter, shots=shots, mixers=("x", "xy"))
    problem = load_problem()
    classical = classical_baselines(problem)
    audit = policy_audit(problem, scenarios=scenarios)
    for filename, rows in (("classical_baselines.csv", classical),
                           ("policy_stability.csv", audit["summary"]),
                           ("selection_stability.csv", audit["selections"]),
                           ("sensitivity_details.csv", audit["details"]),
                           ("quota_policy.csv", audit["quotas"])):
        write_csv(output_dir / filename, rows)
    (output_dir / "perturbed_inputs.json").write_text(json.dumps(audit["inputs"], indent=2), encoding="utf-8")
    runs = json.loads((output_dir / "runs.json").read_text(encoding="utf-8"))
    grouped = []
    for mixer in ("x", "xy"):
        for weight in WEIGHTS:
            for depth in (1, 2):
                rows = [r for r in runs if r["mixer"] == mixer and r["weight"] == weight and r["depth"] == depth]
                record = {"mixer": mixer, "weight": weight, "depth": depth, "runs": len(rows),
                          "converged": sum(r["optimizer_success"] for r in rows)}
                for field in ("feasible_probability", "optimal_probability", "conditional_regret", "two_qubit_gates"):
                    values = np.array([r[field] for r in rows], dtype=float)
                    record.update({f"{field}_mean": float(values.mean()), f"{field}_min": float(values.min()),
                                   f"{field}_max": float(values.max())})
                grouped.append(record)
    write_csv(output_dir / "grouped_summary.csv", grouped)
    manifest["comparison"] = {
        "same_controls": ["problem", "penalty", "Pauli coefficient normalization", "depth grid", "seed list",
                          "COBYLA call budget", "sampler shots", "logical transpilation basis"],
        "different_controls": ["initial state: uniform all vs uniform feasible", "X vs ordered ring XY mixer",
                               "compiled circuit resources including state preparation"],
        "interpretation": "Comparison of complete ansatz variants, not an isolated causal effect of the mixer alone.",
        "costs": "Counts include generic StatePreparation. Equal objective calls do not imply equal quantum resources.",
        "policy_stress_test": {"scenarios": scenarios, "seed": audit["seed"], "relative_perturbation": 0.1,
                               "interpretation": "Synthetic independent score stress tests; not empirical confidence intervals."},
        "quota_audit": "Classical-only policy comparison; no claim that XY enforces the added underserved quota.",
    }
    manifest["sha256"]["benchmark.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "comparison")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--scenarios", type=int, default=500)
    args = parser.parse_args()
    run_benchmark(args.output, maxiter=args.maxiter, shots=args.shots, scenarios=args.scenarios)
