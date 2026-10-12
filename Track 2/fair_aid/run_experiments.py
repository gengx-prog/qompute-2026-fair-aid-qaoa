"""Reproduce QAOA experiments without IBM credentials or remote quantum jobs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from allocation import AllocationProblem, DATA_DIR, load_problem
from mixers import make_feasible_initial_state, make_xy_mixer

HERE = Path(__file__).resolve().parent


def solve_qaoa(problem: AllocationProblem, weight: float, depth: int, seed: int,
               penalty: float = 20, maxiter: int = 160, shots: int = 2048,
               *, mixer: str = "x") -> dict:
    """Train using exact Qiskit expectations; use enumeration only for assessment.

    Each call is one initialization, not a selected best-of-many result. COBYLA's
    best evaluated point is retained even if it reaches its evaluation budget.
    No solution is repaired; infeasible samples remain in all raw denominators.
    """
    if depth < 1 or maxiter < 1 or shots < 1:
        raise ValueError("depth, maxiter, and shots must be positive.")
    if mixer not in {"x", "xy"}:
        raise ValueError("mixer must be 'x' or 'xy'.")
    started = time.perf_counter()
    hamiltonian = problem.hamiltonian(weight, penalty)
    identity = "I" * problem.n
    terms = hamiltonian.to_list()
    offset = float(sum(c.real for label, c in terms if label == identity))
    non_identity = [(label, c) for label, c in terms if label != identity and abs(c) > 1e-12]
    if not non_identity:
        raise ValueError("The cost operator is constant; QAOA cannot rank these decisions.")
    scale = float(max(abs(c) for _, c in non_identity))
    operator = SparsePauliOp.from_list([(label, c / scale) for label, c in non_identity])
    variant_options = ({"mixer_operator": make_xy_mixer(problem.n),
                        "initial_state": make_feasible_initial_state(problem.n, problem.k)}
                       if mixer == "xy" else {})
    ansatz = QAOAAnsatz(cost_operator=operator, reps=depth, flatten=True, **variant_options)
    estimator = StatevectorEstimator(default_precision=0.0, seed=seed)
    rng = np.random.default_rng(seed)
    initial = rng.uniform(0.0, np.pi, ansatz.num_parameters)
    history = []
    best_value, best_params = float("inf"), None

    def objective(parameters):
        nonlocal best_value, best_params
        value = float(estimator.run([(ansatz, operator, parameters)]).result()[0].data.evs)
        history.append(value * scale + offset)
        if value < best_value:
            best_value, best_params = value, np.array(parameters, dtype=float).copy()
        return value

    result = minimize(objective, initial, method="COBYLA",
                      options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1e-5})
    optimized = ansatz.assign_parameters(best_params)
    # Logical all-to-all resource estimate; this is NOT an IBM-device layout.
    circuit = transpile(optimized, basis_gates=["h", "rx", "rz", "cx"],
                        optimization_level=1, seed_transpiler=seed)
    probabilities = Statevector.from_instruction(circuit).probabilities()
    measured = circuit.copy()
    measured.measure_all()
    counts = StatevectorSampler(seed=seed + 100_000).run(
        [measured], shots=shots).result()[0].data.meas.get_counts()

    rows = problem.enumerate(weight, penalty)
    feasible = np.array([r["feasible"] for r in rows], dtype=bool)
    utilities = np.array([r["utility"] for r in rows])
    costs = np.array([r["cost"] for r in rows])
    optimum = float(utilities[feasible].max())
    optimal = feasible & np.isclose(utilities, optimum, rtol=0, atol=1e-9)
    sampled = np.zeros(len(rows), dtype=int)
    for bitstring, count in counts.items():
        sampled[int(bitstring, 2)] += count
    if int(sampled.sum()) != shots or not np.isclose(probabilities.sum(), 1):
        raise AssertionError("Sampler/statevector normalization failed.")
    expected_cost = float(probabilities @ costs)
    if not np.isclose(expected_cost, best_value * scale + offset, atol=1e-7):
        raise AssertionError("Qiskit expectation and independent cost evaluation disagree.")
    feasibility = float(probabilities[feasible].sum())
    if mixer == "xy" and not np.isclose(feasibility, 1.0, atol=1e-9):
        raise AssertionError("The ideal XY circuit leaked outside the feasible subspace.")
    candidates = [r for r in rows if r["feasible"] and sampled[r["state"]] > 0]
    best_sample = max(candidates, key=lambda r: (r["utility"], -r["state"])) if candidates else None
    conditional_regret = (float(probabilities[feasible] @ (optimum - utilities[feasible]) / feasibility)
                          if feasibility > 1e-14 else None)
    return {
        "weight": float(weight), "depth": int(depth), "seed": int(seed), "mixer": mixer,
        "penalty": float(penalty), "penalty_bound": problem.penalty_bound(weight),
        "shots": shots, "exact_utility": optimum,
        "exact_selected": ";".join(r["selected"] for r, flag in zip(rows, optimal) if flag),
        "best_sampled_utility": best_sample["utility"] if best_sample else None,
        "best_sampled_selected": best_sample["selected"] if best_sample else None,
        "best_sampled_regret": max(0.0, optimum - best_sample["utility"]) if best_sample else None,
        "feasible_probability": feasibility,
        "optimal_probability": float(probabilities[optimal].sum()),
        "feasible_rate": float(sampled[feasible].sum() / shots),
        "optimal_rate": float(sampled[optimal].sum() / shots),
        "conditional_regret": conditional_regret, "expected_cost": expected_cost,
        "optimal_given_feasible": float(probabilities[optimal].sum() / feasibility) if feasibility > 1e-14 else None,
        "expected_underserved_given_feasible": float(
            probabilities[feasible] @ np.array([r["underserved_projects"] for r in rows])[feasible]
            / feasibility) if feasibility > 1e-14 else None,
        "uniform_all_feasible_probability": float(feasible.mean()),
        "uniform_all_optimal_probability": float(optimal.mean()),
        "uniform_feasible_optimal_probability": float(optimal.sum() / feasible.sum()),
        "uniform_feasible_conditional_regret": float(np.mean(optimum - utilities[feasible])),
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "nfev": int(result.nfev), "elapsed_seconds": time.perf_counter() - started,
        "circuit_depth": circuit.depth(), "two_qubit_gates": int(circuit.count_ops().get("cx", 0)),
        "scale": scale, "offset": offset,
        "parameter_order": [str(p) for p in ansatz.parameters],
        "initial_parameters": initial.tolist(), "best_parameters": best_params.tolist(),
        "history": history, "probabilities": probabilities.tolist(),
        "counts": counts,
    }


DETAIL_FIELDS = {"parameter_order", "initial_parameters", "best_parameters", "history", "probabilities", "counts"}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("Cannot write an empty result table.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(output_dir: Path, *, weights=(0, 0.5, 1, 2), depths=(1, 2),
              seeds=(11, 29, 47), penalty=20, maxiter=160, shots=2048, mixers=("x",)) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights, depths, seeds, mixers = tuple(weights), tuple(depths), tuple(seeds), tuple(mixers)
    if not weights or not depths or not seeds or not mixers or set(mixers) - {"x", "xy"}:
        raise ValueError("Provide weights, depths, seeds and mixers drawn from {'x', 'xy'}.")
    problem = load_problem()
    started = time.perf_counter()
    exact_rows, penalty_rows, validation_rows = [], [], []
    for weight in weights:
        table = problem.enumerate(weight, penalty)
        feasible = [row for row in table if row["feasible"]]
        optimal_utility = max(row["utility"] for row in feasible)
        exact_rows.extend({"weight": weight, **row,
                           "is_optimal": bool(np.isclose(row["utility"], optimal_utility, rtol=0, atol=1e-9))}
                          for row in feasible)
        matrix = problem.hamiltonian(weight, penalty).to_matrix()
        diagonal = np.diag(matrix).real
        error = float(np.max(np.abs(diagonal - [r["cost"] for r in table])))
        if not np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-10) or error > 1e-8:
            raise AssertionError("Hamiltonian encoding failed the exhaustive validation.")
        validation_rows.append({"weight": weight, "states_checked": len(table), "max_energy_error": error})
        for test_penalty in sorted(set([0.0, 2.0, 8.0, float(penalty), 40.0])):
            candidates = problem.enumerate(weight, test_penalty)
            ground_cost = min(r["cost"] for r in candidates)
            ground = [r for r in candidates if np.isclose(r["cost"], ground_cost, rtol=0, atol=1e-9)]
            penalty_rows.append({
                "weight": weight, "penalty": test_penalty, "bound": problem.penalty_bound(weight),
                "ground_feasible": all(r["feasible"] for r in ground),
                "ground_cost": ground_cost, "ground_selected": ";".join(r["selected"] for r in ground),
                "ground_state_count": len(ground),
                "guaranteed_by_bound": test_penalty > problem.penalty_bound(weight),
            })
    write_csv(output_dir / "exact_solutions.csv", exact_rows)
    write_csv(output_dir / "penalty_sweep.csv", penalty_rows)
    write_csv(output_dir / "encoding_validation.csv", validation_rows)
    runs = []
    total = len(weights) * len(depths) * len(seeds) * len(mixers)
    for mixer in mixers:
        for weight in weights:
            for depth in depths:
                for seed in seeds:
                    run = solve_qaoa(problem, weight, depth, seed, penalty, maxiter, shots, mixer=mixer)
                    runs.append(run)
                    print(f"[{len(runs):02d}/{total}] mixer={mixer}, weight={weight:g}, p={depth}, seed={seed}: "
                          f"P(feasible)={run['feasible_probability']:.3f}, "
                          f"P(optimal)={run['optimal_probability']:.3f}", flush=True)
    write_csv(output_dir / "summary.csv", [{k: v for k, v in run.items() if k not in DETAIL_FIELDS} for run in runs])
    (output_dir / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    source_files = [*sorted(DATA_DIR.glob("*.csv")), HERE / "allocation.py", HERE / "mixers.py",
                    Path(__file__), HERE / "requirements.txt"]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": "synthetic pedagogical scores; no real beneficiary or population records",
        "problem": {"n": problem.n, "k": problem.k, "cost_per_project": 1},
        "weights": list(weights), "depths": list(depths), "seeds": list(seeds),
        "mixers": list(mixers),
        "penalty": penalty, "maxiter": maxiter, "shots_per_run": shots, "runs": len(runs),
        "training": "Qiskit StatevectorEstimator, precision=0, COBYLA; exact ideal simulation",
        "sampling": "Qiskit StatevectorSampler; seed=initialization_seed+100000; no repair",
        "hardware": "none; resources include initial-state preparation, transpiled to h/rx/rz/cx on all-to-all logical connectivity",
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["qiskit", "numpy", "scipy", "matplotlib", "nbformat", "nbclient"]},
        "sha256": {str(path.relative_to(HERE)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in source_files},
        "elapsed_seconds": time.perf_counter() - started,
        "encoding_validation": validation_rows,
        "notes": ["Classical exhaustive reference enumerates all 64 states / 20 feasible sets.",
                  "Exact statevector probabilities and finite-shot rates are distinct fields.",
                  "All initialization runs are reported, including non-converged optimizations.",
                  "Best-of-2048 samples is not evidence of a computational advantage.",
                  "Equity priority is a policy preference, not a fairness constraint or guarantee."],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--quick", action="store_true", help="Four runs: two weights, two depths, one seed.")
    args = parser.parse_args()
    options = {"weights": (0, 2), "seeds": (11,)} if args.quick else {}
    run_suite(args.output, maxiter=args.maxiter, shots=args.shots, **options)
