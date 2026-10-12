"""Independent baseline, policy-audit, and trained QAOA integration checks."""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import transpile as qiskit_transpile

from allocation import AllocationProblem, load_problem
from benchmark import greedy_solution, policy_audit
from run_experiments import solve_qaoa


def direct_utility(indices, benefit, equity, overlap, weight):
    """Scalar arithmetic independent of model/benchmark metric functions."""
    return (sum(float(benefit[i]) + weight * float(equity[i]) for i in indices)
            - sum(float(overlap[i, j]) for i, j in combinations(indices, 2)))


def feasible_sets(problem):
    return sorted(combinations(range(problem.n), problem.k),
                  key=lambda group: sum(1 << i for i in group))


def selected_label(problem, group):
    return ",".join(problem.ids[i] for i in group)


class ClassicalBaselineTests(unittest.TestCase):
    def test_greedy_and_swap_do_not_read_enumeration_and_are_feasible(self):
        problem = load_problem()
        with patch.object(AllocationProblem, "enumerate", side_effect=AssertionError("oracle used")):
            for weight in (0, 0.5, 1, 2):
                for improve in (False, True):
                    with self.subTest(weight=weight, improve=improve):
                        result = greedy_solution(problem, weight, improve=improve)
                        chosen = tuple(problem.ids.index(pid) for pid in result["selected"].split(","))
                        self.assertEqual(len(chosen), problem.k)
                        self.assertEqual(len(set(chosen)), problem.k)
                        self.assertTrue(result["feasible"])
                        expected = direct_utility(chosen, problem.benefit, problem.equity,
                                                  problem.overlap, weight)
                        self.assertAlmostEqual(result["utility"], expected, places=12)
                        self.assertGreaterEqual(result["objective_evaluations"], 15)
                        repeated = greedy_solution(problem, weight, improve=improve)
                        for field in ("selected", "utility", "swaps", "objective_evaluations"):
                            self.assertEqual(result[field], repeated[field])

    def test_swap_terminates_at_a_one_swap_local_optimum(self):
        problem = load_problem()
        for weight in (0, 0.5, 1, 2):
            result = greedy_solution(problem, weight, improve=True)
            plain = greedy_solution(problem, weight)
            chosen = {problem.ids.index(pid) for pid in result["selected"].split(",")}
            self.assertGreaterEqual(result["utility"] + 1e-10, plain["utility"])
            for removed in chosen:
                for added in set(range(problem.n)) - chosen:
                    neighbor = sorted((chosen - {removed}) | {added})
                    value = direct_utility(neighbor, problem.benefit, problem.equity,
                                           problem.overlap, weight)
                    self.assertLessEqual(value, result["utility"] + 1e-10)

    def test_ties_follow_project_order_without_unnecessary_swaps(self):
        original = load_problem()
        tied = replace(original, benefit=np.ones(original.n), equity=np.zeros(original.n),
                       overlap=np.zeros((original.n, original.n)))
        for improve in (False, True):
            result = greedy_solution(tied, weight=1, improve=improve)
            self.assertEqual(result["selected"], "A,B,C")
            self.assertEqual(result["swaps"], 0)


class PolicyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = load_problem()
        cls.weights = (0, 0.5, 1, 2)
        cls.audit = policy_audit(cls.problem, cls.weights, scenarios=12, seed=531)

    def test_scenarios_reproduce_and_respect_perturbation_bounds(self):
        self.assertEqual(self.audit, policy_audit(self.problem, self.weights, scenarios=12, seed=531))
        different = policy_audit(self.problem, self.weights, scenarios=12, seed=532)
        self.assertNotEqual(self.audit["inputs"], different["inputs"])
        for scenario in self.audit["inputs"]:
            for field in ("benefit", "equity", "overlap"):
                values = np.asarray(scenario[field])
                original = getattr(self.problem, field)
                self.assertTrue(np.all(values >= 0.9 * original - 1e-12))
                self.assertTrue(np.all(values <= 1.1 * original + 1e-12))
                np.testing.assert_array_equal(values[original == 0], 0)
            overlap = np.asarray(scenario["overlap"])
            np.testing.assert_allclose(overlap, overlap.T, atol=0)
            np.testing.assert_array_equal(np.diag(overlap), 0)

    def test_all_saved_perturbations_recompute_directly(self):
        problem, groups = self.problem, feasible_sets(self.problem)
        for weight in self.weights:
            original = [direct_utility(g, problem.benefit, problem.equity, problem.overlap, weight)
                        for g in groups]
            baseline = groups[int(np.argmax(original))]
            retained, regrets = [], []
            inclusion = np.zeros(problem.n)
            for inputs in self.audit["inputs"]:
                values = [direct_utility(g, inputs["benefit"], inputs["equity"],
                                         np.asarray(inputs["overlap"]), weight) for g in groups]
                winners = [g for g, value in zip(groups, values) if abs(value - max(values)) < 1e-9]
                regret = max(values) - values[groups.index(baseline)]
                detail = next(r for r in self.audit["details"]
                              if r["weight"] == weight and r["scenario"] == inputs["scenario"])
                self.assertEqual(detail["baseline_selected"], selected_label(problem, baseline))
                self.assertEqual(detail["optimal_selected"],
                                 ";".join(selected_label(problem, g) for g in winners))
                self.assertEqual(detail["baseline_still_optimal"], baseline in winners)
                self.assertAlmostEqual(detail["baseline_regret"], regret, places=11)
                retained.append(baseline in winners)
                regrets.append(regret)
                for winner in winners:
                    inclusion[list(winner)] += 1 / len(winners)
            summary = next(r for r in self.audit["summary"] if r["weight"] == weight)
            self.assertAlmostEqual(summary["baseline_still_optimal_rate"], np.mean(retained))
            self.assertAlmostEqual(summary["mean_baseline_regret"], np.mean(regrets), places=11)
            self.assertAlmostEqual(summary["max_baseline_regret"], max(regrets), places=11)
            observed = [next(r["inclusion_frequency"] for r in self.audit["selections"]
                             if r["weight"] == weight and r["project"] == pid) for pid in problem.ids]
            np.testing.assert_allclose(observed, inclusion / len(retained), atol=1e-12)
            self.assertAlmostEqual(sum(observed), problem.k, places=12)

    def test_quota_frontier_matches_independent_feasible_search(self):
        problem, groups = self.problem, feasible_sets(self.problem)
        for weight in self.weights:
            scores = {g: direct_utility(g, problem.benefit, problem.equity, problem.overlap, weight)
                      for g in groups}
            previous_loss = -1.0
            for quota in range(4):
                allowed = [g for g in groups if sum(problem.underserved[i] for i in g) >= quota]
                best = max(scores[g] for g in allowed)
                expected = {selected_label(problem, g) for g in allowed if abs(scores[g] - best) < 1e-9}
                observed = [r for r in self.audit["quotas"] if r["weight"] == weight and r["quota"] == quota]
                self.assertEqual({r["selected"] for r in observed}, expected)
                for row in observed:
                    self.assertGreaterEqual(row["underserved_projects"], quota)
                    self.assertAlmostEqual(row["utility"], best, places=12)
                    self.assertAlmostEqual(row["utility_loss"], max(scores.values()) - best, places=12)
                    self.assertGreaterEqual(row["utility_loss"] + 1e-10, previous_loss)
                previous_loss = observed[0]["utility_loss"]

    def test_zero_perturbation_keeps_baseline_optimal(self):
        audit = policy_audit(self.problem, scenarios=7, relative_perturbation=0)
        for row in audit["summary"]:
            self.assertEqual(row["baseline_still_optimal_rate"], 1)
            self.assertEqual(row["mean_baseline_regret"], 0)
            self.assertEqual(row["max_baseline_regret"], 0)
        for scenario in audit["inputs"]:
            for field in ("benefit", "equity", "overlap"):
                np.testing.assert_array_equal(scenario[field], getattr(self.problem, field))

    def test_all_ties_share_inclusion_credit(self):
        problem = replace(self.problem, benefit=np.zeros(6), equity=np.zeros(6), overlap=np.zeros((6, 6)))
        audit = policy_audit(problem, weights=(1,), scenarios=2)
        for row in audit["selections"]:
            self.assertAlmostEqual(row["inclusion_frequency"], 0.5, places=12)
        for row in audit["details"]:
            self.assertEqual(len(row["optimal_selected"].split(";")), 20)

    def test_invalid_audit_inputs_fail_early(self):
        for scenarios in (0, -1, 1.5, True, False):
            with self.subTest(scenarios=scenarios), self.assertRaises(ValueError):
                policy_audit(self.problem, scenarios=scenarios)
        for perturbation in (-0.1, 1, 2, float("nan"), float("inf")):
            with self.subTest(perturbation=perturbation), self.assertRaises(ValueError):
                policy_audit(self.problem, scenarios=1, relative_perturbation=perturbation)
        for weights in ((), (-1,), (float("nan"),)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                policy_audit(self.problem, weights=weights, scenarios=1)


class TrainedVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = load_problem()
        captured = {}

        def record_transpile(circuit, **kwargs):
            captured["input"] = circuit.copy()
            captured["compiled"] = qiskit_transpile(circuit, **kwargs)
            return captured["compiled"]

        with patch("run_experiments.transpile", side_effect=record_transpile):
            cls.xy = solve_qaoa(cls.problem, 1, 1, 11, 20, 30, 128, mixer="xy")
        cls.captured = captured
        # Legacy positional parameters retain X-mixer behavior.
        cls.x = solve_qaoa(cls.problem, 1, 1, 11, 20, 30, 128)

    def test_primitive_metrics_match_direct_scores_and_raw_counts(self):
        for run in (self.x, self.xy):
            with self.subTest(mixer=run["mixer"]):
                groups = [tuple(i for i in range(6) if (state >> i) & 1) for state in range(64)]
                utility = np.array([direct_utility(g, self.problem.benefit, self.problem.equity,
                                                  self.problem.overlap, run["weight"]) for g in groups])
                feasible = np.array([len(g) == self.problem.k for g in groups])
                optimum = float(max(utility[feasible]))
                optimal = feasible & np.isclose(utility, optimum, rtol=0, atol=1e-9)
                probabilities = np.asarray(run["probabilities"])
                self.assertTrue(np.all(probabilities >= 0))
                self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=11)
                self.assertEqual(sum(run["counts"].values()), 128)
                sampled = np.zeros(64)
                for bitstring, count in run["counts"].items():
                    self.assertEqual(len(bitstring), 6)
                    self.assertIsInstance(count, int)
                    self.assertGreater(count, 0)
                    sampled[int(bitstring, 2)] = count
                feasible_probability = float(probabilities[feasible].sum())
                self.assertAlmostEqual(run["feasible_probability"], feasible_probability, places=12)
                self.assertAlmostEqual(run["optimal_probability"], probabilities[optimal].sum(), places=12)
                self.assertAlmostEqual(run["feasible_rate"], sampled[feasible].sum() / 128)
                self.assertAlmostEqual(run["optimal_rate"], sampled[optimal].sum() / 128)
                regret = probabilities[feasible] @ (optimum - utility[feasible]) / feasible_probability
                self.assertAlmostEqual(run["conditional_regret"], regret, places=11)
                self.assertAlmostEqual(run["optimal_given_feasible"],
                                       probabilities[optimal].sum() / feasible_probability, places=12)
                costs = -utility + run["penalty"] * np.array([(len(g) - 3)**2 for g in groups])
                self.assertAlmostEqual(run["expected_cost"], probabilities @ costs, places=10)
                self.assertAlmostEqual(run["expected_cost"], min(run["history"]), places=9)
                self.assertEqual(run["nfev"], len(run["history"]))
                self.assertLessEqual(run["nfev"], 30)

    def test_xy_is_feasible_and_counts_full_initial_state_preparation(self):
        self.assertEqual(self.xy["mixer"], "xy")
        self.assertAlmostEqual(self.xy["feasible_probability"], 1.0, places=10)
        self.assertEqual(self.xy["feasible_rate"], 1.0)
        self.assertTrue(all(bitstring.count("1") == 3 for bitstring in self.xy["counts"]))
        self.assertEqual(self.captured["input"].count_ops().get("state_preparation", 0), 1)
        self.assertEqual(self.xy["two_qubit_gates"], self.captured["compiled"].count_ops().get("cx", 0))
        self.assertEqual(self.xy["circuit_depth"], self.captured["compiled"].depth())
        self.assertGreater(self.xy["two_qubit_gates"], self.x["two_qubit_gates"])

    def test_explicit_x_matches_legacy_default(self):
        explicit = solve_qaoa(self.problem, 1, 1, 11, 20, 30, 128, mixer="x")
        self.assertEqual(self.x["mixer"], "x")
        for field in ("probabilities", "counts", "history", "best_parameters", "two_qubit_gates"):
            self.assertEqual(explicit[field], self.x[field])
        with self.assertRaises(ValueError):
            solve_qaoa(self.problem, 1, 1, 11, mixer="unknown")


if __name__ == "__main__":
    unittest.main()
