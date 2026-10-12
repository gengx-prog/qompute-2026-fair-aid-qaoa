"""Independent arithmetic, encoding, and feasibility checks for the aid model.

Run from this directory with ``python -m unittest -v test_allocation``.
No optimizer or stochastic sampling is involved in these regression tests.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from allocation import AllocationProblem, DATA_DIR, int_to_bits, load_problem


def manual_cost(problem, bits, weight, penalty):
    """Scalar definition, deliberately independent of the production methods."""
    reward = sum(
        (float(problem.benefit[i]) + weight * float(problem.equity[i])) * bits[i]
        for i in range(problem.n)
    )
    overlap_loss = sum(
        float(problem.overlap[i, j]) * bits[i] * bits[j]
        for i in range(problem.n)
        for j in range(i + 1, problem.n)
    )
    return -reward + overlap_loss + penalty * (sum(bits) - problem.k) ** 2


class AllocationEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = load_problem()

    def assert_encoding_matches_definition(self, problem, weight, penalty):
        constant, linear, quadratic = problem.qubo(weight, penalty)
        matrix = problem.hamiltonian(weight, penalty).to_matrix()
        np.testing.assert_allclose(matrix.imag, 0, atol=1e-12)
        np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)), atol=1e-12)
        for state in range(2**problem.n):
            # Read q_i from the integer without using the production decoder.
            bits = tuple((state // (2**i)) % 2 for i in range(problem.n))
            expected = manual_cost(problem, bits, weight, penalty)
            qubo_cost = constant + sum(linear[i] * bits[i] for i in range(problem.n))
            qubo_cost += sum(
                quadratic[i, j] * bits[i] * bits[j]
                for i in range(problem.n)
                for j in range(i + 1, problem.n)
            )
            with self.subTest(weight=weight, penalty=penalty, state=state):
                self.assertAlmostEqual(problem.cost(bits, weight, penalty), expected, places=10)
                self.assertAlmostEqual(float(qubo_cost), expected, places=10)
                self.assertAlmostEqual(float(matrix[state, state].real), expected, places=10)

    def test_all_64_states_match_manual_qubo_and_ising_costs(self):
        for weight in (0, 0.5, 1, 2):
            for penalty in (0, 2, 20):
                self.assert_encoding_matches_definition(self.problem, weight, penalty)

    def test_encoding_also_works_when_k_is_not_half_the_qubits(self):
        problem = AllocationProblem(
            ids=("P", "Q", "R"), names=("First", "Second", "Third"),
            benefit=np.array([1.0, 2.0, 3.0]), equity=np.array([1.0, 0.0, 2.0]),
            underserved=np.array([0, 1, 1]),
            overlap=np.array([[0.0, 10.0, 8.0], [10.0, 0.0, 4.0], [8.0, 4.0, 0.0]]),
            k=1,
        )
        self.assert_encoding_matches_definition(problem, weight=0.75, penalty=18)
        # Here the under-selection bound dominates the adjusted benefit bound.
        self.assertAlmostEqual(problem.penalty_bound(0), 17)
        rows = problem.enumerate(0, 17.01)
        minimum = min(row["cost"] for row in rows)
        self.assertTrue(all(row["feasible"] for row in rows if abs(row["cost"] - minimum) < 1e-9))

    def test_exact_optima_and_reported_tradeoff_match_independent_enumeration(self):
        expected = {
            0: ("A,B,D", 22.1, 2.5, 22.1, 1),
            0.5: ("A,B,F", 21.3, 4.5, 23.55, 1),
            1: ("A,E,F", 19.4, 7.0, 26.4, 2),
            2: ("D,E,F", 16.7, 9.0, 34.7, 3),
        }
        for weight, (selected, benefit, equity, utility, underserved) in expected.items():
            rows = self.problem.enumerate(weight, 20)
            feasible = [row for row in rows if row["feasible"]]
            self.assertEqual(len(rows), 64)
            self.assertEqual(len(feasible), 20)
            optimum = max(row["utility"] for row in feasible)
            winners = [row for row in feasible if abs(row["utility"] - optimum) < 1e-9]
            with self.subTest(weight=weight):
                self.assertEqual(len(winners), 1)
                best = winners[0]
                self.assertEqual(best["selected"], selected)
                self.assertAlmostEqual(best["net_benefit"], benefit)
                self.assertAlmostEqual(best["equity_points"], equity)
                self.assertAlmostEqual(best["utility"], utility)
                self.assertAlmostEqual(best["cost"], -utility)
                self.assertEqual(best["underserved_projects"], underserved)

    def test_qiskit_little_endian_bits_select_expected_projects(self):
        expected = {
            "000001": "A", "100000": "F", "001011": "A,B,D",
            "100011": "A,B,F", "110001": "A,E,F", "111000": "D,E,F",
        }
        for label, selected in expected.items():
            with self.subTest(bitstring=label):
                bits = int_to_bits(int(label, 2), 6)
                self.assertEqual(self.problem.metrics(bits)["selected"], selected)
                self.assertEqual("".join(str(int(bit)) for bit in bits[::-1]), label)

    def test_overlap_is_charged_once_per_selected_pair(self):
        metrics = self.problem.metrics([1, 1, 0, 1, 0, 0])
        self.assertAlmostEqual(metrics["benefit"], 24)
        self.assertAlmostEqual(metrics["overlap_loss"], 1.9)
        self.assertAlmostEqual(metrics["net_benefit"], 22.1)


class AllocationPenaltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = load_problem()

    def test_fixed_penalty_20_guarantees_feasible_sweep_minima(self):
        for weight in (0, 0.5, 1, 2):
            with self.subTest(weight=weight):
                self.assertGreater(20, self.problem.penalty_bound(weight))
                rows = self.problem.enumerate(weight, 20)
                best_feasible = min(row["cost"] for row in rows if row["feasible"])
                best_infeasible = min(row["cost"] for row in rows if not row["feasible"])
                self.assertGreater(best_infeasible, best_feasible)

    def test_small_penalties_can_make_the_global_minimum_infeasible(self):
        for weight in (0, 0.5, 1, 2):
            for penalty in (0, 2):
                rows = self.problem.enumerate(weight, penalty)
                minimum = min(row["cost"] for row in rows)
                winners = [row for row in rows if abs(row["cost"] - minimum) < 1e-9]
                with self.subTest(weight=weight, penalty=penalty):
                    self.assertTrue(winners)
                    self.assertTrue(all(not row["feasible"] for row in winners))

    def test_safe_bound_scales_beyond_the_demonstration_weight_range(self):
        for weight in (2.5, 5, 10):
            bound = self.problem.penalty_bound(weight)
            expected_bound = max(
                max(
                    float(self.problem.benefit[i] + weight * self.problem.equity[i]),
                    sum(float(self.problem.overlap[i, j]) for j in range(self.problem.n))
                    - float(self.problem.benefit[i] + weight * self.problem.equity[i]),
                )
                for i in range(self.problem.n)
            )
            with self.subTest(weight=weight):
                self.assertAlmostEqual(bound, expected_bound)
                penalty = bound + 0.01  # Strictly above, as required by the proof.
                rows = self.problem.enumerate(weight, penalty)
                minimum = min(row["cost"] for row in rows)
                self.assertTrue(all(row["feasible"] for row in rows if abs(row["cost"] - minimum) < 1e-9))
                # Every one-bit move toward feasibility satisfies the repair proof.
                for row in rows:
                    if row["feasible"]:
                        continue
                    bits = int_to_bits(row["state"], self.problem.n)
                    target_bit = int(row["count"] > self.problem.k)
                    for i in range(self.problem.n):
                        if bits[i] == target_bit:
                            repaired = bits.copy()
                            repaired[i] = 1 - target_bit
                            self.assertLess(self.problem.cost(repaired, weight, penalty), row["cost"])


class AllocationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = load_problem()

    def test_invalid_weights_and_penalties_are_rejected(self):
        for invalid in (-1, float("nan"), float("inf")):
            for weight, penalty in ((invalid, 20), (1, invalid)):
                with self.subTest(weight=weight, penalty=penalty):
                    with self.assertRaises(ValueError):
                        self.problem.cost([1, 1, 1, 0, 0, 0], weight, penalty)
                    with self.assertRaises(ValueError):
                        self.problem.qubo(weight, penalty)
            with self.assertRaises(ValueError):
                self.problem.penalty_bound(invalid)

    def test_invalid_binary_decisions_are_rejected(self):
        for invalid in ([1, 0], [2, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0],
                        [0.5, 0, 0, 0, 0, 0], [1.9, 0, 0, 0, 0, 0]):
            with self.subTest(bits=invalid):
                with self.assertRaises(ValueError):
                    self.problem.metrics(invalid)

    def test_invalid_integer_state_decodings_are_rejected(self):
        for state, n in ((-1, 6), (64, 6), (1.5, 6), (0, 0), (0, 1.5)):
            with self.subTest(state=state, n=n):
                with self.assertRaises(ValueError):
                    int_to_bits(state, n)

    def test_invalid_project_data_is_rejected(self):
        changes = (
            {"ids": ("A", "A", "C", "D", "E", "F")},
            {"names": ("Only one name",)}, {"k": 0}, {"k": 6}, {"k": 2.5},
            {"benefit": np.array([1.0, 2.0])},
            {"benefit": np.array([-1.0, 2, 3, 4, 5, 6])},
            {"equity": np.array([float("nan"), 2, 3, 4, 5, 6])},
            {"underserved": np.array([0, 0, 0, 1, 1, 0.5])},
            {"overlap": np.zeros((3, 3))},
        )
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    replace(self.problem, **change)

    def test_invalid_overlap_matrices_are_rejected(self):
        for kind in ("asymmetric", "negative", "diagonal", "nonfinite"):
            overlap = self.problem.overlap.copy()
            if kind == "asymmetric":
                overlap[0, 1] += 1
            elif kind == "negative":
                overlap[0, 1] = overlap[1, 0] = -1
            elif kind == "diagonal":
                overlap[0, 0] = 1
            else:
                overlap[0, 1] = overlap[1, 0] = float("inf")
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    replace(self.problem, overlap=overlap)

    def test_loader_rejects_unequal_costs_and_duplicate_or_self_pairs(self):
        for kind in ("unequal_cost", "duplicate", "self_pair"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for filename in ("projects.csv", "overlaps.csv"):
                    shutil.copy2(DATA_DIR / filename, directory / filename)
                if kind == "unequal_cost":
                    with (directory / "projects.csv").open(encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle)
                        fields, rows = reader.fieldnames, list(reader)
                    rows[0]["unit_cost"] = "2"
                    with (directory / "projects.csv").open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                else:
                    with (directory / "overlaps.csv").open("a", encoding="utf-8", newline="") as handle:
                        csv.writer(handle).writerow(("B", "A", 1.2) if kind == "duplicate" else ("A", "A", 1))
                with self.assertRaises(ValueError):
                    load_problem(directory)


if __name__ == "__main__":
    unittest.main()
