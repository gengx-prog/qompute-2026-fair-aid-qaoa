"""Independent checks of the constrained QAOA variant and its gate synthesis.

Run with ``python -m unittest -v test_mixers`` from this directory.
"""
from __future__ import annotations

from math import comb
import unittest

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Operator, Statevector

from allocation import load_problem
from mixers import make_feasible_initial_state, make_xy_mixer


class FeasibleMixerTests(unittest.TestCase):
    def test_six_qubit_initial_state_is_uniform_on_all_twenty_feasible_states(self):
        vector = Statevector.from_instruction(make_feasible_initial_state(6, 3)).data
        feasible = np.array([i.bit_count() == 3 for i in range(64)])
        self.assertEqual(int(feasible.sum()), 20)
        self.assertAlmostEqual(float(np.vdot(vector, vector).real), 1.0, places=12)
        # Fix any synthesis global phase before comparing complex amplitudes.
        aligned = vector * np.exp(-1j * np.angle(vector[7]))
        np.testing.assert_allclose(aligned[feasible], 1 / np.sqrt(20), atol=1e-12)
        np.testing.assert_allclose(aligned[~feasible], 0, atol=1e-12)

    def test_initial_state_other_cardinalities_and_endpoints(self):
        for n, k in ((2, 0), (2, 1), (2, 2), (5, 2), (6, 1), (6, 5)):
            with self.subTest(n=n, k=k):
                probabilities = Statevector.from_instruction(
                    make_feasible_initial_state(n, k)).probabilities()
                expected = np.array([1 / comb(n, k) if i.bit_count() == k else 0
                                     for i in range(2**n)])
                np.testing.assert_allclose(probabilities, expected, atol=1e-12)

    def test_ring_is_connected_and_uses_one_shared_parameter(self):
        for n in (2, 3, 6, 10):
            with self.subTest(n=n):
                mixer = make_xy_mixer(n)
                expected_edges = [(i, i + 1) for i in range(n - 1)]
                if n > 2:
                    expected_edges.append((n - 1, 0))
                self.assertEqual(mixer.num_parameters, 1)
                beta = next(iter(mixer.parameters))
                observed = []
                for instruction in mixer.data:
                    observed.append((instruction.operation.name,
                                     tuple(mixer.find_bit(q).index for q in instruction.qubits)))
                    self.assertEqual(instruction.operation.params, [beta])
                expected = [(gate, edge) for edge in expected_edges for gate in ("rxx", "ryy")]
                self.assertEqual(observed, expected)

    def test_two_qubit_exchange_matches_analytic_rotation(self):
        beta = 0.731
        actual = Operator(make_xy_mixer(2).assign_parameters([beta])).data
        expected = np.eye(4, dtype=complex)
        expected[1, 1] = expected[2, 2] = np.cos(beta)
        expected[1, 2] = expected[2, 1] = -1j * np.sin(beta)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_mixer_preserves_every_hamming_weight_sector(self):
        weights = np.array([i.bit_count() for i in range(64)])
        cross_sector = weights[:, None] != weights[None, :]
        for beta in (0.137, 0.731, 1.943, -0.629):
            with self.subTest(beta=beta):
                unitary = Operator(make_xy_mixer(6).assign_parameters([beta])).data
                np.testing.assert_allclose(unitary[cross_sector], 0, atol=1e-12)
                np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(64), atol=1e-12)

    def test_complete_qaoa_preserves_weight_three_at_both_depths(self):
        problem = load_problem()
        feasible = np.array([i.bit_count() == 3 for i in range(64)])
        for depth in (1, 2):
            for weight in (0, 0.5, 1, 2):
                with self.subTest(depth=depth, weight=weight):
                    ansatz = QAOAAnsatz(
                        cost_operator=problem.hamiltonian(weight, penalty=0),
                        initial_state=make_feasible_initial_state(problem.n, problem.k),
                        mixer_operator=make_xy_mixer(problem.n), reps=depth, flatten=True)
                    self.assertEqual(ansatz.num_parameters, 2 * depth)
                    bound = ansatz.assign_parameters(np.linspace(0.173, 1.391, 2 * depth))
                    # Exercise the same logical basis synthesis used in the suite.
                    compiled = transpile(bound, basis_gates=["h", "rx", "rz", "cx"],
                                         optimization_level=1, seed_transpiler=11)
                    amplitudes = Statevector.from_instruction(compiled).data
                    np.testing.assert_allclose(amplitudes[~feasible], 0, atol=1e-10)
                    self.assertAlmostEqual(float(np.sum(np.abs(amplitudes[feasible])**2)),
                                           1.0, places=10)

    def test_initial_state_has_physical_synthesis_cost_and_no_reset(self):
        initial = make_feasible_initial_state(6, 3)
        compiled = transpile(initial, basis_gates=["h", "rx", "rz", "cx"],
                             optimization_level=1, seed_transpiler=11)
        self.assertGreater(compiled.count_ops().get("cx", 0), 0)
        self.assertNotIn("reset", compiled.count_ops())
        self.assertNotIn("initialize", compiled.count_ops())
        expected = Statevector.from_instruction(initial)
        self.assertTrue(expected.equiv(Statevector.from_instruction(compiled)))

    def test_rejects_invalid_sizes_before_allocating_statevector(self):
        for n in (None, True, np.bool_(True), 1, 0, -2, 11, 1000, 6.0, "6"):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    make_xy_mixer(n)
                with self.assertRaises(ValueError):
                    make_feasible_initial_state(n, 3)
        for k in (None, True, np.bool_(False), -1, 7, 3.0, "3"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                make_feasible_initial_state(6, k)


if __name__ == "__main__":
    unittest.main()
