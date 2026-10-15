"""Verify detector equivalence, fault response, and the path-only compilation."""
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap

from bell_core import build_circuits
from optimization_circuits import BASES, VARIANTS, build_variant, parity_check_circuit


def probabilities(circuit):
    unmeasured = circuit.remove_final_measurements(inplace=False)
    return Statevector.from_instruction(unmeasured).probabilities_dict()


def correlation_and_acceptance(circuit):
    probabilities_by_bits = probabilities(circuit)
    keep = lambda bits: not circuit.metadata["postselect"] or bits[0] == "0"
    accepted = sum(p for bits, p in probabilities_by_bits.items() if keep(bits))
    if accepted < 1e-12:
        return None, 0.0
    correlation = sum(
        (1 if bits[1] == bits[2] else -1) * p
        for bits, p in probabilities_by_bits.items()
        if keep(bits)
    ) / accepted
    return correlation, accepted


def operation_signature(circuit):
    return [
        (
            item.operation.name,
            tuple(circuit.find_bit(q).index for q in item.qubits),
            tuple(circuit.find_bit(c).index for c in item.clbits),
        )
        for item in circuit.data
    ]


class OptimizationCircuitTests(unittest.TestCase):
    def test_bridge_is_the_same_operator_on_arbitrary_three_qubit_inputs(self):
        # Equality of all 8x8 matrix entries includes both ancilla inputs and
        # every coherent superposition, rather than only the target Bell state.
        original = Operator(parity_check_circuit("protected")).data
        bridge = Operator(parity_check_circuit("bridge")).data
        np.testing.assert_allclose(bridge, original, atol=1e-14, rtol=0)

    def test_original_variants_preserve_every_operation_and_measurement(self):
        for fault in (None, "X", "Z"):
            for original in build_circuits(injected_fault=fault):
                candidate = build_variant(
                    original.metadata["variant"],
                    original.metadata["basis"],
                    injected_fault=fault,
                )
                self.assertEqual(operation_signature(candidate), operation_signature(original))

    def test_ideal_bell_correlations_and_fidelity(self):
        for variant in VARIANTS:
            correlations = []
            for basis in BASES:
                value, acceptance = correlation_and_acceptance(build_variant(variant, basis))
                self.assertAlmostEqual(acceptance, 1.0)
                correlations.append(value)
            np.testing.assert_allclose(correlations, [1, -1, 1], atol=1e-14)
            self.assertAlmostEqual((1 + correlations[0] - correlations[1] + correlations[2]) / 4, 1)

    def test_both_detectors_reject_x_fault_in_every_measurement_basis(self):
        for variant in ("protected", "bridge"):
            for basis in BASES:
                value, acceptance = correlation_and_acceptance(
                    build_variant(variant, basis, injected_fault="X")
                )
                self.assertEqual(acceptance, 0.0)
                self.assertIsNone(value)

    def test_phase_fault_is_not_misrepresented_as_detectable(self):
        for variant in VARIANTS:
            correlations = []
            for basis in BASES:
                value, acceptance = correlation_and_acceptance(
                    build_variant(variant, basis, injected_fault="Z")
                )
                self.assertAlmostEqual(acceptance, 1.0)
                correlations.append(value)
            self.assertAlmostEqual((1 + correlations[0] - correlations[1] + correlations[2]) / 4, 0)

    def test_bridge_compiles_without_routing_and_retains_detector(self):
        path = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1]])
        for basis in BASES:
            compiled = transpile(
                build_variant("bridge", basis),
                basis_gates=["u", "cx"],
                coupling_map=path,
                initial_layout=[0, 1, 2],
                routing_method="none",
                optimization_level=3,
                seed_transpiler=11,
            )
            # One Bell-preparation CX plus three detector CXs.  Collapsing the
            # detector into preparation would be an invalid apparent speedup.
            self.assertEqual(compiled.count_ops().get("cx", 0), 4)
            self.assertEqual(compiled.count_ops().get("swap", 0), 0)
            for item in compiled.data:
                if item.operation.name == "cx":
                    endpoints = tuple(compiled.find_bit(q).index for q in item.qubits)
                    self.assertIn(endpoints, path.get_edges())

    def test_terminal_measurements_and_explicit_postselection_metadata(self):
        for variant in VARIANTS:
            circuit = build_variant(variant, "XX")
            self.assertEqual([(r.name, r.size) for r in circuit.cregs], [("meas", 3)])
            self.assertEqual(circuit.metadata["postselect"], variant in ("protected", "bridge"))
            self.assertEqual(circuit.metadata["postselect_bit"], 2)
            self.assertEqual(circuit.metadata["accepted_syndrome"], 0)
            expected = 2 if variant == "baseline_quiet" else 3
            self.assertEqual(circuit.count_ops()["measure"], expected)
            self.assertEqual([item.operation.name for item in circuit.data[-expected:]],
                             ["measure"] * expected)

    def test_quiet_only_removes_unused_ancilla_measurement(self):
        for basis in BASES:
            for fault in (None, "X", "Z"):
                original = build_variant("baseline", basis, injected_fault=fault)
                quiet = build_variant("baseline_quiet", basis, injected_fault=fault)
                self.assertEqual(operation_signature(quiet), operation_signature(original)[:-1])
                self.assertEqual(quiet.num_qubits, 3)
                self.assertEqual(quiet.num_clbits, 3)
                self.assertFalse(quiet.metadata["postselect"])
                self.assertEqual(quiet.count_ops()["barrier"], original.count_ops()["barrier"])

    def test_quiet_three_bit_register_with_current_executor_sampler_offline(self):
        from qiskit_ibm_runtime.executor_sampler import Sampler
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        # Fail any attempted network connection: this is a local fake-backend
        # compatibility check, not a cloud account or QPU operation.
        network_error = AssertionError("This compatibility test must remain offline.")
        with patch("socket.socket.connect", side_effect=network_error), \
                patch("socket.create_connection", side_effect=network_error):
            backend = FakeTorino()
            compiled = transpile(
                [build_variant("baseline_quiet", basis) for basis in BASES],
                backend=backend,
                initial_layout=[0, 1, 2],
                optimization_level=1,
                seed_transpiler=11,
            )
            sampler = Sampler(mode=backend)
            sampler.options.twirling.enable_gates = False
            sampler.options.twirling.enable_measure = False
            result = sampler.run(compiled, shots=64).result()
        self.assertEqual(len(result), 3)
        for pub in result:
            self.assertEqual(pub.data.meas.num_bits, 3)
            counts = pub.data.meas.get_counts()
            self.assertEqual(sum(counts.values()), 64)
            self.assertTrue(all(len(bits) == 3 and bits[0] == "0" for bits in counts))


if __name__ == "__main__":
    unittest.main()
