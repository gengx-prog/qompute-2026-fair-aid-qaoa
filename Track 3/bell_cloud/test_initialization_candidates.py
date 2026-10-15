"""Ideal, target, and explicitly synthetic thermal-model initialization checks."""
import unittest

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer.noise import thermal_relaxation_error
from qiskit_ibm_runtime.fake_provider import FakeTorino

from initialization_candidates import INITIALIZATION_VARIANTS, build_initialization_candidate
from optimization_circuits import BASES, build_variant


def operation_signature(circuit):
    return [
        (item.operation.name,
         tuple(circuit.find_bit(q).index for q in item.qubits),
         tuple(circuit.find_bit(c).index for c in item.clbits),
         tuple(str(p) for p in item.operation.params))
        for item in circuit.data
    ]


class InitializationCandidateTests(unittest.TestCase):
    def test_only_initialization_prefix_changes_and_all_measurements_remain(self):
        for variant in INITIALIZATION_VARIANTS:
            for basis in BASES:
                circuit = build_initialization_candidate(variant, basis)
                baseline = build_variant("baseline", basis)
                self.assertEqual(operation_signature(circuit)[3:], operation_signature(baseline))
                self.assertEqual(circuit.count_ops()["measure"], 3)
                self.assertEqual(circuit.count_ops()["barrier"], 5)
                self.assertFalse(circuit.metadata["postselect"])

    def test_ideal_data_state_and_all_basis_probabilities_match_baseline(self):
        for variant in INITIALIZATION_VARIANTS:
            for basis in BASES:
                candidate = build_initialization_candidate(variant, basis)
                baseline = build_variant("baseline", basis)
                actual = DensityMatrix.from_instruction(candidate.remove_final_measurements(inplace=False))
                expected = DensityMatrix.from_instruction(baseline.remove_final_measurements(inplace=False))
                self.assertAlmostEqual(state_fidelity(actual, expected), 1.0)
                np.testing.assert_allclose(actual.probabilities(), expected.probabilities(), atol=1e-14)

    def test_fake_target_support_and_compilation_retain_initialization(self):
        backend = FakeTorino()
        self.assertTrue(backend.target.instruction_supported("reset", (0,)))
        self.assertTrue(backend.target.instruction_supported("delay", (0,)))
        for variant in INITIALIZATION_VARIANTS:
            for level in (1, 3):
                compiled = transpile(
                    build_initialization_candidate(variant, "ZZ"), backend=backend,
                    initial_layout=[0, 1, 2], optimization_level=level, seed_transpiler=11,
                )
                self.assertEqual(compiled.count_ops().get("cz", 0), 1)
                self.assertEqual(compiled.count_ops()["barrier"], 5)
                if variant == "baseline_reset":
                    self.assertEqual(compiled.count_ops().get("reset", 0), 2)
                else:
                    delays = [item for item in compiled.data if item.operation.name == "delay"]
                    self.assertEqual(len(delays), 2)
                    for item in delays:
                        duration = item.operation.duration
                        seconds = duration * backend.dt if item.operation.unit == "dt" else duration * 1e-6
                        self.assertAlmostEqual(seconds, 0.0002)
                # Target reports timing for reset; delay timing is carried by
                # the instruction.  This also checks offline duration support.
                self.assertGreater(compiled.estimate_duration(backend.target), 0)

    def test_toy_thermal_model_wait_can_help_or_hurt(self):
        # Explicit illustrative assumptions, not a FakeTorino/QPU prediction:
        # independent data excitations, T1=250 us, T2=250 us, equilibrium=2%.
        # Standard ideal simulation treats delay as identity and cannot show
        # relaxation; we apply the corresponding thermal channel explicitly.
        bell = Statevector.from_label("000").evolve(
            build_variant("baseline", "ZZ").remove_final_measurements(inplace=False)
        )
        circuit = build_variant("baseline", "ZZ").remove_final_measurements(inplace=False)
        channel = thermal_relaxation_error(250, 250, 200, excited_state_population=0.02).to_quantumchannel()
        for initial_excitation, expected_direction in [(0.05, "improve"), (0.005, "worsen")]:
            probabilities = [
                ((1-initial_excitation) if not (bits & 1) else initial_excitation)
                * ((1-initial_excitation) if not (bits & 2) else initial_excitation)
                if bits < 4 else 0.0
                for bits in range(8)
            ]
            initial = DensityMatrix(np.diag(probabilities))
            before = state_fidelity(initial.evolve(circuit), bell)
            waited = initial.evolve(channel, qargs=[0]).evolve(channel, qargs=[1])
            after = state_fidelity(waited.evolve(circuit), bell)
            if expected_direction == "improve":
                self.assertGreater(after, before)
            else:
                self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
