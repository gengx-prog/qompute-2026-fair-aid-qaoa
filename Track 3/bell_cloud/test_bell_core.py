"""Physics checks for Bell preparation, parity rejection, and fidelity signs."""
import json
import unittest

from qiskit.quantum_info import Statevector

from bell_core import analyze_counts, build_circuits


def exact_counts(circuits, shots=4096):
    """The test circuits have dyadic probabilities; obtain exact integer counts."""
    counts_list = []
    for circuit in circuits:
        state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
        counts = {bits: round(probability * shots)
                  for bits, probability in state.probabilities_dict().items()
                  if probability > 1e-12}
        if sum(counts.values()) != shots:
            raise AssertionError("Test probabilities did not map to the intended shot count.")
        counts_list.append(counts)
    return counts_list


class BellPhysicsTests(unittest.TestCase):
    def test_ideal_phi_plus_and_yy_sign(self):
        circuits = build_circuits()
        result = analyze_counts(circuits, exact_counts(circuits))
        for variant in ("baseline", "protected"):
            row = result[variant]
            self.assertAlmostEqual(row["fidelity"], 1.0)
            self.assertEqual(row["acceptance_fraction"], 1.0)
            self.assertEqual([row["correlations"][basis]["value"] for basis in ("XX", "YY", "ZZ")],
                             [1.0, -1.0, 1.0])
            self.assertLess(row["fidelity_interval_95"][0], 1.0)
            self.assertAlmostEqual(row["fidelity_interval_95"][1], 1.0)
        json.dumps(result, allow_nan=False)

    def test_x_fault_is_rejected(self):
        circuits = build_circuits(injected_fault="X")
        result = analyze_counts(circuits, exact_counts(circuits))
        self.assertAlmostEqual(result["baseline"]["fidelity"], 0.0)
        self.assertEqual(result["protected"]["acceptance_fraction"], 0.0)
        self.assertIsNone(result["protected"]["fidelity"])
        self.assertIsNone(result["protected"]["fidelity_standard_error"])
        self.assertIsNone(result["protected"]["fidelity_interval_95"])

    def test_z_fault_passes_undetected(self):
        circuits = build_circuits(injected_fault="Z")
        result = analyze_counts(circuits, exact_counts(circuits))
        for variant in ("baseline", "protected"):
            self.assertEqual(result[variant]["acceptance_fraction"], 1.0)
            self.assertAlmostEqual(result[variant]["fidelity"], 0.0)

    def test_classical_mixture_has_one_half_fidelity(self):
        # Equal mixture of |00> and |11>: XX=YY=0, ZZ=1. Matching
        # populations in ZZ alone are insufficient evidence of entanglement.
        transverse = {"000": 256, "001": 256, "010": 256, "011": 256}
        longitudinal = {"000": 512, "011": 512}
        result = analyze_counts(build_circuits(),
                                [transverse, transverse, longitudinal] * 2)
        for variant in ("baseline", "protected"):
            self.assertAlmostEqual(result[variant]["fidelity"], 0.5)
            self.assertGreater(result[variant]["fidelity_standard_error"], 0.0)

    def test_ancilla_is_leftmost_bit_and_baseline_keeps_all_shots(self):
        # q2 q1 q0: 001 is odd DATA parity with ancilla 0; 100 has
        # even DATA parity with ancilla 1. This detects accidental reversal.
        counts = {"001": 3, "100": 7}
        result = analyze_counts(build_circuits(), [counts] * 6)
        baseline = result["baseline"]["correlations"]["XX"]
        protected = result["protected"]["correlations"]["XX"]
        self.assertEqual(baseline["accepted_shots"], 10)
        self.assertAlmostEqual(baseline["value"], 0.4)
        self.assertEqual(protected["accepted_shots"], 3)
        self.assertEqual(protected["value"], -1.0)
        self.assertEqual(protected["acceptance_fraction"], 0.3)

    def test_one_missing_accepted_basis_makes_fidelity_unavailable(self):
        circuits = build_circuits()
        counts = exact_counts(circuits)
        counts[4] = {"100": 4096}
        result = analyze_counts(circuits, counts)
        self.assertIsNone(result["protected"]["fidelity"])
        self.assertIsNotNone(result["protected"]["correlations"]["ZZ"]["value"])
        json.dumps(result, allow_nan=False)

    def test_all_measurements_are_terminal(self):
        for circuit in build_circuits():
            measurement_seen = False
            for instruction in circuit.data:
                if instruction.operation.name == "measure":
                    measurement_seen = True
                elif measurement_seen:
                    self.fail("Operations after measurement would change hardware requirements.")
            self.assertEqual(circuit.num_qubits, 3)
            self.assertEqual([(register.name, register.size) for register in circuit.cregs], [("meas", 3)])

    def test_raw_fidelity_is_not_silently_clipped(self):
        # Deliberately inconsistent correlators XX=-1, YY=+1, ZZ=-1
        # give -0.5, revealing the unconstrained finite-shot estimator.
        counts = [{"001": 10}, {"000": 10}, {"001": 10}] * 2
        result = analyze_counts(build_circuits(), counts)
        self.assertEqual(result["baseline"]["fidelity"], -0.5)


if __name__ == "__main__":
    unittest.main()
