"""Offline fault-branch validation and retained experiment metadata checks."""
import json
import unittest

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from optimization_analysis import analyze_records, summarize_candidate
from optimization_circuits import VARIANTS
from optimize_bell import (
    build_candidate_circuits, compact_summary, records_for, validate_ideal_counts,
)


def ideal_counts(circuit, shots=256):
    state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    counts = {bits: round(float(probability) * shots)
              for bits, probability in state.probabilities_dict().items() if probability > 1e-12}
    if sum(counts.values()) != shots:
        raise AssertionError("Test state probabilities do not yield exact integer counts.")
    return counts


class FaultBranchTests(unittest.TestCase):
    def candidate(self, variant, fault=None):
        return {"id": f"{variant}_{fault or 'clean'}", "variant": variant,
                "layout": [0, 1, 2], "injected_fault": fault}

    def test_all_fault_branches_match_physics_and_preserve_metadata(self):
        for variant in VARIANTS:
            for fault in (None, "X", "Z"):
                with self.subTest(variant=variant, fault=fault):
                    circuits = build_candidate_circuits(self.candidate(variant, fault))
                    counts = [ideal_counts(circuit) for circuit in circuits]
                    for circuit, raw in zip(circuits, counts):
                        validate_ideal_counts(circuit, raw)
                        expected_ancilla = "1" if fault == "X" and variant in ("protected", "bridge") else "0"
                        self.assertTrue(all(bits[0] == expected_ancilla for bits in raw))
                    records = records_for(circuits, counts, "offline-fault-validation")
                    self.assertTrue(all(record["injected_fault"] == fault for record in records))
                    summary = analyze_records(records)[circuits[0].metadata["candidate_id"]]
                    if fault == "X" and variant in ("protected", "bridge"):
                        self.assertIsNone(summary["fidelity"])
                        self.assertEqual(summary["retention"], 0)
                    else:
                        self.assertEqual(summary["fidelity"], 1 if fault is None else 0)
                    json.dumps(summary, allow_nan=False)

    def test_initialization_operations_survive_all_fault_and_basis_branches(self):
        for variant, operation in (("baseline_reset", "reset"), ("baseline_relax_200us", "delay")):
            for fault in (None, "X", "Z"):
                for circuit in build_candidate_circuits(self.candidate(variant, fault)):
                    with self.subTest(variant=variant, fault=fault, basis=circuit.metadata["basis"]):
                        self.assertEqual(circuit.count_ops()[operation], 2)
                        self.assertEqual([item.operation.name for item in circuit.data[:3]],
                                         [operation, operation, "barrier"])
                        self.assertEqual([circuit.find_bit(item.qubits[0]).index
                                          for item in circuit.data[:2]], [0, 1])
                        if operation == "delay":
                            self.assertEqual([(item.operation.duration, item.operation.unit)
                                              for item in circuit.data[:2]], [(200, "us"), (200, "us")])
                        raw = ideal_counts(circuit)
                        self.assertTrue(all(bits[0] == "0" for bits in raw))
                        validate_ideal_counts(circuit, raw)

    def test_omitted_fault_retains_original_clean_behavior(self):
        candidate = self.candidate("baseline")
        del candidate["injected_fault"]
        for circuit in build_candidate_circuits(candidate):
            self.assertIsNone(circuit.metadata["injected_fault"])
            validate_ideal_counts(circuit, ideal_counts(circuit))

    def test_validator_rejects_wrong_syndrome_and_wrong_basis_parity(self):
        circuit = build_candidate_circuits(self.candidate("bridge", "X"))[0]  # XX
        validate_ideal_counts(circuit, {"100": 128, "111": 128})
        for wrong in ({"000": 256}, {"101": 256}, {"100": 255}):
            with self.subTest(counts=wrong), self.assertRaises(ValueError):
                validate_ideal_counts(circuit, wrong)
        clean = build_candidate_circuits(self.candidate("baseline"))[1]  # YY
        with self.assertRaises(ValueError):
            validate_ideal_counts(clean, {"000": 256})
        phase_fault = build_candidate_circuits(self.candidate("baseline", "Z"))[0]  # XX
        with self.assertRaises(ValueError):
            validate_ideal_counts(phase_fault, {"000": 256})

    def test_compiled_bridge_faults_remain_valid_without_routing(self):
        path = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1]])
        for fault in (None, "X", "Z"):
            compiled = transpile(build_candidate_circuits(self.candidate("bridge", fault)),
                                 basis_gates=["u", "cx"], coupling_map=path, initial_layout=[0, 1, 2],
                                 routing_method="none", optimization_level=1, seed_transpiler=11)
            for circuit in compiled:
                self.assertEqual(circuit.count_ops()["cx"], 4)
                validate_ideal_counts(circuit, ideal_counts(circuit))

    def test_entirely_rejected_fault_branch_is_reportable_next_to_clean_branch(self):
        circuits = (build_candidate_circuits(self.candidate("bridge", "X"))
                    + build_candidate_circuits(self.candidate("bridge")))
        records = records_for(circuits, [ideal_counts(circuit) for circuit in circuits], "offline-mixed")
        summary = analyze_records(records)
        compact = compact_summary(summary)
        self.assertEqual(compact[0]["fidelity"], 1)
        self.assertIsNone(compact[1]["fidelity"])
        self.assertEqual(compact[1]["retention"], 0)
        json.dumps({"records": records, "summary": summary, "compact": compact}, allow_nan=False)

    def test_fixed_ten_percent_x_mixture_uses_raw_counts_before_postselection(self):
        # Nine equally sampled clean preparations and one X preparation form
        # the requested controlled mixture. Never average conditional branch
        # fidelities: the all-X protected branch has no accepted observations.
        for variant in ("baseline", "bridge"):
            pooled = {basis: {} for basis in ("XX", "YY", "ZZ")}
            for branch in range(10):
                fault = "X" if branch == 9 else None
                for circuit in build_candidate_circuits(self.candidate(variant, fault)):
                    counts = ideal_counts(circuit, shots=512)
                    validate_ideal_counts(circuit, counts, expected_shots=512)
                    basis_counts = pooled[circuit.metadata["basis"]]
                    for bits, count in counts.items():
                        basis_counts[bits] = basis_counts.get(bits, 0) + count
            summary = summarize_candidate(pooled, postselect_ancilla=variant == "bridge")
            self.assertEqual(summary["total_shots"], 3 * 10 * 512)
            self.assertAlmostEqual(summary["fidelity"], 1.0 if variant == "bridge" else .9)
            self.assertAlmostEqual(summary["retention"], .9 if variant == "bridge" else 1.0)


if __name__ == "__main__":
    unittest.main()
