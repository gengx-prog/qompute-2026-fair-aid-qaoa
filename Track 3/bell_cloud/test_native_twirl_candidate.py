"""Offline validation of a native-CZ ensemble and same-job control compatibility."""
import unittest
from unittest.mock import patch

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime.fake_provider import FakeTorino

from native_twirl_candidate import aggregate_ensemble_counts, make_native_cz_twirl_ensemble
from optimization_circuits import BASES, build_variant


def compact_operator(circuit):
    """The offline test fixes logical wires to physical 0,1,2."""
    result = QuantumCircuit(3)
    result.global_phase = circuit.global_phase
    for item in circuit.data:
        if item.operation.name in ("measure", "barrier"):
            continue
        result.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
    return Operator(result)


class NativeTwirlCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = FakeTorino()
        source = [build_variant("baseline_quiet", basis) for basis in BASES]
        compiled = transpile(source, backend=cls.backend, initial_layout=[0, 1, 2],
                             optimization_level=1, seed_transpiler=11)
        cls.controls = dict(zip(BASES, compiled))
        cls.ensemble = make_native_cz_twirl_ensemble(
            cls.controls, target=cls.backend.target,
            candidate_id="offline_twirl", num_twirls=8, seed=1701,
        )

    def test_each_native_member_is_equivalent_and_uses_supported_gates(self):
        for circuit in self.ensemble:
            self.assertTrue(compact_operator(circuit).equiv(
                compact_operator(self.controls[circuit.metadata["basis"]])))
            self.assertEqual(circuit.count_ops().get("cz", 0), 1)
            self.assertEqual(circuit.count_ops().get("barrier", 0), 4)
            self.assertFalse(circuit.metadata["postselect"])
            for item in circuit.data:
                if item.operation.name == "barrier":
                    continue
                qubits = tuple(circuit.find_bit(q).index for q in item.qubits)
                self.assertTrue(self.backend.target.instruction_supported(item.operation.name, qubits))

    def test_counts_preserve_uniform_weights_and_detect_missing_member(self):
        counts = [{"000": 8, "011": 8} for _ in self.ensemble]
        pooled = aggregate_ensemble_counts(self.ensemble, counts)
        self.assertEqual(pooled, {basis: {"000": 64, "011": 64} for basis in BASES})
        with self.assertRaises(ValueError):
            aggregate_ensemble_counts(self.ensemble[:-1], counts[:-1])
        counts[-1] = {"000": 8, "011": 7}
        with self.assertRaises(ValueError):
            aggregate_ensemble_counts(self.ensemble, counts)

    def test_current_sampler_accepts_controls_and_twirls_in_one_offline_job(self):
        from qiskit_ibm_runtime.executor_sampler import Sampler

        # Service-side twirling stays disabled for the entire job.  Only the
        # hand-generated candidate circuits contain the Pauli randomization.
        circuits = list(self.controls.values()) + self.ensemble
        failure = AssertionError("No network is permitted in this local test.")
        with patch("socket.socket.connect", side_effect=failure), \
                patch("socket.create_connection", side_effect=failure):
            sampler = Sampler(mode=self.backend)
            sampler.options.twirling.enable_gates = False
            sampler.options.twirling.enable_measure = False
            result = sampler.run(circuits, shots=16).result()
        self.assertEqual(len(result), len(circuits))
        counts = [pub.data.meas.get_counts() for pub in result[3:]]
        pooled = aggregate_ensemble_counts(self.ensemble, counts)
        for basis in BASES:
            self.assertEqual(sum(pooled[basis].values()), 128)
            self.assertTrue(all(len(bits) == 3 and bits[0] == "0" for bits in pooled[basis]))


if __name__ == "__main__":
    unittest.main()
