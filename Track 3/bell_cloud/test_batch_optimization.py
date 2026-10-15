"""Offline accounting, integrity, and single-attempt Batch submission checks."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from qiskit import QuantumCircuit, qpy

import batch_optimization as batch_runner

SOURCE = {"optimize_bell.py": "verified-main", "optimization_circuits.py": "verified-circuits",
          "optimization_analysis.py": "verified-analysis"}


class OfflineBatch:
    def __init__(self, backend, max_time):
        self.backend = backend
        self.max_time = max_time
        self.session_id = "offline-batch-id"
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        return False


class BatchOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bell-batch-test-")
        self.campaign = Path(self.temp.name).resolve()
        self.assertEqual(self.campaign.parent, Path(tempfile.gettempdir()).resolve())
        self.hash_patch = patch.object(batch_runner.campaign_runner, "hashes", return_value=SOURCE)
        self.hash_patch.start()
        self.names = [f"scan-{number:02d}" for number in (3, 4, 5, 6)]
        for name in self.names:
            self.make_stage(name)
        self.write(self.campaign / "finished" / "manifest.json",
                   {"job_id": "finished-job", "submission_state": "submitted", "cap_seconds": 90})
        self.write(self.campaign / "finished" / "hardware-usage.json",
                   {"job_id": "finished-job", "usage_status": "complete", "reported_qpu_charge_time_seconds": 48})
        self.write(self.campaign / "queued" / "manifest.json",
                   {"job_id": "queued-job", "submission_state": "submitted", "cap_seconds": 90})
        self.backend = Mock()
        self.backend.name = "offline_backend"
        self.backend.configuration.return_value = SimpleNamespace(simulator=False, rep_delay_range=[0, .001])
        self.backend.status.return_value = SimpleNamespace(operational=True, status_msg="active")
        self.service = Mock()
        self.service.backend.return_value = self.backend

    def tearDown(self):
        self.hash_patch.stop()
        # Verify the exact uniquely created temporary path before its recursive
        # cleanup; no workspace or computed external path is ever removed.
        self.assertEqual(Path(self.temp.name).resolve(), self.campaign)
        self.assertEqual(self.campaign.parent, Path(tempfile.gettempdir()).resolve())
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_stage(self, name):
        folder = self.campaign / name
        (folder / "isa").mkdir(parents=True)
        circuits = []
        for basis in ("XX", "YY", "ZZ"):
            circuit = QuantumCircuit(3, 3, name=f"{name}_{basis}")
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.measure(range(3), range(3))
            circuits.append(circuit)
        path = folder / "isa" / "circuits.qpy"
        with path.open("wb") as handle:
            qpy.dump(circuits, handle)
        manifest = {"stage": name, "submission_state": "prepared", "job_id": None,
                    "backend": "offline_backend", "local_validation_passed": True,
                    "source_sha256": SOURCE, "isa_qpy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "shots": 1024, "circuit_count": 3, "cap_seconds": 90,
                    "estimated_system_seconds": 25, "rep_delay_seconds": .00025, "job_options": {}}
        self.write(folder / "manifest.json", manifest)
        self.write(folder / "local-validation.json", {"ideal_passed": True, "compiled_circuit_count": 3})

    def mutate(self, name, **updates):
        path = self.campaign / name / "manifest.json"
        row = json.loads(path.read_text())
        row.update(updates)
        self.write(path, row)

    def test_budget_reserves_pending_caps_and_all_four_planned_jobs(self):
        _, _, budget = batch_runner.preflight(self.names, self.campaign)
        self.assertEqual(budget["known_final_used_seconds"], 53)
        self.assertEqual(budget["pending_caps_reserved_seconds"], 90)
        self.assertEqual(budget["uncommitted_seconds"], 457)
        self.assertEqual(budget["planned_caps_seconds"], 360)
        self.assertEqual(budget["worst_case_used_plus_reserve_seconds"], 513)

    def test_pending_reported_zero_still_reserves_full_cap(self):
        self.write(self.campaign / "queued" / "hardware-usage.json",
                   {"job_id": "queued-job", "usage_status": "pending", "reported_qpu_charge_time_seconds": 0})
        self.assertEqual(batch_runner.budget_reservations(self.campaign)["pending_caps_reserved_seconds"], 90)

    def test_excess_budget_blocks_before_account_lookup(self):
        self.write(self.campaign / "finished" / "hardware-usage.json",
                   {"job_id": "finished-job", "usage_status": "complete", "reported_qpu_charge_time_seconds": 200})
        with patch.object(batch_runner.campaign_runner, "service") as service, self.assertRaises(ValueError):
            batch_runner.submit_batch(self.names, campaign=self.campaign)
        service.assert_not_called()

    def test_ambiguous_existing_submission_blocks(self):
        self.mutate("queued", job_id=None)
        with self.assertRaises(ValueError):
            batch_runner.preflight(self.names, self.campaign)

    def test_duplicate_and_path_escape_stage_names_block(self):
        for names in ([self.names[0]] * 2, ["../outside"], []):
            with self.subTest(names=names), self.assertRaises(ValueError):
                batch_runner.preflight(names, self.campaign)

    def test_changed_source_or_qpy_is_rejected(self):
        self.mutate(self.names[0], source_sha256={"wrong": "hash"})
        with self.assertRaises(ValueError):
            batch_runner.preflight(self.names, self.campaign)
        self.mutate(self.names[0], source_sha256=SOURCE)
        with (self.campaign / self.names[0] / "isa" / "circuits.qpy").open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaises(ValueError):
            batch_runner.preflight(self.names, self.campaign)

    def test_missing_validation_and_wrong_cap_are_rejected(self):
        self.mutate(self.names[0], local_validation_passed=False)
        with self.assertRaises(ValueError):
            batch_runner.preflight(self.names, self.campaign)
        self.mutate(self.names[0], local_validation_passed=True, cap_seconds=121)
        with self.assertRaises(ValueError):
            batch_runner.preflight(self.names, self.campaign)

    def make_sampler_factory(self, *, fail_on=None, expected_cap=90):
        samplers = []

        def factory(mode):
            options = SimpleNamespace(max_execution_time=None, execution=SimpleNamespace(rep_delay=None),
                                      twirling=SimpleNamespace(enable_gates=None, enable_measure=None))
            sampler = Mock(options=options)
            samplers.append(sampler)
            index = len(samplers)

            def run(circuits, shots):
                self.assertIsInstance(mode, OfflineBatch)
                self.assertEqual(mode.max_time, 600)
                self.assertEqual(options.max_execution_time, expected_cap)
                self.assertEqual(options.execution.rep_delay, .00025)
                self.assertFalse(options.twirling.enable_gates)
                self.assertFalse(options.twirling.enable_measure)
                self.assertEqual(shots, 1024)
                stage = circuits[0].name.rsplit("_", 1)[0]
                manifest = json.loads((self.campaign / stage / "manifest.json").read_text())
                self.assertEqual(manifest["submission_state"], "attempted_check_platform_before_retry")
                self.assertEqual(manifest["batch_id"], "offline-batch-id")
                if index == fail_on:
                    raise TimeoutError("Simulated response lost after possible acceptance")
                return Mock(job_id=Mock(return_value=f"new-job-{index}"))

            sampler.run.side_effect = run
            return sampler

        return factory, samplers

    def test_four_capped_runs_in_one_batch_with_network_blocked_and_no_duplicates(self):
        factory, samplers = self.make_sampler_factory()
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), \
                patch.object(batch_runner.campaign_runner, "service", return_value=self.service), redirect_stdout(StringIO()):
            ledger = batch_runner.submit_batch(self.names, campaign=self.campaign,
                                              batch_factory=OfflineBatch, sampler_factory=factory)
        self.assertEqual(len(samplers), 4)
        self.assertTrue(all(sampler.run.call_count == 1 for sampler in samplers))
        self.assertEqual(ledger["job_ids"], ["new-job-1", "new-job-2", "new-job-3", "new-job-4"])
        self.assertEqual(ledger["batch_ttl_wall_seconds"], 600)
        self.assertEqual(ledger["submission_state"], "submitted_and_batch_closed_to_new_jobs")
        self.assertFalse((self.campaign / "batch-submission.lock").exists())
        self.assertEqual(batch_runner.budget_reservations(self.campaign)["uncommitted_seconds"], 97)
        with patch.object(batch_runner.campaign_runner, "service") as service, self.assertRaises(ValueError):
            batch_runner.submit_batch(self.names, campaign=self.campaign)
        service.assert_not_called()

    def test_lost_response_stops_remaining_jobs_and_keeps_attempt_lock(self):
        factory, samplers = self.make_sampler_factory(fail_on=2)
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), \
                patch.object(batch_runner.campaign_runner, "service", return_value=self.service), redirect_stdout(StringIO()), \
                self.assertRaises(TimeoutError):
            batch_runner.submit_batch(self.names, campaign=self.campaign,
                                      batch_factory=OfflineBatch, sampler_factory=factory)
        self.assertEqual(len(samplers), 2)
        self.assertTrue(all(sampler.run.call_count == 1 for sampler in samplers))
        self.assertTrue((self.campaign / "batch-submission.lock").exists())
        state2 = json.loads((self.campaign / self.names[1] / "manifest.json").read_text())
        state3 = json.loads((self.campaign / self.names[2] / "manifest.json").read_text())
        self.assertEqual(state2["submission_state"], "attempted_check_platform_before_retry")
        self.assertIsNone(state2["job_id"])
        self.assertEqual(state3["submission_state"], "prepared")
        with self.assertRaises(ValueError):
            batch_runner.preflight([self.names[2]], self.campaign)

    def test_two_120_second_confirmation_jobs_reserve_and_propagate_manifest_caps(self):
        names = self.names[:2]
        for name in names:
            self.mutate(name, cap_seconds=120, estimated_system_seconds=95)
        _, _, budget = batch_runner.preflight(names, self.campaign)
        self.assertEqual(budget["planned_caps_seconds"], 240)
        self.assertEqual(budget["worst_case_used_plus_reserve_seconds"], 393)
        factory, samplers = self.make_sampler_factory(expected_cap=120)
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), \
                patch.object(batch_runner.campaign_runner, "service", return_value=self.service), redirect_stdout(StringIO()):
            ledger = batch_runner.submit_batch(names, campaign=self.campaign,
                                              batch_factory=OfflineBatch, sampler_factory=factory)
        self.assertEqual(len(samplers), 2)
        self.assertEqual(ledger["per_job_execution_caps_seconds"], {name: 120 for name in names})
        self.assertEqual(batch_runner.budget_reservations(self.campaign)["pending_caps_reserved_seconds"], 330)


if __name__ == "__main__":
    unittest.main()
