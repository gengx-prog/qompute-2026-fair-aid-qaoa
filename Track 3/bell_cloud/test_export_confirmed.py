"""Offline evidence-binding checks for the confirmed-result package."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from qiskit import QuantumCircuit, qasm3, qpy

from export_confirmed import canonical_digest, digest, export
from optimization_analysis import analyze_records, compare_candidates, confirm_candidate, confirm_replicates


class ExportConfirmedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bell-export-test-")
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.stage_names = ["confirm-01-a", "confirm-01-b"]
        self.definitions = [
            {"id": "original_baseline", "variant": "baseline", "layout": [0, 1, 2], "optimization_level": 1},
            {"id": "winner", "variant": "baseline_reset", "layout": [1, 0, 2], "optimization_level": 1},
            {"id": "same_layout_baseline", "variant": "baseline", "layout": [1, 0, 2], "optimization_level": 1},
        ]
        plan = {"candidate_id": "winner", "candidate_definitions": self.definitions,
                "stage_names": self.stage_names, "backend": "offline", "shots_per_basis": 16384,
                "attribution_baseline_id": "same_layout_baseline", "spec_sha256": {}}
        replications = []
        for index, stage in enumerate(self.stage_names):
            folder, job = self.root / stage, f"offline-job-{index}"
            records, circuits = [], []
            for definition in self.definitions:
                name = definition["id"]
                successes = 14000 if name == "original_baseline" else 16000
                for basis in ("XX", "YY", "ZZ"):
                    good, bad = ("001", "000") if basis == "YY" else ("000", "001")
                    records.append({"candidate_id": name, "basis": basis, "job_id": job,
                                    "variant": definition["variant"], "layout": definition["layout"],
                                    "postselect_ancilla": False,
                                    "counts": {good: successes, bad: 16384 - successes}})
                    circuit = QuantumCircuit(3, 3, name=f"{name}_{basis}")
                    circuit.h(0)
                    circuit.cx(0, 1)
                    circuit.measure(range(3), range(3))
                    circuit.metadata = {"candidate_id": name, "basis": basis, "variant": definition["variant"],
                                        "initial_layout": definition["layout"]}
                    circuits.append(circuit)
            for kind in ("logical", "isa"):
                (folder / kind).mkdir(parents=True)
                with (folder / kind / "circuits.qpy").open("wb") as handle:
                    qpy.dump(circuits, handle)
                for circuit in circuits:
                    (folder / kind / f"{circuit.name}.qasm").write_text(qasm3.dumps(circuit), encoding="utf-8")
                self.write(folder / kind / "resources.json", [])
            spec = {"stage": stage, "shots": 16384, "candidates": self.definitions}
            self.write(folder / "spec.json", spec)
            plan["spec_sha256"][stage] = canonical_digest(spec)
            self.write(folder / "manifest.json", {"job_id": job, "backend": "offline", "shots": 16384,
                                                    "rep_delay_seconds": .00025, "cap_seconds": 120,
                                                    "isa_qpy_sha256": digest(folder / "isa" / "circuits.qpy")})
            self.write(folder / "hardware-results.json", {"job_id": job, "records": records})
            self.write(folder / "hardware-usage.json", {"job_id": job, "job_status": "DONE", "usage_status": "complete",
                                                          "reported_qpu_charge_time_seconds": 4})
            summaries = analyze_records(records)
            report = confirm_candidate(summaries, "winner", "original_baseline", screening_job_ids=["screen-job"], alpha=.05 / 3)
            report.update(stage=stage, records_sha256=canonical_digest(records),
                          attribution_comparison=compare_candidates(summaries["winner"], summaries["same_layout_baseline"]))
            replications.append(report)
        self.plan_path = self.root / "confirmation-plan-01.json"
        self.write(self.plan_path, plan)
        aggregate = confirm_replicates(replications)
        aggregate.update(frozen_plan_file=str(self.plan_path), frozen_plan_sha256=canonical_digest(plan))
        self.report_path = self.root / "confirmation-attempt-01.json"
        self.write(self.report_path, aggregate)

    def tearDown(self):
        self.assertEqual(Path(self.temp.name).resolve(), self.root)
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def change(self, path, callback):
        value = json.loads(path.read_text())
        callback(value)
        self.write(path, value)

    def assert_rejected(self):
        with self.assertRaises(ValueError), redirect_stdout(StringIO()):
            export(self.report_path, self.root / "package")
        self.assertFalse((self.root / "package").exists())

    def test_valid_package_binds_all_sources_and_keeps_attribution_secondary(self):
        output = self.root / "package"
        with redirect_stdout(StringIO()):
            export(self.report_path, output)
        config = json.loads((output / "best-config.json").read_text())
        self.assertEqual(config["confirmed_against"], ["original_baseline"])
        self.assertEqual(config["secondary_attribution_control"], "same_layout_baseline")
        self.assertTrue(all(row["improvement"] == 0 for row in config["secondary_attribution_comparisons"]))
        for row in json.loads((output / "evidence-index.json").read_text()):
            self.assertEqual(digest(output / row["packaged_file"]), row["sha256"])
        self.assertTrue((output / self.stage_names[0] / "isa" / "circuits.qpy").exists())

    def test_changed_measurement_records_are_rejected(self):
        path = self.root / self.stage_names[0] / "hardware-results.json"
        self.change(path, lambda row: row["records"][0]["counts"].update({"000": 13999}))
        self.assert_rejected()

    def test_export_accepts_historical_windows_plan_path_after_relocation(self):
        self.change(self.report_path, lambda row: row.update(
            frozen_plan_file=r"G:\Qompute-2026-main\results\confirmation-plan-01.json"))
        with redirect_stdout(StringIO()):
            export(self.report_path, self.root / "relocated-windows")
        self.assertTrue((self.root / "relocated-windows" / self.plan_path.name).is_file())

    def test_export_accepts_stale_posix_plan_path_after_relocation(self):
        self.change(self.report_path, lambda row: row.update(
            frozen_plan_file="/another/computer/results/confirmation-plan-01.json"))
        with redirect_stdout(StringIO()):
            export(self.report_path, self.root / "relocated-posix")
        self.assertTrue((self.root / "relocated-posix" / self.plan_path.name).is_file())

    def test_relocated_plan_still_requires_the_frozen_hash(self):
        self.change(self.report_path, lambda row: row.update(
            frozen_plan_file=r"G:\old\confirmation-plan-01.json"))
        self.change(self.plan_path, lambda row: row.update(backend="changed"))
        self.assert_rejected()

    def test_changed_spec_is_rejected(self):
        self.change(self.root / self.stage_names[0] / "spec.json", lambda row: row.update(shots=8192))
        self.assert_rejected()

    def test_stage_cannot_borrow_other_replication_job_id(self):
        self.change(self.root / self.stage_names[0] / "manifest.json", lambda row: row.update(job_id="offline-job-1"))
        self.assert_rejected()

    def test_usage_from_wrong_job_is_rejected(self):
        self.change(self.root / self.stage_names[0] / "hardware-usage.json", lambda row: row.update(job_id="unrelated-job"))
        self.assert_rejected()

    def test_usage_without_done_state_is_rejected(self):
        self.change(self.root / self.stage_names[0] / "hardware-usage.json", lambda row: row.update(job_status="RUNNING"))
        self.assert_rejected()

    def test_changed_submitted_qpy_is_rejected(self):
        with (self.root / self.stage_names[0] / "isa" / "circuits.qpy").open("ab") as handle:
            handle.write(b"changed")
        self.assert_rejected()

    def test_native_qasm_must_match_the_frozen_qpy(self):
        path = self.root / self.stage_names[0] / "isa" / "winner_XX.qasm"
        path.write_text(path.read_text() + "// modified after submission\n", encoding="utf-8")
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
