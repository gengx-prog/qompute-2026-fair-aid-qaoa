"""Offline protocol checks; all data below are synthetic, never QPU results."""
import copy
from datetime import datetime, timezone
import json
import unittest

from stress_protocol import analyze_job, confirm_jobs, freeze_protocol, make_specs


def synthetic_job(plan, spec, job_id, *, ineffective_bridge=False, imperfect=False):
    records = []
    for candidate in spec["candidates"]:
        for basis in ("XX", "YY", "ZZ"):
            fault = candidate["injected_fault"] == "X"
            # Phi+ correlations: +XX,-YY,+ZZ. X on one data bit gives Psi+:
            # +XX,+YY,-ZZ. The ideal detector writes the odd syndrome to ancilla.
            even = (basis != "YY") if not fault else (basis != "ZZ")
            ancilla = int(fault and candidate["variant"] == "bridge" and not ineffective_bridge)
            right = "00" if even else "01"
            counts = {f"{ancilla}{right}": 512}
            if imperfect:
                counts[f"{ancilla}{right}"] = 500
                wrong = "01" if even else "00"
                counts[f"{ancilla}{wrong}"] = 12
            records.append({"candidate_id": candidate["id"], "basis": basis,
                            "layout": plan["layout"], "variant": candidate["variant"],
                            "injected_fault": candidate["injected_fault"],
                            "postselect_ancilla": candidate["variant"] == "bridge",
                            "job_id": job_id, "counts": counts})
    data = {"job_id": job_id, "backend": plan["backend"], "records": records, "usage": {}}
    manifest = {"job_id": job_id, "stage": spec["stage"], "shots": 512,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "statistics_plan": dict(plan),
                "source_sha256": dict(plan["source_sha256"])}
    return spec, data, manifest


class StressProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = freeze_protocol([7, 6, 5], ["stress-test-01", "stress-test-02"], ["excluded-screen-job"])
        self.specs = make_specs(self.plan)

    def test_frozen_specs_have_equal_channel_and_exact_acquisition_cost(self):
        for spec in self.specs:
            self.assertEqual(len(spec["candidates"]), 20)
            self.assertEqual(len({row["id"] for row in spec["candidates"]}), 20)
            for variant in ("baseline", "bridge"):
                rows = [row for row in spec["candidates"] if row["variant"] == variant]
                self.assertEqual(sum(row["injected_fault"] is None for row in rows), 9)
                self.assertEqual(sum(row["injected_fault"] == "X" for row in rows), 1)
            self.assertEqual(3 * len(spec["candidates"]) * spec["shots"], 30720)

    def test_ideal_mixture_pools_before_postselection_and_retains_costs(self):
        result = analyze_job(self.plan, *synthetic_job(self.plan, self.specs[0], "new-job-1"))
        self.assertAlmostEqual(result["scenarios"]["p0"]["baseline"]["fidelity"], 1)
        self.assertAlmostEqual(result["scenarios"]["p10"]["baseline"]["fidelity"], 0.9)
        bridge = result["scenarios"]["p10"]["bridge"]
        self.assertAlmostEqual(bridge["fidelity"], 1)
        self.assertAlmostEqual(bridge["retention"], 0.9)
        self.assertEqual(bridge["discarded_shots"], 1536)
        self.assertTrue(result["primary"]["passes"])
        self.assertFalse(result["ambient_optimization_success"])
        self.assertEqual(result["cost"]["total_shots"], 30720)
        self.assertEqual(result["cost"]["p0_subset_shots"], 27648)

    def test_bootstrap_preserves_shared_sample_dependence_and_is_deterministic(self):
        inputs = synthetic_job(self.plan, self.specs[0], "new-job-1", imperfect=True)
        first = analyze_job(self.plan, *inputs)
        second = analyze_job(self.plan, *inputs)
        self.assertEqual(first, second)
        self.assertGreater(first["shared_samples"]["delta_bootstrap_correlation"], 0.5)
        self.assertEqual(first["bootstrap"]["replicates"], 5000)

    def test_two_new_jobs_both_must_pass_and_cannot_be_reused(self):
        inputs = [synthetic_job(self.plan, spec, f"new-job-{i}") for i, spec in enumerate(self.specs)]
        result = confirm_jobs(self.plan, inputs)
        self.assertTrue(result["confirmed"])
        failed = synthetic_job(self.plan, self.specs[1], "new-job-2", ineffective_bridge=True)
        self.assertFalse(confirm_jobs(self.plan, [inputs[0], failed])["confirmed"])
        duplicate = synthetic_job(self.plan, self.specs[1], "new-job-0")
        with self.assertRaises(ValueError):
            confirm_jobs(self.plan, [inputs[0], duplicate])

    def test_missing_records_screening_reuse_and_changed_spec_are_rejected(self):
        inputs = synthetic_job(self.plan, self.specs[0], "excluded-screen-job")
        with self.assertRaises(ValueError):
            analyze_job(self.plan, *inputs)
        spec, data, manifest = synthetic_job(self.plan, self.specs[0], "new-job-1")
        bad_data = copy.deepcopy(data)
        bad_data["records"].pop()
        with self.assertRaises(ValueError):
            analyze_job(self.plan, spec, bad_data, manifest)
        bad_spec = copy.deepcopy(spec)
        bad_spec["candidates"][-1]["injected_fault"] = None
        with self.assertRaises(ValueError):
            analyze_job(self.plan, bad_spec, data, manifest)

    def test_zero_accepted_bridge_is_unavailable_and_cannot_pass(self):
        spec, data, manifest = synthetic_job(self.plan, self.specs[0], "new-job-1")
        for record in data["records"]:
            if record["variant"] == "bridge":
                record["counts"] = {"100": 512}
        result = analyze_job(self.plan, spec, data, manifest)
        self.assertIsNone(result["scenarios"]["p10"]["bridge"]["fidelity"])
        self.assertIsNone(result["primary"]["confidence_interval_99"])
        self.assertFalse(result["primary"]["passes"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
