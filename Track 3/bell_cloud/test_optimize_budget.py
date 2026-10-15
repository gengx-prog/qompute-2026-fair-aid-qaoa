"""Regression checks for the campaign's cross-job spending guard."""
import json
from pathlib import Path
import tempfile
import unittest

from optimize_bell import remaining_budget, validate_budget


class CampaignBudgetTests(unittest.TestCase):
    def test_execution_reservation_and_work_limits(self):
        validate_budget(8192, 9, 90, 595, 58)
        for args in [(8192, 9, 90, 99, 58), (8192, 9, 121, 595, 58),
                     (8192, 9, 90, 595, 91), (16385, 9, 90, 595, 58)]:
            with self.assertRaises(ValueError):
                validate_budget(*args)

    def test_completed_jobs_deduct_actual_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, charge in (("a", 7), ("b", 23)):
                stage = root / name
                stage.mkdir()
                (stage / "manifest.json").write_text(json.dumps({"job_id": name, "submission_state": "submitted"}))
                (stage / "hardware-usage.json").write_text(json.dumps({"usage_status": "complete", "reported_qpu_charge_time_seconds": charge}))
            self.assertEqual(remaining_budget(root), 565)

    def test_pending_or_unknown_charge_blocks_next_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "a"
            stage.mkdir()
            (stage / "manifest.json").write_text(json.dumps({"job_id": "a", "submission_state": "submitted"}))
            with self.assertRaises(ValueError):
                remaining_budget(root)
            (stage / "hardware-usage.json").write_text(json.dumps({"usage_status": "pending", "reported_qpu_charge_time_seconds": 0}))
            with self.assertRaises(ValueError):
                remaining_budget(root)

    def test_ambiguous_submission_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "a"
            stage.mkdir()
            (stage / "manifest.json").write_text(json.dumps({"job_id": None, "submission_state": "attempted_check_platform_before_retry"}))
            with self.assertRaises(ValueError):
                remaining_budget(root)

    def test_read_only_preparation_reserves_pending_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "a"
            stage.mkdir()
            (stage / "manifest.json").write_text(json.dumps({"job_id": "a", "submission_state": "submitted", "cap_seconds": 90}))
            self.assertEqual(remaining_budget(root, reserve_pending=True), 505)
            with self.assertRaises(ValueError):
                remaining_budget(root)


if __name__ == "__main__":
    unittest.main()
