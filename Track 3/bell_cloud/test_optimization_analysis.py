"""Independent analytic examples for candidate selection and confirmation."""
import json
import math
import unittest

from optimization_analysis import (
    analyze_records, compare_candidates, confirm_candidate, confirm_replicates,
    screen_candidates, summarize_candidate,
)


def counts_for(successes, shots, *, rejected=0):
    counts = {}
    for basis in ("XX", "YY", "ZZ"):
        good, bad = ("001", "000") if basis == "YY" else ("000", "001")
        counts[basis] = {good: successes, bad: shots - successes}
        if rejected:
            counts[basis]["100"] = rejected
    return counts


def records_for(name, successes, *, shots=8192, job="confirmation-1", protected=False,
                layout=(0, 1, 2), rejected=0):
    return [{"metadata": {"candidate_id": name, "basis": basis, "layout": list(layout),
                          "postselect_ancilla": protected}, "job_id": job, "counts": counts}
            for basis, counts in counts_for(successes, shots, rejected=rejected).items()]


def good_confirmation(job="confirmation-1", *, alpha=0.05 / 3):
    rows = analyze_records(records_for("baseline", 6500, job=job) +
                           records_for("candidate", 7600, job=job))
    return confirm_candidate(rows, "candidate", "baseline", screening_job_ids=["screen-1"], alpha=alpha)


class OptimizationAnalysisTests(unittest.TestCase):
    def test_bell_fidelity_sign_and_independent_standard_error(self):
        result = summarize_candidate(counts_for(80, 100))
        # p=.8 in each of three independently measured success events:
        # F=(3p-1)/2=.7; Var(F)=3*p*(1-p)/(4n)=.0012.
        self.assertAlmostEqual(result["fidelity"], 0.7)
        self.assertAlmostEqual(result["fidelity_standard_error"], math.sqrt(0.0012))
        self.assertEqual([result["correlations"][b]["value"] for b in ("XX", "YY", "ZZ")],
                         [0.6000000000000001, -0.6, 0.6000000000000001])
        self.assertEqual(result["retention"], 1)
        self.assertLess(result["fidelity_interval"][0], result["fidelity"])
        self.assertGreater(result["fidelity_interval"][1], result["fidelity"])

    def test_boundary_uncertainty_is_nonzero_despite_zero_plugin_se(self):
        perfect = summarize_candidate(counts_for(20, 20))
        self.assertEqual(perfect["fidelity"], 1)
        self.assertEqual(perfect["fidelity_standard_error"], 0)
        self.assertLess(perfect["fidelity_interval"][0], 1)
        difference = compare_candidates(perfect, perfect)
        self.assertLess(difference["improvement_lower"], 0)
        self.assertGreater(difference["improvement_interval"][1], 0)

    def test_no_clipping_of_inconsistent_fidelity_or_difference(self):
        wrong = summarize_candidate(counts_for(0, 100))
        perfect = summarize_candidate(counts_for(100, 100))
        self.assertEqual(wrong["fidelity"], -0.5)
        self.assertEqual(compare_candidates(perfect, wrong)["improvement"], 1.5)
        json.dumps(wrong, allow_nan=False)

    def test_leftmost_ancilla_selection_preserves_raw_counts_and_retention(self):
        raw = counts_for(10, 10, rejected=30)
        result = summarize_candidate(raw, postselect_ancilla=True)
        self.assertEqual(result["fidelity"], 1)
        self.assertEqual((result["total_shots"], result["accepted_shots"]), (120, 30))
        self.assertEqual(result["retention"], 0.25)
        self.assertEqual(result["raw_counts"], raw)
        unselected = summarize_candidate(raw)
        self.assertNotEqual(result["fidelity"], unselected["fidelity"])

    def test_metadata_controls_postselection_not_candidate_name(self):
        rows = analyze_records(records_for("protected_in_name_only", 10, shots=10, rejected=10,
                                           protected=False) +
                               records_for("name_without_hint", 10, shots=10, rejected=10,
                                           protected=True))
        self.assertEqual(rows["protected_in_name_only"]["retention"], 1)
        self.assertEqual(rows["name_without_hint"]["retention"], 0.5)

    def test_missing_accepted_basis_returns_unavailable_without_nan(self):
        counts = counts_for(20, 20)
        counts["YY"] = {"100": 20}
        result = summarize_candidate(counts, postselect_ancilla=True)
        self.assertIsNone(result["fidelity"])
        self.assertIsNone(result["fidelity_interval"])
        self.assertIsNone(compare_candidates(result, summarize_candidate(counts_for(10, 20)))["improvement"])
        json.dumps(result, allow_nan=False)

    def test_invalid_counts_and_missing_or_duplicate_bases_are_rejected(self):
        for value in (-1, 1.5, True):
            counts = counts_for(1, 2)
            counts["XX"] = {"000": value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_candidate(counts)
        for key in ("00", "0000", "00x", 0):
            counts = counts_for(1, 2)
            counts["XX"] = {key: 2}
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_candidate(counts)
        records = records_for("a", 1, shots=2)
        with self.assertRaises(ValueError):
            analyze_records(records[:-1])
        with self.assertRaises(ValueError):
            analyze_records(records + [records[0]])
        records[0]["job_id"] = "other-job"
        with self.assertRaises(ValueError):
            analyze_records(records)

    def test_difference_standard_error_is_independent_sum_and_familywise_widens(self):
        a = summarize_candidate(counts_for(90, 100))
        b = summarize_candidate(counts_for(80, 100))
        single = compare_candidates(a, b)
        multiple = compare_candidates(a, b, comparison_count=5)
        self.assertAlmostEqual(single["improvement"], 0.15)
        self.assertAlmostEqual(single["improvement_standard_error"],
                               math.sqrt(a["fidelity_standard_error"]**2 + b["fidelity_standard_error"]**2))
        self.assertLess(multiple["improvement_lower"], single["improvement_lower"])
        self.assertGreater(multiple["improvement_interval"][1], single["improvement_interval"][1])

    def test_screening_ranks_but_does_not_claim_confirmation(self):
        summaries = analyze_records(records_for("base", 6000, job="screen") +
                                    records_for("better", 7000, job="screen") +
                                    records_for("best", 7500, job="screen"))
        screen = screen_candidates(summaries, "base")
        self.assertEqual(screen["selected_candidate_id"], "best")
        self.assertEqual(screen["phase"], "exploratory_screening")
        self.assertEqual(screen["familywise_comparison_count"], 2)
        self.assertNotIn("confirmed_improvement", screen)
        self.assertEqual(screen["screening_job_ids"], ["screen"])

    def test_confirmation_rejects_screening_reuse_or_different_control_job(self):
        summaries = analyze_records(records_for("base", 6000) + records_for("candidate", 7500))
        with self.assertRaises(ValueError):
            confirm_candidate(summaries, "candidate", "base", screening_job_ids=["confirmation-1"])
        summaries["base"]["job_id"] = "another-job"
        with self.assertRaises(ValueError):
            confirm_candidate(summaries, "candidate", "base", screening_job_ids=["screen"])

    def test_protected_requires_unprotected_same_layout_control(self):
        summaries = analyze_records(records_for("base", 6000) +
                                    records_for("protected", 7500, protected=True, layout=(5, 6, 7)) +
                                    records_for("local_base", 7600, layout=(5, 6, 7)))
        with self.assertRaises(ValueError):
            confirm_candidate(summaries, "protected", "base", screening_job_ids=["screen"])
        with self.assertRaises(ValueError):
            confirm_candidate(summaries, "protected", "base", screening_job_ids=["screen"],
                              matching_baseline_id="base")
        report = confirm_candidate(summaries, "protected", "base", screening_job_ids=["screen"],
                                   matching_baseline_id="local_base", alpha=0.05 / 3)
        self.assertTrue(report["comparisons"]["base"]["passes"])
        self.assertFalse(report["comparisons"]["local_base"]["passes"])
        self.assertFalse(report["confirmed_improvement"])
        self.assertEqual(report["comparisons"]["base"]["comparison_count"], 2)

    def test_significant_but_smaller_than_half_point_does_not_pass(self):
        summaries = analyze_records(records_for("base", 5000000, shots=10000000) +
                                    records_for("candidate", 5020000, shots=10000000))
        report = confirm_candidate(summaries, "candidate", "base", screening_job_ids=["screen"])
        row = report["comparisons"]["base"]
        self.assertAlmostEqual(row["improvement"], 0.003)
        self.assertGreater(row["improvement_lower"], 0)
        self.assertFalse(report["confirmed_improvement"])

    def test_replicated_confirmation_requires_two_fresh_passing_jobs(self):
        first, second = good_confirmation(), good_confirmation("confirmation-2")
        self.assertAlmostEqual(first["alpha"], 0.05 / 3)
        self.assertAlmostEqual(first["confidence"], 1 - 0.05 / 3)
        result = confirm_replicates([first, second])
        self.assertTrue(result["confirmed_improvement"])
        self.assertEqual(result["job_ids"], ["confirmation-1", "confirmation-2"])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            confirm_replicates([first, first])
        with self.assertRaises(ValueError):
            confirm_replicates([first])
        with self.assertRaises(ValueError):
            confirm_replicates([first, good_confirmation("confirmation-2", alpha=0.05)])
        second["confirmed_improvement"] = False
        self.assertFalse(confirm_replicates([first, second])["confirmed_improvement"])

    def test_looser_confirmation_rule_is_rejected(self):
        summaries = analyze_records(records_for("base", 6000) + records_for("candidate", 7500))
        for options in ({"alpha": 0.1}, {"minimum_improvement": 0.001}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                confirm_candidate(summaries, "candidate", "base", screening_job_ids=["screen"], **options)


if __name__ == "__main__":
    unittest.main()
