"""Read an existing Bell campaign and export inspectable, offline summaries.

No accounts, network access, submissions, or source artifacts are modified.
This command may be rerun while a campaign is incomplete; unfinished or unknown
billing is explicitly incomplete, never silently replaced with zero.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from optimization_analysis import analyze_records, confirm_replicates

BASES = ("XX", "YY", "ZZ")
CSV_FIELDS = (
    "stage", "candidate_id", "variant", "layout", "unordered_data_pair", "phase",
    "fidelity", "fidelity_standard_error", "confidence", "ci_lower", "ci_upper",
    "retention", "postselected", "shots_per_basis", "total_shots", "accepted_shots",
    "native_2q_min", "native_2q_max", "native_2q_by_basis", "depth_min", "depth_max",
    "depth_by_basis", "optimization_level", "job_id", "backend", "source_file",
)


def read_json(path, warnings, *, optional=False):
    path = Path(path)
    if optional and not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.append({"file": str(path), "issue": f"{type(error).__name__}: {error}"})
        return None


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _phase(stage, manifest):
    declared = str((manifest.get("statistics_plan") or {}).get("phase", ""))
    if declared == "controlled_noise":
        return "controlled_noise"
    if stage.startswith(("confirm", "confirmation")) or "confirmation" in declared:
        return "confirmation"
    if stage.startswith(("scan", "screen")) or "exploratory" in declared:
        return "exploratory"
    return "unclassified"


def _pair(layout):
    return tuple(sorted(layout[:2])) if isinstance(layout, list) and len(layout) >= 2 else None


def _usage_summary(campaign, manifests, warnings):
    statistical_plan = read_json(campaign / "statistical-plan.json", warnings, optional=True) or {}
    initial = statistical_plan.get("used_before_campaign", 5.0)
    budget = statistical_plan.get("budget_seconds_total", 600.0)
    if not _number(initial) or not _number(budget):
        raise ValueError("Initial usage and total budget must be nonnegative finite numbers.")
    ledger = []
    submitted = defaultdict(list)
    for stage, manifest in manifests.items():
        job_id = manifest.get("job_id")
        state = manifest.get("submission_state", "unknown")
        if state == "prepared" and not job_id:
            ledger.append({"stage": stage, "job_id": None, "state": "not_submitted", "final_charge_seconds": None})
            continue
        usage = read_json(campaign / stage / "hardware-usage.json", warnings, optional=True) or {}
        final = (usage.get("usage_status") == "complete" and _number(usage.get("reported_qpu_charge_time_seconds"))
                 and job_id is not None and usage.get("job_id") == job_id)
        row = {"stage": stage, "job_id": job_id, "state": state,
               "job_status": usage.get("job_status"), "usage_status": usage.get("usage_status", "unknown"),
               "reported_charge_seconds": usage.get("reported_qpu_charge_time_seconds"),
               "final_charge_seconds": usage["reported_qpu_charge_time_seconds"] if final else None}
        ledger.append(row)
        submitted[job_id or f"ambiguous:{stage}"].append(row)
    known, incomplete = float(initial), []
    for key, rows in submitted.items():
        final_values = {row["final_charge_seconds"] for row in rows if row["final_charge_seconds"] is not None}
        if len(final_values) == 1:
            known += next(iter(final_values))
            if len(rows) > 1:
                warnings.append({"job_id": key, "issue": "Same submitted job appears in several stages; charge counted once."})
        else:
            incomplete.append(key)
            if len(final_values) > 1:
                warnings.append({"job_id": key, "issue": "Conflicting final charges; total accounting is incomplete."})
    complete = not incomplete
    return {"budget_seconds": float(budget), "initial_used_seconds": float(initial),
            "initial_usage_source": "statistical-plan.json used_before_campaign, or documented default of 5 seconds",
            "known_final_used_seconds": known, "accounting_complete": complete,
            "total_used_seconds": known if complete else None,
            "remaining_seconds": float(budget) - known if complete else None,
            "remaining_seconds_upper_bound": float(budget) - known,
            "unsettled_or_ambiguous_jobs": incomplete, "jobs": ledger,
            "note": "Only final unique-job charges enter known usage. Pending or missing usage is unknown, not zero. The remaining upper bound is not permission to submit."}


def _candidate_rows(campaign, manifests, warnings):
    rows, stages = [], []
    for stage, manifest in manifests.items():
        folder = campaign / stage
        phase = _phase(stage, manifest)
        result_path = folder / "hardware-results.json"
        results = read_json(result_path, warnings, optional=True)
        if results is None:
            stages.append({"stage": stage, "status": "results_not_available", "phase": phase,
                           "job_id": manifest.get("job_id")})
            continue
        if not manifest.get("job_id") or results.get("job_id") != manifest["job_id"]:
            warnings.append({"stage": stage, "issue": "Manifest and result job IDs do not agree; results excluded."})
            continue
        records = results.get("records")
        if not isinstance(records, list) or any(record.get("job_id") != manifest["job_id"] for record in records):
            warnings.append({"stage": stage, "issue": "Missing records or inconsistent record job IDs; results excluded."})
            continue
        try:
            summaries = analyze_records(records)
        except (KeyError, TypeError, ValueError) as error:
            warnings.append({"stage": stage, "issue": f"Invalid measurement records: {error}"})
            continue
        spec = read_json(folder / "spec.json", warnings, optional=True) or {}
        definitions = {candidate["id"]: candidate for candidate in spec.get("candidates", [])}
        candidate_order = {candidate["id"]: index for index, candidate in enumerate(spec.get("candidates", []))}
        resources = read_json(folder / "isa" / "resources.json", warnings)
        resource_map = {row["name"]: row for row in resources} if isinstance(resources, list) else {}
        for name, summary in summaries.items():
            definition = definitions.get(name, {})
            resource_rows = {basis: resource_map.get(f"{name}_{basis}") for basis in BASES}
            gate_counts = {basis: row.get("two_qubit_gates") if row else None for basis, row in resource_rows.items()}
            depths = {basis: row.get("depth") if row else None for basis, row in resource_rows.items()}
            if any(value is None for value in [*gate_counts.values(), *depths.values()]):
                warnings.append({"stage": stage, "candidate_id": name, "issue": "Some ISA resource counts are unavailable."})
            complete_gates = all(value is not None for value in gate_counts.values())
            complete_depths = all(value is not None for value in depths.values())
            ci = summary["fidelity_interval"]
            pair = _pair(summary.get("layout"))
            rows.append({
                "stage": stage, "candidate_id": name, "variant": summary.get("variant", definition.get("variant")),
                "layout": summary.get("layout"), "unordered_data_pair": list(pair) if pair else None,
                "phase": phase, "fidelity": summary["fidelity"],
                "fidelity_standard_error": summary["fidelity_standard_error"], "confidence": summary["confidence"],
                "ci_lower": ci[0] if ci else None, "ci_upper": ci[1] if ci else None,
                "retention": summary["retention"], "postselected": summary["postselected"],
                "shots_per_basis": {basis: summary["correlations"][basis]["total_shots"] for basis in BASES},
                "total_shots": summary["total_shots"], "accepted_shots": summary["accepted_shots"],
                "native_2q_min": min(gate_counts.values()) if complete_gates else None,
                "native_2q_max": max(gate_counts.values()) if complete_gates else None,
                "native_2q_by_basis": gate_counts,
                "depth_min": min(depths.values()) if complete_depths else None,
                "depth_max": max(depths.values()) if complete_depths else None, "depth_by_basis": depths,
                "optimization_level": definition.get("optimization_level", 1),
                "job_id": manifest["job_id"], "backend": results.get("backend", manifest.get("backend")),
                "source_file": str(result_path.resolve()),
                "stage_created_utc": manifest.get("created_utc", ""),
                "candidate_order": candidate_order.get(name, len(candidate_order)),
            })
        stages.append({"stage": stage, "status": "results_available", "phase": phase,
                       "job_id": manifest["job_id"], "candidate_count": len(summaries)})
    rows.sort(key=lambda row: (row["stage_created_utc"], row["stage"], row["candidate_order"], row["candidate_id"]))
    return rows, stages


def _exploratory_groups(rows):
    groups, other = defaultdict(list), []
    for row in rows:
        if row["phase"] != "exploratory":
            continue
        pair = _pair(row["layout"])
        if row["variant"] == "baseline" and not row["postselected"] and pair is not None:
            groups[pair].append(row)
        else:
            other.append(row)
    representatives, drift_rows, drift_groups = [], [], []
    for pair, observations in sorted(groups.items()):
        reference = observations[0]
        representatives.append(reference)
        stable = []
        for row in observations:
            same_recipe = row["layout"] == reference["layout"] and row["optimization_level"] == reference["optimization_level"]
            if same_recipe and row["fidelity"] is not None:
                stable.append(row["fidelity"])
            drift_rows.append({**row, "pair_observation_count": len(observations),
                               "reference_stage": reference["stage"], "reference_candidate_id": reference["candidate_id"],
                               "same_recipe_as_reference": same_recipe,
                               "delta_from_first": row["fidelity"] - reference["fidelity"]
                               if row["fidelity"] is not None and reference["fidelity"] is not None else None})
        drift_groups.append({"unordered_data_pair": list(pair), "observation_count": len(observations),
                             "representative_stage": reference["stage"], "representative_candidate_id": reference["candidate_id"],
                             "same_recipe_fidelity_min": min(stable) if stable else None,
                             "same_recipe_fidelity_max": max(stable) if stable else None,
                             "observations": [{key: row[key] for key in ("stage", "candidate_id", "layout", "optimization_level",
                                                                         "job_id", "fidelity", "fidelity_standard_error", "retention")}
                                              for row in observations]})
    # Other circuit variants are distinct configurations; exact configuration
    # repeats likewise use the first observation instead of a selected maximum.
    seen = set()
    for row in other:
        key = (row["variant"], tuple(row["layout"] or []), row["optimization_level"], row["postselected"])
        if key not in seen:
            representatives.append(row)
            seen.add(key)
    ranked = sorted((row for row in representatives if row["fidelity"] is not None),
                    key=lambda row: row["fidelity"], reverse=True)
    return ranked, drift_rows, drift_groups, set(groups)


def _coverage(campaign, observed_pairs, warnings):
    plan = read_json(campaign / "edge-scan-plan.json", warnings, optional=True)
    if plan is None:
        return {"plan_available": False, "observed_unique_baseline_pairs": len(observed_pairs), "complete": None}
    planned = {tuple(sorted(row["data_pair"])) for row in plan.get("coverage", [])}
    matched = planned & observed_pairs
    return {"plan_available": True, "planned_unique_pairs": len(planned),
            "declared_total_unique_undirected_edges": plan.get("total_unique_undirected_edges"),
            "observed_planned_pairs": len(matched), "observed_unique_baseline_pairs": len(observed_pairs),
            "fraction_complete": len(matched) / len(planned) if planned else None,
            "missing_pairs": [list(pair) for pair in sorted(planned - observed_pairs)],
            "observed_unplanned_pairs": [list(pair) for pair in sorted(observed_pairs - planned)],
            "complete": bool(planned) and matched == planned,
            "scope": plan.get("statistics_plan", {}).get("scope"),
            "note": "Coverage counts unordered data pairs with completed exploratory XX/YY/ZZ records. Repeated controls do not add coverage; only the prescribed orientations were scanned."}


def _confirmations(campaign, warnings):
    files, verified = [], []
    for path in sorted(campaign.glob("confirmation-*.json")):
        report = read_json(path, warnings)
        if not isinstance(report, dict):
            continue
        files.append({"file": str(path.resolve()), "report": report})
        if report.get("phase") == "replicated_confirmation":
            try:
                checked = confirm_replicates(report["replications"],
                                             required_replicates=report.get("required_replicates", 2),
                                             max_candidate_attempts=report.get("max_candidate_attempts", 3),
                                             family_alpha=report.get("family_alpha", 0.05))
                if checked["confirmed_improvement"]:
                    verified.append({"file": str(path.resolve()), "candidate_id": checked["candidate_id"],
                                     "job_ids": checked["job_ids"]})
            except (KeyError, TypeError, ValueError) as error:
                warnings.append({"file": str(path), "issue": f"Confirmation report could not be verified: {error}"})
    return {"status": "confirmed" if verified else "not_confirmed", "verified_candidates": verified,
            "reports": files,
            "note": "A screening maximum or one passing replication is not confirmed. Confirmation requires the recorded prespecified independent replications; absence of confirmation does not prove equal performance."}


def _controlled_noise(campaign, rows, stages, warnings):
    """Keep injected-noise mechanism evidence separate from ambient optimization."""
    controlled_stages = [row for row in stages if row.get("phase") == "controlled_noise"]
    observations = [row for row in rows if row["phase"] == "controlled_noise"]
    report_path = campaign / "controlled-noise-report.json"
    report = read_json(report_path, warnings, optional=True)
    per_stage_reports = []
    for stage in controlled_stages:
        path = campaign / stage["stage"] / "stress-analysis.json"
        value = read_json(path, warnings, optional=True)
        if value is not None:
            per_stage_reports.append({"stage": stage["stage"], "file": str(path.resolve()), "report": value})
    confirmed = False
    if report is not None:
        if (not isinstance(report, dict) or report.get("phase") != "controlled_noise"
                or report.get("ambient_optimization_success") is not False):
            warnings.append({"file": str(report_path),
                             "issue": "Controlled-noise report has missing or inconsistent scope; it does not establish ambient improvement."})
        elif report.get("confirmed") is True:
            jobs = report.get("jobs", [])
            reported_ids = [job.get("job_id") for job in jobs]
            observed_ids = {row["job_id"] for row in controlled_stages if row["status"] == "results_available"}
            confirmed = (len(jobs) == 2 and len(set(reported_ids)) == 2
                         and all(isinstance(job, str) and job for job in reported_ids)
                         and set(reported_ids) <= observed_ids
                         and all(job.get("primary", {}).get("passes") is True for job in jobs))
            if not confirmed:
                warnings.append({"file": str(report_path),
                                 "issue": "A claimed mechanism confirmation lacks two distinct passing jobs with available controlled-noise evidence."})
    return {"phase": "controlled_noise", "stages": controlled_stages,
            "candidate_observations": observations, "per_stage_reports": per_stage_reports,
            "report_file": str(report_path.resolve()) if report is not None else None, "report": report,
            "mechanism_confirmed": confirmed, "ambient_optimization_success": False,
            "scope": "Controlled injected-noise mechanism verification only.",
            "note": "Controlled-noise stages never enter ambient rankings or edge coverage. Fully rejected postselected branches retain null fidelity/CI values. Their QPU charges remain in the same 600-second campaign budget."}


def build_summary(campaign):
    """Read a partial or complete campaign; return JSON data and CSV row lists."""
    campaign = Path(campaign).resolve()
    if not campaign.is_dir():
        raise ValueError(f"Campaign directory does not exist: {campaign}")
    warnings, manifests = [], {}
    for path in sorted(campaign.glob("*/manifest.json")):
        manifest = read_json(path, warnings)
        if isinstance(manifest, dict):
            manifests[path.parent.name] = manifest
        else:
            # A concurrently written or damaged manifest can conceal a paid
            # submission. Keep an unknown ledger entry rather than dropping it.
            manifests[path.parent.name] = {"submission_state": "unknown_manifest_unreadable"}
    rows, stages = _candidate_rows(campaign, manifests, warnings)
    ranked, drift_rows, drift_groups, observed_pairs = _exploratory_groups(rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "campaign": str(campaign),
        "candidate_observation_count": len(rows), "stages": stages,
        "usage": _usage_summary(campaign, manifests, warnings),
        "coverage": _coverage(campaign, observed_pairs, warnings),
        "exploratory_top10": ranked[:10],
        "exploratory_ranking_note": "Exploratory only, subject to selection bias. Baseline repeats sharing an unordered data pair are one grouping, represented by the earliest stage and spec order, never its best repeat. Counts are not pooled. Circuit variants are grouped by variant/layout/optimization level. Confirmation observations are excluded.",
        "baseline_pair_groups": drift_groups,
        "drift_note": "Repeated controls are displayed separately without pooling or treating them as independent new candidates. Different orientations, ancilla choices, or optimization levels are labeled recipe changes and cannot alone establish temporal drift.",
        "confirmation": _confirmations(campaign, warnings),
        "controlled_noise": _controlled_noise(campaign, rows, stages, warnings),
        "candidates": rows, "warnings": warnings,
        "uncertainty_note": "Candidate intervals combine three Bonferroni-adjusted Wilson bounds with approximate coverage. They describe shot uncertainty, not calibration drift or systematic error. Values are not clipped; retention is reported for postselection.",
    }
    return summary, rows, drift_rows


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def export_summary(campaign, output=None):
    campaign = Path(campaign).resolve()
    output = Path(output).resolve() if output else campaign / "analysis-summary"
    summary, rows, drift_rows = build_summary(campaign)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "candidate-observations.csv", rows, CSV_FIELDS)
    _write_csv(output / "baseline-drift.csv", drift_rows,
               [*CSV_FIELDS, "pair_observation_count", "reference_stage", "reference_candidate_id",
                "same_recipe_as_reference", "delta_from_first"])
    (output / "optimization-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                                  allow_nan=False) + "\n", encoding="utf-8")
    return summary, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path, help="Existing campaign directory.")
    parser.add_argument("--output", type=Path, help="Summary directory; defaults to <campaign>/analysis-summary.")
    args = parser.parse_args()
    summary, output = export_summary(args.campaign, args.output)
    print(json.dumps({"output": str(output), "candidate_observations": summary["candidate_observation_count"],
                      "coverage": summary["coverage"], "usage": {key: summary["usage"][key] for key in
                       ("accounting_complete", "known_final_used_seconds", "total_used_seconds", "remaining_seconds")},
                      "confirmation": summary["confirmation"]["status"], "warnings": len(summary["warnings"])},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
