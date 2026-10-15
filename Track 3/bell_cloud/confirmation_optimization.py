"""Freeze and evaluate Bell confirmation plans using local files only.

Each candidate attempt is fixed before either confirmation job is prepared.
Two distinct jobs, each with a frozen 8192 or 16384 shots per basis, must improve fidelity by
at least 0.005 with a positive Bonferroni-Wilson difference lower bound. Alpha
is .05/3 per candidate attempt, with at most three attempts. A postselected
candidate additionally needs an unprotected control on the same layout.

This helper creates specs and evaluates recorded results. It has no account,
network, batch, or submission functionality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import optimization_analysis as analysis

DEFAULT_SHOTS = 16384
ALLOWED_SHOTS = (8192, 16384)
DEFAULT_CAMPAIGN = Path(__file__).resolve().parent / "results" / "optimization-20261015"
ALPHA_PER_ATTEMPT = 0.05 / 3
MAX_ATTEMPTS = 3
MINIMUM_IMPROVEMENT = 0.005
ATTRIBUTION_VARIANTS = {"baseline_quiet", "baseline_reset", "baseline_relax_200us"}
ATTRIBUTION_NOTE = (
    "The combined configuration can succeed against the original baseline. "
    "Its same-layout ordinary baseline is an additional comparison to assess "
    "the added operation or readout change, not an extra primary success requirement."
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=False)
        handle.write("\n")


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _analysis_hash():
    return hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest()


def _stage(campaign, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Stage names must be simple directory names.")
    folder = (campaign / name).resolve()
    if folder.parent != campaign:
        raise ValueError("Stage resolves outside the selected campaign.")
    return folder


def _completed_stage(campaign, name):
    folder = _stage(campaign, name)
    manifest = read(folder / "manifest.json")
    result = read(folder / "hardware-results.json")
    job_id = manifest.get("job_id")
    if not isinstance(job_id, str) or not job_id or result.get("job_id") != job_id:
        raise ValueError(f"Manifest/result job IDs disagree for {name}.")
    usage = read(folder / "hardware-usage.json")
    if (usage.get("job_id") != job_id or str(usage.get("job_status", "")).upper()
            not in {"DONE", "COMPLETED", "JOBSTATUS.DONE"} or usage.get("usage_status") != "complete"):
        raise ValueError(f"Stage {name} must have final successful results and settled usage.")
    records = result.get("records")
    if not isinstance(records, list) or any(row.get("job_id") != job_id for row in records):
        raise ValueError(f"Missing or inconsistent measurement records for {name}.")
    return manifest, result, analysis.analyze_records(records)


def _exploratory(name, manifest):
    phase = str((manifest.get("statistics_plan") or {}).get("phase", ""))
    return (not name.startswith(("confirm-", "confirmation-")) and
            (name.startswith(("screen", "scan", "refine")) or "exploratory" in phase))


def _selection_jobs(campaign):
    jobs, evidence = [], []
    for path in sorted(campaign.glob("*/hardware-results.json")):
        name = path.parent.name
        if not _exploratory(name, read(path.parent / "manifest.json")):
            continue
        manifest, result, _ = _completed_stage(campaign, name)
        jobs.append(manifest["job_id"])
        evidence.append({"stage": name, "job_id": manifest["job_id"],
                         "records_sha256": _canonical_hash(result["records"])})
    return sorted(set(jobs)), evidence


def freeze_plan(campaign, source_stage, candidate_id, stage_names, *, attempt=1,
                original_baseline_id="original_baseline", matching_baseline_id=None,
                shots=DEFAULT_SHOTS):
    """Freeze an attempt and create its two specs without contacting IBM.

    Returns paths to ``confirmation-plan-NN.json`` and the two ``*-spec.json``
    files. Existing plans/specs/stages are never overwritten. A second or third
    attempt requires a fully evaluated, unsuccessful prior attempt.
    """
    campaign = Path(campaign).resolve()
    if type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
        raise ValueError("Candidate attempt must be 1, 2, or 3.")
    if type(shots) is not int or shots not in ALLOWED_SHOTS:
        raise ValueError("Freeze either 8192 or 16384 shots per basis.")
    amendment_path = campaign / "confirmation-power-amendment.json"
    amendment = read(amendment_path) if amendment_path.exists() else None
    if amendment is not None and amendment.get("revised_shots_per_basis") != shots:
        raise ValueError("The campaign's registered power amendment fixes its confirmation shot count.")
    if shots == 16384 and amendment is None:
        raise ValueError("Register confirmation-power-amendment.json before freezing the enlarged sample size.")
    stage_names = list(stage_names)
    if len(stage_names) != 2 or len(set(stage_names)) != 2:
        raise ValueError("Freeze exactly two distinct confirmation stage names.")
    if any(not name.startswith(("confirm-", "confirmation-")) for name in stage_names):
        raise ValueError("Confirmation stages must start with confirm- or confirmation-.")
    prior_jobs = []
    for index in range(1, attempt):
        prior_path = campaign / f"confirmation-attempt-{index:02d}.json"
        if not prior_path.exists():
            raise ValueError("Finish and report every earlier candidate attempt before freezing the next.")
        prior = read(prior_path)
        checked = analysis.confirm_replicates(prior["replications"])
        if prior.get("shots_per_basis") != shots:
            raise ValueError("The confirmation shot count cannot change after inspecting an earlier attempt.")
        if checked["confirmed_improvement"]:
            raise ValueError("An earlier attempt already succeeded; the frozen search protocol is complete.")
        prior_jobs.extend(checked["job_ids"])
    if attempt == 1:
        for path in campaign.glob("*/manifest.json"):
            if path.parent.name.startswith(("confirm-", "confirmation-")) and read(path).get("job_id"):
                raise ValueError("Freeze the first confirmation plan before any holdout job is submitted.")
    existing = list(campaign.glob("confirmation-plan-*.json"))
    if any(read(path).get("attempt", 0) >= attempt for path in existing):
        raise ValueError("This or a later attempt has already been frozen; do not overwrite or refreeze it.")
    manifest, result, summaries = _completed_stage(campaign, source_stage)
    if not _exploratory(source_stage, manifest):
        raise ValueError("Select the candidate from an exploratory screening stage.")
    spec = read(_stage(campaign, source_stage) / "spec.json")
    definitions = {row["id"]: row for row in spec["candidates"]}
    if candidate_id not in summaries or candidate_id not in definitions or original_baseline_id not in summaries:
        raise ValueError("Candidate and original baseline must both exist in the source-stage evidence.")
    if candidate_id == original_baseline_id or summaries[original_baseline_id]["postselected"]:
        raise ValueError("Use a candidate distinct from the original unprotected baseline.")
    candidate = summaries[candidate_id]
    if candidate["fidelity"] is None:
        raise ValueError("A candidate with an unavailable screening fidelity cannot be frozen.")
    original = definitions[original_baseline_id]
    if original.get("variant") != "baseline" or original.get("layout") != [0, 1, 2]:
        raise ValueError("The original control must remain the original baseline on physical layout [0,1,2].")
    controls = [original_baseline_id]
    attribution_baseline_id = None
    if candidate["postselected"] or candidate.get("variant") in ATTRIBUTION_VARIANTS:
        candidate_level = definitions[candidate_id].get("optimization_level", 1)
        valid = [name for name, row in summaries.items()
                 if not row["postselected"] and row.get("variant") == "baseline"
                 and row.get("layout") == candidate.get("layout")
                 and definitions[name].get("optimization_level", 1) == candidate_level]
        if matching_baseline_id is None:
            matching_baseline_id = original_baseline_id if original_baseline_id in valid else next(iter(valid), None)
        if matching_baseline_id not in valid:
            raise ValueError("Protected or modified-baseline candidates require an ordinary baseline with the same layout and optimization level.")
        if matching_baseline_id not in controls:
            controls.append(matching_baseline_id)
        if not candidate["postselected"]:
            attribution_baseline_id, matching_baseline_id = matching_baseline_id, None
    elif matching_baseline_id is not None:
        raise ValueError("A same-layout postselection control is only needed for a protected candidate.")
    screening_jobs, evidence = _selection_jobs(campaign)
    if manifest["job_id"] not in screening_jobs:
        raise ValueError("The selection source was not included among completed screening jobs.")
    plan_path = campaign / f"confirmation-plan-{attempt:02d}.json"
    spec_paths = [campaign / f"{name}-spec.json" for name in stage_names]
    for name, path in zip(stage_names, spec_paths):
        if _stage(campaign, name).exists() or path.exists():
            raise ValueError("Confirmation stages/specs must not exist before freezing the candidate.")
    ids = [original_baseline_id, candidate_id, *[name for name in controls if name != original_baseline_id]]
    selected_definitions = [deepcopy(definitions[name]) for name in ids]
    cap_seconds = 90 if len(ids) <= 2 else 120
    statistic_plan = {
        "phase": "independent_confirmation", "attempt": attempt,
        "maximum_candidate_confirmation_attempts": MAX_ATTEMPTS,
        "confirmation_replicates_per_candidate": 2, "confirmation_shots_per_basis": shots,
        "minimum_fidelity_improvement": MINIMUM_IMPROVEMENT, "alpha_per_attempt": ALPHA_PER_ATTEMPT,
        "comparison": "Same-job original baseline; postselected candidate additionally compares with the frozen same-layout baseline.",
        "selection": "Candidate and both confirmation stages frozen before either new job; both replications must pass.",
        "scope": "Independent confirmation among tested configurations, not proof of global optimality.",
    }
    specs = [{"stage": name, "backend": manifest["backend"], "shots": shots,
              "local_shots": 4096, "cap_seconds": cap_seconds,
              "order_seed": 20261015 + 100 * attempt + index,
              "candidates": deepcopy(selected_definitions), "statistics_plan": deepcopy(statistic_plan),
              "job_options": deepcopy(spec.get("job_options", {})),
              "confirmation_plan": plan_path.name}
             for index, name in enumerate(stage_names, 1)]
    plan = {
        "phase": "confirmation_plan", "attempt": attempt,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "candidate_id": candidate_id,
        "source_stage": source_stage, "source_job_id": manifest["job_id"],
        "source_records_sha256": _canonical_hash(result["records"]),
        "source_spec_sha256": _canonical_hash(spec),
        "screening_job_ids": screening_jobs, "screening_evidence": evidence,
        "prior_confirmation_job_ids": sorted(set(prior_jobs)),
        "excluded_confirmation_job_ids": sorted(set(screening_jobs + prior_jobs)),
        "original_baseline_id": original_baseline_id, "matching_baseline_id": matching_baseline_id,
        "attribution_baseline_id": attribution_baseline_id,
        "control_ids": controls, "backend": manifest["backend"],
        "stage_names": stage_names, "candidate_definitions": selected_definitions,
        "expected_postselection": {name: summaries[name]["postselected"] for name in ids},
        "shots_per_basis": shots, "cap_seconds_per_job": cap_seconds, "alpha_per_attempt": ALPHA_PER_ATTEMPT,
        "minimum_improvement": MINIMUM_IMPROVEMENT, "required_replicates": 2,
        "max_candidate_attempts": MAX_ATTEMPTS, "statistics_plan": statistic_plan,
        "analysis_source_sha256": _analysis_hash(),
        "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "specs": {name: item for name, item in zip(stage_names, specs)},
        "spec_sha256": {name: _canonical_hash(item) for name, item in zip(stage_names, specs)},
        "screening_candidate_snapshot": candidate,
        "sample_size_amendment": {
            "original_proposed_shots": 8192, "frozen_shots": shots,
            "timing": "Frozen before the first holdout job, then unchanged across candidate attempts.",
            "rationale": "Additional independent samples improve power under the conservative interval; alpha, improvement threshold, and maximum attempts are unchanged.",
        },
        "registered_power_amendment": {"file": amendment_path.name,
                                        "sha256": hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
                                        "content": amendment} if amendment is not None else None,
        "attribution_note": ATTRIBUTION_NOTE if attribution_baseline_id else None,
        "quiet_attribution_note": ATTRIBUTION_NOTE if attribution_baseline_id and candidate.get("variant") == "baseline_quiet" else None,
        "note": "Immutable selection record. Preserve all failed attempts. Two later independent jobs must both pass; do not pool or choose passing subsets.",
    }
    # Freeze the decision before creating specs. If writing a spec fails, this
    # plan remains immutable evidence and must be inspected, not silently redone.
    _write(plan_path, plan, exclusive=True)
    for path, item in zip(spec_paths, specs):
        _write(path, item, exclusive=True)
    return {"plan": str(plan_path), "specs": [str(path) for path in spec_paths],
            "candidate_id": candidate_id, "control_ids": controls,
            "screening_job_ids": screening_jobs}


def evaluate_plan(plan_path, *, stage_names=None):
    """Evaluate exactly the two frozen stages; write summary-discoverable reports."""
    plan_path = Path(plan_path).resolve()
    campaign, plan = plan_path.parent, read(plan_path)
    if plan.get("phase") != "confirmation_plan" or plan.get("analysis_source_sha256") != _analysis_hash():
        raise ValueError("Missing frozen plan or analysis code changed since freezing.")
    if stage_names is not None and list(stage_names) != plan["stage_names"]:
        raise ValueError("Result stages must be exactly the two stages frozen in the plan, in order.")
    shots = plan.get("shots_per_basis")
    if (shots not in ALLOWED_SHOTS or plan.get("alpha_per_attempt") != ALPHA_PER_ATTEMPT
            or plan.get("minimum_improvement") != MINIMUM_IMPROVEMENT
            or plan.get("required_replicates") != 2 or plan.get("max_candidate_attempts") != MAX_ATTEMPTS):
        raise ValueError("The frozen confirmation rule was altered.")
    registered = plan.get("registered_power_amendment")
    if registered is not None:
        amendment_path = campaign / registered["file"]
        if hashlib.sha256(amendment_path.read_bytes()).hexdigest() != registered["sha256"]:
            raise ValueError("The registered power amendment changed after the plan was frozen.")
    source_folder = _stage(campaign, plan["source_stage"])
    if _canonical_hash(read(source_folder / "spec.json")) != plan["source_spec_sha256"]:
        raise ValueError("The source specification changed after selection was frozen.")
    if _canonical_hash(read(source_folder / "hardware-results.json")["records"]) != plan["source_records_sha256"]:
        raise ValueError("The source measurement records changed after selection was frozen.")
    frozen_time = datetime.fromisoformat(plan["frozen_utc"].replace("Z", "+00:00"))
    if frozen_time.tzinfo is None:
        raise ValueError("The freeze time must include its timezone.")
    definitions = {row["id"]: row for row in plan["candidate_definitions"]}
    reports, observed_jobs = [], []
    for name in plan["stage_names"]:
        folder = _stage(campaign, name)
        stage_spec = read(folder / "spec.json")
        if _canonical_hash(stage_spec) != plan["spec_sha256"][name]:
            raise ValueError(f"Stage {name} does not match its frozen specification.")
        manifest, result, summaries = _completed_stage(campaign, name)
        if manifest.get("backend") != plan["backend"] or manifest.get("shots") != shots:
            raise ValueError("Confirmation backend or shot count differs from the frozen plan.")
        attempted_time = datetime.fromisoformat(str(manifest.get("attempted_utc", "")).replace("Z", "+00:00"))
        if attempted_time.tzinfo is None or attempted_time <= frozen_time:
            raise ValueError("Each holdout submission must occur strictly after the plan was frozen.")
        if set(summaries) != set(definitions):
            raise ValueError("Confirmation candidate/control set differs from the frozen plan.")
        for candidate_id, summary in summaries.items():
            definition = definitions[candidate_id]
            if (summary["layout"] != definition["layout"] or summary["variant"] != definition["variant"]
                    or summary["postselected"] != plan["expected_postselection"][candidate_id]):
                raise ValueError("Recorded candidate metadata differs from the frozen circuit definition.")
            if any(row["total_shots"] != shots for row in summary["correlations"].values()):
                raise ValueError("Each candidate must contain exactly the frozen shot count in each basis.")
        resource_rows = read(folder / "isa" / "resources.json")
        resource_map = {row["name"]: row for row in resource_rows}
        resources = {}
        for candidate_id in definitions:
            resources[candidate_id] = {}
            for basis in analysis.BASES:
                row = resource_map.get(f"{candidate_id}_{basis}")
                if row is None:
                    raise ValueError("Missing compiled-resource evidence for a confirmation circuit.")
                resources[candidate_id][basis] = {
                    "native_two_qubit_gates": row["two_qubit_gates"], "depth": row["depth"],
                    "measured_output_wires": row.get("measured_output_wires"),
                }
        report = analysis.confirm_candidate(summaries, plan["candidate_id"], plan["original_baseline_id"],
                                             screening_job_ids=plan["excluded_confirmation_job_ids"],
                                             matching_baseline_id=plan["matching_baseline_id"], alpha=ALPHA_PER_ATTEMPT,
                                             minimum_improvement=MINIMUM_IMPROVEMENT)
        report.update(stage=name, frozen_plan_file=str(plan_path), attempt=plan["attempt"],
                      records_sha256=_canonical_hash(result["records"]), shots_per_basis=shots,
                      candidate_summaries=summaries, compiled_resources=resources)
        attribution_id = plan.get("attribution_baseline_id")
        if attribution_id:
            report["attribution_comparison"] = analysis.compare_candidates(
                summaries[plan["candidate_id"]], summaries[attribution_id], alpha=ALPHA_PER_ATTEMPT,
                comparison_count=2)
            report["attribution_note"] = "Secondary same-layout comparison for interpretation; not an extra primary success requirement or a separate multiplicity-controlled discovery claim. If its difference does not support improvement, do not attribute the combined configuration's gain to the added operation or readout change."
        reports.append(report)
        observed_jobs.append(manifest["job_id"])
    if len(set(observed_jobs)) != 2:
        raise ValueError("The two confirmation stages reused one job.")
    aggregate = analysis.confirm_replicates(reports)
    aggregate.update(attempt=plan["attempt"], frozen_plan_file=str(plan_path),
                     frozen_plan_sha256=_canonical_hash(plan), source_stage=plan["source_stage"],
                     screening_job_ids=plan["screening_job_ids"],
                     prior_confirmation_job_ids=plan["prior_confirmation_job_ids"],
                     shots_per_basis=shots, sample_size_amendment=plan["sample_size_amendment"],
                     registered_power_amendment=plan["registered_power_amendment"],
                     attribution_note=plan.get("attribution_note", plan.get("quiet_attribution_note")),
                     quiet_attribution_note=plan.get("quiet_attribution_note"),
                     evaluated_utc=datetime.now(timezone.utc).isoformat())
    final_path = campaign / f"confirmation-attempt-{plan['attempt']:02d}.json"
    for index, report in enumerate(reports, 1):
        _write(campaign / f"confirmation-attempt-{plan['attempt']:02d}-replicate-{index}.json", report)
    _write(final_path, aggregate)
    return {"report": str(final_path), "candidate_id": plan["candidate_id"],
            "confirmed_improvement": aggregate["confirmed_improvement"], "job_ids": observed_jobs,
            "replications": [{"stage": report["stage"], "passes": report["confirmed_improvement"],
                              "comparisons": report["comparisons"]} for report in reports]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("plan", help="Freeze selection and generate two specs locally.")
    freeze.add_argument("--campaign", default=DEFAULT_CAMPAIGN, type=Path)
    freeze.add_argument("--source-stage", required=True)
    freeze.add_argument("--candidate", required=True)
    freeze.add_argument("--stages", nargs=2, required=True)
    freeze.add_argument("--attempt", type=int, default=1)
    freeze.add_argument("--original-baseline", default="original_baseline")
    freeze.add_argument("--matching-baseline")
    freeze.add_argument("--shots", type=int, choices=ALLOWED_SHOTS, default=DEFAULT_SHOTS)
    evaluate = sub.add_parser("evaluate", help="Evaluate the two recorded confirmation results locally.")
    evaluate.add_argument("--plan", required=True, type=Path)
    evaluate.add_argument("--stages", nargs=2, help="Optional explicit stage names; must match the frozen plan.")
    args = parser.parse_args()
    if args.action == "plan":
        output = freeze_plan(args.campaign, args.source_stage, args.candidate, args.stages,
                             attempt=args.attempt, original_baseline_id=args.original_baseline,
                             matching_baseline_id=args.matching_baseline, shots=args.shots)
    else:
        output = evaluate_plan(args.plan, stage_names=args.stages)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
