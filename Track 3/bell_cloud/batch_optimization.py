"""Submit already locally validated Bell scan stages once in an IBM Batch.

The requested batch TTL is 600 wall-clock seconds, not a QPU-time allowance. Every job
has a separate validated 1..120-second execution cap. Budget checks reserve caps for
already pending jobs and all planned jobs, plus ten seconds of unused budget.
This module never automatically retries. A failed/ambiguous attempt retains a
lock and its ledger for inspection; already submitted jobs must not be resent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import optimize_bell as campaign_runner
from run_bell import save_json

TOTAL_BUDGET_SECONDS = 600.0
INITIAL_USED_SECONDS = 5.0
MAX_JOB_CAP_SECONDS = 120
UNUSED_RESERVE_SECONDS = 10
BATCH_TTL_SECONDS = 600


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def budget_reservations(campaign=None):
    """Conservatively account for final charges and known pending-job caps.

    No job lookup occurs. Ambiguous submissions with no job ID, bad manifests,
    or contradictory duplicate-job accounting block another submission.
    """
    campaign = Path(campaign if campaign is not None else campaign_runner.CAMPAIGN).resolve()
    jobs = {}
    for path in sorted(campaign.glob("*/manifest.json")):
        manifest = campaign_runner.read(path)
        job_id = manifest.get("job_id")
        state = manifest.get("submission_state")
        if state == "prepared" and not job_id:
            continue
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"Ambiguous prior submission at {path}; inspect its platform job before continuing.")
        usage_path = path.parent / "hardware-usage.json"
        usage = campaign_runner.read(usage_path) if usage_path.exists() else {}
        if usage and usage.get("job_id") != job_id:
            raise ValueError(f"Usage job ID mismatch at {usage_path}.")
        if usage.get("usage_status") == "complete":
            charge = usage.get("reported_qpu_charge_time_seconds")
            if not _number(charge):
                raise ValueError(f"Invalid final job charge at {usage_path}.")
            entry = {"job_id": job_id, "state": "final", "seconds": float(charge)}
        else:
            cap = manifest.get("cap_seconds")
            if type(cap) is not int or not 1 <= cap <= 120:
                raise ValueError(f"Cannot safely reserve a pending-job cap at {path}.")
            entry = {"job_id": job_id, "state": "pending_cap_reserved", "seconds": cap}
        if job_id in jobs and jobs[job_id] != entry:
            raise ValueError(f"Contradictory accounting for duplicate job {job_id}.")
        jobs[job_id] = entry
    final = INITIAL_USED_SECONDS + sum(row["seconds"] for row in jobs.values() if row["state"] == "final")
    pending = sum(row["seconds"] for row in jobs.values() if row["state"] == "pending_cap_reserved")
    return {"total_budget_seconds": TOTAL_BUDGET_SECONDS, "initial_used_seconds": INITIAL_USED_SECONDS,
            "known_final_used_seconds": final, "pending_caps_reserved_seconds": pending,
            "uncommitted_seconds": TOTAL_BUDGET_SECONDS - final - pending,
            "jobs": list(jobs.values()),
            "note": "Pending actual usage is unknown; its full execution cap is reserved, not counted as zero."}


def _stage_folder(campaign, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Stage names must be simple directory names, without path components.")
    folder = (campaign / name).resolve()
    if folder.parent != campaign:
        raise ValueError("Stage directory resolves outside the selected campaign.")
    return folder


def _load_prepared_stage(campaign, name, expected_hashes):
    folder = _stage_folder(campaign, name)
    manifest = campaign_runner.read(folder / "manifest.json")
    if manifest.get("submission_state") != "prepared" or manifest.get("job_id"):
        raise ValueError(f"Stage {name} was already attempted or submitted; it cannot be resent.")
    if manifest.get("local_validation_passed") is not True:
        raise ValueError(f"Stage {name} has not passed local validation.")
    validation = campaign_runner.read(folder / "local-validation.json")
    if validation.get("ideal_passed") is not True:
        raise ValueError(f"Stage {name} is missing successful compiled-circuit validation.")
    if manifest.get("source_sha256") != expected_hashes:
        raise ValueError(f"Sources changed since preparing {name}; prepare a new stage before submission.")
    isa_path = folder / "isa" / "circuits.qpy"
    if hashlib.sha256(isa_path.read_bytes()).hexdigest() != manifest.get("isa_qpy_sha256"):
        raise ValueError(f"Compiled circuits changed after validation for {name}.")
    shots, count = manifest.get("shots"), manifest.get("circuit_count")
    if type(shots) is not int or not 64 <= shots <= 16384:
        raise ValueError(f"Invalid shot count for {name}.")
    if type(count) is not int or not 3 <= count <= 144:
        raise ValueError(f"Invalid circuit count for {name}.")
    cap = manifest.get("cap_seconds")
    if type(cap) is not int or not 1 <= cap <= MAX_JOB_CAP_SECONDS:
        raise ValueError("Each batch job must have an explicit 1..120-second execution cap.")
    estimate = manifest.get("estimated_system_seconds")
    if not _number(estimate) or estimate > cap:
        raise ValueError(f"Planning estimate exceeds the execution cap for {name}.")
    if not _number(manifest.get("rep_delay_seconds")):
        raise ValueError(f"Invalid repetition delay for {name}.")
    if manifest.get("job_options") not in ({}, None):
        raise ValueError("This scan batch supports only the locally validated default job options.")
    circuits = campaign_runner.load_circuits(folder / "isa")
    if len(circuits) != count or validation.get("compiled_circuit_count") != count:
        raise ValueError(f"Compiled circuit count does not match the validation for {name}.")
    return {"stage": name, "folder": folder, "manifest": manifest, "circuits": circuits}


def preflight(stage_names, campaign=None):
    """Validate everything locally before creating an account connection or batch."""
    campaign = Path(campaign if campaign is not None else campaign_runner.CAMPAIGN).resolve()
    stage_names = list(stage_names)
    if not 1 <= len(stage_names) <= 4 or len(set(stage_names)) != len(stage_names):
        raise ValueError("Provide one to four distinct prepared stage names.")
    if (campaign / "batch-submission.lock").exists():
        raise ValueError("A batch lock exists. Inspect its ledger and jobs; do not automatically resubmit.")
    current_hashes = campaign_runner.hashes()
    stages = [_load_prepared_stage(campaign, name, current_hashes) for name in stage_names]
    if len({stage["manifest"].get("backend") for stage in stages}) != 1 or not stages[0]["manifest"].get("backend"):
        raise ValueError("All batch stages must target the same explicit backend.")
    budget = budget_reservations(campaign)
    planned = sum(stage["manifest"]["cap_seconds"] for stage in stages)
    if planned + UNUSED_RESERVE_SECONDS > budget["uncommitted_seconds"]:
        raise ValueError("Planned job caps plus the ten-second reserve exceed the uncommitted QPU budget.")
    return campaign, stages, {**budget, "planned_caps_seconds": planned,
                              "unused_reserve_seconds": UNUSED_RESERVE_SECONDS,
                              "worst_case_used_plus_reserve_seconds":
                                  budget["known_final_used_seconds"] + budget["pending_caps_reserved_seconds"] + planned + UNUSED_RESERVE_SECONDS}


def submit_batch(stage_names, *, campaign=None, batch_factory=None, sampler_factory=None):
    """Submit each validated stage once; return the persisted batch ledger.

    Factories permit offline testing. Normal calls use the installed Runtime
    Batch and client Sampler, obtaining the backend through optimize_bell.service.
    """
    campaign, stages, budget = preflight(stage_names, campaign)
    if batch_factory is None:
        from qiskit_ibm_runtime import Batch
        batch_factory = Batch
    if sampler_factory is None:
        from qiskit_ibm_runtime.executor_sampler import Sampler
        sampler_factory = Sampler
    service = campaign_runner.service()
    backend = service.backend(stages[0]["manifest"]["backend"])
    status = backend.status()
    if backend.configuration().simulator or not status.operational or status.status_msg.lower() != "active":
        raise ValueError("The selected real backend is unavailable; nothing was submitted.")
    repetition_range = backend.configuration().rep_delay_range
    for stage in stages:
        if not min(repetition_range) <= stage["manifest"]["rep_delay_seconds"] <= max(repetition_range):
            raise ValueError(f"Repetition delay is outside the backend range for {stage['stage']}.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ledger_path = campaign / f"batch-{stamp}.json"
    lock_path = campaign / "batch-submission.lock"
    # Exclusive creation prevents two batch controllers from racing. The root
    # controller must also stop any previous sequential submit process first.
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump({"ledger": str(ledger_path), "created_utc": datetime.now(timezone.utc).isoformat()}, handle)
    ledger = {"created_utc": datetime.now(timezone.utc).isoformat(), "batch_id": None,
              "backend": backend.name, "batch_ttl_wall_seconds": BATCH_TTL_SECONDS,
              "per_job_execution_caps_seconds": {row["stage"]: row["manifest"]["cap_seconds"] for row in stages},
              "budget_preflight": budget,
              "submission_state": "creating_batch", "job_ids": [],
              "batch_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "stages": [{"stage": row["stage"], "job_id": None, "state": "planned"} for row in stages],
              "note": "TTL is batch wall-clock lifetime, not QPU allocation. Per-job execution caps and reserved budget apply. No automatic retries; existing queued jobs are not cancelled."}
    save_json(ledger_path, ledger)
    try:
        with batch_factory(backend=backend, max_time=BATCH_TTL_SECONDS) as batch:
            if not isinstance(batch.session_id, str) or not batch.session_id:
                raise ValueError("Batch creation returned no ID; stop before submitting jobs.")
            ledger["batch_id"] = batch.session_id
            ledger["submission_state"] = "submitting"
            save_json(ledger_path, ledger)
            for index, stage in enumerate(stages):
                # Recheck local state and integrity immediately before spending.
                refreshed = _load_prepared_stage(campaign, stage["stage"], campaign_runner.hashes())
                remaining = budget_reservations(campaign)
                pending_batch_caps = sum(row["manifest"]["cap_seconds"] for row in stages[index:])
                if pending_batch_caps + UNUSED_RESERVE_SECONDS > remaining["uncommitted_seconds"]:
                    raise ValueError("Budget changed while preparing the batch; remaining submissions stopped.")
                manifest, path = refreshed["manifest"], refreshed["folder"] / "manifest.json"
                sampler = sampler_factory(mode=batch)
                sampler.options.max_execution_time = manifest["cap_seconds"]
                sampler.options.execution.rep_delay = manifest["rep_delay_seconds"]
                sampler.options.twirling.enable_gates = False
                sampler.options.twirling.enable_measure = False
                attempted = datetime.now(timezone.utc).isoformat()
                manifest.update(submission_state="attempted_check_platform_before_retry", attempted_utc=attempted,
                                batch_id=batch.session_id, batch_ledger=str(ledger_path))
                ledger["stages"][index].update(state="attempted_check_platform_before_retry", attempted_utc=attempted)
                save_json(path, manifest)
                save_json(ledger_path, ledger)
                # Exactly one run call; any exception propagates, never retries.
                job = sampler.run(refreshed["circuits"], shots=manifest["shots"])
                job_id = job.job_id()
                if not isinstance(job_id, str) or not job_id:
                    raise ValueError("Submission returned no job ID; inspect the batch before any further action.")
                manifest.update(job_id=job_id, submission_state="submitted")
                save_json(path, manifest)
                ledger["stages"][index].update(job_id=job_id, state="submitted")
                ledger["job_ids"].append(job_id)
                save_json(ledger_path, ledger)
                print(json.dumps({"stage": stage["stage"], "job_id": job_id, "batch_id": batch.session_id,
                                  "execution_cap_seconds": manifest["cap_seconds"]}), flush=True)
        ledger["submission_state"] = "submitted_and_batch_closed_to_new_jobs"
        ledger["finished_submitting_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(ledger_path, ledger)
        lock_path.unlink()
        return ledger
    except BaseException as error:
        ledger["submission_state"] = "stopped_no_retry_inspect_platform"
        ledger["error_type"] = type(error).__name__
        save_json(ledger_path, ledger)
        # Retain the lock even if a response was lost; the remote job may exist.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", help="Already prepared stage names, e.g. scan-03 scan-04 scan-05 scan-06.")
    parser.add_argument("--campaign", type=Path, help="Campaign directory; defaults to optimize_bell.CAMPAIGN.")
    parser.add_argument("--check-only", action="store_true", help="Offline preflight only; no account access or submissions.")
    args = parser.parse_args()
    if args.check_only:
        _, _, budget = preflight(args.stages, args.campaign)
        print(json.dumps(budget, indent=2))
    else:
        ledger = submit_batch(args.stages, campaign=args.campaign)
        print(json.dumps({"batch_id": ledger["batch_id"], "job_ids": ledger["job_ids"],
                          "submission_state": ledger["submission_state"]}, indent=2))


if __name__ == "__main__":
    main()
