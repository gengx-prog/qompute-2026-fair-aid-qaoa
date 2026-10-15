"""Package a successfully replicated configuration without accessing IBM."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from qiskit import qasm3, qpy

from optimization_analysis import confirm_replicates


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def verify_candidate_qasm(folder, candidate_id, definition):
    """Bind exported text to the QPY circuits, including the hashed submitted ISA."""
    for kind in ("logical", "isa"):
        with (folder / kind / "circuits.qpy").open("rb") as handle:
            circuits = qpy.load(handle)
        for basis in ("XX", "YY", "ZZ"):
            name = f"{candidate_id}_{basis}"
            matches = [circuit for circuit in circuits if circuit.name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {kind} QPY circuit named {name}.")
            circuit, metadata = matches[0], matches[0].metadata or {}
            if (metadata.get("candidate_id") != candidate_id or metadata.get("basis") != basis
                    or metadata.get("variant") != definition["variant"]
                    or metadata.get("initial_layout") != definition["layout"]):
                raise ValueError(f"The {kind} QPY circuit metadata differs from the frozen candidate.")
            if (folder / kind / f"{name}.qasm").read_text(encoding="utf-8") != qasm3.dumps(circuit):
                raise ValueError(f"The exported {kind} QASM differs from its recorded QPY circuit: {name}.")


def export(report_path: Path, output: Path):
    report_path = report_path.resolve()
    campaign = report_path.parent
    report = read(report_path)
    if report.get("phase") != "replicated_confirmation" or not report.get("confirmed_improvement"):
        raise ValueError("Only an independently replicated improvement can be exported as confirmed.")
    verified = confirm_replicates(report["replications"])
    if not verified["confirmed_improvement"] or verified["candidate_id"] != report["candidate_id"]:
        raise ValueError("The two replication reports do not confirm the same candidate.")
    if (len(report.get("job_ids", [])) != 2 or set(report["job_ids"]) != set(verified["job_ids"])):
        raise ValueError("Top-level report job IDs differ from the two verified replications.")
    plan_name = Path(report["frozen_plan_file"])
    plan_path = plan_name if plan_name.is_absolute() else campaign / plan_name
    plan = read(plan_path)
    canonical_hash = canonical_digest(plan)
    if canonical_hash != report["frozen_plan_sha256"]:
        raise ValueError("Frozen plan changed after evaluation.")
    candidate_id = plan["candidate_id"]
    if candidate_id != report["candidate_id"] or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate_id):
        raise ValueError("Invalid or inconsistent candidate identifier.")
    definitions = [row for row in plan["candidate_definitions"] if row["id"] == candidate_id]
    if len(definitions) != 1 or len(plan["stage_names"]) != 2:
        raise ValueError("Invalid frozen candidate definition or replication count.")
    rep_by_stage = {row.get("stage"): row for row in report["replications"]}
    if len(rep_by_stage) != 2 or set(rep_by_stage) != set(plan["stage_names"]):
        raise ValueError("Replication reports must bind exactly to the two frozen stages.")
    if output.exists():
        raise ValueError("Output exists; preserve the existing evidence instead of overwriting it.")

    # Check all inputs before creating the output folder.
    copies = [(report_path, Path(report_path.name)), (plan_path, Path(plan_path.name))]
    rep_metadata = []
    for stage in plan["stage_names"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", stage):
            raise ValueError("Invalid stage identifier.")
        folder = campaign / stage
        manifest = read(folder / "manifest.json")
        result = read(folder / "hardware-results.json")
        usage = read(folder / "hardware-usage.json")
        spec = read(folder / "spec.json")
        replication = rep_by_stage[stage]
        if manifest["job_id"] != replication["job_id"] or result["job_id"] != manifest["job_id"]:
            raise ValueError("Evidence job IDs differ from the confirmation report.")
        records = result.get("records")
        if (not isinstance(records, list) or canonical_digest(records) != replication.get("records_sha256")
                or any(row.get("job_id") != manifest["job_id"] for row in records)):
            raise ValueError("Measurement records changed after confirmation evaluation.")
        if canonical_digest(spec) != plan["spec_sha256"][stage]:
            raise ValueError("Stage specification differs from the frozen confirmation plan.")
        if manifest["backend"] != plan["backend"] or manifest["shots"] != plan["shots_per_basis"]:
            raise ValueError("Stage backend or shot count differs from the frozen plan.")
        if (usage.get("job_id") != manifest["job_id"] or str(usage.get("job_status", "")).upper()
                not in {"DONE", "COMPLETED", "JOBSTATUS.DONE"}):
            raise ValueError("Usage evidence must belong to this successfully completed job.")
        charge = usage.get("reported_qpu_charge_time_seconds")
        if (usage.get("usage_status") != "complete" or isinstance(charge, bool)
                or not isinstance(charge, (int, float)) or not math.isfinite(charge) or charge < 0):
            raise ValueError("Wait for finalized usage before exporting the final package.")
        if digest(folder / "isa" / "circuits.qpy") != manifest.get("isa_qpy_sha256"):
            raise ValueError("Submitted ISA QPY changed after local validation.")
        verify_candidate_qasm(folder, candidate_id, definitions[0])
        rep_metadata.append({"stage": stage, "job_id": manifest["job_id"],
                             "shots_per_basis": manifest["shots"],
                             "rep_delay_seconds": manifest["rep_delay_seconds"],
                             "execution_cap_seconds": manifest["cap_seconds"],
                             "charged_seconds": usage["reported_qpu_charge_time_seconds"]})
        for filename in ("spec.json", "manifest.json", "hardware-results.json", "hardware-usage.json"):
            copies.append((folder / filename, Path(stage) / filename))
        for kind in ("logical", "isa"):
            copies.append((folder / kind / "circuits.qpy", Path(stage) / kind / "circuits.qpy"))
            for basis in ("XX", "YY", "ZZ"):
                filename = f"{candidate_id}_{basis}.qasm"
                copies.append((folder / kind / filename, Path(stage) / kind / filename))
            copies.append((folder / kind / "resources.json", Path(stage) / kind / "resources.json"))
    index = []
    for source, relative in copies:
        if not source.is_file():
            raise ValueError(f"Missing evidence: {source}")
        index.append({"source": str(source.relative_to(campaign)),
                      "packaged_file": relative.as_posix(), "sha256": digest(source)})
    output.mkdir(parents=True)
    for source, relative in copies:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for entry in index:
        if digest(output / entry["packaged_file"]) != entry["sha256"]:
            raise ValueError("A packaged evidence file does not match its indexed source hash.")
    configuration = {
        "backend": plan["backend"], "candidate": definitions[0],
        "confirmed_against": sorted(report["replications"][0]["comparisons"]),
        "secondary_attribution_control": plan.get("attribution_baseline_id"),
        "secondary_attribution_comparisons": [row.get("attribution_comparison") for row in report["replications"]],
        "confirmation_report": report_path.name, "replications": rep_metadata,
        "language": "OpenQASM 3; logical and IBM native ISA circuits included for XX, YY and ZZ",
        "interpretation": "Independently confirmed measured improvement among tested configurations; not a global optimum or an isolated preparation-fidelity claim.",
        "reproduction": "The per-replication spec records candidate and controls. Use a new stage name when preparing a new run with optimize_bell.py; authenticate locally and check the remaining quota. Do not resubmit an existing manifest.",
        "postselection": "Only protected/bridge variants filter ancilla-zero shots. All baseline variants retain every shot. baseline_quiet does not measure the auxiliary qubit.",
        "credentials_included": False,
    }
    for filename, value in (("best-config.json", configuration), ("evidence-index.json", index)):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "candidate": candidate_id,
                      "replication_jobs": report["job_ids"], "copied_files": len(index)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.report, args.output)
