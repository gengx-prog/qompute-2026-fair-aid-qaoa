"""Prepare, validate, and execute a budgeted multi-candidate Bell experiment.

Each stage is prepared and simulated locally before one explicit submission.
Stages are sequential; unknown/pending usage or an ambiguous submission blocks
the next job. Credentials come from run_bell.connect, never from result files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral
import random
from pathlib import Path
from types import SimpleNamespace

from qiskit import qpy
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from optimization_circuits import BASES, VARIANTS, build_variant
from run_bell import collect_job_usage, connect, export_circuits, resources, save_json

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "results" / "optimization-20261015"
SOURCE_NAMES = ("optimize_bell.py", "optimization_circuits.py", "optimization_analysis.py")


def hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compact_summary(summary):
    return [{"candidate": name, "fidelity": row["fidelity"], "retention": row["retention"]}
            for name, row in sorted(summary.items(), key=lambda item: (
                item[1]["fidelity"] is not None,
                item[1]["fidelity"] if item[1]["fidelity"] is not None else -math.inf), reverse=True)]


def service():
    return connect(SimpleNamespace(account=None, instance=None))


def remaining_budget(campaign=CAMPAIGN, *, reserve_pending=False):
    """A prior completed 5-second job and all campaign jobs share 600 seconds."""
    used = 5.0
    for path in campaign.glob("*/manifest.json"):
        manifest = read(path)
        if manifest.get("submission_state") == "prepared":
            continue
        if not manifest.get("job_id"):
            raise ValueError(f"Ambiguous submission at {path}; inspect IBM before continuing.")
        usage_path = path.parent / "hardware-usage.json"
        if not usage_path.exists():
            if reserve_pending:
                cap = manifest.get("cap_seconds")
                if type(cap) is not int or not 1 <= cap <= 120:
                    raise ValueError("Cannot reserve an invalid pending execution cap.")
                used += cap
                continue
            raise ValueError(f"Fetch usage before the next submission: {path.parent.name}")
        usage = read(usage_path)
        if usage.get("usage_status") != "complete":
            if reserve_pending:
                cap = manifest.get("cap_seconds")
                if type(cap) is not int or not 1 <= cap <= 120:
                    raise ValueError("Cannot reserve an invalid pending execution cap.")
                used += cap
                continue
            raise ValueError("A prior job's charge is not final; another job is blocked.")
        seconds = usage.get("reported_qpu_charge_time_seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Missing valid actual usage; another job is blocked.")
        used += seconds
    return 600.0 - used


def validate_budget(shots, circuit_count, cap, remaining, estimate):
    if not all(type(value) in (int, float) and math.isfinite(value) and value >= 0
               for value in (remaining, estimate)):
        raise ValueError("Budget and estimate must be finite nonnegative numbers.")
    if type(shots) is not int or not 64 <= shots <= 16384:
        raise ValueError("Use 64..16384 shots per circuit.")
    if type(circuit_count) is not int or not 3 <= circuit_count <= 144:
        raise ValueError("Use 3..144 circuits in one job.")
    if type(cap) is not int or not 1 <= cap <= 120:
        raise ValueError("Execution cap must be 1..120 seconds.")
    if cap + 10 > remaining:
        raise ValueError("Execution cap plus 10-second reserve exceeds remaining total budget.")
    if estimate > cap:
        raise ValueError("Planning estimate exceeds job execution cap; reduce work.")


def load_circuits(folder):
    with (folder / "circuits.qpy").open("rb") as handle:
        return qpy.load(handle)


def records_for(circuits, counts, job_id):
    records = []
    for circuit, counts_row in zip(circuits, counts):
        meta = circuit.metadata
        records.append({"candidate_id": meta["candidate_id"], "basis": meta["basis"],
                        "layout": meta["initial_layout"], "variant": meta["variant"],
                        "postselect_ancilla": meta["postselect"],
                        "injected_fault": meta.get("injected_fault"),
                        "job_id": job_id, "counts": counts_row})
    return records


def build_candidate_circuits(candidate):
    """Build all three analysis settings, including any explicit test fault."""
    circuits = [build_variant(candidate["variant"], basis,
                              injected_fault=candidate.get("injected_fault")) for basis in BASES]
    for circuit in circuits:
        circuit.name = candidate["id"] + "_" + circuit.metadata["basis"]
        circuit.metadata.update(candidate_id=candidate["id"], initial_layout=candidate["layout"],
                                optimization_level=candidate.get("optimization_level", 1))
    return circuits


def validate_ideal_counts(circuit, counts, *, expected_shots=256):
    """Verify the full ideal outcome support, before any ancilla postselection.

    No fault prepares Phi+ (XX+, YY-, ZZ+); X on q0 prepares Psi+
    (XX+, YY+, ZZ-), and Z prepares Phi- (XX-, YY+, ZZ+). The two
    parity detectors flag X with ancilla=1 and retain Z with ancilla=0.
    Thus an entirely rejected X branch is a valid local physics result.
    """
    meta = circuit.metadata or {}
    fault, basis, variant = meta.get("injected_fault"), meta.get("basis"), meta.get("variant")
    if fault not in (None, "X", "Z") or basis not in BASES or variant not in VARIANTS:
        raise ValueError("Ideal validation requires a known variant, basis, and None/X/Z fault.")
    if type(expected_shots) is not int or expected_shots <= 0:
        raise ValueError("Expected ideal shot count must be a positive integer.")
    if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0
           for count in counts.values()) or sum(counts.values()) != expected_shots:
        raise ValueError("Incomplete or invalid local ideal results.")
    negative_basis = {None: "YY", "X": "ZZ", "Z": "XX"}[fault]
    expected_even = basis != negative_basis
    expected_ancilla = "1" if fault == "X" and variant in ("protected", "bridge") else "0"
    for bits, count in counts.items():
        if not isinstance(bits, str):
            raise ValueError("Ideal counts must use three-bit meas strings.")
        bits = bits.replace(" ", "")
        if len(bits) != 3 or any(bit not in "01" for bit in bits):
            raise ValueError("Ideal counts must use three-bit meas strings.")
        if count and (bits[0] != expected_ancilla or (bits[1] == bits[2]) != expected_even):
            raise ValueError(f"Compiled circuit failed fault-aware Bell/ancilla check: {circuit.name}")


def prepare(spec_path):
    from optimization_analysis import analyze_records

    spec = read(spec_path)
    stage = CAMPAIGN / spec["stage"]
    if stage.exists():
        raise ValueError("Stage already exists; choose a new stage name, never overwrite evidence.")
    candidates = spec["candidates"]
    ids = [row["id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique.")
    remaining = remaining_budget(reserve_pending=True)
    validate_budget(spec["shots"], 3 * len(candidates), spec["cap_seconds"], remaining, 0)
    backend = service().backend(spec["backend"])
    status = backend.status()
    if backend.configuration().simulator or not status.operational or status.status_msg.lower() != "active":
        raise ValueError("Selected real QPU is not active.")
    configuration = backend.configuration()
    rep_delay = float(configuration.default_rep_delay)
    rep_range = configuration.rep_delay_range
    if not min(rep_range) <= rep_delay <= max(rep_range):
        raise ValueError("Invalid device repetition delay.")
    logical, compiled = [], []
    for candidate in candidates:
        layout = candidate["layout"]
        if len(layout) != 3 or len(set(layout)) != 3 or min(layout) < 0 or max(layout) >= backend.num_qubits:
            raise ValueError("Invalid initial physical layout.")
        circuits = build_candidate_circuits(candidate)
        manager = generate_preset_pass_manager(backend=backend, initial_layout=layout,
                                               optimization_level=candidate.get("optimization_level", 1),
                                               seed_transpiler=11)
        logical.extend(circuits)
        compiled.extend(manager.run(circuits))
    # Record a predetermined randomized circuit order to reduce ordering bias.
    order = list(range(len(compiled)))
    random.Random(spec.get("order_seed", 731)).shuffle(order)
    logical = [logical[i] for i in order]
    compiled = [compiled[i] for i in order]
    durations = [float(c.estimate_duration(backend.target, unit="s")) for c in compiled]
    estimate = 20 + 2 * spec["shots"] * sum(rep_delay + duration for duration in durations)
    validate_budget(spec["shots"], len(compiled), spec["cap_seconds"], remaining, estimate)
    stage.mkdir(parents=True)
    export_circuits(stage / "logical", logical)
    export_circuits(stage / "isa", compiled)
    save_json(stage / "spec.json", spec)
    save_json(stage / "backend-snapshot.json", {
        "backend": backend.name, "captured_utc": datetime.now(timezone.utc).isoformat(),
        "num_qubits": backend.num_qubits, "coupling_edges": list(backend.coupling_map.get_edges()),
        "properties": backend.properties().to_dict(), "default_rep_delay_seconds": rep_delay})
    print(f"Locally validating {len(compiled)} ISA circuits; no QPU submission.", flush=True)
    ideal = AerSimulator(method="density_matrix", max_parallel_threads=4)
    ideal_result = ideal.run(compiled, shots=256, seed_simulator=17).result()
    for i, circuit in enumerate(compiled):
        validate_ideal_counts(circuit, ideal_result.get_counts(i), expected_shots=256)
    noisy = AerSimulator.from_backend(backend, method="density_matrix", max_parallel_threads=4)
    noisy_result = noisy.run(compiled, shots=spec.get("local_shots", 4096), seed_simulator=19).result()
    noisy_counts = [noisy_result.get_counts(i) for i in range(len(compiled))]
    local_records = records_for(logical, noisy_counts, "LOCAL_CALIBRATION_MODEL")
    local_summary = analyze_records(local_records)
    save_json(stage / "local-validation.json", {
        "ideal_passed": True, "compiled_circuit_count": len(compiled),
        "noise_kind": "Aer model constructed from compilation-time IBM calibration; not real measurements",
        "noisy_records": local_records, "noisy_summary": local_summary})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "stage": spec["stage"],
        "backend": backend.name, "shots": spec["shots"], "circuit_count": len(compiled),
        "cap_seconds": spec["cap_seconds"], "rep_delay_seconds": rep_delay,
        "estimated_system_seconds": estimate,
        "estimate_note": "20s overhead + 2 * shots * sum(circuit duration + explicitly set repetition delay); heuristic, not guarantee",
        "remaining_seconds_before_submission": remaining, "source_sha256": hashes(),
        "budget_value_kind": "Uncommitted lower bound after reserving caps of known pending jobs; submission checks again.",
        "isa_qpy_sha256": hashlib.sha256((stage / "isa" / "circuits.qpy").read_bytes()).hexdigest(),
        "submission_state": "prepared", "job_id": None, "local_validation_passed": True,
        "statistics_plan": spec.get("statistics_plan"), "job_options": spec.get("job_options", {}),
    }
    save_json(stage / "manifest.json", manifest)
    print(json.dumps({"stage": spec["stage"], "circuits": len(compiled),
                      "estimated_seconds": estimate, "cap": spec["cap_seconds"],
                      "rep_delay": rep_delay, "local_summary": compact_summary(local_summary)}, default=str), flush=True)


def submit(stage_name):
    from qiskit_ibm_runtime.executor_sampler import Sampler

    stage = CAMPAIGN / stage_name
    path = stage / "manifest.json"
    manifest = read(path)
    if manifest["submission_state"] != "prepared" or not manifest["local_validation_passed"]:
        raise ValueError("Stage was attempted already or lacks successful local validation.")
    if hashes() != manifest["source_sha256"]:
        raise ValueError("Source changed since local validation; prepare a new stage.")
    if hashlib.sha256((stage / "isa" / "circuits.qpy").read_bytes()).hexdigest() != manifest["isa_qpy_sha256"]:
        raise ValueError("Compiled circuits changed since validation.")
    remaining = remaining_budget()
    validate_budget(manifest["shots"], manifest["circuit_count"], manifest["cap_seconds"],
                    remaining, manifest["estimated_system_seconds"])
    backend = service().backend(manifest["backend"])
    status = backend.status()
    if not status.operational or status.status_msg.lower() != "active":
        raise ValueError("Device became unavailable; no submission.")
    circuits = load_circuits(stage / "isa")
    sampler = Sampler(mode=backend)
    sampler.options.max_execution_time = manifest["cap_seconds"]
    sampler.options.execution.rep_delay = manifest["rep_delay_seconds"]
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    if manifest["job_options"].get("dynamical_decoupling", False):
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XX"
    manifest["submission_state"] = "attempted_check_platform_before_retry"
    manifest["attempted_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(path, manifest)
    job = sampler.run(circuits, shots=manifest["shots"])
    manifest.update(job_id=job.job_id(), submission_state="submitted")
    save_json(path, manifest)
    print(json.dumps({"job_id": job.job_id(), "stage": stage_name, "cap": manifest["cap_seconds"]}))


def fetch(stage_name):
    from optimization_analysis import analyze_records

    stage = CAMPAIGN / stage_name
    manifest = read(stage / "manifest.json")
    if not manifest.get("job_id"):
        raise ValueError("No job ID; do not submit blindly after an ambiguous failure.")
    job = service().job(manifest["job_id"])
    status = str(job.status()).upper()
    usage = collect_job_usage(job)
    save_json(stage / "hardware-usage.json", {"job_id": job.job_id(), "job_status": status, **usage})
    if status not in ("DONE", "COMPLETED", "JOBSTATUS.DONE"):
        print(json.dumps({"status": status, "job_id": job.job_id()}))
        return
    circuits = load_circuits(stage / "logical")
    result = job.result()
    if len(result) != len(circuits):
        raise ValueError("PUB count mismatch.")
    counts = [pub.data.meas.get_counts() for pub in result]
    if any(sum(row.values()) != manifest["shots"] for row in counts):
        raise ValueError("Incomplete measurement counts.")
    records = records_for(circuits, counts, job.job_id())
    summary = analyze_records(records)
    save_json(stage / "hardware-results.json", {
        "job_id": job.job_id(), "backend": manifest["backend"], "records": records,
        "summary": summary, "usage": usage, "runtime_metadata": result.metadata,
        "pub_metadata": [pub.metadata for pub in result]})
    print(json.dumps({"status": status, "job_id": job.job_id(),
                      "charge_seconds": usage["reported_qpu_charge_time_seconds"],
                      "uncommitted_seconds_after_reservations": remaining_budget(reserve_pending=True),
                      "summary": compact_summary(summary)}, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "fetch"))
    parser.add_argument("target", help="Spec JSON path for prepare, stage name otherwise")
    args = parser.parse_args()
    {"prepare": prepare, "submit": submit, "fetch": fetch}[args.action](args.target)


if __name__ == "__main__":
    main()
