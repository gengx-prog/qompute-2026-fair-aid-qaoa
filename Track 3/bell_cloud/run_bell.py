"""Bell-state verification and ZZ-parity postselection, locally or on IBM QPUs.

This is an introductory experiment and optional hardware extension, not the
PDF's complete three-architecture benchmark. No cloud calls in local mode.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from getpass import getpass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys

from qiskit import qasm3, qpy, transpile
from qiskit.exceptions import QiskitError
from qiskit.transpiler import generate_preset_pass_manager

from bell_core import analyze_counts, build_circuits

HERE = Path(__file__).resolve().parent
HARDWARE_MAX_SECONDS = 120
HARDWARE_MAX_SHOTS = 4096
HARDWARE_CIRCUIT_COUNT = 6


def validate_hardware_budget(shots, max_execution_time, circuit_count=None):
    """Reject oversized work before any account lookup or submission."""
    if type(shots) is not int or not 1 <= shots <= HARDWARE_MAX_SHOTS:
        raise ValueError(f"Hardware --shots must be between 1 and {HARDWARE_MAX_SHOTS}.")
    if type(max_execution_time) is not int or not 1 <= max_execution_time <= HARDWARE_MAX_SECONDS:
        raise ValueError(f"--max-execution-time must be between 1 and {HARDWARE_MAX_SECONDS} seconds.")
    if circuit_count is not None and circuit_count != HARDWARE_CIRCUIT_COUNT:
        raise ValueError(f"This bounded experiment requires exactly {HARDWARE_CIRCUIT_COUNT} circuits.")


def estimate_hardware_usage(backend, circuits, shots):
    """Planning heuristic, not an IBM billing prediction or a guaranteed bound.

    Include the target's circuit duration and maximum advertised repetition
    delay, double the shot time, and allow 30 seconds of fixed overhead. Runtime
    processing, resets, and calibration can differ from these assumptions; only
    the server-side max_execution_time option supplies an execution limit.
    """
    durations = []
    fallback_names = []
    for circuit in circuits:
        try:
            duration = float(circuit.estimate_duration(backend.target, unit="s"))
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Unavailable target duration.")
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, QiskitError):
            duration = 0.001
            fallback_names.append(circuit.name)
        durations.append(duration)
    configuration = backend.configuration()
    rep_delay_range = getattr(configuration, "rep_delay_range", None)
    rep_delay = max(rep_delay_range) if rep_delay_range else 0.001
    if not isinstance(rep_delay, (int, float)) or not math.isfinite(rep_delay) or rep_delay < 0:
        rep_delay = 0.001
    shot_seconds = shots * sum(duration + rep_delay for duration in durations)
    return {
        "kind": "conservative_planning_heuristic_not_guaranteed",
        "estimated_system_seconds": 30.0 + 2.0 * shot_seconds,
        "circuit_durations_seconds": durations,
        "repetition_delay_assumption_seconds": rep_delay,
        "duration_fallback_circuits": fallback_names,
        "formula": "30 seconds fixed overhead + 2 * shots * sum(circuit duration + max repetition delay)",
        "fallback_assumption": "Unavailable circuit duration or repetition delay is assumed to be 1 millisecond each.",
        "note": "Not a provider quote or upper bound; excludes queue and wall-clock waiting. Runtime execution cap is enforced separately.",
    }


def submit_bounded_job(backend, circuits, shots, max_execution_time, sampler_factory=None):
    """Make exactly one submission, with no automatic retry or Session."""
    validate_hardware_budget(shots, max_execution_time, len(circuits))
    if sampler_factory is None:
        from qiskit_ibm_runtime.executor_sampler import Sampler

        sampler_factory = Sampler
    sampler = sampler_factory(mode=backend)
    sampler.options.max_execution_time = max_execution_time
    return sampler.run(circuits, shots=shots)


def collect_job_usage(job):
    """Preserve IBM metrics, without mistaking pending billing's zero for free usage."""
    record = {"captured_utc": datetime.now(timezone.utc).isoformat()}
    try:
        record["metrics"] = job.metrics()
    except Exception as error:
        record["metrics_error"] = f"{type(error).__name__}: {error}"
    try:
        record["usage_method_seconds"] = job.usage()
    except Exception as error:
        record["usage_error"] = f"{type(error).__name__}: {error}"
    usage = record.get("metrics", {}).get("usage", {})
    record["usage_status"] = usage.get("status", "unknown")
    record["reported_qpu_charge_time_seconds"] = usage.get("qpu_charge_time_seconds")
    record["note"] = "Usage can remain pending after results arrive; a zero from usage() is not proof of zero charge. Fetch again to update billing metrics."
    return record


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def provenance():
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {name: version(name) for name in
                     ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy")},
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ("bell_core.py", "run_bell.py")},
        "scope": "Starter Bell experiment; not the official three-model benchmark.",
    }


def new_output(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(args.output) if args.output else HERE / "results" / f"{args.mode}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def resources(circuit):
    active = set()
    measured = {}
    two_qubit = 0
    for item in circuit.data:
        if item.operation.name == "barrier":
            continue
        if item.operation.name == "measure":
            measured[f"meas[{circuit.find_bit(item.clbits[0]).index}]"] = circuit.find_bit(item.qubits[0]).index
            continue
        active.update(circuit.find_bit(q).index for q in item.qubits)
        two_qubit += int(item.operation.num_qubits == 2)
    return {
        "name": circuit.name, "depth": circuit.depth(),
        "allocated_qubits": circuit.num_qubits,
        "gate_active_qubits": sorted(active),
        "measured_output_wires": measured,
        "used_physical_qubits": sorted(active | set(measured.values())),
        "two_qubit_gates": two_qubit,
        "gate_counts": dict(circuit.count_ops()),
        "explicit_swap_gates": int(circuit.count_ops().get("swap", 0)),
        "routing_note": "Native SWAPs may be decomposed; explicit SWAP count is not routing overhead.",
    }


def export_circuits(folder, circuits):
    folder.mkdir(parents=True, exist_ok=True)
    for circuit in circuits:
        (folder / f"{circuit.name}.qasm").write_text(qasm3.dumps(circuit), encoding="utf-8")
        (folder / f"{circuit.name}.txt").write_text(
            str(circuit.draw(output="text", idle_wires=False, fold=110)), encoding="utf-8")
    with (folder / "circuits.qpy").open("wb") as handle:
        qpy.dump(circuits, handle)
    save_json(folder / "resources.json", [resources(c) for c in circuits])


def print_summary(label, summary):
    print(label)
    for variant, row in summary.items():
        value = "unavailable (no accepted shots)" if row["fidelity"] is None else f"{row['fidelity']:.5f}"
        print(f"  {variant}: fidelity={value}; acceptance={row['acceptance_fraction']:.3%}")
        print(f"    approximate 95% shot-noise interval: {row['fidelity_interval_95']}")


def run_local(args):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    output = new_output(args)
    circuits = build_circuits()
    export_circuits(output / "logical", circuits)
    # Synthetic, homogeneous teaching noise; these are not device calibrations.
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(args.one_qubit_error, 1), ["h", "sdg"])
    noise.add_all_qubit_quantum_error(depolarizing_error(args.two_qubit_error, 2), ["cx"])
    r = args.readout_error
    noise.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    compiled = transpile(circuits, basis_gates=["h", "sdg", "cx"], optimization_level=0)
    export_circuits(output / "compiled", compiled)
    manifest = provenance()
    manifest.update({
        "mode": "local", "shots_per_circuit": args.shots, "seed": args.seed,
        "noise": {"kind": "synthetic depolarization after gates and symmetric readout flips",
                  "one_qubit_depolarizing_parameter": args.one_qubit_error,
                  "two_qubit_depolarizing_parameter": args.two_qubit_error,
                  "readout_flip_probability": r,
                  "note": "No thermal relaxation, routing, measured calibration, or official noise specification."},
    })
    for label, model in (("ideal", None), ("synthetic_noisy", noise)):
        backend = AerSimulator(method="density_matrix", noise_model=model)
        result = backend.run(compiled, shots=args.shots, seed_simulator=args.seed).result()
        counts = [result.get_counts(i) for i in range(len(compiled))]
        summary = analyze_counts(circuits, counts)
        save_json(output / f"{label}.json", {"raw_counts": counts, "summary": summary})
        print_summary(label, summary)
    save_json(output / "manifest.json", manifest)
    print(f"Saved: {output.resolve()}")


def connect(args):
    from qiskit_ibm_runtime import QiskitRuntimeService

    instance = args.instance or os.environ.get("IBM_QUANTUM_INSTANCE")
    if args.account:
        kwargs = {"name": args.account}
        if instance:
            kwargs["instance"] = instance
        return QiskitRuntimeService(**kwargs)
    if not instance:
        instance = input("IBM Quantum instance CRN (from your account): ").strip()
    if not instance:
        raise ValueError("An instance CRN is required. Choose your intended instance in IBM Quantum Platform.")
    token = os.environ.get("IBM_QUANTUM_API_KEY") or getpass("IBM Quantum API key (hidden, not saved): ")
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)


def run_hardware(args):
    validate_hardware_budget(args.shots, args.max_execution_time)
    service = connect(args)
    backend = (service.backend(args.backend) if args.backend else
               service.least_busy(operational=True, simulator=False, min_num_qubits=3))
    if backend.configuration().simulator:
        raise ValueError("The hardware command requires a real backend.")
    circuits = build_circuits()
    layout = args.layout
    if layout is None:
        # Fix a common initial placement for both variants using a connected path.
        edges = list(backend.coupling_map.get_edges())
        neighbours = {}
        for a, b in edges:
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)
        middle = next((q for q in sorted(neighbours) if len(neighbours[q]) >= 2), None)
        if middle is None:
            raise ValueError("No connected three-qubit path found. Supply --layout explicitly.")
        ends = sorted(neighbours[middle])[:2]
        layout = [ends[0], middle, ends[1]]
    if len(set(layout)) != 3 or min(layout) < 0 or max(layout) >= backend.num_qubits:
        raise ValueError("--layout must contain three distinct valid physical qubit IDs.")
    pm = generate_preset_pass_manager(backend=backend, initial_layout=layout,
                                      optimization_level=1, seed_transpiler=args.seed)
    compiled = pm.run(circuits)
    validate_hardware_budget(args.shots, args.max_execution_time, len(compiled))
    usage_estimate = estimate_hardware_usage(backend, compiled, args.shots)
    if usage_estimate["estimated_system_seconds"] > args.max_execution_time:
        raise ValueError("Planning estimate exceeds the execution cap; reduce --shots. No job was submitted.")
    output = new_output(args)
    export_circuits(output / "logical", circuits)
    export_circuits(output / "isa", compiled)
    properties = backend.properties()
    save_json(output / "backend-snapshot.json", {
        "backend": backend.name,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "num_qubits": backend.num_qubits,
        "coupling_edges": list(backend.coupling_map.get_edges()),
        "operation_names": list(backend.operation_names),
        "dt": backend.dt,
        "properties": properties.to_dict() if properties is not None else None,
        "note": "Read at compilation time; calibration may change before job execution.",
    })
    manifest = provenance()
    manifest.update({
        "mode": "hardware", "backend": backend.name,
        "shots_per_circuit": args.shots, "initial_layout": layout,
        "seed_transpiler": args.seed, "optimization_level": 1,
        "circuit_order": [c.metadata for c in circuits],
        "job_id": None,
        "max_execution_time_seconds": args.max_execution_time,
        "usage_estimate": usage_estimate,
        "submission_policy": "One job containing six circuits; no automatic retries or sessions.",
        "execution_limit_note": "Runtime caps system execution time, not queue time or wall-clock completion. Cancellation may produce no usable result.",
        "port_note": "Initial placement is fixed; routed output wires are recorded in isa/resources.json. This does not implement the PDF's fixed-output-port benchmark.",
        "statistics_note": "Raw basis-rotated estimates include preparation, analysis-gate and readout errors; no measurement mitigation.",
    })
    save_json(output / "manifest.json", manifest)
    print(f"Backend: {backend.name}; initial physical layout: {layout}")
    print(f"Prepared {len(compiled)} circuits x {args.shots} shots = {len(compiled) * args.shots} shots.")
    print(f"Runtime execution cap: {args.max_execution_time} seconds; planning estimate: "
          f"{usage_estimate['estimated_system_seconds']:.2f} seconds (not guaranteed; queue excluded).")
    print(f"Saved: {output.resolve()}")
    if not args.submit:
        print("Preparation only. Add --submit to send these circuits to the selected QPU.")
        return
    # Save the attempt before the network call: if its response is lost, inspect
    # the platform's job list instead of resubmitting and spending quota twice.
    manifest["submission_attempted_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["submission_state"] = "attempted_check_platform_before_any_retry"
    save_json(output / "manifest.json", manifest)
    job = submit_bounded_job(backend, compiled, args.shots, args.max_execution_time)
    manifest["job_id"] = job.job_id()
    manifest["submission_state"] = "submitted"
    save_json(output / "manifest.json", manifest)
    print(f"Submitted job: {job.job_id()}")
    print(f'Fetch later (PowerShell): & "{sys.executable}" "{Path(__file__).resolve()}" '
          f'fetch --manifest "{(output / "manifest.json").resolve()}"')


def fetch_result(args):
    path = Path(args.manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    job_id = manifest.get("job_id")
    if not job_id:
        raise ValueError("This manifest has no submitted job ID.")
    service = connect(args)
    job = service.job(job_id)
    status = str(job.status()).upper()
    print(f"Job {job_id}: {status}")
    usage = collect_job_usage(job)
    save_json(path.parent / "hardware-usage.json", {"job_id": job_id, "job_status": status, **usage})
    if status not in {"DONE", "COMPLETED", "JOBSTATUS.DONE"}:
        print("No results downloaded. If queued or running, repeat this fetch command later.")
        return
    with (path.parent / "logical" / "circuits.qpy").open("rb") as handle:
        circuits = qpy.load(handle)
    result = job.result()
    if len(result) != len(circuits):
        raise ValueError("Returned PUB count does not match the submitted manifest.")
    counts = [pub.data.meas.get_counts() for pub in result]
    summary = analyze_counts(circuits, counts)
    save_json(path.parent / "hardware-results.json", {
        "job_id": job_id, "backend": manifest["backend"],
        "raw_counts": counts, "summary": summary,
        "runtime_metadata": result.metadata,
        "pub_metadata": [pub.metadata for pub in result],
        "usage": usage,
        "uncertainty_scope": "Finite-shot intervals only; excludes drift and systematic measurement error.",
    })
    print_summary("REAL HARDWARE", summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    local = sub.add_parser("local", help="Run ideal and synthetic-noise simulations; no account needed.")
    hardware = sub.add_parser("hardware", help="Compile for a real QPU; add --submit to run on it.")
    fetch = sub.add_parser("fetch", help="Retrieve a submitted job without submitting another one.")
    for command in (local, hardware):
        command.add_argument("--shots", type=int, default=2048)
        command.add_argument("--seed", type=int, default=11)
        command.add_argument("--output", help="New output directory; must not exist.")
    for command in (hardware, fetch):
        command.add_argument("--instance", help="IBM instance CRN; otherwise prompt or use environment.")
        command.add_argument("--account", help="Explicitly named saved Qiskit account instead of an API-key prompt.")
    local.add_argument("--one-qubit-error", type=float, default=0.002)
    local.add_argument("--two-qubit-error", type=float, default=0.02)
    local.add_argument("--readout-error", type=float, default=0.02)
    hardware.add_argument("--backend", help="An accessible IBM QPU name; default is least busy.")
    hardware.add_argument("--layout", nargs=3, type=int, metavar=("DATA0", "DATA1", "ANCILLA"))
    hardware.add_argument("--submit", action="store_true", help="Submit six circuits; uses the selected account's QPU allocation.")
    hardware.add_argument("--max-execution-time", type=int, default=HARDWARE_MAX_SECONDS,
                          help="Runtime system-execution limit in seconds (1-120; default 120), excluding queue time.")
    fetch.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "shots") and args.shots <= 0:
        parser.error("--shots must be positive.")
    if args.mode == "local":
        for name in ("one_qubit_error", "two_qubit_error", "readout_error"):
            if not 0 <= getattr(args, name) <= 1:
                parser.error(f"{name} must be between zero and one.")
    if args.mode == "hardware":
        try:
            validate_hardware_budget(args.shots, args.max_execution_time)
        except ValueError as error:
            parser.error(str(error))
    return args


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    arguments = parse_args()
    {"local": run_local, "hardware": run_hardware, "fetch": fetch_result}[arguments.mode](arguments)
