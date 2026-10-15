"""Offline, preregistered controlled-X-noise mechanism test and analysis.

This is NOT an ambient-noise optimization claim.  A single frozen physical
layout compares baseline and bridge under the same p=0 and p=0.10 X channel.
Two new, independently acquired jobs must each pass the fixed p=0.10 endpoint.
The module only reads/writes local files; it has no cloud execution functions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
BASES = ("XX", "YY", "ZZ")
VARIANTS = ("baseline", "bridge")
BITSTRINGS = tuple(f"{index:03b}" for index in range(8))
SOURCE_NAMES = ("optimize_bell.py", "optimization_circuits.py", "optimization_analysis.py", "stress_protocol.py")
SHOTS = 512
BOOTSTRAP_REPLICATES = 5000
MINIMUM_DELTA = 0.005
SCOPE = "Controlled X-noise mechanism verification only; not ambient optimization."


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_hashes():
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def freeze_protocol(layout, stage_names, excluded_job_ids, *, backend="ibm_marrakesh", cap_seconds=45):
    """Freeze one layout, two future stages, screening exclusions, and source hashes."""
    layout, stage_names = list(layout), list(stage_names)
    if (len(layout) != 3 or any(type(q) is not int or q < 0 for q in layout)
            or len(set(layout)) != 3):
        raise ValueError("Freeze exactly three distinct nonnegative physical qubit IDs.")
    if (len(stage_names) != 2 or len(set(stage_names)) != 2
            or any(not isinstance(s, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", s) for s in stage_names)):
        raise ValueError("Freeze exactly two distinct safe future stage names.")
    excluded = sorted(set(excluded_job_ids))
    if not excluded or any(not isinstance(job, str) or not job for job in excluded):
        raise ValueError("List the prior screening job IDs explicitly.")
    if cap_seconds not in (45, 60):
        raise ValueError("Use a frozen 45- or 60-second execution cap per job.")
    plan = {
        "version": 1, "phase": "controlled_noise", "scope": SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend, "layout": layout, "stage_names": stage_names,
        "excluded_screening_job_ids": excluded, "source_sha256": _source_hashes(),
        "p_values": [0.0, 0.1], "variants": list(VARIANTS),
        "shots_per_circuit": SHOTS, "no_fault_copies": 9, "x_fault_copies": 1,
        "circuits_per_job": 60, "total_shots_per_job": 30720,
        "cap_seconds": cap_seconds,
        "order_seeds": [91371, 91372], "bootstrap_seed": 246810,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "confidence": 0.99,
        "minimum_delta": MINIMUM_DELTA,
        "primary_endpoint": "F_bridge(p=0.10) minus F_baseline(p=0.10)",
        "pass_rule": "point delta >= 0.005 and the two-sided 99% bootstrap lower bound > 0",
        "confirmation_rule": "Both frozen new distinct non-screening jobs must pass; no extra retry selection.",
        "p0_role": "Descriptive ambient result only; shares all no-fault samples with p=0.10.",
        "ambient_optimization_success": False,
        "fault_location": "X on physical layout[0], after Bell preparation and before parity checking.",
        "statistical_scope": "Fixed 9:1 stratified mixture; finite-shot conditional inference, not random Bernoulli fault allocation.",
    }
    plan["protocol_id"] = _digest(plan)
    return plan


def validate_protocol(plan, *, check_sources=True):
    body = {key: value for key, value in plan.items() if key != "protocol_id"}
    if plan.get("protocol_id") != _digest(body):
        raise ValueError("Frozen protocol contents changed.")
    fixed = {"phase": "controlled_noise", "p_values": [0.0, 0.1],
             "shots_per_circuit": SHOTS, "no_fault_copies": 9, "x_fault_copies": 1,
             "bootstrap_replicates": BOOTSTRAP_REPLICATES, "confidence": 0.99,
             "minimum_delta": MINIMUM_DELTA, "ambient_optimization_success": False}
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError("This analyzer only supports the declared fixed stress protocol.")
    if check_sources and plan["source_sha256"] != _source_hashes():
        raise ValueError("Source changed after protocol freeze; do not reinterpret the frozen study.")


def make_specs(plan):
    """Return the two ordinary optimize_bell specs; do not create or submit jobs."""
    validate_protocol(plan)
    specs = []
    for index, stage in enumerate(plan["stage_names"]):
        candidates = []
        for variant in VARIANTS:
            for copy in range(10):
                fault = None if copy < 9 else "X"
                branch = "nofault" if fault is None else "X"
                candidates.append({
                    "id": f"stress_{variant}_{branch}_{copy:02d}",
                    "variant": variant, "layout": list(plan["layout"]),
                    "optimization_level": 1, "injected_fault": fault,
                    "stress_branch": branch,
                })
        specs.append({
            "stage": stage, "backend": plan["backend"], "shots": SHOTS,
            "local_shots": SHOTS, "cap_seconds": plan["cap_seconds"],
            "order_seed": plan["order_seeds"][index], "candidates": candidates,
            "statistics_plan": dict(plan),
        })
    return specs


def _normalized_counts(counts):
    if not isinstance(counts, dict):
        raise ValueError("Raw counts must be a dictionary.")
    vector = np.zeros(8, dtype=np.int64)
    for key, count in counts.items():
        bits = key.replace(" ", "") if isinstance(key, str) else ""
        if len(bits) != 3 or any(bit not in "01" for bit in bits):
            raise ValueError("Expected three-bit ancilla,data1,data0 raw count keys.")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Counts must be nonnegative integers.")
        vector[int(bits, 2)] += int(count)
    if int(vector.sum()) != SHOTS:
        raise ValueError("Each recorded circuit must have exactly 512 shots.")
    return vector


def _parse_utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("A timezone-aware provenance timestamp is required.")
    return stamp


def _validate_and_group(plan, spec, data, manifest):
    validate_protocol(plan)
    expected_specs = {item["stage"]: item for item in make_specs(plan)}
    if spec.get("stage") not in expected_specs or spec != expected_specs[spec["stage"]]:
        raise ValueError("Stage spec differs from the frozen single-layout protocol.")
    job_id = data.get("job_id")
    if (not isinstance(job_id, str) or not job_id
            or job_id in plan["excluded_screening_job_ids"]):
        raise ValueError("The result must come from a new non-screening job.")
    if data.get("backend") != plan["backend"] or manifest.get("job_id") != job_id:
        raise ValueError("Backend/job provenance does not match.")
    if manifest.get("stage") != spec["stage"] or manifest.get("shots") != SHOTS:
        raise ValueError("Manifest stage/shots do not match the frozen spec.")
    if manifest.get("statistics_plan", {}).get("protocol_id") != plan["protocol_id"]:
        raise ValueError("The execution manifest did not record this frozen protocol.")
    if _parse_utc(manifest["created_utc"]) < _parse_utc(plan["created_utc"]):
        raise ValueError("Confirmation preparation predates protocol freeze.")
    for name in SOURCE_NAMES[:-1]:
        if manifest.get("source_sha256", {}).get(name) != plan["source_sha256"][name]:
            raise ValueError("Execution source hashes differ from the frozen protocol.")
    candidates = {candidate["id"]: candidate for candidate in spec["candidates"]}
    records = data.get("records", [])
    if len(records) != 60:
        raise ValueError("Exactly 60 circuit records are required; no selective omission.")
    grouped = np.zeros((2, 3, 2, 8), dtype=np.int64)
    seen = set()
    for record in records:
        meta = {**record.get("metadata", {}), **record}
        candidate_id, basis = meta.get("candidate_id"), meta.get("basis")
        if candidate_id not in candidates or basis not in BASES or (candidate_id, basis) in seen:
            raise ValueError("Unknown, missing or duplicated candidate/basis record.")
        seen.add((candidate_id, basis))
        candidate = candidates[candidate_id]
        variant, fault = candidate["variant"], candidate["injected_fault"]
        if (meta.get("layout") != plan["layout"] or meta.get("variant") != variant
                or meta.get("job_id") != job_id
                or meta.get("postselect_ancilla") is not (variant == "bridge")):
            raise ValueError("Record layout/variant/job/postselection differs from spec.")
        if "injected_fault" in meta and meta["injected_fault"] != fault:
            raise ValueError("Recorded injected fault differs from the frozen branch.")
        grouped[VARIANTS.index(variant), BASES.index(basis), int(fault == "X")] += _normalized_counts(meta["counts"])
    if seen != {(candidate_id, basis) for candidate_id in candidates for basis in BASES}:
        raise ValueError("A candidate/basis record is missing.")
    if not np.all(grouped[:, :, 0].sum(axis=-1) == 4608) or not np.all(grouped[:, :, 1].sum(axis=-1) == 512):
        raise ValueError("The acquisition did not retain the fixed 9:1 branch allocation.")
    return grouped


def _metrics(counts):
    """counts shape (..., variant=2, basis=3, bitstring=8)."""
    keep = np.array([[1] * 8, [1] * 4 + [0] * 4], dtype=np.int64)
    parity = np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.int64)
    retained = counts * keep[:, None, :]
    accepted = retained.sum(axis=-1)
    signed = (retained * parity).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = signed / accepted
        fidelity = (1 + correlations[..., 0] - correlations[..., 1] + correlations[..., 2]) / 4
        retention = accepted.sum(axis=-1) / counts.sum(axis=(-1, -2))
    return fidelity, retention, accepted, correlations


def _finite_float(value):
    return float(value) if np.isfinite(value) else None


def _interval(values):
    # A missing accepted basis invalidates the interval; do not drop bootstrap
    # draws until a favorable conditional distribution remains.
    if not np.all(np.isfinite(values)):
        return None
    return [float(value) for value in np.quantile(values, [0.005, 0.995])]


def analyze_job(plan, spec, data, manifest):
    """Analyze one independent job, retaining shared p0/p10 sampling dependence."""
    grouped = _validate_and_group(plan, spec, data, manifest)
    stage_index = plan["stage_names"].index(spec["stage"])
    seed = plan["bootstrap_seed"] + stage_index * 100003
    rng = np.random.default_rng(seed)
    samples = np.empty((BOOTSTRAP_REPLICATES, 2, 3, 2, 8), dtype=np.int64)
    for variant in range(2):
        for basis in range(3):
            for branch in range(2):
                vector = grouped[variant, basis, branch]
                shots = int(vector.sum())
                samples[:, variant, basis, branch] = rng.multinomial(shots, vector / shots, size=BOOTSTRAP_REPLICATES)
    scenarios, draws = {}, {}
    for name, raw, boot in (
        ("p0", grouped[:, :, 0], samples[:, :, :, 0]),
        ("p10", grouped.sum(axis=2), samples.sum(axis=3)),
    ):
        fidelity, retention, accepted, correlations = _metrics(raw)
        boot_fidelity, boot_retention, _, _ = _metrics(boot)
        delta = fidelity[1] - fidelity[0]
        delta_draws = boot_fidelity[:, 1] - boot_fidelity[:, 0]
        draws[name] = delta_draws
        row = {"delta_fidelity": _finite_float(delta), "delta_interval_99": _interval(delta_draws)}
        for vi, variant in enumerate(VARIANTS):
            total = int(raw[vi].sum())
            kept = int(accepted[vi].sum())
            row[variant] = {
                "fidelity": _finite_float(fidelity[vi]),
                "fidelity_kind": "conditional on measured ancilla=0" if vi else "all shots",
                "fidelity_interval_99": _interval(boot_fidelity[:, vi]),
                "postselected": bool(vi), "retention": float(retention[vi]),
                "retention_interval_99": _interval(boot_retention[:, vi]),
                "accepted_shots": kept, "discarded_shots": total - kept, "total_shots": total,
                "per_basis": {basis: {
                    "correlation": _finite_float(correlations[vi, bi]),
                    "accepted_shots": int(accepted[vi, bi]),
                    "total_shots": int(raw[vi, bi].sum()),
                    "raw_counts_before_postselection": {bits: int(raw[vi, bi, bit]) for bit, bits in enumerate(BITSTRINGS)},
                } for bi, basis in enumerate(BASES)},
            }
        scenarios[name] = row
    primary = scenarios["p10"]
    interval = primary["delta_interval_99"]
    passes = (primary["delta_fidelity"] is not None and interval is not None
              and primary["delta_fidelity"] >= MINIMUM_DELTA and interval[0] > 0)
    correlation = None
    if (np.all(np.isfinite(draws["p0"])) and np.all(np.isfinite(draws["p10"]))
            and np.std(draws["p0"]) > 1e-14 and np.std(draws["p10"]) > 1e-14):
        correlation = float(np.corrcoef(draws["p0"], draws["p10"])[0, 1])
    return {
        "phase": "controlled_noise", "protocol_id": plan["protocol_id"], "scope": SCOPE,
        "stage": spec["stage"], "job_id": data["job_id"], "backend": plan["backend"],
        "layout": plan["layout"], "ambient_optimization_success": False,
        "primary": {"p": 0.1, "delta_fidelity": primary["delta_fidelity"],
                    "confidence_interval_99": interval, "minimum_delta": MINIMUM_DELTA,
                    "passes": bool(passes)},
        "scenarios": scenarios,
        "shared_samples": {"p0_is_subset_of_p10": True,
                           "delta_bootstrap_correlation": correlation,
                           "note": "The same resampled no-fault counts enter both p values; p0 is descriptive, not a second test."},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": seed,
                      "method": "Independent multinomial resampling within variant x basis x fault-branch, then combine shared draws.",
                      "interval": "two-sided 99% percentile; approximate, may degenerate at observed boundaries",
                      "pairing": "Same-job matched protocol; distinct circuits are not per-shot paired observations.",
                      "limits": "Assumes exchangeability within branch; excludes time drift, systematic readout bias and calibration uncertainty."},
        "cost": {"circuit_count": 60, "total_shots": int(grouped.sum()),
                 "p0_subset_shots": int(grouped[:, :, 0].sum()),
                 "p10_shots": int(grouped.sum()), "subset_costs_are_not_additive": True,
                 "logical_qubits": 3, "logical_cx_per_circuit": {"baseline": 1, "bridge": 4},
                 "native_resources": "Use the saved ISA resources; routing may change physical gate and qubit costs.",
                 "extra_x_gates_per_fault_branch_circuit": 1, "runtime_usage": data.get("usage")},
        "branch_counts": {variant: {basis: {
            "nofault": {bits: int(grouped[vi, bi, 0, bit]) for bit, bits in enumerate(BITSTRINGS)},
            "X": {bits: int(grouped[vi, bi, 1, bit]) for bit, bits in enumerate(BITSTRINGS)},
        } for bi, basis in enumerate(BASES)} for vi, variant in enumerate(VARIANTS)},
    }


def confirm_jobs(plan, job_inputs):
    """Reanalyze exactly the two frozen independent jobs; no best-of retries."""
    validate_protocol(plan)
    if len(job_inputs) != 2:
        raise ValueError("Exactly two independent replicated jobs are required.")
    analyses = [analyze_job(plan, spec, data, manifest) for spec, data, manifest in job_inputs]
    if {row["stage"] for row in analyses} != set(plan["stage_names"]):
        raise ValueError("The two frozen stage names are both required.")
    if len({row["job_id"] for row in analyses}) != 2:
        raise ValueError("Reusing a job is not independent replication.")
    analyses.sort(key=lambda row: plan["stage_names"].index(row["stage"]))
    confirmed = all(row["primary"]["passes"] for row in analyses)
    return {
        "phase": "controlled_noise", "protocol_id": plan["protocol_id"], "scope": SCOPE,
        "confirmed": confirmed, "status": "mechanism_confirmed" if confirmed else "mechanism_not_confirmed",
        "ambient_optimization_success": False, "layout": plan["layout"], "p_values": plan["p_values"],
        "confirmation_rule": plan["confirmation_rule"], "jobs": analyses,
        "total_shots": sum(row["cost"]["total_shots"] for row in analyses),
        "interpretation": "A pass supports conditional filtering of this declared X-error channel, not ambient-noise improvement, arbitrary-noise protection, or increased unconditional correct-output throughput.",
    }


def _job_input(stage_dir):
    folder = Path(stage_dir)
    return (_read(folder / "spec.json"), _read(folder / "hardware-results.json"), _read(folder / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Write a new local protocol and two execution specs; no cloud calls.")
    freeze.add_argument("--layout", type=int, nargs=3, required=True)
    freeze.add_argument("--stages", nargs=2, default=["stress-01", "stress-02"])
    freeze.add_argument("--exclude-job-ids", nargs="+", required=True)
    freeze.add_argument("--backend", default="ibm_marrakesh")
    freeze.add_argument("--cap-seconds", type=int, choices=(45, 60), default=45)
    freeze.add_argument("--output-dir", required=True)
    for name in ("analyze", "confirm"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--stage-dir", nargs=2 if name == "confirm" else None, required=True)
        command.add_argument("--output")
    args = parser.parse_args()
    if args.command == "freeze":
        plan = freeze_protocol(args.layout, args.stages, args.exclude_job_ids,
                               backend=args.backend, cap_seconds=args.cap_seconds)
        folder = Path(args.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        outputs = [(folder / "controlled-noise-plan.json", plan)]
        outputs += [(folder / f"{spec['stage']}-spec.json", spec) for spec in make_specs(plan)]
        if any(path.exists() for path, _ in outputs):
            raise FileExistsError("Protocol/spec output exists; preserve the frozen evidence.")
        for path, value in outputs:
            _write_new(path, value)
        print(json.dumps({"protocol_id": plan["protocol_id"], "files": [str(path) for path, _ in outputs]}))
    else:
        plan = _read(args.plan)
        if args.command == "analyze":
            result = analyze_job(plan, *_job_input(args.stage_dir))
            output = Path(args.output) if args.output else Path(args.stage_dir) / "stress-analysis.json"
        else:
            result = confirm_jobs(plan, [_job_input(folder) for folder in args.stage_dir])
            output = Path(args.output) if args.output else Path(args.plan).parent / "controlled-noise-report.json"
        _write_new(output, result)
        print(json.dumps({"saved": str(output), "confirmed": result.get("confirmed"), "primary": result.get("primary")}))


if __name__ == "__main__":
    main()
