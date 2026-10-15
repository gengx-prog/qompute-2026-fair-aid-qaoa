"""Bell candidate screening and independent confirmation, without cloud access.

Protocol (fix before looking at hardware results): screen candidates, freeze
one candidate for an attempt, and run that candidate and its controls in NEW
jobs. A confirmed improvement requires a raw fidelity increase >= 0.005 and a
two-sided difference interval strictly above zero. The planned protocol requires
two independent passing replications and allows at most three candidate attempts,
each using alpha <= .05/3. A postselected candidate
must also pass against its unprotected control on the same physical layout.
Screening is exploratory, even when a screening interval excludes zero.

Primary intervals propagate Bonferroni-adjusted Wilson binomial bounds. Their
coverage is approximate, with conservative endpoint propagation; they are not
exact confidence guarantees. Independent-shot plug-in standard errors and normal
intervals are also reported. Neither method includes drift or readout bias.
Raw fidelities and fidelity differences are never clipped to physical ranges.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from numbers import Integral
from statistics import NormalDist

BASES = ("XX", "YY", "ZZ")
MINIMUM_IMPROVEMENT = 0.005
DEFAULT_CONFIDENCE = 0.95
PRIMARY_INTERVAL = "bonferroni_wilson"
UNCERTAINTY_NOTE = (
    "Finite independent-shot uncertainty, conditional on accepted counts. "
    "Wilson intervals have approximate coverage; Bonferroni adjustment and "
    "endpoint propagation are conservative. Plug-in SE can be zero at sample "
    "boundaries. Excludes drift, measurement bias, and other systematic errors. "
    "Postselected fidelity describes retained trials, not all attempted trials. "
    "The three bases must measure the same prepared state. Estimates are not clipped."
)


def _confidence(value):
    if isinstance(value, bool) or not 0 < value < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    return float(value)


def _resolve_confidence(confidence, alpha):
    if alpha is not None:
        _confidence(alpha)
        if confidence != DEFAULT_CONFIDENCE and abs(confidence - (1 - alpha)) > 1e-12:
            raise ValueError("confidence and alpha disagree.")
        confidence = 1 - alpha
    return _confidence(confidence)


def _wilson(successes, trials, z):
    fraction = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (fraction + z2 / (2 * trials)) / denominator
    radius = z * sqrt(fraction * (1 - fraction) / trials + z2 / (4 * trials**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _basis_summary(counts, basis, postselect_ancilla):
    if not isinstance(counts, Mapping):
        raise ValueError("counts must map three-bit strings to nonnegative integers.")
    raw = {}
    total = accepted = even = 0
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str):
            raise ValueError("Count keys must be strings in ancilla, data1, data0 order.")
        bits = bitstring.replace(" ", "")
        if len(bits) != 3 or any(bit not in "01" for bit in bits):
            raise ValueError("Count keys must contain exactly three binary digits.")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Shot counts must be nonnegative integers.")
        count = int(count)
        raw[bitstring] = count
        total += count
        if postselect_ancilla and bits[0] != "0":
            continue
        accepted += count
        even += count if bits[1] == bits[2] else 0
    # Phi+ has positive XX/ZZ and negative YY correlations. A 'success'
    # therefore means odd data parity for YY and even parity otherwise.
    successes = accepted - even if basis == "YY" else even
    probability = successes / accepted if accepted else None
    correlation = 2 * even / accepted - 1 if accepted else None
    return {
        "raw_counts": raw, "total_shots": total, "accepted_shots": accepted,
        "even_parity_shots": even, "successes": successes,
        "success_probability": probability, "value": correlation,
        "standard_error": sqrt(max(0.0, 1 - correlation**2) / accepted) if accepted else None,
        "acceptance_fraction": accepted / total if total else None,
    }


def summarize_candidate(basis_counts, *, postselect_ancilla=False, confidence=DEFAULT_CONFIDENCE, alpha=None):
    """Summarize {XX: counts, YY: counts, ZZ: counts}; leftmost bit is ancilla.

    ``postselect_ancilla=True`` retains ancilla-zero shots. The caller supplies
    this from circuit metadata; the candidate's name never controls selection.
    """
    confidence = _resolve_confidence(confidence, alpha)
    if type(postselect_ancilla) is not bool:
        raise ValueError("postselect_ancilla must be an explicit bool.")
    if set(basis_counts) != set(BASES):
        raise ValueError("Each candidate must contain exactly XX, YY, and ZZ.")
    rows = {basis: _basis_summary(basis_counts[basis], basis, postselect_ancilla) for basis in BASES}
    total = sum(row["total_shots"] for row in rows.values())
    accepted = sum(row["accepted_shots"] for row in rows.values())
    fidelity = standard_error = interval = normal_interval = None
    if all(row["accepted_shots"] for row in rows.values()):
        fidelity = (sum(row["success_probability"] for row in rows.values()) - 1) / 2
        variance = sum(row["success_probability"] * (1 - row["success_probability"])
                       / row["accepted_shots"] for row in rows.values()) / 4
        standard_error = sqrt(variance)
        z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
        normal_interval = [fidelity - z * standard_error, fidelity + z * standard_error]
        z_family = NormalDist().inv_cdf(1 - (1 - confidence) / (2 * 3))
        intervals = [_wilson(row["successes"], row["accepted_shots"], z_family) for row in rows.values()]
        interval = [(sum(bounds[0] for bounds in intervals) - 1) / 2,
                    (sum(bounds[1] for bounds in intervals) - 1) / 2]
    return {
        "fidelity": fidelity, "fidelity_standard_error": standard_error,
        "fidelity_interval": interval, "fidelity_normal_interval": normal_interval,
        "confidence": confidence, "alpha": 1 - confidence, "interval_method": PRIMARY_INTERVAL,
        "correlations": rows, "raw_counts": {basis: rows[basis]["raw_counts"] for basis in BASES},
        "total_shots": total, "accepted_shots": accepted,
        "retention": accepted / total if total else None,
        "acceptance_fraction": accepted / total if total else None,
        "postselected": postselect_ancilla, "uncertainty_note": UNCERTAINTY_NOTE,
    }


def analyze_records(records, *, confidence=DEFAULT_CONFIDENCE, alpha=None):
    """Group circuit records into candidate summaries.

    A record is ``{'candidate_id': str, 'basis': 'XX'|'YY'|'ZZ', 'counts':
    dict, 'job_id': str, 'layout': [q0,q1,anc], 'postselect_ancilla': bool}``.
    Metadata fields can instead be nested under ``metadata``; flat fields win.
    ``postselect`` is accepted as an alias, or ``variant='protected'`` when no
    explicit flag exists. Job and layout must agree across a candidate's bases.
    """
    confidence = _resolve_confidence(confidence, alpha)
    grouped = {}
    for record in records:
        data = {**record.get("metadata", {}), **{key: value for key, value in record.items() if key != "metadata"}}
        candidate_id, basis = data.get("candidate_id"), data.get("basis")
        if not isinstance(candidate_id, str) or not candidate_id or basis not in BASES:
            raise ValueError("Each record requires a nonempty candidate_id and a valid basis.")
        flag = data.get("postselect_ancilla", data.get("postselect", data.get("variant") == "protected"))
        if type(flag) is not bool:
            raise ValueError("postselect_ancilla/postselect metadata must be bool.")
        layout = list(data["layout"]) if data.get("layout") is not None else None
        metadata = {"candidate_id": candidate_id, "job_id": data.get("job_id"),
                    "layout": layout, "postselected": flag, "variant": data.get("variant")}
        if candidate_id not in grouped:
            grouped[candidate_id] = {"metadata": metadata, "counts": {}}
        group = grouped[candidate_id]
        if group["metadata"] != metadata:
            raise ValueError(f"Inconsistent job/layout/postselection across {candidate_id} bases.")
        if basis in group["counts"]:
            raise ValueError(f"Duplicate {candidate_id}/{basis}; do not pool jobs silently.")
        group["counts"][basis] = data["counts"]
    if not grouped:
        raise ValueError("Provide at least one candidate.")
    return {name: {**summarize_candidate(group["counts"],
                                        postselect_ancilla=group["metadata"]["postselected"],
                                        confidence=confidence), **group["metadata"]}
            for name, group in grouped.items()}


def compare_candidates(candidate, baseline, *, confidence=DEFAULT_CONFIDENCE, alpha=None, comparison_count=1):
    """Independent-shot fidelity difference, with simultaneous comparison bounds.

    Six Wilson bounds use alpha/(6*comparison_count), and the propagated
    difference interval is the primary interval. The normal approximation is
    supplementary. ``comparison_count`` may adjust screening for many choices.
    """
    confidence = _resolve_confidence(confidence, alpha)
    if isinstance(comparison_count, bool) or not isinstance(comparison_count, Integral) or comparison_count < 1:
        raise ValueError("comparison_count must be a positive integer.")
    output = {"candidate_id": candidate.get("candidate_id"), "baseline_id": baseline.get("candidate_id"),
              "confidence": confidence, "alpha": 1 - confidence, "comparison_count": int(comparison_count),
              "interval_method": PRIMARY_INTERVAL, "improvement": None,
              "improvement_standard_error": None, "improvement_interval": None,
              "improvement_lower": None, "improvement_normal_interval": None}
    if candidate["fidelity"] is None or baseline["fidelity"] is None:
        output["unavailable_reason"] = "At least one basis has no accepted shots."
        return output
    improvement = candidate["fidelity"] - baseline["fidelity"]
    stderr = sqrt(candidate["fidelity_standard_error"]**2 + baseline["fidelity_standard_error"]**2)
    alpha = (1 - confidence) / comparison_count
    z_normal = NormalDist().inv_cdf(1 - alpha / 2)
    z_wilson = NormalDist().inv_cdf(1 - alpha / (2 * 6))
    intervals = []
    for summary in (candidate, baseline):
        intervals.append([_wilson(summary["correlations"][basis]["successes"],
                                  summary["correlations"][basis]["accepted_shots"], z_wilson)
                          for basis in BASES])
    interval = [
        (sum(row[0] for row in intervals[0]) - sum(row[1] for row in intervals[1])) / 2,
        (sum(row[1] for row in intervals[0]) - sum(row[0] for row in intervals[1])) / 2,
    ]
    output.update({"improvement": improvement, "improvement_standard_error": stderr,
                   "improvement_interval": interval, "improvement_lower": interval[0],
                   "improvement_normal_interval": [improvement - z_normal * stderr,
                                                   improvement + z_normal * stderr]})
    return output


def _controls(summaries, candidate_id, original_baseline_id, matching_baseline_id):
    candidate, baseline = summaries[candidate_id], summaries[original_baseline_id]
    if candidate_id == original_baseline_id or baseline["postselected"]:
        raise ValueError("The original baseline must be a distinct unpostselected candidate.")
    controls = [original_baseline_id]
    if candidate["postselected"]:
        if not matching_baseline_id or matching_baseline_id not in summaries:
            raise ValueError("A protected candidate requires its same-layout unprotected baseline.")
        matching = summaries[matching_baseline_id]
        if matching["postselected"] or candidate.get("layout") is None or matching.get("layout") != candidate["layout"]:
            raise ValueError("The protected candidate's control must be unprotected on the same layout.")
        if matching_baseline_id not in controls:
            controls.append(matching_baseline_id)
    return controls


def screen_candidates(summaries, original_baseline_id, *, matching_baselines=None,
                      candidate_ids=None, minimum_improvement=MINIMUM_IMPROVEMENT,
                      confidence=DEFAULT_CONFIDENCE, alpha=None, familywise=True):
    """Rank exploratory candidates; selection is not an improvement claim.

    ``matching_baselines`` maps protected candidate IDs to unprotected control
    IDs. Specify ``candidate_ids`` to exclude control-only candidates. A
    candidate qualifies for confirmation by point improvement >= the threshold
    against every required control; its screening CI need not exclude zero.
    """
    confidence = _resolve_confidence(confidence, alpha)
    matching_baselines = matching_baselines or {}
    candidate_ids = list(candidate_ids) if candidate_ids is not None else [key for key in summaries if key != original_baseline_id]
    if minimum_improvement < MINIMUM_IMPROVEMENT:
        raise ValueError("The prespecified minimum improvement cannot be below 0.005.")
    controls = {name: _controls(summaries, name, original_baseline_id, matching_baselines.get(name)) for name in candidate_ids}
    comparison_count = max(1, sum(map(len, controls.values()))) if familywise else 1
    ranking = []
    for name in candidate_ids:
        comparisons = {control: compare_candidates(summaries[name], summaries[control], confidence=confidence,
                                                   comparison_count=comparison_count) for control in controls[name]}
        eligible = all(row["improvement"] is not None and row["improvement"] >= minimum_improvement
                       for row in comparisons.values())
        ranking.append({"candidate_id": name, "fidelity": summaries[name]["fidelity"],
                        "retention": summaries[name]["retention"], "comparisons": comparisons,
                        "eligible_for_confirmation": eligible})
    ranking.sort(key=lambda row: (row["fidelity"] is not None,
                                  row["fidelity"] if row["fidelity"] is not None else float("-inf")), reverse=True)
    selected = next((row["candidate_id"] for row in ranking if row["eligible_for_confirmation"]), None)
    return {"phase": "exploratory_screening", "ranking": ranking,
            "selected_candidate_id": selected, "minimum_improvement": minimum_improvement,
            "confidence": confidence, "alpha": 1 - confidence, "familywise_comparison_count": comparison_count,
            "screening_job_ids": sorted({row["job_id"] for row in summaries.values() if row.get("job_id")}),
            "note": "Selection from many candidates is optimistic. Freeze at most one candidate, then independently confirm in one new job with its controls. Screening never establishes an improvement."}


def confirm_candidate(summaries, candidate_id, original_baseline_id, *, screening_job_ids,
                      matching_baseline_id=None, minimum_improvement=MINIMUM_IMPROVEMENT,
                      confidence=DEFAULT_CONFIDENCE, alpha=None):
    """Evaluate one independent replication for the previously frozen candidate.

    Controls and candidate must share one fresh job ID absent from all screening
    job IDs. Multiple required comparisons get simultaneous Bonferroni-Wilson
    bounds. Pass alpha=.05/3 for the planned maximum of three candidate attempts;
    aggregate the two required replications with ``confirm_replicates``.
    """
    confidence = _resolve_confidence(confidence, alpha)
    if minimum_improvement < MINIMUM_IMPROVEMENT:
        raise ValueError("The prespecified minimum improvement cannot be below 0.005.")
    if confidence < DEFAULT_CONFIDENCE:
        raise ValueError("Confirmation confidence cannot be below 0.95.")
    if not screening_job_ids or any(not isinstance(job, str) or not job for job in screening_job_ids):
        raise ValueError("Supply nonempty screening job IDs to verify independent confirmation.")
    controls = _controls(summaries, candidate_id, original_baseline_id, matching_baseline_id)
    job_ids = {summaries[name].get("job_id") for name in [candidate_id, *controls]}
    if len(job_ids) != 1 or None in job_ids or "" in job_ids or job_ids.intersection(screening_job_ids):
        raise ValueError("Confirmation candidate and controls must share one new job, independent of all screening jobs.")
    comparisons = {control: compare_candidates(summaries[candidate_id], summaries[control], confidence=confidence,
                                               comparison_count=len(controls)) for control in controls}
    for row in comparisons.values():
        row["passes"] = (row["improvement"] is not None and row["improvement"] >= minimum_improvement
                         and row["improvement_lower"] > 0)
    confirmed = all(row["passes"] for row in comparisons.values())
    return {"phase": "independent_confirmation", "candidate_id": candidate_id,
            "job_id": next(iter(job_ids)), "screening_job_ids": sorted(set(screening_job_ids)),
            "minimum_improvement": minimum_improvement, "confidence": confidence, "alpha": 1 - confidence,
            "primary_interval": PRIMARY_INTERVAL, "comparisons": comparisons,
            "confirmed_improvement": confirmed,
            "interpretation": "This replication passes the prespecified shot-noise criterion." if confirmed else
                              "An improvement was not established; this does not prove equal performance.",
            "scope": "One replication of a previously frozen candidate. Aggregate the prespecified independent replications before declaring success. This does not establish the globally optimal method.",
            "uncertainty_note": UNCERTAINTY_NOTE}


def confirm_replicates(reports, *, required_replicates=2, max_candidate_attempts=3,
                       family_alpha=0.05):
    """Require every prespecified replication to pass without pooling or picking.

    Each candidate attempt uses alpha <= family_alpha/max_candidate_attempts.
    Both replications must pass; they are not alternative opportunities to pass.
    The caller must retain all candidate attempts and enforce the attempt limit.
    """
    reports = list(reports)
    if (type(required_replicates) is not int or required_replicates < 2 or
            type(max_candidate_attempts) is not int or not 1 <= max_candidate_attempts <= 3):
        raise ValueError("Require at least two replications and one to three candidate attempts.")
    _confidence(family_alpha)
    if family_alpha > 0.05:
        raise ValueError("Family-wise alpha cannot exceed 0.05.")
    if len(reports) != required_replicates:
        raise ValueError("Provide exactly all prespecified replication reports; do not select a passing subset.")
    candidates = {row.get("candidate_id") for row in reports}
    jobs = {row.get("job_id") for row in reports}
    if (len(candidates) != 1 or None in candidates or len(jobs) != required_replicates
            or any(not isinstance(job, str) or not job for job in jobs)):
        raise ValueError("Replications must evaluate the same candidate in distinct jobs.")
    if any(row.get("phase") != "independent_confirmation" or row.get("alpha", 1) > family_alpha / max_candidate_attempts + 1e-12
           for row in reports):
        raise ValueError("Each replication must use independent_confirmation and the prespecified attempt-adjusted alpha.")
    control_sets = {tuple(sorted(row["comparisons"])) for row in reports}
    if len(control_sets) != 1:
        raise ValueError("Replications must include the same prespecified controls.")
    passed = all(row["confirmed_improvement"] for row in reports)
    return {"phase": "replicated_confirmation", "candidate_id": next(iter(candidates)),
            "job_ids": sorted(jobs), "required_replicates": required_replicates,
            "max_candidate_attempts": max_candidate_attempts, "family_alpha": family_alpha,
            "alpha_per_attempt": family_alpha / max_candidate_attempts,
            "confirmed_improvement": passed, "replications": reports,
            "interpretation": "Both independent replications pass the prespecified improvement criterion." if passed else
                              "A repeatable improvement was not established.",
            "multiplicity_note": "At most three prespecified candidate attempts; Bonferroni alpha across attempts. Every replication must pass. All failed attempts must remain in the report; the caller enforces the attempt limit.",
            "scope": "Best supported tested candidate, not proof of a globally optimal method."}
