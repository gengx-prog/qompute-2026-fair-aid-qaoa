"""Bell-state circuits and transparent, shot-based fidelity estimates.

The protected circuit detects odd ZZ parity and discards that syndrome. It
does not correct errors or detect a single Z error. All measurements occur
at the end: ancilla measurement commutes with the data analysis rotations.
"""
from __future__ import annotations

from math import sqrt
from numbers import Integral
from statistics import NormalDist

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

BASES = ("XX", "YY", "ZZ")
VARIANTS = ("baseline", "protected")
UNCERTAINTY_NOTE = (
    "Standard errors describe independent shot noise only, conditional on the "
    "accepted count; plug-in errors may be zero at boundaries. Fidelity bounds "
    "combine three Bonferroni-adjusted Wilson intervals (approximate 95% joint "
    "coverage). They exclude calibration drift, readout bias, gate errors, and "
    "other systematic uncertainty. Fidelity assumes the same prepared state "
    "across XX, YY, and ZZ settings. Raw estimates are not clipped."
)


def build_circuits(*, injected_fault: str | None = None) -> list[QuantumCircuit]:
    """Return baseline XX/YY/ZZ then protected XX/YY/ZZ circuits.

    q0/q1 are data; q2 is an initially zero parity ancilla. Each result has
    one ``meas`` register whose strings read q2 q1 q0. ``injected_fault`` is
    an optional deterministic X or Z on q0 after preparation, for tests.
    """
    if injected_fault not in (None, "X", "Z"):
        raise ValueError("injected_fault must be None, 'X', or 'Z'.")
    circuits = []
    for variant in VARIANTS:
        for basis in BASES:
            circuit = QuantumCircuit(
                QuantumRegister(3, "q"), ClassicalRegister(3, "meas"),
                name=f"{variant}_{basis}",
            )
            circuit.metadata = {"variant": variant, "basis": basis}
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.barrier()
            if injected_fault == "X":
                circuit.x(0)
            elif injected_fault == "Z":
                circuit.z(0)
            circuit.barrier()
            if variant == "protected":
                circuit.cx(0, 2)
                circuit.cx(1, 2)
            circuit.barrier()
            if basis == "XX":
                circuit.h(0)
                circuit.h(1)
            elif basis == "YY":
                # S-dagger followed by H measures Y in the Z readout basis.
                circuit.sdg(0)
                circuit.h(0)
                circuit.sdg(1)
                circuit.h(1)
            circuit.barrier()
            circuit.measure(range(3), range(3))
            circuits.append(circuit)
    return circuits


def _wilson(successes: int, trials: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial probability, including boundaries."""
    fraction = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (fraction + z2 / (2 * trials)) / denominator
    radius = z * sqrt(fraction * (1 - fraction) / trials + z2 / (4 * trials**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _summarize_counts(counts: dict[str, int], *, postselect: bool) -> dict:
    total = accepted = even = 0
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str):
            raise ValueError("Count keys must be three-bit strings in q2 q1 q0 order.")
        bits = bitstring.replace(" ", "")
        if len(bits) != 3 or any(bit not in "01" for bit in bits):
            raise ValueError("Count keys must be three-bit strings in q2 q1 q0 order.")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Shot counts must be nonnegative integers.")
        count = int(count)
        total += count
        if postselect and bits[0] != "0":
            continue
        accepted += count
        if bits[1] == bits[2]:
            even += count

    correlation = None if accepted == 0 else 2.0 * even / accepted - 1.0
    stderr = None if accepted == 0 else sqrt(max(0.0, 1.0 - correlation**2) / accepted)
    interval = None
    if accepted:
        # Family-wise alpha=.05 over three correlators; each two-sided
        # interval has alpha=.05/3. No spurious zero-width boundary interval.
        z = NormalDist().inv_cdf(1.0 - 0.05 / 6)
        lower, upper = _wilson(even, accepted, z)
        interval = [2.0 * lower - 1.0, 2.0 * upper - 1.0]
    return {
        "value": correlation,
        "standard_error": stderr,
        "simultaneous_interval_95": interval,
        "even_parity_shots": even,
        "accepted_shots": accepted,
        "total_shots": total,
        "acceptance_fraction": accepted / total if total else None,
    }


def analyze_counts(circuits: list[QuantumCircuit], counts_list: list[dict[str, int]]) -> dict:
    """Analyze six independent measurement settings without altering raw counts.

    Fidelity to Phi+ is (1 + XX - YY + ZZ)/4. Protected correlations are
    conditioned on measured ancilla zero; baseline uses every shot. Return
    JSON-serializable results under ``baseline`` and ``protected``. Missing
    accepted data in any setting yields None for fidelity and its uncertainty.
    """
    if len(circuits) != 6 or len(counts_list) != len(circuits):
        raise ValueError("Provide exactly the six circuits and one count dictionary per circuit.")
    summaries = {variant: {} for variant in VARIANTS}
    for circuit, counts in zip(circuits, counts_list):
        metadata = circuit.metadata or {}
        variant, basis = metadata.get("variant"), metadata.get("basis")
        if variant not in VARIANTS or basis not in BASES:
            raise ValueError("Each circuit needs valid variant and basis metadata.")
        if basis in summaries[variant]:
            raise ValueError(f"Duplicate measurement setting: {variant}/{basis}.")
        summaries[variant][basis] = _summarize_counts(counts, postselect=variant == "protected")

    result = {}
    for variant in VARIANTS:
        correlations = summaries[variant]
        if set(correlations) != set(BASES):
            raise ValueError(f"Missing measurement settings for {variant}.")
        total = sum(row["total_shots"] for row in correlations.values())
        accepted = sum(row["accepted_shots"] for row in correlations.values())
        fidelity = stderr = interval = None
        if all(correlations[basis]["value"] is not None for basis in BASES):
            xx, yy, zz = (correlations[basis] for basis in BASES)
            fidelity = (1.0 + xx["value"] - yy["value"] + zz["value"]) / 4.0
            stderr = sqrt(sum(row["standard_error"] ** 2 for row in (xx, yy, zz))) / 4.0
            bounds = [row["simultaneous_interval_95"] for row in (xx, yy, zz)]
            interval = [
                (1.0 + bounds[0][0] - bounds[1][1] + bounds[2][0]) / 4.0,
                (1.0 + bounds[0][1] - bounds[1][0] + bounds[2][1]) / 4.0,
            ]
        result[variant] = {
            "fidelity": fidelity,
            "fidelity_standard_error": stderr,
            "fidelity_interval_95": interval,
            "correlations": correlations,
            "accepted_shots": accepted,
            "total_shots": total,
            "acceptance_fraction": accepted / total if total else None,
            "postselected": variant == "protected",
            "uncertainty_note": UNCERTAINTY_NOTE,
        }
    return result
