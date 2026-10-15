"""Comparable Bell experiments with a nearest-neighbour parity detector.

The bridge detector implements exactly the same three-qubit unitary as the
original two-CX parity detector.  It replaces a triangle of required couplings
with the path q0--q1--q2.  It can therefore reduce routing overhead on a line,
although improvement on a noisy QPU must be measured rather than assumed.

All variants retain the original Bell preparation and stage barriers.  In
particular, the preparation CX must not cancel with the first detector CX:
that cancellation would remove an error-detection opportunity.

The baseline_quiet candidate omits the unused ancilla readout while retaining
the three-bit result register.  Whether this reduces readout crosstalk is an
experimental question; the circuit change alone does not establish a benefit.
"""
from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

BASES = ("XX", "YY", "ZZ")
VARIANTS = ("baseline", "baseline_quiet", "protected", "bridge", "baseline_reset", "baseline_relax_200us")


def _append_parity_check(circuit: QuantumCircuit, variant: str) -> None:
    if variant == "protected":
        circuit.cx(0, 2)
        circuit.cx(1, 2)
    elif variant == "bridge":
        # (a, b, c) -> (a, b XOR a, c)
        #           -> (a, b XOR a, c XOR b XOR a)
        #           -> (a, b, c XOR b XOR a).
        # This equality holds for an arbitrary three-qubit input, including
        # ancilla=1; it is not an identity peculiar to the ideal Bell state.
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(0, 1)


def parity_check_circuit(variant: str) -> QuantumCircuit:
    """Return just the detector, for whole-operator verification and inspection."""
    if variant not in ("protected", "bridge"):
        raise ValueError("A parity detector must be 'protected' or 'bridge'.")
    circuit = QuantumCircuit(3, name=f"{variant}_parity")
    _append_parity_check(circuit, variant)
    return circuit


def build_variant(
    variant: str,
    basis: str,
    *,
    injected_fault: str | None = None,
) -> QuantumCircuit:
    """Build one three-bit terminal-measurement circuit.

    q0 and q1 carry Bell data; q2 is the initially-zero parity ancilla.  Counts
    use the single ``meas`` register and read q2 q1 q0.  For baseline_quiet,
    meas[2] is unmeasured and remains classical zero; it must not be interpreted
    as an observed ancilla outcome.  Protected variants keep measured q2=0.
    The optional X/Z fault on q0 occurs after Bell preparation
    and before the detector; it is for deterministic verification, not an
    additional component of the ordinary hardware experiment.

    ``metadata['postselect']`` is explicit because the existing starter's
    ``analyze_counts`` only recognizes baseline/protected.  A caller evaluating
    bridge must use this metadata or deliberately map bridge to protected for
    analysis; it must never silently count rejected bridge shots.  Likewise,
    baseline_quiet uses the same unconditional analysis as baseline.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}.")
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}.")
    if injected_fault not in (None, "X", "Z"):
        raise ValueError("injected_fault must be None, 'X', or 'Z'.")

    circuit = QuantumCircuit(
        QuantumRegister(3, "q"),
        ClassicalRegister(3, "meas"),
        name=f"{variant}_{basis}",
    )
    circuit.metadata = {
        "variant": variant,
        "basis": basis,
        "postselect": variant in ("protected", "bridge"),
        "postselect_bit": 2,
        "accepted_syndrome": 0,
        "injected_fault": injected_fault,
    }
    if variant == "baseline_reset":
        circuit.reset([0, 1])
        circuit.barrier()
        circuit.metadata["initialization_hypothesis"] = "Extra reset after service initialization may change residual excitation; benefit is not assumed."
    elif variant == "baseline_relax_200us":
        circuit.delay(200, 0, unit="us")
        circuit.delay(200, 1, unit="us")
        circuit.barrier()
        circuit.metadata["initialization_hypothesis"] = "200us pre-preparation wait may relax residual excitation or heat an initially colder state; benefit is not assumed."
        circuit.metadata["extra_initial_delay_seconds"] = 0.0002
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.barrier()
    if injected_fault == "X":
        circuit.x(0)
    elif injected_fault == "Z":
        circuit.z(0)
    circuit.barrier()
    _append_parity_check(circuit, variant)
    circuit.barrier()
    if basis == "XX":
        circuit.h(0)
        circuit.h(1)
    elif basis == "YY":
        circuit.sdg(0)
        circuit.h(0)
        circuit.sdg(1)
        circuit.h(1)
    circuit.barrier()
    measured = range(2) if variant == "baseline_quiet" else range(3)
    circuit.measure(measured, measured)
    return circuit


def build_variants(
    variants: tuple[str, ...] = VARIANTS,
    *,
    injected_fault: str | None = None,
) -> list[QuantumCircuit]:
    """Return XX, YY, ZZ circuits for each requested variant, in that order."""
    return [
        build_variant(variant, basis, injected_fault=injected_fault)
        for variant in variants
        for basis in BASES
    ]
