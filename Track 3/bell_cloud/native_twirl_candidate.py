"""Offline candidate construction for a controlled native-CZ twirling study.

This module does not contact an account or execute a backend.  Its candidates
may be mixed with unchanged controls in ONE Sampler job with service-side
twirling disabled.  Each basis uses the same fixed ensemble of preparation
circuits, so the three correlations still describe one ensemble of states.
An improvement is not assumed; extra Pauli gates can also add noise.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral

from qiskit import QuantumCircuit
from qiskit.circuit import pauli_twirl_2q_gates
from qiskit.transpiler import Target

BASES = ("XX", "YY", "ZZ")


def _prefix_signature(circuit: QuantumCircuit) -> tuple:
    """Preparation/check instructions through the third original barrier."""
    operations = []
    barriers = 0
    for item in circuit.data:
        operation = item.operation
        operations.append((
            operation.name,
            tuple(circuit.find_bit(q).index for q in item.qubits),
            tuple(str(value) for value in operation.params),
        ))
        if operation.name == "barrier":
            barriers += 1
            if barriers == 3:
                return tuple(operations)
    raise ValueError("Expected three preparation/check boundaries before basis analysis.")


def make_native_cz_twirl_ensemble(
    isa_by_basis: dict[str, QuantumCircuit],
    *,
    target: Target,
    candidate_id: str,
    num_twirls: int = 16,
    seed: int = 1701,
) -> list[QuantumCircuit]:
    """Return basis-major native circuits, with metadata identifying each twirl.

    Inputs must be baseline/quiet/bridge circuits already compiled for the
    target, with the original stage barriers retained.  Use equal shots for
    every circuit.  ``target`` converts the inserted Pauli gates into allowed
    native single-qubit gates, without another layout/routing pass.  Keep the
    original controls unchanged and do not enable global Sampler twirling.

    Metadata ``candidate_id`` groups the ensemble; ``basis`` and ``twirl_index``
    identify individual results.  Existing analyzers that forbid duplicate
    bases need ``aggregate_ensemble_counts`` before their normal analysis.
    """
    if set(isa_by_basis) != set(BASES):
        raise ValueError("Provide exactly the XX, YY and ZZ compiled circuits.")
    if isinstance(num_twirls, bool) or not isinstance(num_twirls, Integral) or num_twirls < 2:
        raise ValueError("num_twirls must be an integer of at least two.")
    if not candidate_id or not isinstance(candidate_id, str):
        raise ValueError("candidate_id must be a nonempty string.")
    reference = _prefix_signature(isa_by_basis["XX"])
    if any(_prefix_signature(circuit) != reference for circuit in isa_by_basis.values()):
        raise ValueError("All measurement bases must share the same compiled preparation/check.")
    generated = {}
    for basis in BASES:
        original = isa_by_basis[basis]
        if not original.count_ops().get("cz", 0):
            raise ValueError("Expected at least one native CZ to twirl.")
        # Restarting from the same seed matches the ensemble across bases;
        # the explicit prefix check below verifies that property in the SDK.
        generated[basis] = pauli_twirl_2q_gates(
            original, twirling_gate="cz", seed=seed,
            num_twirls=int(num_twirls), target=target,
        )
        for index, circuit in enumerate(generated[basis]):
            circuit.name = f"{candidate_id}_{basis}_twirl{index:03d}"
            circuit.metadata = {
                **(original.metadata or {}),
                "candidate_id": candidate_id,
                "basis": basis,
                "twirl_index": index,
                "twirl_count": int(num_twirls),
                "twirl_seed": int(seed),
                "twirling_gate": "cz",
                "service_twirling_required": False,
            }
    for index in range(num_twirls):
        reference = _prefix_signature(generated["XX"][index])
        if any(_prefix_signature(generated[basis][index]) != reference for basis in BASES):
            raise ValueError("Generated twirls differ across measurement-basis preparations.")
    return [circuit for basis in BASES for circuit in generated[basis]]


def aggregate_ensemble_counts(
    circuits: list[QuantumCircuit], counts_list: list[dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Pool raw counts per basis after verifying a balanced, complete ensemble.

    No readout correction or postselection is performed here.  A subsequent
    analyzer must preserve the input variant's explicit postselection flag.
    Statistical claims are conditional on this fixed ensemble; confirm using
    independent data and preferably a separately frozen randomization seed.
    """
    if not circuits or len(circuits) != len(counts_list):
        raise ValueError("Provide one count dictionary for every ensemble circuit.")
    first = circuits[0].metadata
    identity_keys = ("candidate_id", "twirl_count", "twirl_seed", "postselect")
    identity = tuple(first.get(key) for key in identity_keys)
    seen = set()
    pooled = {basis: Counter() for basis in BASES}
    shots_per_circuit = None
    for circuit, counts in zip(circuits, counts_list):
        meta = circuit.metadata or {}
        if tuple(meta.get(key) for key in identity_keys) != identity:
            raise ValueError("Counts mix different ensembles or postselection rules.")
        basis, index = meta.get("basis"), meta.get("twirl_index")
        if basis not in BASES or (basis, index) in seen:
            raise ValueError("Each basis/twirl pair must occur exactly once.")
        if not isinstance(index, int) or not 0 <= index < meta["twirl_count"]:
            raise ValueError("Invalid twirl index.")
        seen.add((basis, index))
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in counts.values()):
            raise ValueError("Raw shot counts must be nonnegative integers.")
        shots = sum(counts.values())
        if shots <= 0 or (shots_per_circuit is not None and shots != shots_per_circuit):
            raise ValueError("Every twirl must use the same positive number of shots.")
        shots_per_circuit = shots
        pooled[basis].update(counts)
    expected = {(basis, index) for basis in BASES for index in range(first["twirl_count"])}
    if seen != expected:
        raise ValueError("Incomplete ensemble; do not silently reweight retained twirls.")
    return {basis: dict(counts) for basis, counts in pooled.items()}
