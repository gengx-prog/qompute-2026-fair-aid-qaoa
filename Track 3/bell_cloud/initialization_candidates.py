"""Offline initialization candidates; no backend access or execution.

Compare these with baseline in the same job, keeping global init_qubits=True
and rep_delay identical.  An additional reset may be redundant or imperfect.
Waiting only reduces residual excitation when it exceeds thermal equilibrium;
it can increase excitation when the initialized state is colder than that.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from optimization_circuits import build_variant

INITIALIZATION_VARIANTS = ("baseline_reset", "baseline_relax_200us")


def build_initialization_candidate(variant: str, basis: str) -> QuantumCircuit:
    """Prepend explicit data reset or 200-us parallel idle to the old baseline.

    All three qubits remain measured, without postselection.  Stage barriers
    prevent optimization from folding initialization into state preparation.
    The extra initialization barrier supplements all four original barriers.
    """
    if variant not in INITIALIZATION_VARIANTS:
        raise ValueError(f"variant must be one of {INITIALIZATION_VARIANTS}.")
    baseline = build_variant("baseline", basis)
    circuit = QuantumCircuit(*baseline.qregs, *baseline.cregs, name=f"{variant}_{basis}")
    if variant == "baseline_reset":
        circuit.reset([0, 1])
        assumption = "Global initialization is already enabled; a second data reset may be redundant or add errors."
    else:
        circuit.delay(200, 0, unit="us")
        circuit.delay(200, 1, unit="us")
        assumption = "Waiting helps only if residual excitation exceeds thermal equilibrium; it can instead heat the data."
    circuit.barrier()
    circuit.compose(baseline, inplace=True)
    circuit.metadata = {
        **baseline.metadata,
        "variant": variant,
        "postselect": False,
        "initialization_assumption": assumption,
        "required_common_init_qubits": True,
        "required_common_rep_delay_seconds": 0.00025,
        "data_wait_us": 200 if variant == "baseline_relax_200us" else 0,
        "benefit_status": "Hypothesis to test; no assumed improvement.",
    }
    return circuit
