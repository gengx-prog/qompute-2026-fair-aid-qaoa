"""Feasibility-preserving circuits for the equal-cost aid allocation problem.

The ring XY mixer keeps the number of selected projects fixed. It is an
ordered product of two-qubit exchange rotations, not the exact exponential
of the sum of all ring interactions. The initial Dicke state depends only on
``n`` and ``k``; no utility scores or known optimum enter its construction.

References (IBM Quantum API):
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RXXGate
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RYYGate
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.StatePreparation
"""
from __future__ import annotations

from math import comb, sqrt
from numbers import Integral

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import StatePreparation


def _validate_n(n: int) -> int:
    # StatePreparation consumes an explicit 2**n vector. Keep the same small
    # instance limit as AllocationProblem, before any exponential allocation.
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or not 2 <= n <= 10:
        raise ValueError("n must be an integer between 2 and 10.")
    return int(n)


def make_xy_mixer(n: int) -> QuantumCircuit:
    """Return one ring sweep, parameterized by a single shared ``beta``.

    Apply edges (0,1), (1,2), ..., (n-2,n-1), (n-1,0), in that order.
    For n=2 the sole edge is applied once. Each edge applies RXX(beta)
    then RYY(beta), giving exp[-i beta (XX+YY)/2] because the two generators
    on the same edge commute. This exchanges |01> and |10> while fixing
    |00> and |11>. Adjacent edges need not commute, so this particular
    ordered product defines the variational mixer (one Trotter sweep).

    Feasibility holds after complete exchange pairs in an ideal circuit;
    individual RXX/RYY gates and noisy hardware can leave the subspace.
    """
    n = _validate_n(n)
    beta = Parameter("beta")
    mixer = QuantumCircuit(n, name="XY_ring")
    edges = [(i, i + 1) for i in range(n - 1)]
    if n > 2:
        edges.append((n - 1, 0))
    for first, second in edges:
        mixer.rxx(beta, first, second)
        mixer.ryy(beta, first, second)
    return mixer


def make_feasible_initial_state(n: int, k: int) -> QuantumCircuit:
    """Prepare the uniform, positive-amplitude Dicke state of weight ``k``.

    All C(n,k) feasible bitstrings receive amplitude 1/sqrt(C(n,k)). This
    is a small-instance reference preparation using a dense amplitude vector
    and Qiskit's general StatePreparation synthesis, not an efficient Dicke
    preparation algorithm. Include this circuit when transpiling and counting
    resources; it is not a free initial state. Reported logical all-to-all
    gate counts do not include routing to a particular hardware topology.
    The endpoint cases k=0 and k=n are supported for validation as well.
    """
    n = _validate_n(n)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or not 0 <= k <= n:
        raise ValueError("k must be an integer between 0 and n.")
    k = int(k)
    amplitude = 1.0 / sqrt(comb(n, k))
    amplitudes = np.zeros(2**n, dtype=complex)
    for state in range(2**n):
        if state.bit_count() == k:
            amplitudes[state] = amplitude
    initial = QuantumCircuit(n, name=f"Dicke_{n}_{k}")
    initial.append(StatePreparation(amplitudes), range(n))
    return initial
