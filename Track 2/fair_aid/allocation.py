"""Six-qubit, synthetic aid allocation with independently checked QUBO encoding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np
from qiskit.quantum_info import SparsePauliOp

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class AllocationProblem:
    ids: tuple[str, ...]
    names: tuple[str, ...]
    benefit: np.ndarray
    equity: np.ndarray
    underserved: np.ndarray
    overlap: np.ndarray  # Symmetric; each pair is charged exactly once.
    k: int = 3

    @property
    def n(self) -> int:
        return len(self.ids)

    def __post_init__(self):
        n = len(self.ids)
        if not 2 <= n <= 10 or not isinstance(self.k, (int, np.integer)) or not 0 < self.k < n:
            raise ValueError("Use 2–10 projects and integer 0 < k < n for this small-instance solver.")
        if len(set(self.ids)) != n or len(self.names) != n:
            raise ValueError("Project IDs must be unique and names must match IDs.")
        for values in (self.benefit, self.equity, self.underserved):
            if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("Invalid project scores.")
        if not np.all(np.isin(self.underserved, [0, 1])):
            raise ValueError("underserved must be binary.")
        if (self.overlap.shape != (n, n) or not np.all(np.isfinite(self.overlap))
                or np.any(self.overlap < 0)
                or not np.allclose(self.overlap, self.overlap.T)
                or not np.allclose(np.diag(self.overlap), 0)):
            raise ValueError("Overlap must be nonnegative, symmetric, and zero on the diagonal.")

    def metrics(self, bits: np.ndarray) -> dict:
        bits = np.asarray(bits)
        if bits.shape != (self.n,) or not np.all(np.isin(bits, [0, 1])):
            raise ValueError("Expected one binary decision per project.")
        bits = bits.astype(int)
        overlap_loss = float(bits @ self.overlap @ bits / 2)
        return {
            "selected": ",".join(pid for pid, bit in zip(self.ids, bits) if bit),
            "count": int(bits.sum()),
            "benefit": float(self.benefit @ bits),
            "overlap_loss": overlap_loss,
            "net_benefit": float(self.benefit @ bits - overlap_loss),
            "equity_points": float(self.equity @ bits),
            "underserved_projects": int(self.underserved @ bits),
            "feasible": bool(bits.sum() == self.k),
        }

    def cost(self, bits: np.ndarray, weight: float, penalty: float) -> float:
        self._validate_parameters(weight, penalty)
        m = self.metrics(bits)
        return float(-m["net_benefit"] - weight * m["equity_points"]
                     + penalty * (m["count"] - self.k) ** 2)

    @staticmethod
    def _validate_parameters(weight: float, penalty: float):
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(penalty) or penalty < 0:
            raise ValueError("Equity weight and constraint penalty must be finite and nonnegative.")

    def penalty_bound(self, weight: float) -> float:
        """A strictly greater penalty makes every infeasible state improvable.

        Repair under-selection by adding a project, over-selection by removing
        one. The largest possible increase in the unpenalized cost per flip is
        max_i(max(w_i, sum_j o_ij - w_i)), w_i=b_i+weight*e_i.
        Each repair reduces the squared constraint violation by at least one.
        """
        self._validate_parameters(weight, 0)
        scores = self.benefit + weight * self.equity
        return float(np.max(np.maximum(scores, self.overlap.sum(axis=1) - scores)))

    def qubo(self, weight: float, penalty: float) -> tuple[float, np.ndarray, np.ndarray]:
        """Return c, a, Q such that C=c+a@x+sum_{i<j} Q[i,j]*x[i]*x[j]."""
        self._validate_parameters(weight, penalty)
        linear = -(self.benefit + weight * self.equity) + penalty * (1 - 2 * self.k)
        quadratic = np.triu(self.overlap + 2 * penalty, k=1)
        return float(penalty * self.k**2), linear, quadratic

    def hamiltonian(self, weight: float, penalty: float) -> SparsePauliOp:
        """Use x_i=(I-Z_i)/2, with project i mapped to Qiskit qubit i."""
        constant, linear, quadratic = self.qubo(weight, penalty)
        offset = constant + linear.sum() / 2 + quadratic.sum() / 4
        z = -linear / 2 - (quadratic.sum(axis=0) + quadratic.sum(axis=1)) / 4
        terms = [("I" * self.n, complex(offset))]
        for i, coefficient in enumerate(z):
            label = ["I"] * self.n
            label[self.n - 1 - i] = "Z"
            terms.append(("".join(label), complex(coefficient)))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if quadratic[i, j]:
                    label = ["I"] * self.n
                    label[self.n - 1 - i] = label[self.n - 1 - j] = "Z"
                    terms.append(("".join(label), complex(quadratic[i, j] / 4)))
        return SparsePauliOp.from_list(terms).simplify()

    def enumerate(self, weight: float, penalty: float) -> list[dict]:
        rows = []
        for state in range(2**self.n):
            bits = int_to_bits(state, self.n)
            m = self.metrics(bits)
            rows.append({"state": state, "bitstring": format(state, f"0{self.n}b"), **m,
                         "utility": m["net_benefit"] + weight * m["equity_points"],
                         "cost": self.cost(bits, weight, penalty)})
        return rows


def int_to_bits(state: int, n: int) -> np.ndarray:
    """Qiskit integer convention: qubit 0 is the least significant bit."""
    if not isinstance(n, int) or n < 1 or not isinstance(state, (int, np.integer)) or not 0 <= state < 2**n:
        raise ValueError("State must be an integer in [0, 2**n), with positive integer n.")
    return np.array([(state >> i) & 1 for i in range(n)], dtype=int)


def load_problem(data_dir: Path = DATA_DIR, k: int = 3) -> AllocationProblem:
    with (data_dir / "projects.csv").open(encoding="utf-8-sig", newline="") as handle:
        projects = list(csv.DictReader(handle))
    if any(float(p["unit_cost"]) != 1 for p in projects):
        raise ValueError("This model assumes equal project costs of one budget unit.")
    ids = tuple(p["id"] for p in projects)
    index = {pid: i for i, pid in enumerate(ids)}
    overlap = np.zeros((len(projects), len(projects)))
    seen = set()
    with (data_dir / "overlaps.csv").open(encoding="utf-8-sig", newline="") as handle:
        for pair in csv.DictReader(handle):
            i, j = index[pair["first"]], index[pair["second"]]
            key = tuple(sorted((i, j)))
            if i == j or key in seen:
                raise ValueError("Duplicate or self-overlap pair.")
            seen.add(key)
            overlap[i, j] = overlap[j, i] = float(pair["penalty"])
    return AllocationProblem(
        ids=ids, names=tuple(p["name"] for p in projects),
        benefit=np.array([float(p["benefit"]) for p in projects]),
        equity=np.array([float(p["equity"]) for p in projects]),
        underserved=np.array([int(p["underserved"]) for p in projects]),
        overlap=overlap, k=k,
    )
