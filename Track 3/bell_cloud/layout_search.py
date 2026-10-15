"""Rank three-qubit layouts from local CZ/readout calibration, without cloud calls.

The score is an additive calibration heuristic, not predicted Bell fidelity.
For an error estimate e, its contribution is -log(1-e). Baseline uses one CZ
and the two data readouts. Parity uses the three logical CNOT interactions,
the cheapest SWAP route confined to these three physical qubits, and all
three readouts. A SWAP costs three CZs in this deliberately simple model.
Bridge requires the ordered path DATA0--DATA1--ANCILLA and uses three CZs
on the data edge plus one CZ on the ancilla edge, without SWAPs.
Single-qubit gates, idle errors, correlations, mitigation, and calibration
drift are omitted. Hardware compilation remains the authoritative resource
count. Selection uses calibration only, never the observed Bell experiment.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "results/ibm-20261015/backend-snapshot.json"
OBJECTIVES = ("baseline", "parity", "balanced", "bridge")


def _valid_error(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value < 1)


def _cost(error: float) -> float:
    return -math.log1p(-error)


def _calibration(snapshot: Mapping) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    properties = snapshot.get("properties") or {}
    readout = {}
    for index, entries in enumerate(properties.get("qubits", [])):
        fields = {entry["name"]: entry.get("value") for entry in entries}
        value = fields.get("readout_error")
        if fields.get("operational", True) and _valid_error(value):
            readout[index] = float(value)
    available = {tuple(sorted(edge)) for edge in snapshot.get("coupling_edges", [])
                 if len(edge) == 2 and edge[0] != edge[1]}
    gates = {}
    for gate in properties.get("gates", []):
        qubits = gate.get("qubits", [])
        if gate.get("gate") != "cz" or len(qubits) != 2 or qubits[0] == qubits[1]:
            continue
        key = tuple(sorted(qubits))
        parameters = {entry["name"]: entry.get("value") for entry in gate.get("parameters", [])}
        value = parameters.get("gate_error")
        if key not in available or not all(q in readout for q in key) or not _valid_error(value):
            continue
        # CZ is symmetric. If both orientations have entries, conservatively
        # use their larger reported error rather than silently selecting one.
        gates[key] = max(gates.get(key, 0.0), float(value))
    return readout, gates


def _parity_route(layout: tuple[int, int, int], errors: Mapping[tuple[int, int], float]) -> dict:
    """Dijkstra search over role permutations and the three required interactions."""
    edges = {edge: error for edge, error in errors.items() if set(edge) <= set(layout)}
    sequence = ((0, 1), (0, 2), (1, 2))
    initial = (0, layout)
    distance = {initial: (0.0, 0)}
    queue = [(0.0, 0, 0, layout, ())]
    while queue:
        score, gate_count, stage, placement, actions = heapq.heappop(queue)
        state = (stage, placement)
        if (score, gate_count) != distance[state]:
            continue
        if stage == len(sequence):
            counts = {}
            for kind, edge in actions:
                key = f"{edge[0]}-{edge[1]}"
                counts[key] = counts.get(key, 0) + (3 if kind == "swap" else 1)
            return {
                "cz_log_error_score": score,
                "estimated_cz_count": gate_count,
                "cz_counts_by_edge": counts,
                "estimated_swap_count": sum(kind == "swap" for kind, _ in actions),
                "final_logical_to_physical": list(placement),
            }

        transitions = []
        first, second = sequence[stage]
        edge = tuple(sorted((placement[first], placement[second])))
        if edge in edges:
            transitions.append((stage + 1, placement, edge, 1, "gate"))
        for swap_edge in sorted(edges):
            a, b = swap_edge
            swapped = tuple(b if q == a else a if q == b else q for q in placement)
            transitions.append((stage, swapped, swap_edge, 3, "swap"))
        for next_stage, next_placement, edge, multiplicity, kind in transitions:
            next_cost = (score + multiplicity * _cost(edges[edge]), gate_count + multiplicity)
            next_state = (next_stage, next_placement)
            if next_cost < distance.get(next_state, (math.inf, math.inf)):
                distance[next_state] = next_cost
                heapq.heappush(queue, (*next_cost, next_stage, next_placement, actions + ((kind, edge),)))
    raise ValueError("The selected three physical qubits are not connected.")


def rank_layouts(snapshot_or_path=DEFAULT_SNAPSHOT, *, limit: int = 5,
                 objective: str = "balanced") -> list[dict]:
    """Return ranked candidates with layout=[DATA0, DATA1, ANCILLA].

    ``objective`` is baseline, parity, balanced (the arithmetic mean of
    baseline/parity log-error scores), or bridge. Bridge requires edges
    DATA0--DATA1 and DATA1--ANCILLA, with DATA1 physically in the middle;
    its score is 3*CZ(DATA0,DATA1)+CZ(DATA1,ANCILLA)+three readout costs.
    One best role assignment per physical triple
    is retained, so reversed paths and duplicate directed CZ entries do not
    consume the shortlist. Ties are resolved deterministically. Returned
    layouts have an adjacent data pair and an ancilla adjacent to that pair.
    Unknown, invalid, or missing readout/CZ errors are excluded, never zeroed.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    snapshot = (snapshot_or_path if isinstance(snapshot_or_path, Mapping)
                else json.loads(Path(snapshot_or_path).read_text(encoding="utf-8")))
    readout, errors = _calibration(snapshot)
    neighbors = {}
    for first, second in errors:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)

    best_by_triple = {}
    for pair in sorted(errors):
        for first, second in (pair, pair[::-1]):
            ancillas = (neighbors[first] | neighbors[second]) - {first, second}
            for ancilla in sorted(ancillas):
                ancilla_edge = tuple(sorted((second, ancilla)))
                if objective == "bridge" and ancilla_edge not in errors:
                    continue
                layout = (first, second, ancilla)
                baseline_score = _cost(errors[pair]) + _cost(readout[first]) + _cost(readout[second])
                route = _parity_route(layout, errors)
                parity_score = route["cz_log_error_score"] + sum(_cost(readout[q]) for q in layout)
                balanced_score = (baseline_score + parity_score) / 2
                bridge_score = (3 * _cost(errors[pair]) + _cost(errors[ancilla_edge])
                                + sum(_cost(readout[q]) for q in layout)
                                if ancilla_edge in errors else None)
                bridge_estimate = ({"estimated_cz_count": 4, "estimated_swap_count": 0,
                                    "cz_counts_by_edge": {f"{pair[0]}-{pair[1]}": 3,
                                                          f"{ancilla_edge[0]}-{ancilla_edge[1]}": 1},
                                    "final_logical_to_physical": list(layout)}
                                   if ancilla_edge in errors else None)
                scores = {"baseline": baseline_score, "parity": parity_score,
                          "balanced": balanced_score, "bridge": bridge_score}
                candidate = {
                    "layout": list(layout),
                    "objective": objective,
                    "score": scores[objective],
                    "baseline_score": baseline_score,
                    "parity_score": parity_score,
                    "balanced_score": balanced_score,
                    "bridge_score": bridge_score,
                    "readout_errors": {str(q): readout[q] for q in layout},
                    "cz_errors": {f"{a}-{b}": error for (a, b), error in sorted(errors.items())
                                  if a in layout and b in layout},
                    "baseline_cz_count": 1,
                    "parity_route_estimate": route,
                    "bridge_route_estimate": bridge_estimate,
                    "backend": snapshot.get("backend"),
                    "calibration_captured_utc": snapshot.get("captured_utc"),
                    "calibration_last_update": (snapshot.get("properties") or {}).get("last_update_date"),
                }
                key = tuple(sorted(layout))
                tie_key = lambda row: (row["score"], row["balanced_score"], tuple(row["layout"]))
                if key not in best_by_triple or tie_key(candidate) < tie_key(best_by_triple[key]):
                    best_by_triple[key] = candidate
    result = sorted(best_by_triple.values(),
                    key=lambda row: (row["score"], row["balanced_score"], row["layout"]))[:limit]
    for rank, candidate in enumerate(result, 1):
        candidate["rank"] = rank
    return result


def rank_target_layouts(target, *, limit: int = 5, objective: str = "balanced") -> list[dict]:
    """Rank an already available Qiskit Target; no backend refresh or cloud calls.

    Requires native CZ and classified ``measure`` instruction errors. This
    deliberately avoids pretending that directional ECR/CX calibration is CZ.
    """
    if "cz" not in target.operation_names or "measure" not in target.operation_names:
        raise ValueError("This scoring model requires native CZ and measure error properties.")
    qubits = []
    for q in range(target.num_qubits):
        instruction = target["measure"].get((q,))
        error = instruction.error if instruction is not None else None
        qubits.append([{"name": "readout_error", "value": error}])
    gates, coupling = [], []
    for pair, instruction in target["cz"].items():
        if pair is None or len(pair) != 2:
            continue
        coupling.append(list(pair))
        gates.append({"gate": "cz", "qubits": list(pair), "parameters": [
            {"name": "gate_error", "value": instruction.error if instruction is not None else None}
        ]})
    return rank_layouts({"coupling_edges": coupling, "properties": {"qubits": qubits, "gates": gates}},
                        limit=limit, objective=objective)


def _self_test() -> bool:
    """Small checks of connectivity, duplicate removal, and routing cost."""
    import unittest

    def example():
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        return {
            "coupling_edges": [list(edge) for edge in edges for edge in (edge, edge[::-1])],
            "properties": {
                "qubits": [[{"name": "readout_error", "value": error}]
                           for error in (.03, .02, .004, .005, .006)],
                "gates": [{"gate": "cz", "qubits": list(edge), "parameters": [
                    {"name": "gate_error", "value": error}]} for edge, error in
                    zip(edges, (.02, .01, .002, .003))],
            },
        }

    class LayoutChecks(unittest.TestCase):
        def test_connected_unique_triples(self):
            snapshot = example()
            rows = rank_layouts(snapshot, limit=100)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({tuple(sorted(row["layout"])) for row in rows}), len(rows))
            edges = {tuple(sorted(edge)) for edge in snapshot["coupling_edges"]}
            for row in rows:
                a, b, c = row["layout"]
                self.assertIn(tuple(sorted((a, b))), edges)
                self.assertTrue(tuple(sorted((a, c))) in edges or tuple(sorted((b, c))) in edges)

        def test_good_data_pair_ranks_first(self):
            top = rank_layouts(example(), objective="baseline")[0]
            self.assertEqual(set(top["layout"][:2]), {2, 3})

        def test_missing_calibration_is_not_free(self):
            snapshot = example()
            snapshot["properties"]["qubits"][2] = []
            self.assertEqual(rank_layouts(snapshot), [])

        def test_path_requires_swap_triangle_does_not(self):
            chain = _parity_route((0, 1, 2), {(0, 1): .01, (1, 2): .01})
            triangle = _parity_route((0, 1, 2), {(0, 1): .01, (1, 2): .01, (0, 2): .01})
            self.assertEqual(chain["estimated_cz_count"], 6)
            self.assertEqual(chain["estimated_swap_count"], 1)
            self.assertEqual(triangle["estimated_cz_count"], 3)
            self.assertEqual(triangle["estimated_swap_count"], 0)

        def test_bridge_enforces_middle_qubit_and_exact_edge_multiplicities(self):
            snapshot = example()
            readout, errors = _calibration(snapshot)
            rows = rank_layouts(snapshot, limit=100, objective="bridge")
            self.assertEqual(len(rows), 3)
            for row in rows:
                first, middle, ancilla = row["layout"]
                edge01 = tuple(sorted((first, middle)))
                edge12 = tuple(sorted((middle, ancilla)))
                self.assertIn(edge01, errors)
                self.assertIn(edge12, errors)
                expected = 3 * _cost(errors[edge01]) + _cost(errors[edge12])
                expected += sum(_cost(readout[q]) for q in row["layout"])
                self.assertAlmostEqual(row["score"], expected)
                self.assertEqual(row["bridge_route_estimate"]["estimated_cz_count"], 4)
                self.assertEqual(row["bridge_route_estimate"]["estimated_swap_count"], 0)

    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LayoutChecks)).wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--objective", choices=OBJECTIVES, default="balanced")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(0 if _self_test() else 1)
    print(json.dumps(rank_layouts(args.snapshot, limit=args.limit, objective=args.objective), indent=2,
                     allow_nan=False))
