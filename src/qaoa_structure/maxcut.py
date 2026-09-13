from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class MaxCutSolution:
    optimum: float
    optimal_bitstrings: tuple[int, ...]


def cut_value(graph: nx.Graph, bits: int | list[int] | tuple[int, ...] | np.ndarray) -> float:
    """Return the unweighted MaxCut value of a bit assignment.

    Integer assignments use little-endian convention: node i is bit i.
    """
    if isinstance(bits, (int, np.integer)):
        assignment = int(bits)
        return float(
            sum(
                ((assignment >> u) & 1) != ((assignment >> v) & 1)
                for u, v in graph.edges()
            )
        )
    arr = np.asarray(bits, dtype=int)
    return float(sum(arr[u] != arr[v] for u, v in graph.edges()))


def cost_spectrum(graph: nx.Graph) -> np.ndarray:
    """MaxCut cost for all computational-basis states in little-endian order."""
    n = graph.number_of_nodes()
    indices = np.arange(1 << n, dtype=np.uint64)
    costs = np.zeros(1 << n, dtype=float)
    for u, v in graph.edges():
        bu = (indices >> np.uint64(u)) & np.uint64(1)
        bv = (indices >> np.uint64(v)) & np.uint64(1)
        costs += (bu != bv)
    return costs


def brute_force_maxcut(graph: nx.Graph) -> MaxCutSolution:
    """Compute the exact MaxCut value and all optimal bitstrings."""
    costs = cost_spectrum(graph)
    optimum = float(costs.max(initial=0.0))
    idx = tuple(int(i) for i in np.flatnonzero(np.isclose(costs, optimum)))
    return MaxCutSolution(optimum=optimum, optimal_bitstrings=idx)


def expectation_from_state(state: np.ndarray, graph: nx.Graph) -> float:
    probabilities = np.abs(np.asarray(state, dtype=complex)) ** 2
    return float(np.dot(probabilities, cost_spectrum(graph)))


def optimal_probability(state: np.ndarray, solution: MaxCutSolution) -> float:
    probabilities = np.abs(np.asarray(state, dtype=complex)) ** 2
    if not solution.optimal_bitstrings:
        return 0.0
    return float(probabilities[list(solution.optimal_bitstrings)].sum())
