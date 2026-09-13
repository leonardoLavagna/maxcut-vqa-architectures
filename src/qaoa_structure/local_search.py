from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .maxcut import cut_value


@dataclass
class LocalSearchResult:
    bits: np.ndarray
    value: float
    flips: int
    n_evaluations: int


def single_flip_local_search(graph: nx.Graph, seed: int | None = None) -> LocalSearchResult:
    """Best-improvement one-vertex-flip local search from a random cut.

    ``n_evaluations`` counts explicit cut-value evaluations, including the
    initial assignment and every candidate flip inspected.
    """
    rng = np.random.default_rng(seed)
    n = graph.number_of_nodes()
    bits = rng.integers(0, 2, size=n, dtype=np.int8)
    value = cut_value(graph, bits)
    n_evaluations = 1
    flips = 0
    while True:
        best_gain = 0.0
        best_k = None
        for k in range(n):
            trial = bits.copy()
            trial[k] ^= 1
            trial_value = cut_value(graph, trial)
            n_evaluations += 1
            gain = trial_value - value
            if gain > best_gain + 1e-12:
                best_gain, best_k = gain, k
        if best_k is None:
            break
        bits[best_k] ^= 1
        value += best_gain
        flips += 1
    return LocalSearchResult(
        bits=bits,
        value=float(value),
        flips=flips,
        n_evaluations=n_evaluations,
    )
