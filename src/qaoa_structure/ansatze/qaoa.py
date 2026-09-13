from __future__ import annotations

import networkx as nx
import numpy as np

from ..maxcut import cost_spectrum
from ..statevector import apply_single_qubit, plus_state, rx


def qaoa_state(graph: nx.Graph, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Exact statevector for standard unweighted MaxCut QAOA with X mixer.

    We use H_C = sum_(i,j in E) (I-Z_i Z_j)/2 and
    H_M = sum_i X_i. Therefore exp(-i beta H_M) applies Rx(2 beta)
    independently to every qubit.
    """
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have equal length.")
    n = graph.number_of_nodes()
    costs = cost_spectrum(graph)
    state = plus_state(n)
    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * costs)
        gate = rx(2.0 * beta)
        for q in range(n):
            state = apply_single_qubit(state, gate, q)
    return state
