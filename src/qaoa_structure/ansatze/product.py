from __future__ import annotations

import networkx as nx
import numpy as np

from ..statevector import apply_single_qubit, rx, ry, rz, zero_state


def product_expectation(graph: nx.Graph, theta: np.ndarray) -> float:
    """Exact MaxCut expectation for independent Rx or Ry rotations.

    For either axis, <Z_i> = cos(theta_i), hence the objective is
    1/2 sum_(i,j in E) [1 - cos(theta_i) cos(theta_j)].
    """
    theta = np.asarray(theta, dtype=float)
    if len(theta) != graph.number_of_nodes():
        raise ValueError("theta must contain one parameter per graph node.")
    z = np.cos(theta)
    return float(0.5 * sum(1.0 - z[u] * z[v] for u, v in graph.edges()))


def product_probabilities(theta: np.ndarray) -> np.ndarray:
    """Computational-basis probabilities for independent Rx or Ry rotations."""
    theta = np.asarray(theta, dtype=float)
    probs = np.array([1.0], dtype=float)
    # little-endian: append higher-index qubits on the left in Kronecker order
    for t in theta:
        p = np.array([np.cos(t / 2) ** 2, np.sin(t / 2) ** 2], dtype=float)
        probs = np.kron(p, probs)
    return probs


def product_state(theta: np.ndarray, axis: str = "y", phi: np.ndarray | None = None) -> np.ndarray:
    """Construct Rx, Ry, or Ry-then-Rz product states for validation/metrics."""
    theta = np.asarray(theta, dtype=float)
    state = zero_state(len(theta))
    if axis not in {"x", "y", "yz"}:
        raise ValueError("axis must be 'x', 'y', or 'yz'.")
    if axis == "yz":
        if phi is None:
            raise ValueError("phi is required for the 'yz' ansatz.")
        phi = np.asarray(phi, dtype=float)
        if len(phi) != len(theta):
            raise ValueError("phi and theta must have the same length.")
    for i, t in enumerate(theta):
        state = apply_single_qubit(state, rx(t) if axis == "x" else ry(t), i)
        if axis == "yz":
            state = apply_single_qubit(state, rz(float(phi[i])), i)
    return state
