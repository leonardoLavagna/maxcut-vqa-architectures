from __future__ import annotations

import networkx as nx
import numpy as np


def ry_objective(graph: nx.Graph, theta: np.ndarray) -> float:
    """Closed-form MaxCut expectation for the product-Ry ansatz."""
    theta = np.asarray(theta, dtype=float)
    if len(theta) != graph.number_of_nodes():
        raise ValueError("theta must contain one angle per graph node.")
    z = np.cos(theta)
    return float(0.5 * sum(1.0 - z[u] * z[v] for u, v in graph.edges()))


def ry_gradient(graph: nx.Graph, theta: np.ndarray) -> np.ndarray:
    """Analytical gradient of the product-Ry MaxCut objective."""
    theta = np.asarray(theta, dtype=float)
    n = graph.number_of_nodes()
    if len(theta) != n:
        raise ValueError("theta must contain one angle per graph node.")
    c = np.cos(theta)
    s = np.sin(theta)
    grad = np.zeros(n, dtype=float)
    for k in range(n):
        grad[k] = 0.5 * s[k] * sum(c[j] for j in graph.neighbors(k))
    return grad


def ry_hessian(graph: nx.Graph, theta: np.ndarray) -> np.ndarray:
    """Analytical Hessian of the product-Ry MaxCut objective."""
    theta = np.asarray(theta, dtype=float)
    n = graph.number_of_nodes()
    if len(theta) != n:
        raise ValueError("theta must contain one angle per graph node.")
    c = np.cos(theta)
    s = np.sin(theta)
    h = np.zeros((n, n), dtype=float)
    for k in range(n):
        h[k, k] = 0.5 * c[k] * sum(c[j] for j in graph.neighbors(k))
    for u, v in graph.edges():
        value = -0.5 * s[u] * s[v]
        h[u, v] = value
        h[v, u] = value
    return h


def bitstring_to_angles(bits: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    """Map computational-basis bits 0/1 to Ry angles 0/pi."""
    bits = np.asarray(bits, dtype=int)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 and 1.")
    return np.pi * bits.astype(float)


def single_flip_gains(
    graph: nx.Graph,
    bits: np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    """Cut-value change produced by flipping each vertex once."""
    bits = np.asarray(bits, dtype=int)
    n = graph.number_of_nodes()
    if len(bits) != n:
        raise ValueError("bits must contain one value per graph node.")
    spins = 1 - 2 * bits  # bit 0 -> +1; bit 1 -> -1
    gains = np.zeros(n, dtype=float)
    for k in range(n):
        gains[k] = spins[k] * sum(spins[j] for j in graph.neighbors(k))
    return gains
