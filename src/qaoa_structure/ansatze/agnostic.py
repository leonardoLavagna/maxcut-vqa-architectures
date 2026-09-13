from __future__ import annotations

import numpy as np

from ..statevector import apply_single_qubit, apply_two_qubit, rxx, ry, rz, zero_state


def agnostic_entangling_state(
    theta: np.ndarray,
    phi: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """One-layer problem-agnostic entangling ansatz.

    Local Ry rotations are followed by Rz rotations, then a fixed nearest-neighbour
    chain of RXX gates. The entangling connectivity is independent of the MaxCut
    graph, giving 2n + (n-1) = 3n-1 trainable parameters.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    n = len(theta)
    if len(phi) != n or len(alpha) != max(0, n - 1):
        raise ValueError("Expected len(phi)=n and len(alpha)=n-1.")
    state = zero_state(n)
    for i in range(n):
        state = apply_single_qubit(state, ry(theta[i]), i)
        state = apply_single_qubit(state, rz(phi[i]), i)
    for i in range(n - 1):
        state = apply_two_qubit(state, rxx(alpha[i]), i, i + 1)
    return state
