from __future__ import annotations

import numpy as np


def zero_state(n: int) -> np.ndarray:
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0
    return state


def plus_state(n: int) -> np.ndarray:
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex
    )


def rxx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    xx = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        dtype=complex,
    )
    return c * np.eye(4, dtype=complex) - 1j * s * xx


def apply_single_qubit(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a one-qubit gate using vectorized little-endian index blocks."""
    out = np.asarray(state, dtype=complex).copy()
    stride = 1 << qubit
    period = stride << 1
    blocks = out.reshape(-1, period)
    a0 = blocks[:, :stride].copy()
    a1 = blocks[:, stride:].copy()
    blocks[:, :stride] = gate[0, 0] * a0 + gate[0, 1] * a1
    blocks[:, stride:] = gate[1, 0] * a0 + gate[1, 1] * a1
    return out


def apply_two_qubit(state: np.ndarray, gate: np.ndarray, q0: int, q1: int) -> np.ndarray:
    """Apply a two-qubit gate using vectorized basis-index gathering."""
    if q0 == q1:
        raise ValueError("Two-qubit gate requires distinct qubits.")
    q_lo, q_hi = sorted((q0, q1))
    out = np.asarray(state, dtype=complex).copy()
    mask0, mask1 = 1 << q_lo, 1 << q_hi
    indices = np.arange(len(out), dtype=np.int64)
    base = indices[((indices & mask0) == 0) & ((indices & mask1) == 0)]
    idx0 = base
    idx1 = base | mask0
    idx2 = base | mask1
    idx3 = base | mask0 | mask1
    # Gate basis uses |q_hi q_lo> = |00>,|01>,|10>,|11>.
    vec = np.vstack((out[idx0], out[idx1], out[idx2], out[idx3]))
    transformed = gate @ vec
    out[idx0], out[idx1], out[idx2], out[idx3] = transformed
    return out
