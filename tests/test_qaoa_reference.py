import networkx as nx
import numpy as np
from scipy.linalg import expm

from qaoa_structure.ansatze.qaoa import qaoa_state
from qaoa_structure.maxcut import cost_spectrum


X = np.array([[0, 1], [1, 0]], dtype=complex)
I = np.eye(2, dtype=complex)


def kron_for_qubit(op, q, n):
    # Big-endian Kronecker construction corresponding to little-endian q labels.
    mats = [op if k == q else I for k in reversed(range(n))]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def test_qaoa_matches_independent_dense_matrix_reference():
    g = nx.path_graph(3)
    gammas = np.array([0.37, 1.11])
    betas = np.array([0.29, 0.63])
    state = np.full(8, 1 / np.sqrt(8), dtype=complex)
    hc = np.diag(cost_spectrum(g))
    hm = sum(kron_for_qubit(X, q, 3) for q in range(3))
    for gamma, beta in zip(gammas, betas):
        state = expm(-1j * beta * hm) @ (expm(-1j * gamma * hc) @ state)
    actual = qaoa_state(g, gammas, betas)
    # Allow an irrelevant global phase.
    phase = np.vdot(state, actual)
    assert np.isclose(abs(phase), 1.0, atol=1e-11)
