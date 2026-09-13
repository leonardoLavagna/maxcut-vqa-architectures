import networkx as nx
import numpy as np

from qaoa_structure.landscape import (
    bitstring_to_angles,
    ry_gradient,
    ry_hessian,
    single_flip_gains,
)


def test_basis_curvature_matches_half_single_flip_gain():
    g = nx.erdos_renyi_graph(7, 0.5, seed=31)
    bits = np.array([0, 1, 0, 0, 1, 1, 0], dtype=int)
    theta = bitstring_to_angles(bits)
    h = ry_hessian(g, theta)
    gains = single_flip_gains(g, bits)
    assert np.allclose(np.diag(h), 0.5 * gains, atol=1e-12)
    assert np.allclose(h - np.diag(np.diag(h)), 0.0, atol=1e-12)


def test_uniform_point_has_zero_gradient_and_hessian_minus_half_adjacency():
    g = nx.cycle_graph(6)
    theta = np.full(6, np.pi / 2)
    assert np.allclose(ry_gradient(g, theta), 0.0, atol=1e-12)
    adjacency = nx.to_numpy_array(g, nodelist=range(6))
    h = ry_hessian(g, theta)
    assert np.allclose(h, -0.5 * adjacency, atol=1e-12)
    eigvals = np.linalg.eigvalsh(h)
    assert eigvals.min() < -1e-12
    assert eigvals.max() > 1e-12


def test_strict_one_flip_maximum_has_negative_definite_basis_hessian():
    g = nx.complete_graph(4)
    # Balanced K4 cut: every one-vertex flip reduces the cut from 4 to 3.
    bits = np.array([0, 0, 1, 1], dtype=int)
    gains = single_flip_gains(g, bits)
    h = ry_hessian(g, bitstring_to_angles(bits))
    assert np.all(gains < 0)
    assert np.all(np.diag(h) < 0)
    assert np.all(np.linalg.eigvalsh(h) < 0)


def test_non_local_maximum_has_positive_basis_curvature_direction():
    g = nx.complete_graph(4)
    bits = np.array([0, 0, 0, 1], dtype=int)
    gains = single_flip_gains(g, bits)
    h = ry_hessian(g, bitstring_to_angles(bits))
    assert np.any(gains > 0)
    assert np.any(np.diag(h) > 0)
