import networkx as nx
import numpy as np

from qaoa_structure.ansatze.product import product_state
from qaoa_structure.maxcut import expectation_from_state


def test_rx_ry_pointwise_objective_equivalence():
    rng = np.random.default_rng(12)
    g = nx.erdos_renyi_graph(8, 0.5, seed=3)
    for _ in range(10):
        theta = rng.uniform(0, 2*np.pi, size=8)
        fx = expectation_from_state(product_state(theta, "x"), g)
        fy = expectation_from_state(product_state(theta, "y"), g)
        assert np.isclose(fx, fy, atol=1e-12)
