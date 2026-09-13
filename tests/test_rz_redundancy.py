import networkx as nx
import numpy as np

from qaoa_structure.ansatze.product import product_state
from qaoa_structure.maxcut import expectation_from_state


def test_rz_parameters_are_objective_redundant_without_later_noncommuting_gates():
    rng = np.random.default_rng(13)
    g = nx.random_regular_graph(3, 8, seed=5)
    for _ in range(10):
        theta = rng.uniform(0, 2*np.pi, size=8)
        phi = rng.uniform(0, 2*np.pi, size=8)
        fy = expectation_from_state(product_state(theta, "y"), g)
        fyz = expectation_from_state(product_state(theta, "yz", phi), g)
        assert np.isclose(fy, fyz, atol=1e-12)
