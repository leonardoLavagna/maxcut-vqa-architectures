import networkx as nx
import numpy as np

from qaoa_structure.ansatze.product import product_expectation, product_state
from qaoa_structure.maxcut import expectation_from_state


def test_product_closed_form_matches_statevector():
    rng = np.random.default_rng(7)
    g = nx.erdos_renyi_graph(7, 0.5, seed=11)
    theta = rng.uniform(0, 2*np.pi, size=7)
    exact = product_expectation(g, theta)
    state = product_state(theta, axis="y")
    assert np.isclose(exact, expectation_from_state(state, g), atol=1e-12)
