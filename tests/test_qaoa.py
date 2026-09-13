import networkx as nx
import numpy as np

from qaoa_structure.ansatze.qaoa import qaoa_state
from qaoa_structure.maxcut import expectation_from_state


def test_qaoa_p0_is_uniform_cut_expectation():
    g = nx.erdos_renyi_graph(8, 0.5, seed=23)
    state = qaoa_state(g, np.array([]), np.array([]))
    assert np.isclose(expectation_from_state(state, g), g.number_of_edges()/2, atol=1e-12)


def test_qaoa_state_is_normalized():
    g = nx.cycle_graph(6)
    state = qaoa_state(g, np.array([0.4, 1.2]), np.array([0.7, 0.2]))
    assert np.isclose(np.vdot(state, state).real, 1.0, atol=1e-12)
