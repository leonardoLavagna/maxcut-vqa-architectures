import networkx as nx
import numpy as np

from qaoa_structure.maxcut import brute_force_maxcut, cut_value, cost_spectrum


def test_complete_graph_known_optima():
    for n in range(2, 9):
        sol = brute_force_maxcut(nx.complete_graph(n))
        assert sol.optimum == (n * n) // 4


def test_bipartite_graph_cuts_every_edge():
    g = nx.complete_bipartite_graph(3, 4)
    sol = brute_force_maxcut(g)
    assert sol.optimum == g.number_of_edges()


def test_cost_spectrum_matches_direct_cut_value():
    g = nx.cycle_graph(5)
    costs = cost_spectrum(g)
    for z in range(1 << 5):
        assert costs[z] == cut_value(g, z)
