from pathlib import Path

from qaoa_structure.graphs import generate_benchmark_graphs
from qaoa_structure.runner import _run_seed, build_graphs, load_config


def test_full_graph_count_and_ids_are_unique():
    graphs = generate_benchmark_graphs()
    assert len(graphs) == 95
    assert len({g.graph_id for g in graphs}) == 95


def test_graph_generation_is_reproducible():
    a = generate_benchmark_graphs(node_sizes=(6, 8), er_instances=2, regular_instances=2)
    b = generate_benchmark_graphs(node_sizes=(6, 8), er_instances=2, regular_instances=2)
    for ga, gb in zip(a, b):
        assert ga.graph_id == gb.graph_id
        assert sorted(ga.graph.edges()) == sorted(gb.graph.edges())


def test_restart_seed_is_architecture_independent_and_repeatable():
    graph = generate_benchmark_graphs(node_sizes=(6,), er_instances=1, regular_instances=0)[1]
    s1 = _run_seed(20260815, graph, 3)
    s2 = _run_seed(20260815, graph, 3)
    assert s1 == s2
