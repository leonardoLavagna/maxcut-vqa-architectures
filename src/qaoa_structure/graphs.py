from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GraphInstance:
    """A graph plus immutable metadata used to reproduce an experiment."""

    graph_id: str
    family: str
    n: int
    instance_index: int
    seed: int | None
    graph: nx.Graph


def _derived_seed(master_seed: int, family_code: int, n: int, index: int) -> int:
    """Create a deterministic 32-bit seed without depending on Python hash()."""
    ss = np.random.SeedSequence([master_seed, family_code, n, index])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def generate_benchmark_graphs(
    node_sizes: Iterable[int] = (4, 6, 8, 10, 12),
    er_instances: int = 10,
    regular_instances: int = 10,
    er_probability: float = 0.5,
    regular_degree: int = 3,
    master_seed: int = 20260815,
) -> list[GraphInstance]:
    """Generate the frozen unweighted benchmark used by the study.

    The benchmark contains one complete graph for each node size, ``er_instances``
    Erdős--Rényi graphs for each size, and ``regular_instances`` random regular
    graphs for sizes where they are both feasible and not intentionally omitted.
    We omit n=4 for degree-3 regular graphs because it is necessarily K4 and would
    duplicate the complete-graph control.
    """
    out: list[GraphInstance] = []
    for n in node_sizes:
        g = nx.complete_graph(n)
        out.append(GraphInstance(f"complete_n{n}", "complete", n, 0, None, g))

        for idx in range(er_instances):
            seed = _derived_seed(master_seed, 1, n, idx)
            g = nx.erdos_renyi_graph(n, er_probability, seed=seed)
            graph_id = f"er_n{n}_i{idx:02d}"
            out.append(GraphInstance(graph_id, "erdos_renyi", n, idx, seed, g))

        feasible = n > regular_degree and (n * regular_degree) % 2 == 0
        if n != 4 and feasible:
            for idx in range(regular_instances):
                seed = _derived_seed(master_seed, 2, n, idx)
                g = nx.random_regular_graph(regular_degree, n, seed=seed)
                graph_id = f"reg{regular_degree}_n{n}_i{idx:02d}"
                out.append(GraphInstance(graph_id, "regular", n, idx, seed, g))
    return out
