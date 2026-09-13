"""Run a small deterministic installation check.

This script is intentionally separate from the paper benchmark. It runs three
representative architectures on a six-node graph with a small evaluation budget
so that a new installation can be checked quickly.
"""

from __future__ import annotations

from dataclasses import asdict

import networkx as nx

from qaoa_structure.experiment import run_variational


def main() -> None:
    graph = nx.erdos_renyi_graph(6, 0.5, seed=42)
    models = [("ry", None), ("agnostic_xx", None), ("qaoa", 1)]

    for offset, (ansatz, depth) in enumerate(models):
        record, _ = run_variational(
            graph,
            graph_id="smoke_er_n6",
            graph_family="erdos_renyi",
            ansatz=ansatz,
            p=depth,
            seed=20260815 + offset,
            max_evaluations=100,
        )
        print(asdict(record))


if __name__ == "__main__":
    main()
