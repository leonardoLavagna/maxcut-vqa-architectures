"""Benchmark orchestration and deterministic seed management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .experiment import run_variational
from .graphs import GraphInstance, generate_benchmark_graphs
from .local_search import single_flip_local_search
from .maxcut import brute_force_maxcut
from .metrics import approximation_ratio
from .results import append_jsonl, write_json


_FAMILY_CODE = {"complete": 1, "erdos_renyi": 2, "regular": 3}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run_seed(master_seed: int, instance: GraphInstance, restart: int) -> int:
    """Return the deterministic restart seed shared by all architectures.

    Sharing a seed for a given graph/restart pair aligns the initial local-angle
    draws across the product-state and graph-agnostic controls. The seed is
    derived with ``SeedSequence`` rather than Python's process-dependent hash.
    """
    seed_sequence = np.random.SeedSequence(
        [
            int(master_seed),
            100 + _FAMILY_CODE[instance.family],
            int(instance.n),
            int(instance.instance_index),
            int(restart),
        ]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _models(config: dict[str, Any], n: int) -> list[tuple[str, int | None]]:
    """Build the architecture schedule for graphs of size ``n``.

    Fixed QAOA depths and the parameter-count-matched depth ``p=n/2`` may
    overlap. Duplicate depths are removed while preserving the configured order.
    """
    models: list[tuple[str, int | None]] = []

    for name in config["ansatze"].get("principal", []):
        models.append((str(name), None))
    for name in config["ansatze"].get("controls", []):
        models.append((str(name), None))

    qaoa_depths = [int(p) for p in config["ansatze"].get("qaoa_depths", [])]
    if config["ansatze"].get("qaoa_parameter_matched", False):
        if n % 2:
            raise ValueError("Parameter-matched QAOA with 2p=n requires an even node count.")
        qaoa_depths.append(n // 2)

    seen_depths: set[int] = set()
    for depth in qaoa_depths:
        if depth not in seen_depths:
            models.append(("qaoa", depth))
            seen_depths.add(depth)

    return models


def build_graphs(config: dict[str, Any]) -> list[GraphInstance]:
    """Generate the graph suite described by a loaded configuration."""
    graph_config = config["graphs"]
    return generate_benchmark_graphs(
        node_sizes=config["node_sizes"],
        er_instances=int(graph_config["erdos_renyi"]["instances_per_n"]),
        regular_instances=int(graph_config["random_regular"]["instances_per_n"]),
        er_probability=float(graph_config["erdos_renyi"]["probability"]),
        regular_degree=int(graph_config["random_regular"]["degree"]),
        master_seed=int(config["master_seed"]),
    )


def _load_completed_keys(path: Path, kind: str) -> set[tuple[Any, ...]]:
    """Read keys from an existing JSONL output when resuming a benchmark."""
    if not path.exists():
        return set()

    keys: set[tuple[Any, ...]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if kind == "variational":
                keys.add(
                    (
                        row["graph_id"],
                        row["ansatz"],
                        row.get("p"),
                        int(row["restart"]),
                        int(row["seed"]),
                    )
                )
            elif kind == "local":
                keys.add((row["graph_id"], int(row["restart"]), int(row["seed"])))
            else:
                raise ValueError(f"Unknown result kind: {kind}")
    return keys


def _clear_outputs(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def run_benchmark(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> None:
    """Run the configured benchmark and write append-only JSONL result files.

    Parameters
    ----------
    config_path:
        YAML file defining graphs, architectures, optimizer settings and seeds.
    output_dir:
        Directory receiving raw run records, optimization histories and the
        classical local-search baseline.
    resume:
        If ``True``, completed records are read from the existing output files
        and skipped. If ``False``, the three raw output files are recreated.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    write_json(output_dir / "resolved_config.json", config)

    runs_path = output_dir / "runs.jsonl"
    histories_path = output_dir / "histories.jsonl"
    local_search_path = output_dir / "local_search.jsonl"
    raw_paths = (runs_path, histories_path, local_search_path)

    if not resume:
        _clear_outputs(raw_paths)

    run_keys = _load_completed_keys(runs_path, "variational") if resume else set()
    history_keys = (
        _load_completed_keys(histories_path, "variational") if resume else set()
    )
    local_keys = _load_completed_keys(local_search_path, "local") if resume else set()

    optimizer = config["optimizer"]
    success_threshold = float(config.get("success_threshold", 0.99))
    restarts = int(config["restarts"])
    master_seed = int(config["master_seed"])
    use_local_search = bool(
        config.get("classical_baselines", {}).get("single_flip_local_search", False)
    )

    for instance in build_graphs(config):
        solution = brute_force_maxcut(instance.graph)

        for restart in range(restarts):
            seed = _run_seed(master_seed, instance, restart)

            local_key = (instance.graph_id, restart, seed)
            if use_local_search and local_key not in local_keys:
                local_result = single_flip_local_search(instance.graph, seed=seed)
                append_jsonl(
                    local_search_path,
                    {
                        "graph_id": instance.graph_id,
                        "graph_family": instance.family,
                        "n": instance.n,
                        "restart": restart,
                        "seed": seed,
                        "value": local_result.value,
                        "approximation_ratio": approximation_ratio(
                            local_result.value, solution.optimum
                        ),
                        "flips": local_result.flips,
                        "n_evaluations": local_result.n_evaluations,
                    },
                )
                local_keys.add(local_key)

            for ansatz, depth in _models(config, instance.n):
                run_key = (instance.graph_id, ansatz, depth, restart, seed)
                if run_key in run_keys and run_key in history_keys:
                    continue

                record, history = run_variational(
                    instance.graph,
                    graph_id=instance.graph_id,
                    graph_family=instance.family,
                    ansatz=ansatz,
                    p=depth,
                    seed=seed,
                    restart=restart,
                    max_evaluations=int(optimizer["max_evaluations"]),
                    rhobeg=float(optimizer["rhobeg"]),
                    tol=float(optimizer["tol"]),
                    success_threshold=success_threshold,
                    solution=solution,
                )

                if run_key not in run_keys:
                    append_jsonl(runs_path, record)
                    run_keys.add(run_key)

                if run_key not in history_keys:
                    append_jsonl(
                        histories_path,
                        {
                            "graph_id": instance.graph_id,
                            "ansatz": ansatz,
                            "p": depth,
                            "restart": restart,
                            "seed": seed,
                            "optimum": solution.optimum,
                            "objective_history": history,
                        },
                    )
                    history_keys.add(run_key)
