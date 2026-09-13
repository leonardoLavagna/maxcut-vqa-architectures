from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import networkx as nx
import numpy as np

from .ansatze.agnostic import agnostic_entangling_state
from .ansatze.product import product_expectation, product_state
from .ansatze.qaoa import qaoa_state
from .maxcut import MaxCutSolution, brute_force_maxcut, cost_spectrum
from .metrics import approximation_ratio, p_opt_from_state, success_at_threshold
from .optimization import maximize_cobyla


@dataclass
class RunRecord:
    graph_id: str
    graph_family: str
    n: int
    ansatz: str
    p: int | None
    restart: int
    seed: int
    n_parameters: int
    n_effective_parameters: int | None
    initial_objective: float
    final_objective: float
    approximation_ratio: float
    p_opt: float
    n_evaluations: int
    optimizer_success: bool
    target_success: bool
    runtime_seconds: float


def _initial_parameters(ansatz: str, n: int, p: int | None, rng: np.random.Generator) -> np.ndarray:
    if ansatz in {"rx", "ry"}:
        return rng.uniform(0, 2 * np.pi, size=n)
    if ansatz == "ryrz":
        return rng.uniform(0, 2 * np.pi, size=2 * n)
    if ansatz == "agnostic_xx":
        return rng.uniform(0, 2 * np.pi, size=3 * n - 1)
    if ansatz == "qaoa":
        if p is None:
            raise ValueError("p is required for QAOA.")
        gammas = rng.uniform(0, 2 * np.pi, size=p)
        betas = rng.uniform(0, np.pi, size=p)
        return np.concatenate([gammas, betas])
    raise ValueError(f"Unknown ansatz: {ansatz}")


def run_variational(
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    ansatz: str,
    seed: int,
    restart: int = 0,
    p: int | None = None,
    max_evaluations: int = 1000,
    rhobeg: float = 1.0,
    tol: float = 1e-4,
    success_threshold: float = 0.99,
    solution: MaxCutSolution | None = None,
) -> tuple[RunRecord, list[float]]:
    n = graph.number_of_nodes()
    solution = brute_force_maxcut(graph) if solution is None else solution
    costs = cost_spectrum(graph)
    rng = np.random.default_rng(seed)
    x0 = _initial_parameters(ansatz, n, p, rng)

    def state_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        if ansatz == "rx":
            state = product_state(x, "x")
        elif ansatz == "ry":
            state = product_state(x, "y")
        elif ansatz == "ryrz":
            state = product_state(x[:n], "yz", x[n:])
        elif ansatz == "agnostic_xx":
            state = agnostic_entangling_state(x[:n], x[n:2*n], x[2*n:])
        elif ansatz == "qaoa":
            assert p is not None
            state = qaoa_state(graph, x[:p], x[p:])
        else:
            raise ValueError(ansatz)
        probabilities = np.abs(state) ** 2
        return float(np.dot(probabilities, costs)), state

    if ansatz in {"rx", "ry", "ryrz"}:
        # The exact closed form is shared by Rx, Ry, and Ry-Rz; for Ry-Rz the
        # final n variables are deliberately objective-redundant flat directions.
        def objective(x: np.ndarray) -> float:
            return product_expectation(graph, x[:n])
    else:
        def objective(x: np.ndarray) -> float:
            return state_objective(x)[0]

    initial_value = objective(x0)
    start = perf_counter()
    opt = maximize_cobyla(
        objective,
        x0,
        max_evaluations=max_evaluations,
        rhobeg=rhobeg,
        tol=tol,
    )
    runtime = perf_counter() - start
    final_value, final_state = state_objective(opt.x)
    ratio = approximation_ratio(final_value, solution.optimum)
    p_opt = p_opt_from_state(final_state, solution)

    if ansatz in {"rx", "ry", "ryrz"}:
        n_effective = n
    elif ansatz == "agnostic_xx":
        n_effective = None
    else:
        n_effective = 2 * int(p or 0)

    record = RunRecord(
        graph_id=graph_id,
        graph_family=graph_family,
        n=n,
        ansatz=ansatz,
        p=p,
        restart=restart,
        seed=seed,
        n_parameters=len(x0),
        n_effective_parameters=n_effective,
        initial_objective=float(initial_value),
        final_objective=float(final_value),
        approximation_ratio=float(ratio),
        p_opt=float(p_opt),
        n_evaluations=opt.n_evaluations,
        optimizer_success=opt.success,
        target_success=success_at_threshold(ratio, success_threshold),
        runtime_seconds=float(runtime),
    )
    return record, opt.history
