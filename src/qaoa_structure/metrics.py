from __future__ import annotations

import numpy as np

from .maxcut import MaxCutSolution, optimal_probability


def approximation_ratio(expectation: float, optimum: float) -> float:
    if optimum <= 0:
        return 1.0 if np.isclose(expectation, 0.0) else float("nan")
    return float(expectation / optimum)


def success_at_threshold(ratio: float, threshold: float = 0.99) -> bool:
    return bool(ratio >= threshold)


def p_opt_from_state(state: np.ndarray, solution: MaxCutSolution) -> float:
    return optimal_probability(state, solution)
