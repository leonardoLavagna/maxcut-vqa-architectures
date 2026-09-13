from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    x: np.ndarray
    value: float
    n_evaluations: int
    success: bool
    message: str
    history: list[float]


def maximize_cobyla(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    max_evaluations: int = 1000,
    rhobeg: float = 1.0,
    tol: float = 1e-4,
) -> OptimizationResult:
    """Maximize an objective with COBYLA while recording every evaluation."""
    history: list[float] = []

    def wrapped(x: np.ndarray) -> float:
        value = float(objective(np.asarray(x, dtype=float)))
        history.append(value)
        return -value

    result = minimize(
        wrapped,
        np.asarray(x0, dtype=float),
        method="COBYLA",
        options={"maxiter": int(max_evaluations), "rhobeg": float(rhobeg), "tol": float(tol)},
    )
    return OptimizationResult(
        x=np.asarray(result.x, dtype=float),
        value=float(-result.fun),
        n_evaluations=len(history),
        success=bool(result.success),
        message=str(result.message),
        history=history,
    )
