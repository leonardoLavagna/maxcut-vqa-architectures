import numpy as np

from qaoa_structure.optimization import maximize_cobyla


def test_cobyla_respects_objective_evaluation_cap():
    result = maximize_cobyla(
        lambda x: -float(np.sum((x - 1.0) ** 2)),
        np.zeros(4),
        max_evaluations=25,
    )
    assert result.n_evaluations <= 25
    assert result.n_evaluations == len(result.history)
