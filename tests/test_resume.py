from pathlib import Path

import yaml

from qaoa_structure.runner import run_benchmark


def test_resume_does_not_duplicate_completed_runs(tmp_path: Path):
    config = {
        "master_seed": 7,
        "node_sizes": [2],
        "graphs": {
            "erdos_renyi": {"instances_per_n": 0, "probability": 0.5},
            "random_regular": {"instances_per_n": 0, "degree": 3},
            "complete": {"instances_per_n": 1},
        },
        "ansatze": {"principal": ["ry"], "controls": [], "qaoa_depths": []},
        "optimizer": {"name": "COBYLA", "max_evaluations": 30, "rhobeg": 1.0, "tol": 1e-4},
        "restarts": 1,
        "primary_evaluation": "exact_statevector",
        "success_threshold": 0.99,
        "classical_baselines": {"single_flip_local_search": True, "random_cut_reference": True},
        "checkpoints": [10, 30],
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "results"

    run_benchmark(cfg, out)
    run_benchmark(cfg, out, resume=True)

    assert len((out / "runs.jsonl").read_text().splitlines()) == 1
    assert len((out / "histories.jsonl").read_text().splitlines()) == 1
    assert len((out / "local_search.jsonl").read_text().splitlines()) == 1
