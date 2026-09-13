"""Run lightweight integrity checks on the published benchmark dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qaoa_structure.runner import build_graphs, load_config


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "final"
CONFIG_PATH = ROOT / "experiments" / "config" / "final.yaml"

EXPECTED_GRAPHS = 95
EXPECTED_VARIATIONAL_RUNS = 3590
EXPECTED_HISTORIES = 3590
EXPECTED_LOCAL_SEARCH_RUNS = 475


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def variational_keys(data: pd.DataFrame) -> set[tuple[object, ...]]:
    return {
        (
            row.graph_id,
            row.ansatz,
            None if pd.isna(row.p) else int(row.p),
            int(row.restart),
            int(row.seed),
        )
        for row in data.itertuples(index=False)
    }


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    runs = load_jsonl(RESULTS_DIR / "runs.jsonl")
    histories = load_jsonl(RESULTS_DIR / "histories.jsonl")
    local_search = load_jsonl(RESULTS_DIR / "local_search.jsonl")

    check(len(runs) == EXPECTED_VARIATIONAL_RUNS, "Unexpected number of runs.")
    check(len(histories) == EXPECTED_HISTORIES, "Unexpected number of histories.")
    check(
        len(local_search) == EXPECTED_LOCAL_SEARCH_RUNS,
        "Unexpected number of local-search runs.",
    )
    check(runs["graph_id"].nunique() == EXPECTED_GRAPHS, "Unexpected graph count.")

    run_keys = variational_keys(runs)
    history_keys = variational_keys(histories)
    check(len(run_keys) == len(runs), "Duplicate variational run keys detected.")
    check(run_keys == history_keys, "Run records and optimization histories do not match.")

    tolerance = 1e-10
    check(
        runs["approximation_ratio"].between(-tolerance, 1.0 + tolerance).all(),
        "Approximation ratio outside [0, 1].",
    )
    check(
        runs["p_opt"].between(-tolerance, 1.0 + tolerance).all(),
        "Optimal-solution probability outside [0, 1].",
    )

    config = load_config(CONFIG_PATH)
    resolved_config = json.loads((RESULTS_DIR / "resolved_config.json").read_text())
    check(config == resolved_config, "Stored resolved_config.json differs from final.yaml.")

    expected_graph_ids = {instance.graph_id for instance in build_graphs(config)}
    observed_graph_ids = set(runs["graph_id"])
    check(
        expected_graph_ids == observed_graph_ids,
        "Stored graph IDs do not match the frozen config.",
    )

    exact_controls = json.loads(
        (RESULTS_DIR / "final_exact_control_checks.json").read_text(encoding="utf-8")
    )
    check(exact_controls["n_matched_runs"] == 475, "Rx/Ry control count is incorrect.")
    check(
        np.isclose(exact_controls["max_abs_initial_objective_difference"], 0.0),
        "Rx/Ry initial objectives are not identical.",
    )
    check(
        np.isclose(exact_controls["max_abs_final_objective_difference"], 0.0),
        "Rx/Ry final objectives are not identical.",
    )
    check(
        np.isclose(exact_controls["max_abs_p_opt_difference"], 0.0),
        "Rx/Ry optimal probabilities are not identical.",
    )
    check(
        exact_controls["max_abs_evaluation_difference"] == 0,
        "Rx/Ry evaluation counts are not identical.",
    )

    overall = pd.read_csv(RESULTS_DIR / "final_overall_summary.csv").set_index("model")
    print("Benchmark integrity checks passed.")
    print(f"  graphs:                {runs['graph_id'].nunique()}")
    print(f"  variational runs:      {len(runs)}")
    print(f"  optimization histories:{len(histories):>7}")
    print(f"  local-search runs:     {len(local_search)}")
    print("  mean approximation ratios:")
    for model in ["ry", "agnostic_xx", "qaoa_p1", "qaoa_p3", "qaoa_p5"]:
        value = overall.loc[model, "approximation_ratio"]
        print(f"    {model:12s} {value:.6f}")


if __name__ == "__main__":
    main()
