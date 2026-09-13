"""Regenerate the summary tables reported in the paper from stored raw results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "final"
BOOTSTRAP_SEED = 20260815
N_BOOTSTRAP = 10_000
SUCCESS_THRESHOLD = 0.99
BASE_MODELS = ["ry", "rx", "ryrz", "agnostic_xx", "qaoa_p1", "qaoa_p3", "qaoa_p5"]
STOCHASTIC_FAMILIES = ["erdos_renyi", "regular"]
CHECKPOINTS = [50, 100, 250, 500, 1000]


def load_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSON Lines file into a DataFrame."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def model_label(row: pd.Series) -> str:
    """Map a raw run record to the labels used by the analysis tables."""
    if row["ansatz"] == "qaoa":
        return f"qaoa_p{int(row['p'])}"
    return str(row["ansatz"])


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for a sample mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan

    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def stratified_bootstrap_ci(
    block: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    *,
    strata: str = "n",
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """Bootstrap a mean while preserving the benchmark's fixed size design.

    Graph instances are resampled independently within each node-count stratum.
    Every bootstrap replicate therefore retains the original number of graphs at
    each size instead of allowing the size distribution itself to fluctuate.
    """
    groups = [
        group[metric].to_numpy(dtype=float)
        for _, group in block.groupby(strata, sort=True)
    ]
    if sum(len(group) for group in groups) < 2:
        return np.nan, np.nan

    means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        samples = [
            group[rng.integers(0, len(group), size=len(group))]
            for group in groups
        ]
        means[index] = np.concatenate(samples).mean()

    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_groups(
    data: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
    *,
    stratify_by: str | None = None,
) -> pd.DataFrame:
    """Summarize graph-level metrics and attach 95% bootstrap intervals."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for key, block in data.groupby(group_columns, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        summary: dict[str, object] = dict(zip(group_columns, key_tuple))
        summary["n_graphs"] = block["graph_id"].nunique()

        for metric in metrics:
            values = block[metric].to_numpy(dtype=float)
            summary[f"{metric}_mean"] = values.mean()

            if stratify_by is not None and stratify_by in block.columns:
                low, high = stratified_bootstrap_ci(
                    block,
                    metric,
                    rng,
                    strata=stratify_by,
                )
            else:
                low, high = bootstrap_ci(values, rng)

            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high

        rows.append(summary)

    return pd.DataFrame(rows)


def graph_level_variational_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Average restarts so each graph is one inferential unit."""
    return runs.groupby(
        ["graph_id", "graph_family", "n", "model"], as_index=False
    ).agg(
        approximation_ratio=("approximation_ratio", "mean"),
        p_opt=("p_opt", "mean"),
        target_success=("target_success", "mean"),
        n_evaluations=("n_evaluations", "mean"),
        optimizer_success=("optimizer_success", "mean"),
    )


def graph_level_local_search_summary(local_search: pd.DataFrame) -> pd.DataFrame:
    """Average local-search restarts at graph level."""
    return local_search.groupby(
        ["graph_id", "graph_family", "n"], as_index=False
    ).agg(
        approximation_ratio=("approximation_ratio", "mean"),
        target_success=(
            "approximation_ratio",
            lambda values: float((values >= SUCCESS_THRESHOLD).mean()),
        ),
        n_evaluations=("n_evaluations", "mean"),
    )


def parameter_matched_qaoa(runs: pd.DataFrame) -> pd.DataFrame:
    """Return graph-level QAOA results with ``2p=n``."""
    matched = runs[(runs["ansatz"] == "qaoa") & (runs["p"] == runs["n"] / 2)].copy()
    matched["model"] = "qaoa_match"
    return matched.groupby(
        ["graph_id", "graph_family", "n", "model"], as_index=False
    ).agg(
        approximation_ratio=("approximation_ratio", "mean"),
        p_opt=("p_opt", "mean"),
        target_success=("target_success", "mean"),
        n_evaluations=("n_evaluations", "mean"),
        optimizer_success=("optimizer_success", "mean"),
    )


def paired_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    metric: str,
    family: str,
) -> dict[str, object]:
    """Summarize graph-wise paired differences for one graph family."""
    left_block = left[left["graph_family"] == family][["graph_id", "n", metric]].rename(
        columns={metric: "left"}
    )
    right_block = right[right["graph_family"] == family][
        ["graph_id", "n", metric]
    ].rename(columns={metric: "right"})

    paired = left_block.merge(
        right_block,
        on=["graph_id", "n"],
        validate="one_to_one",
    )
    paired["difference"] = paired["left"] - paired["right"]

    seed_offset = sum(map(ord, left_name + right_name + metric + family))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    low, high = stratified_bootstrap_ci(
        paired,
        "difference",
        rng,
        strata="n",
    )

    differences = paired["difference"].to_numpy(dtype=float)
    return {
        "left": left_name,
        "right": right_name,
        "metric": metric,
        "graph_family": family,
        "n_graphs": len(differences),
        "mean_difference": differences.mean(),
        "ci_low": low,
        "ci_high": high,
    }


def write_overall_summaries(graph: pd.DataFrame) -> None:
    base = graph[graph["model"].isin(BASE_MODELS)]

    overall = base.groupby("model", as_index=False).agg(
        n_graphs=("graph_id", "nunique"),
        approximation_ratio=("approximation_ratio", "mean"),
        p_opt=("p_opt", "mean"),
        target_success=("target_success", "mean"),
        n_evaluations=("n_evaluations", "mean"),
        optimizer_success=("optimizer_success", "mean"),
    )
    overall.to_csv(RESULTS_DIR / "final_overall_summary.csv", index=False)

    summarize_groups(
        base,
        ["model", "n"],
        ["approximation_ratio", "p_opt", "target_success", "n_evaluations"],
    ).to_csv(RESULTS_DIR / "final_by_n_summary.csv", index=False)

    summarize_groups(
        base,
        ["model", "graph_family"],
        ["approximation_ratio", "p_opt", "target_success", "n_evaluations"],
        stratify_by="n",
    ).to_csv(RESULTS_DIR / "final_by_family_summary.csv", index=False)


def write_parameter_matched_summary(
    graph: pd.DataFrame,
    matched_qaoa: pd.DataFrame,
) -> None:
    ry_graph = graph[graph["model"] == "ry"].copy()
    combined = pd.concat([ry_graph, matched_qaoa], ignore_index=True)

    summarize_groups(
        combined,
        ["model", "n"],
        ["approximation_ratio", "p_opt", "target_success", "n_evaluations"],
    ).to_csv(RESULTS_DIR / "final_parameter_matched_by_n.csv", index=False)


def write_local_search_summary(local_graph: pd.DataFrame) -> None:
    summarize_groups(
        local_graph,
        ["n"],
        ["approximation_ratio", "target_success", "n_evaluations"],
    ).to_csv(RESULTS_DIR / "final_local_search_by_n.csv", index=False)


def write_exact_control_checks(runs: pd.DataFrame) -> None:
    """Check the matched Rx/Ry equivalence used as an exact control."""
    rx = runs[runs["model"] == "rx"].set_index(["graph_id", "restart", "seed"])
    ry = runs[runs["model"] == "ry"].set_index(["graph_id", "restart", "seed"])
    common = rx.index.intersection(ry.index)

    control = {
        "n_matched_runs": int(len(common)),
        "max_abs_initial_objective_difference": float(
            np.max(
                np.abs(
                    rx.loc[common, "initial_objective"].to_numpy()
                    - ry.loc[common, "initial_objective"].to_numpy()
                )
            )
        ),
        "max_abs_final_objective_difference": float(
            np.max(
                np.abs(
                    rx.loc[common, "final_objective"].to_numpy()
                    - ry.loc[common, "final_objective"].to_numpy()
                )
            )
        ),
        "max_abs_p_opt_difference": float(
            np.max(
                np.abs(
                    rx.loc[common, "p_opt"].to_numpy()
                    - ry.loc[common, "p_opt"].to_numpy()
                )
            )
        ),
        "max_abs_evaluation_difference": int(
            np.max(
                np.abs(
                    rx.loc[common, "n_evaluations"].to_numpy()
                    - ry.loc[common, "n_evaluations"].to_numpy()
                )
            )
        ),
    }
    (RESULTS_DIR / "final_exact_control_checks.json").write_text(
        json.dumps(control, indent=2) + "\n",
        encoding="utf-8",
    )


def write_paired_comparisons(
    graph: pd.DataFrame,
    matched_qaoa: pd.DataFrame,
    local_graph: pd.DataFrame,
) -> None:
    comparisons: list[dict[str, object]] = []
    ry_graph = graph[graph["model"] == "ry"]

    for family in STOCHASTIC_FAMILIES:
        for left_name, right_name in [
            ("ry", "qaoa_p5"),
            ("ry", "agnostic_xx"),
            ("ryrz", "ry"),
        ]:
            left = graph[graph["model"] == left_name]
            right = graph[graph["model"] == right_name]
            for metric in [
                "approximation_ratio",
                "p_opt",
                "target_success",
                "n_evaluations",
            ]:
                comparisons.append(
                    paired_summary(
                        left,
                        right,
                        left_name,
                        right_name,
                        metric,
                        family,
                    )
                )

        for metric in [
            "approximation_ratio",
            "p_opt",
            "target_success",
            "n_evaluations",
        ]:
            comparisons.append(
                paired_summary(
                    ry_graph,
                    matched_qaoa,
                    "ry",
                    "qaoa_match",
                    metric,
                    family,
                )
            )

        for metric in ["approximation_ratio", "target_success", "n_evaluations"]:
            comparisons.append(
                paired_summary(
                    ry_graph,
                    local_graph,
                    "ry",
                    "local_search",
                    metric,
                    family,
                )
            )

    pd.DataFrame(comparisons).to_csv(
        RESULTS_DIR / "final_paired_differences.csv",
        index=False,
    )


def write_checkpoint_summary(histories: pd.DataFrame) -> None:
    """Evaluate convergence at common objective-evaluation checkpoints.

    If a COBYLA run terminates before a checkpoint, its last available value is
    carried forward, matching the convention used in the manuscript.
    """
    rows: list[dict[str, object]] = []

    for _, record in histories.iterrows():
        model = (
            str(record["ansatz"])
            if record["ansatz"] != "qaoa"
            else f"qaoa_p{int(record['p'])}"
        )
        if model not in BASE_MODELS:
            continue

        history = np.asarray(record["objective_history"], dtype=float)
        optimum = float(record["optimum"])

        for checkpoint in CHECKPOINTS:
            history_index = min(checkpoint, len(history)) - 1
            rows.append(
                {
                    "graph_id": record["graph_id"],
                    "model": model,
                    "restart": int(record["restart"]),
                    "checkpoint": checkpoint,
                    "approximation_ratio": float(history[history_index] / optimum),
                }
            )

    checkpoint_runs = pd.DataFrame(rows)
    checkpoint_graphs = checkpoint_runs.groupby(
        ["graph_id", "model", "checkpoint"], as_index=False
    )["approximation_ratio"].mean()

    checkpoint_summary = checkpoint_graphs.groupby(
        ["model", "checkpoint"], as_index=False
    ).agg(
        approximation_ratio=("approximation_ratio", "mean"),
        n_graphs=("graph_id", "nunique"),
    )
    checkpoint_summary.to_csv(
        RESULTS_DIR / "final_checkpoint_summary.csv",
        index=False,
    )


def write_integrity_summary(
    runs: pd.DataFrame,
    histories: pd.DataFrame,
    local_search: pd.DataFrame,
) -> None:
    run_keys = runs[["graph_id", "ansatz", "p", "restart", "seed"]].astype(str)
    integrity = {
        "n_runs": int(len(runs)),
        "n_histories": int(len(histories)),
        "n_local_search_runs": int(len(local_search)),
        "n_graphs": int(runs["graph_id"].nunique()),
        "unique_run_keys": int(run_keys.agg("|".join, axis=1).nunique()),
        "ar_min": float(runs["approximation_ratio"].min()),
        "ar_max": float(runs["approximation_ratio"].max()),
        "popt_min": float(runs["p_opt"].min()),
        "popt_max": float(runs["p_opt"].max()),
    }
    (RESULTS_DIR / "final_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    runs = load_jsonl(RESULTS_DIR / "runs.jsonl")
    histories = load_jsonl(RESULTS_DIR / "histories.jsonl")
    local_search = load_jsonl(RESULTS_DIR / "local_search.jsonl")
    runs["model"] = runs.apply(model_label, axis=1)

    graph = graph_level_variational_summary(runs)
    local_graph = graph_level_local_search_summary(local_search)
    matched_qaoa = parameter_matched_qaoa(runs)

    write_overall_summaries(graph)
    write_parameter_matched_summary(graph, matched_qaoa)
    write_local_search_summary(local_graph)
    write_exact_control_checks(runs)
    write_paired_comparisons(graph, matched_qaoa, local_graph)
    write_checkpoint_summary(histories)
    write_integrity_summary(runs, histories, local_search)

    print("Summary tables regenerated in results/final/.")


if __name__ == "__main__":
    main()
