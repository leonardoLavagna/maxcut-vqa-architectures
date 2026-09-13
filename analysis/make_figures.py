"""Regenerate the manuscript figures from the stored benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "final"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_SEED = 20260815
N_BOOTSTRAP = 5_000

MODEL_DISPLAY = {
    "ry": r"$R_y$ product",
    "agnostic_xx": r"Agnostic $R_yR_z+R_{XX}$",
    "qaoa_p1": r"QAOA $p=1$",
    "qaoa_p3": r"QAOA $p=3$",
    "qaoa_p5": r"QAOA $p=5$",
}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def model_label(row: pd.Series) -> str:
    if row["ansatz"] == "qaoa":
        return f"qaoa_p{int(row['p'])}"
    return str(row["ansatz"])


def bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """Return a 95% percentile bootstrap interval for the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def graph_level_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Average the five optimization restarts for each graph/model pair."""
    return runs.groupby(
        ["graph_id", "graph_family", "n", "model"], as_index=False
    ).agg(
        approximation_ratio=("approximation_ratio", "mean"),
        p_opt=("p_opt", "mean"),
        target_success=("target_success", "mean"),
        n_evaluations=("n_evaluations", "mean"),
    )


def series_with_ci(
    block: pd.DataFrame,
    value_column: str,
    *,
    seed_offset: int,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Collect node-size means and asymmetric bootstrap errors."""
    node_sizes: list[int] = []
    means: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []

    for n, group in block.groupby("n"):
        values = group[value_column].to_numpy(dtype=float)
        mean = float(values.mean())
        low, high = bootstrap_ci(values, seed=BOOTSTRAP_SEED + seed_offset + int(n))

        node_sizes.append(int(n))
        means.append(mean)
        lower_errors.append(mean - low)
        upper_errors.append(high - mean)

    return node_sizes, means, lower_errors, upper_errors


def figure_approximation_ratio(graph: pd.DataFrame) -> None:
    """Approximation ratio versus graph size for the two stochastic families."""
    models = ["ry", "agnostic_xx", "qaoa_p1", "qaoa_p3", "qaoa_p5"]
    families = [
        ("erdos_renyi", r"Erdős–Rényi $G(n,0.5)$"),
        ("regular", "Random 3-regular"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)

    for axis, (family, title) in zip(axes, families):
        family_block = graph[graph["graph_family"] == family]

        for model_index, model in enumerate(models):
            model_block = family_block[family_block["model"] == model]
            x, y, yerr_low, yerr_high = series_with_ci(
                model_block,
                "approximation_ratio",
                seed_offset=model_index * 100,
            )
            axis.errorbar(
                x,
                y,
                yerr=[yerr_low, yerr_high],
                marker="o",
                capsize=2,
                label=MODEL_DISPLAY[model],
            )

        axis.set_title(title)
        axis.set_xlabel("Number of vertices $n$")
        axis.set_ylim(0.72, 1.01)
        axis.grid(alpha=0.2)

    axes[0].set_ylabel(r"Expected approximation ratio $\mathrm{AR}_{\mathrm{exp}}$")
    axes[1].legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "fig_ar_stochastic.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def parameter_matched_differences(
    runs: pd.DataFrame,
    graph: pd.DataFrame,
) -> pd.DataFrame:
    """Graph-wise AR difference between Ry and QAOA with 2p=n."""
    matched = runs[(runs["ansatz"] == "qaoa") & (runs["p"] == runs["n"] / 2)].copy()
    matched_graph = matched.groupby(
        ["graph_id", "graph_family", "n"], as_index=False
    ).agg(qaoa_ar=("approximation_ratio", "mean"))

    ry_graph = graph[graph["model"] == "ry"][
        ["graph_id", "graph_family", "n", "approximation_ratio"]
    ].rename(columns={"approximation_ratio": "ry_ar"})

    paired = ry_graph.merge(
        matched_graph,
        on=["graph_id", "graph_family", "n"],
        validate="one_to_one",
    )
    paired["difference"] = paired["ry_ar"] - paired["qaoa_ar"]
    return paired


def figure_parameter_matched(runs: pd.DataFrame, graph: pd.DataFrame) -> None:
    paired = parameter_matched_differences(runs, graph)
    families = [("erdos_renyi", "Erdos-Renyi"), ("regular", "3-regular")]

    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for family_index, (family, title) in enumerate(families):
        block = paired[paired["graph_family"] == family]
        x, y, yerr_low, yerr_high = series_with_ci(
            block,
            "difference",
            seed_offset=1000 + family_index * 100,
        )
        axis.errorbar(
            x,
            y,
            yerr=[yerr_low, yerr_high],
            marker="o",
            capsize=3,
            label=title,
        )

    axis.axhline(0, linewidth=1)
    axis.set_xlabel("Number of vertices $n$")
    axis.set_ylabel(r"Paired $\mathrm{AR}_{R_y}-\mathrm{AR}_{\mathrm{QAOA},\,p=n/2}$")
    axis.set_title("Parameter-count-matched comparison ($2p=n$)")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "fig_parameter_matched.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def redundancy_differences(graph: pd.DataFrame) -> pd.DataFrame:
    ryrz = graph[graph["model"] == "ryrz"][
        ["graph_id", "graph_family", "n", "approximation_ratio", "n_evaluations"]
    ].rename(
        columns={
            "approximation_ratio": "ryrz_ar",
            "n_evaluations": "ryrz_evaluations",
        }
    )
    ry = graph[graph["model"] == "ry"][
        ["graph_id", "graph_family", "n", "approximation_ratio", "n_evaluations"]
    ].rename(
        columns={
            "approximation_ratio": "ry_ar",
            "n_evaluations": "ry_evaluations",
        }
    )

    paired = ryrz.merge(
        ry,
        on=["graph_id", "graph_family", "n"],
        validate="one_to_one",
    )
    paired = paired[paired["graph_family"].isin(["erdos_renyi", "regular"])].copy()
    paired["ar_difference"] = paired["ryrz_ar"] - paired["ry_ar"]
    paired["evaluation_difference"] = (
        paired["ryrz_evaluations"] - paired["ry_evaluations"]
    )
    return paired


def figure_redundancy(graph: pd.DataFrame) -> None:
    paired = redundancy_differences(graph)
    panels = [
        (
            "ar_difference",
            r"$\mathrm{AR}_{R_yR_z}-\mathrm{AR}_{R_y}$",
            "Objective effect",
            2000,
        ),
        (
            "evaluation_difference",
            r"Extra objective evaluations ($R_yR_z-R_y$)",
            "Optimization-cost effect",
            2100,
        ),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (metric, ylabel, title, seed_offset) in zip(axes, panels):
        x, y, yerr_low, yerr_high = series_with_ci(
            paired,
            metric,
            seed_offset=seed_offset,
        )
        axis.errorbar(x, y, yerr=[yerr_low, yerr_high], marker="o", capsize=3)
        axis.axhline(0, linewidth=1)
        axis.set_xlabel("Number of vertices $n$")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "fig_redundancy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def figure_convergence() -> None:
    checkpoints = pd.read_csv(RESULTS_DIR / "final_checkpoint_summary.csv")
    models = ["ry", "agnostic_xx", "qaoa_p3", "qaoa_p5"]

    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    for model in models:
        block = checkpoints[checkpoints["model"] == model]
        axis.plot(
            block["checkpoint"],
            block["approximation_ratio"],
            marker="o",
            label=MODEL_DISPLAY[model],
        )

    axis.set_xscale("log")
    axis.set_xticks(
        [50, 100, 250, 500, 1000],
        labels=["50", "100", "250", "500", "1000"],
    )
    axis.set_xlabel("Objective-evaluation budget")
    axis.set_ylabel(r"Mean expected approximation ratio $\mathrm{AR}_{\mathrm{exp}}$")
    axis.set_title("Finite-budget convergence (all benchmark graphs)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "fig_convergence.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    runs = load_jsonl(RESULTS_DIR / "runs.jsonl")
    runs["model"] = runs.apply(model_label, axis=1)
    graph = graph_level_summary(runs)

    figure_approximation_ratio(graph)
    figure_parameter_matched(runs, graph)
    figure_redundancy(graph)
    figure_convergence()

    print(f"Figures regenerated in {FIGURES_DIR.relative_to(ROOT)}/.")


if __name__ == "__main__":
    main()
