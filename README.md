# MaxCut VQA architectures

Code and frozen numerical results for the manuscript:

> **From Problem-Structured QAOA to Problem-Agnostic Variational MaxCut Solvers: Representation, Redundancy, and Optimization Landscapes**  
> Leonardo Lavagna and Massimo Panella

The repository contains only the material needed to inspect the implementation, verify the stored benchmark, regenerate the reported summaries and figures, or rerun the experiment from the frozen configuration.

## Structure

```text
.
├── analysis/                 # verification, summary, and figure scripts
├── experiments/
│   ├── config/final.yaml     # frozen benchmark configuration
│   ├── run_final.py          # full benchmark / resume
│   └── run_smoke.py          # small installation check
├── results/final/            # raw benchmark records and derived reference tables
├── src/qaoa_structure/       # MaxCut, ansatz, optimizer, and runner implementation
├── tests/                    # analytical and reproducibility tests
├── pyproject.toml
└── requirements-reproducibility.txt
```

## Setup

Python 3.10 or newer is supported. The stored benchmark was produced with Python 3.13.5.

For a normal editable installation:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

To reproduce the original software environment more closely:

```bash
pip install -r requirements-reproducibility.txt
pip install -e . --no-deps
pytest -q
```

The test suite contains 22 tests covering the analytical controls and the main reproducibility assumptions, including the exact `Rx`/`Ry` equivalence, `Rz` redundancy, the closed-form product-state objective, QAOA state preparation, deterministic graph generation, optimizer accounting, and resumable execution.

## Check the stored benchmark

The complete final raw outputs are included in `results/final/`. A lightweight integrity check can be run with:

```bash
python analysis/verify_results.py
```

The expected dataset contains 95 graphs, 3,590 variational runs with matching optimization histories, and 475 single-vertex-flip local-search runs.

To regenerate the derived tables and figures from the stored raw data:

```bash
python analysis/summarize_final.py
python analysis/make_figures.py
```

The summary script overwrites the derived `final_*.csv` and `final_*.json` files in `results/final/`. Figures are written to `figures/` and are not required for running the benchmark.

A small deterministic smoke test is also available:

```bash
python experiments/run_smoke.py
```

It is intended only to check the installation and is not part of the reported numerical experiment.

## Rerun the benchmark

The full experiment is specified by `experiments/config/final.yaml` and can be run with:

```bash
python experiments/run_final.py
```

An interrupted run can be resumed with:

```bash
python experiments/run_final.py --resume
```

The frozen configuration uses:

- graph sizes `n = 4, 6, 8, 10, 12`;
- complete, Erdős-Rényi `G(n,0.5)`, and random 3-regular graphs;
- product-state `Ry`, `Rx`, and `RyRz` circuits;
- a graph-agnostic `RyRz + RXX` circuit;
- QAOA at `p = 1, 3, 5`, plus the parameter-matched case `2p = n` when needed;
- five deterministic restarts;
- COBYLA with at most 1,000 objective evaluations;
- checkpoints at 50, 100, 250, 500, and 1,000 evaluations;
- master seed `20260815`.

For each graph/restart pair, the variational architectures use the same deterministic restart seed where the parameterizations permit a matched initialization.

## Stored results

`results/final/` contains three raw JSONL files:

- `runs.jsonl`: final record for each variational optimization;
- `histories.jsonl`: objective-evaluation history for each variational run;
- `local_search.jsonl`: classical single-vertex-flip baseline runs.

`resolved_config.json` records the configuration stored with the run. The `final_*.csv` and `final_*.json` files are deterministic reference outputs generated from the raw records by `analysis/summarize_final.py`.

The stored data reproduce the main mean expected approximation ratios reported in the manuscript: 0.9784 for the `Ry` product state, 0.9811 for the graph-agnostic ansatz, and 0.7954, 0.8908, and 0.9202 for QAOA at `p=1`, `p=3`, and `p=5`, respectively. The single-vertex-flip baseline gives a mean approximation ratio of 0.9684.
