# SemCEB

A benchmark pipeline for running selectivity cardinality estimation algorithms and plotting the results.

## Installation

Clone the repository and install the project in editable mode from the project root:

```bash
pip install -e .
```

This installs the dependencies defined in `pyproject.toml` and makes the `SemCEB` command available.

The editable install means that local code changes are picked up immediately. This is useful when modifying the provided algorithm template in `runner/algorithms/my_selectivity_estimation_algorithm.py`.

**Note:**  
For custom queries, models other than the currently configured OpenAI models in `config.toml`, or new LLM-based algorithms, create a local `.env` file from `.env.example` and add the required API keys or credentials.

## Modes

### `run`

Runs the configured algorithms on the queries.

```bash
SemCEB run  # alternative: python run.py run
```

or, because `run` is the default mode:

```bash
SemCEB  # alternative: python run.py
```

Uses:

```text
config.toml
queries/queries.jsonl
runner/algorithms/
```

Writes raw benchmark results to:

```text
results/raw/result.jsonl
```

### `plot`

Creates result summaries and plots from the raw benchmark results.

```bash
SemCEB plot  # alternative: python run.py plot
```

Uses:

```text
results/raw/result.jsonl
```

Writes plots and summary tables to:

```text
results/plots
results/tables
```

## Implementing your own algorithm

The provided algorithm template is located at:

```text
runner/algorithms/my_selectivity_estimation_algorithm.py
```

This is the main file intended for users to modify. You can implement your own selectivity estimation logic there while keeping the rest of the benchmark pipeline unchanged.

After editing the algorithm file, run the benchmark with:

```bash
SemCEB run  # alternative: python run.py run
```

Then generate plots and summary tables with:

```bash
SemCEB plot  # alternative: python run.py plot
```

No reinstall is needed after changing the algorithm file, as long as the project was installed in editable mode with:

```bash
pip install -e .
```

## Configuration

The benchmark is configured in `config.toml`.

Use this file to select which algorithms should run and to adjust benchmark or algorithm-specific settings. To exclude an algorithm from a run, comment out or remove its corresponding `[[algorithms]]` blocks.

The size of the loaded datasets can be controlled via the `scale_factor` setting. The selected subset is shuffled deterministically, so repeated runs with the same input data and scale factor use the same rows in the same order.
