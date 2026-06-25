# SemCEB: A Cardinality Estimation Benchmark for Semantic Operators

[![arXiv](https://img.shields.io/badge/arXiv-2606.23081-b31b1b.svg)](https://arxiv.org/abs/2606.23081)
[![Python format check](https://github.com/utndatasystems/SemCEB/actions/workflows/python-format.yml/badge.svg?branch=main)](https://github.com/utndatasystems/SemCEB/actions/workflows/python-format.yml)
[![Python 3.10](https://github.com/utndatasystems/SemCEB/actions/workflows/python-version-check-3-10.yml/badge.svg?branch=main)](https://github.com/utndatasystems/SemCEB/actions/workflows/python-version-check-3-10.yml)
[![Python 3.11](https://github.com/utndatasystems/SemCEB/actions/workflows/python-version-check-3-11.yml/badge.svg?branch=main)](https://github.com/utndatasystems/SemCEB/actions/workflows/python-version-check-3-11.yml)
[![Python 3.12](https://github.com/utndatasystems/SemCEB/actions/workflows/python-version-check-3-12.yml/badge.svg?branch=main)](https://github.com/utndatasystems/SemCEB/actions/workflows/python-version-check-3-12.yml)


SemCEB provides a benchmark pipeline for running cardinality estimation algorithms and plotting the results.

## Installation

Clone the repository and install the project in editable mode from the project root.

The editable install means that local code changes under `src/semceb/` are picked up immediately. This is useful when modifying the provided algorithm template in `src/semceb/algorithms/custom_algorithm_template.py`.

<details>
<summary>Linux</summary>

```bash
git clone --recurse-submodules https://github.com/utndatasystems/SemCEB.git
cd SemCEB

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

</details>

<details>
<summary>macOS</summary>

```bash
git clone --recurse-submodules https://github.com/utndatasystems/SemCEB.git
cd SemCEB

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

</details>

<details>
<summary>Windows</summary>

```bash
git clone --recurse-submodules https://github.com/utndatasystems/SemCEB.git
cd SemCEB

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

</details>

Verify the installation by printing the available commands:   
```bash
semceb
```

**Note:**   
The provided `config.toml` is configured to use an OpenAI model for LLM-based semantic operators. Therefore, an OpenAI API key is required when running the benchmark with the configuration included in this repository.   
Create a local `.env` file from `.env.example` and configure the required `OPENAI_API_KEY` value there. If you configure or implement other LLM providers, add the corresponding API keys or credentials to the same `.env` file.


## Modes

### `run`

Runs the configured algorithms on the benchmark queries.

```bash
semceb run
```

Uses:

```text
config.toml
benchmark_queries/queries.jsonl
```

Writes raw benchmark results to:

```text
results/raw/result.jsonl
```

### `plot`

Creates result summaries, plots, and tables from the raw benchmark results.

```bash
semceb plot
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
src/semceb/algorithms/custom_algorithm_template.py
```

This is the main file intended for users to modify. You can implement your own cardinality estimation logic there while keeping the rest of the benchmark pipeline unchanged.

After editing the algorithm file, run the benchmark with:

```bash
semceb run
```

Then generate plots and summary tables with:

```bash
semceb plot
```

No reinstall is needed after changing files under `src/semceb/`, as long as the project was installed in editable mode; see [Installation](#installation).

## Configuration

The benchmark is configured in `config.toml`.

Use this file to select which algorithms should run and to adjust benchmark or algorithm-specific settings. To exclude an algorithm from a run, comment out or remove its corresponding `[[algorithms]]` blocks.

The `scale_factor` setting defines how many rows are loaded from the main dataset table. Related tables are filtered to match the selected rows. Rows are shuffled deterministically before selection.

## Citation

If you use this work, please cite the corresponding paper:

arXiv: <https://arxiv.org/abs/2606.23081>

```
@article{zimmerer2026semceb,
      title={{SemCEB}: A Cardinality Estimation Benchmark for Semantic Operators}, 
      author={Andreas Zimmerer and Claudius Kühn and Yang Li and Mihail Stoian and Renata Borovica-Gajic and Andreas Kipf},
      year={2026},
      eprint={2606.23081},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2606.23081}, 
}
```
