from __future__ import annotations

# Example usage:
# python analyze_data.py amazon-reviews/processed/Arts_Crafts_and_Sewing__raw_5core/products_filtered_with_embeddings.parquet \
#   --columns=product_title_embeddings_qwen_qwen3_embedding_0_6b,description_json_embeddings_qwen_qwen3_embedding_0_6b,features_json_embeddings_qwen_qwen3_embedding_0_6b,details_json_embeddings_qwen_qwen3_embedding_0_6b,main_image_local_embeddings_google_siglip2_base_patch16_224
#
# python analyze_data.py amazon-reviews/processed/Arts_Crafts_and_Sewing__raw_5core/reviews_filtered_with_embeddings.parquet \
#   --columns=review_title_embeddings_qwen_qwen3_embedding_0_6b,review_text_embeddings_qwen_qwen3_embedding_0_6b
#
import argparse
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from scipy.stats import kurtosis, skew
    from sklearn.neighbors import NearestNeighbors
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency: numpy, pyarrow, scipy, and scikit-learn are "
        "required to analyze embedding columns. Install project dependencies "
        "again or install the missing packages manually."
    ) from error

from utils.console import console


DEFAULT_K_VALUES = (5, 10, 20)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Validate that selected parquet columns contain embeddings.",
    )
    parser.add_argument(
        "parquet_file",
        type=Path,
        help="Path to a parquet file.",
    )
    parser.add_argument(
        "--columns",
        required=True,
        help="Comma-separated list of one or more column names to validate.",
    )
    return parser.parse_args()


def parse_columns(raw_columns: str) -> list[str]:
    """Parse a comma-separated column list."""

    columns = [column.strip() for column in raw_columns.split(",")]
    columns = [column for column in columns if column]

    if not columns:
        raise ValueError("Expected at least one column in --columns.")

    return columns


def is_embedding_type(data_type: pa.DataType) -> bool:
    """Return whether an Arrow type is an array of float32 or float64."""

    if not (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    ):
        return False

    value_type = data_type.value_type
    return pa.types.is_float32(value_type) or pa.types.is_float64(value_type)


def validate_embedding_columns(
    parquet_file: Path,
    columns: list[str],
) -> None:
    """Validate that all requested columns exist and contain embeddings."""

    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    schema = pq.read_schema(parquet_file)

    missing_columns = [column for column in columns if column not in schema.names]
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Column(s) not found in parquet schema: {missing}")

    invalid_columns: list[str] = []
    for column in columns:
        column_type = schema.field(column).type
        if not is_embedding_type(column_type):
            invalid_columns.append(f"{column} ({column_type})")

    if invalid_columns:
        invalid = ", ".join(invalid_columns)
        raise TypeError(
            "Expected embedding columns to have Arrow type ARRAY<FLOAT> or "
            f"ARRAY<DOUBLE>. Invalid column(s): {invalid}"
        )


def gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient for a non-negative vector."""

    sorted_values = np.sort(values)
    total = sorted_values.sum()
    if total == 0:
        return 0.0

    n_values = len(sorted_values)
    index = np.arange(1, n_values + 1, dtype=np.float64)
    return float(
        (2 * np.sum(index * sorted_values) / (n_values * total))
        - (n_values + 1) / n_values
    )


def compute_knn_distance_coefficient(
    embeddings: np.ndarray,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    normalize: bool = True,
    progress: Progress | None = None,
    task_id: Any = None,
) -> dict[int, dict[str, float]]:
    """Compute kNN-based local density statistics for multiple k values."""

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}."
        )

    n_rows = embeddings.shape[0]
    if n_rows < 2:
        raise ValueError("At least two embeddings are required for kNN analysis.")

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embeddings contain zero vectors; cannot normalize.")
        embeddings = embeddings / norms

    results: dict[int, dict[str, float]] = {}

    for k in k_values:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")
        if k >= n_rows:
            raise ValueError(
                f"k={k} is invalid for {n_rows} embeddings; k must be smaller "
                "than the number of rows."
            )

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        distances = distances[:, 1:]

        density = 1.0 / distances.mean(axis=1)
        density_mean = density.mean()

        results[k] = {
            "density_skewness": float(skew(density)),
            "density_kurtosis": float(kurtosis(density)),
            "density_cv": float(density.std() / density_mean),
            "density_gini": float(gini(density)),
            "mean_knn_distance": float(distances.mean()),
        }

        if progress is not None and task_id is not None:
            progress.advance(task_id)

    return results


def load_embedding_column(parquet_file: Path, column: str) -> np.ndarray:
    """Load one embedding column from parquet into a dense NumPy matrix."""

    table = pq.read_table(parquet_file, columns=[column])
    values = table.column(column)

    if values.null_count:
        raise ValueError(
            f"Column '{column}' contains {values.null_count} null embeddings."
        )

    embeddings = values.to_pylist()
    matrix = np.asarray(embeddings, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"Column '{column}' does not form a 2D embedding matrix; got "
            f"shape {matrix.shape}."
        )

    return matrix


def analyze_columns(
    parquet_file: Path,
    columns: list[str],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, dict[str, Any]]:
    """Compute embedding statistics for all requested columns."""

    validate_embedding_columns(parquet_file, columns)

    results: dict[str, dict[str, Any]] = {}
    for column in columns:
        embeddings = load_embedding_column(parquet_file, column)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"Computing kNN statistics for [cyan]{column}[/cyan]",
                total=len(k_values),
            )
            results[column] = {
                "num_rows": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "knn_density": compute_knn_distance_coefficient(
                    embeddings,
                    k_values=k_values,
                    progress=progress,
                    task_id=task_id,
                ),
            }

    return results


def print_results(results: dict[str, dict[str, Any]]) -> None:
    """Print column analysis results."""

    for column, stats in results.items():
        print(f"Column: {column}")
        print(f"  rows: {stats['num_rows']}")
        print(f"  embedding_dim: {stats['embedding_dim']}")

        for k, metrics in stats["knn_density"].items():
            print(f"  k={k}")
            for name, value in metrics.items():
                print(f"    {name}: {value:.6f}")


def main() -> None:
    """Run the CLI."""

    args = parse_args()
    columns = parse_columns(args.columns)
    results = analyze_columns(args.parquet_file, columns)
    print_results(results)


if __name__ == "__main__":
    main()
