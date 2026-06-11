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
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from scipy.stats import kurtosis, skew
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    import umap.umap_ as umap
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency: matplotlib, numpy, pyarrow, rich, scipy, "
        "scikit-learn, and umap-learn are required to analyze embedding "
        "columns. Install project dependencies again or install the missing "
        "packages manually."
    ) from error


DEFAULT_K_VALUES = (1, 5, 10, 100)
PLOT_OUTPUT_DIR = Path("results") / "plots" / "dataset_analysis"
console = Console()


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
        help=(
            "Comma-separated list of one or more column names to validate. "
            "If omitted, all embedding columns are detected automatically."
        ),
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


def infer_embedding_columns(parquet_file: Path) -> list[str]:
    """Infer embedding columns from the parquet schema."""

    schema = pq.read_schema(parquet_file)
    columns = [
        field.name
        for field in schema
        if is_embedding_type(field.type)
    ]

    if not columns:
        raise ValueError(
            "No embedding columns found. Expected array-like columns with "
            "float or double values."
        )

    return columns


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
    """Compute the Gini coefficient for a vector."""

    adjusted_values = values.astype(np.float64, copy=True)
    min_value = adjusted_values.min()
    if min_value < 0:
        adjusted_values = adjusted_values - min_value

    sorted_values = np.sort(adjusted_values)
    total = sorted_values.sum()
    if total == 0:
        return 0.0

    n_values = len(sorted_values)
    index = np.arange(1, n_values + 1, dtype=np.float64)
    return float(
        (2 * np.sum(index * sorted_values) / (n_values * total))
        - (n_values + 1) / n_values
    )


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length."""

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embeddings contain zero vectors; cannot normalize.")
    return embeddings / norms


def compute_knn_similarity_coefficient(
    embeddings: np.ndarray,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    normalize: bool = True,
    progress: Progress | None = None,
    task_id: Any = None,
) -> dict[str, Any]:
    """Compute cosine-similarity-based kNN statistics for multiple k values."""

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}."
        )

    n_rows = embeddings.shape[0]
    if n_rows < 2:
        raise ValueError("At least two embeddings are required for kNN analysis.")

    if normalize:
        embeddings = normalize_embeddings(embeddings)

    summary: dict[int, dict[str, float]] = {}
    pointwise_min_similarity: dict[int, np.ndarray] = {}

    for k in k_values:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")
        if k >= n_rows:
            raise ValueError(
                f"k={k} is invalid for {n_rows} embeddings; k must be smaller "
                "than the number of rows."
            )

        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        cosine_distances = distances[:, 1:]
        cosine_similarities = 1.0 - cosine_distances
        pointwise_min_similarity[k] = cosine_similarities.min(axis=1)

        mean_similarity = cosine_similarities.mean(axis=1)
        mean_similarity_mean = mean_similarity.mean()

        summary[k] = {
            "similarity_skewness": float(skew(mean_similarity)),
            "similarity_kurtosis": float(kurtosis(mean_similarity)),
            "similarity_cv": float(mean_similarity.std() / mean_similarity_mean),
            "similarity_gini": float(gini(mean_similarity)),
            "mean_knn_similarity": float(cosine_similarities.mean()),
        }

        if progress is not None and task_id is not None:
            progress.advance(task_id)

    return {
        "summary": summary,
        "pointwise_min_similarity": pointwise_min_similarity,
    }


def load_embedding_column(parquet_file: Path, column: str) -> tuple[np.ndarray, int]:
    """Load one embedding column from parquet into a dense NumPy matrix."""

    table = pq.read_table(parquet_file, columns=[column])
    values = table.column(column)

    embeddings = values.to_pylist()
    filtered_embeddings = [embedding for embedding in embeddings if embedding is not None]
    dropped_nulls = len(embeddings) - len(filtered_embeddings)

    if not filtered_embeddings:
        raise ValueError(
            f"Column '{column}' contains no non-null embeddings after filtering."
        )

    matrix = np.asarray(filtered_embeddings, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"Column '{column}' does not form a 2D embedding matrix; got "
            f"shape {matrix.shape}."
        )

    return matrix, dropped_nulls


def sanitize_filename(name: str) -> str:
    """Convert a column name into a filesystem-safe filename stem."""

    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def create_umap_scatter_plot_figure(
    umap_projection: np.ndarray,
    point_colors_by_k: dict[int, np.ndarray],
    output_path: Path,
    title: str,
    k_values: tuple[int, ...],
) -> None:
    """Create horizontally packed UMAP scatter subplots for all k values."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    color_min = min(point_colors_by_k[k].min() for k in k_values)
    color_max = max(point_colors_by_k[k].max() for k in k_values)

    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 6))
    if len(k_values) == 1:
        axes = [axes]

    scatter = None
    for index, k in enumerate(k_values):
        ax = axes[index]
        scatter = ax.scatter(
            umap_projection[:, 0],
            umap_projection[:, 1],
            c=point_colors_by_k[k],
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            s=8,
            alpha=0.7,
            linewidths=0,
        )
        ax.set_title(f"k={k}")
        ax.set_xlabel("UMAP 1")
        if index == 0:
            ax.set_ylabel("UMAP 2")

    assert scatter is not None
    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.9)
    colorbar.set_label("Min kNN similarity")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def create_sorted_knn_similarity_plot_figure(
    point_similarities_by_k: dict[int, np.ndarray],
    output_path: Path,
    title: str,
    k_values: tuple[int, ...],
) -> None:
    """Create horizontally packed sorted similarity subplots for all k values."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_min = min(np.min(point_similarities_by_k[k]) for k in k_values)
    y_max = max(np.max(point_similarities_by_k[k]) for k in k_values)

    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 4.5))
    if len(k_values) == 1:
        axes = [axes]

    for index, k in enumerate(k_values):
        ax = axes[index]
        sorted_similarities = np.sort(point_similarities_by_k[k])
        ax.plot(np.arange(len(sorted_similarities)), sorted_similarities, linewidth=1.5)
        ax.set_title(f"k={k}")
        ax.set_xlabel("Point index (sorted)")
        if index == 0:
            ax.set_ylabel("Min kNN similarity")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def compute_umap_projection(embeddings: np.ndarray) -> np.ndarray:
    """Compute a 2D UMAP projection after PCA preprocessing."""

    normalized_embeddings = normalize_embeddings(embeddings)

    n_rows, n_dims = normalized_embeddings.shape
    n_pca_components = min(50, n_dims, n_rows)
    reduced_embeddings = PCA(n_components=n_pca_components).fit_transform(
        normalized_embeddings
    )

    n_neighbors = min(15, max(2, n_rows - 1))
    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=42,
    ).fit_transform(reduced_embeddings)


def analyze_columns(
    parquet_file: Path,
    columns: list[str],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, dict[str, Any]]:
    """Compute embedding statistics for all requested columns."""

    validate_embedding_columns(parquet_file, columns)

    results: dict[str, dict[str, Any]] = {}
    for column in columns:
        embeddings, dropped_nulls = load_embedding_column(parquet_file, column)
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
            knn_results = compute_knn_similarity_coefficient(
                embeddings,
                k_values=k_values,
                progress=progress,
                task_id=task_id,
            )

        plot_paths: dict[str, str] = {}
        umap_projection = compute_umap_projection(embeddings)
        scatter_plot_path = PLOT_OUTPUT_DIR / (
            f"{parquet_file.stem}__{sanitize_filename(column)}__scatter.pdf"
        )
        create_umap_scatter_plot_figure(
            umap_projection=umap_projection,
            point_colors_by_k=knn_results["pointwise_min_similarity"],
            output_path=scatter_plot_path,
            title=f"{column} ({parquet_file.stem})",
            k_values=k_values,
        )
        sorted_plot_path = PLOT_OUTPUT_DIR / (
            f"{parquet_file.stem}__{sanitize_filename(column)}__sorted.pdf"
        )
        create_sorted_knn_similarity_plot_figure(
            point_similarities_by_k=knn_results["pointwise_min_similarity"],
            output_path=sorted_plot_path,
            title=f"{column} ({parquet_file.stem}) sorted similarities",
            k_values=k_values,
        )
        plot_paths["scatter"] = str(scatter_plot_path)
        plot_paths["sorted"] = str(sorted_plot_path)

        results[column] = {
            "num_rows": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "dropped_null_embeddings": dropped_nulls,
            "knn_density": knn_results["summary"],
            "plot_paths": plot_paths,
        }

    return results


def print_results(results: dict[str, dict[str, Any]]) -> None:
    """Print column analysis results."""

    for column, stats in results.items():
        print(f"Column: {column}")
        print(f"  rows: {stats['num_rows']}")
        print(f"  embedding_dim: {stats['embedding_dim']}")
        print(f"  dropped_null_embeddings: {stats['dropped_null_embeddings']}")
        print(f"  scatter_plot_path: {stats['plot_paths']['scatter']}")
        print(f"  sorted_plot_path: {stats['plot_paths']['sorted']}")

        for k, metrics in stats["knn_density"].items():
            print(f"  k={k}")
            for name, value in metrics.items():
                print(f"    {name}: {value:.6f}")


def main() -> None:
    """Run the CLI."""

    args = parse_args()
    if args.columns:
        columns = parse_columns(args.columns)
    else:
        columns = infer_embedding_columns(args.parquet_file)
        print(
            "Detected embedding columns: "
            + ", ".join(columns)
        )
    results = analyze_columns(args.parquet_file, columns)
    print_results(results)


if __name__ == "__main__":
    main()
