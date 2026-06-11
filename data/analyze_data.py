from __future__ import annotations

# Example usage:
# python analyze_data.py amazon-reviews/processed/Arts_Crafts_and_Sewing__raw_5core/products_filtered_with_embeddings.parquet \
#   --columns=product_title_embeddings_qwen_qwen3_embedding_0_6b,description_json_embeddings_qwen_qwen3_embedding_0_6b,features_json_embeddings_qwen_qwen3_embedding_0_6b,details_json_embeddings_qwen_qwen3_embedding_0_6b,main_image_local_embeddings_google_siglip2_base_patch16_224
#
# python analyze_data.py amazon-reviews/processed/Arts_Crafts_and_Sewing__raw_5core/reviews_filtered_with_embeddings.parquet \
#   --columns=review_title_embeddings_qwen_qwen3_embedding_0_6b,review_text_embeddings_qwen_qwen3_embedding_0_6b
#
import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import hdbscan
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.stats import kurtosis, linregress, skew
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    import umap.umap_ as umap
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency: hdbscan, matplotlib, numpy, pyarrow, rich, "
        "scipy, scikit-learn, and umap-learn are required to analyze "
        "embedding columns. Install project dependencies again or install "
        "the missing packages manually."
    ) from error


DEFAULT_K_VALUES = (1, 5, 10, 100)
PLOT_OUTPUT_DIR = Path("results") / "plots" / "dataset_analysis"
IMBALANCE_UMAP_COMPONENTS = 20
IMBALANCE_UMAP_NEIGHBORS = 15
IMBALANCE_HDBSCAN_MIN_SAMPLES = 5
IMBALANCE_STABILITY_RUNS = 3
IMBALANCE_NOISE_RADIUS_PERCENTILE = 30.0
IMBALANCE_NOISE_KNN_K = 5
console = Console()


@dataclass
class ImbalanceReport:
    """Semantic imbalance outputs for one embedding column."""

    n_samples: int
    n_clusters: int
    n_primary_clusters: int
    n_micro_clusters: int
    n_singletons: int
    raw_noise_fraction: float
    cluster_sizes: np.ndarray
    labels: np.ndarray
    label_tiers: np.ndarray
    embedding_2d: np.ndarray
    gini: float
    lir: float
    alpha: float
    largest_share: float
    gini_per_seed: list[float] = field(default_factory=list)
    n_clusters_per_seed: list[int] = field(default_factory=list)

    @property
    def gini_std(self) -> float:
        return float(np.std(self.gini_per_seed)) if self.gini_per_seed else 0.0

    @property
    def n_clusters_std(self) -> float:
        return float(np.std(self.n_clusters_per_seed)) if self.n_clusters_per_seed else 0.0


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


def zipf_alpha(cluster_sizes: np.ndarray) -> float:
    """Fit a Zipf exponent on descending cluster sizes."""

    sizes = np.sort(cluster_sizes.astype(np.float64))[::-1]
    sizes = sizes[sizes > 0]
    if len(sizes) < 2:
        return 0.0

    log_ranks = np.log(np.arange(1, len(sizes) + 1))
    log_sizes = np.log(sizes)
    slope, *_ = linregress(log_ranks, log_sizes)
    return float(-slope)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length."""

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embeddings contain zero vectors; cannot normalize.")
    return embeddings / norms


def validate_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Clean one embedding matrix before downstream analysis."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {matrix.shape}.")

    if not np.isfinite(matrix).all():
        warnings.warn("NaN/Inf detected in embeddings; replacing with 0.", stacklevel=2)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    zero_variance_dims = matrix.var(axis=0) == 0
    if zero_variance_dims.any():
        warnings.warn(
            f"Dropping {zero_variance_dims.sum()} zero-variance dimensions.",
            stacklevel=2,
        )
        matrix = matrix[:, ~zero_variance_dims]

    return matrix


def compute_umap_embedding(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    random_state: int,
    min_dist: float = 0.0,
) -> np.ndarray:
    """Compute a UMAP embedding with cosine metric."""

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        min_dist=min_dist,
        random_state=random_state,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return reducer.fit_transform(embeddings)


def compute_cluster_sizes(labels: np.ndarray) -> np.ndarray:
    """Return descending cluster sizes."""

    if labels.max() < 0:
        return np.array([1], dtype=np.int64)
    counts = np.bincount(labels)
    return np.sort(counts[counts > 0])[::-1]


def resolve_noise_labels(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    noise_radius_percentile: float = IMBALANCE_NOISE_RADIUS_PERCENTILE,
    noise_knn_k: int = IMBALANCE_NOISE_KNN_K,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Resolve HDBSCAN noise into micro-clusters and singletons."""

    resolved_labels = labels.copy()
    label_tiers = np.where(labels >= 0, np.int8(0), np.int8(-1))

    noise_indices = np.where(resolved_labels == -1)[0]
    n_noise = len(noise_indices)
    if n_noise == 0:
        return resolved_labels, label_tiers, 0, 0

    noise_points = reduced_embeddings[noise_indices]

    cluster_ids = np.unique(labels[labels >= 0])
    intra_cluster_distances: list[float] = []
    for cluster_id in cluster_ids:
        cluster_points = reduced_embeddings[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        intra_cluster_distances.extend(
            np.linalg.norm(cluster_points - centroid, axis=1).tolist()
        )

    if intra_cluster_distances:
        threshold = float(
            np.percentile(intra_cluster_distances, noise_radius_percentile)
        )
    else:
        fallback_k = min(5, n_noise - 1)
        if fallback_k > 0:
            fallback_distances, _ = (
                NearestNeighbors(n_neighbors=fallback_k)
                .fit(noise_points)
                .kneighbors(noise_points)
            )
            threshold = float(np.median(fallback_distances[:, -1]))
        else:
            threshold = 0.0

    component_labels = np.arange(n_noise)
    if n_noise > 1 and threshold > 0:
        graph_k = min(noise_knn_k, n_noise - 1)
        distances, indices = (
            NearestNeighbors(n_neighbors=graph_k)
            .fit(noise_points)
            .kneighbors(noise_points)
        )

        rows: list[int] = []
        cols: list[int] = []
        for row_index in range(n_noise):
            for neighbor_index in range(graph_k):
                if distances[row_index, neighbor_index] <= threshold:
                    target_index = indices[row_index, neighbor_index]
                    rows.extend([row_index, target_index])
                    cols.extend([target_index, row_index])

        if rows:
            adjacency = csr_matrix(
                (np.ones(len(rows)), (rows, cols)),
                shape=(n_noise, n_noise),
            )
            _, component_labels = connected_components(adjacency, directed=False)

    next_cluster_id = int(resolved_labels.max()) + 1
    component_sizes = np.bincount(component_labels)
    n_micro_clusters = int((component_sizes > 1).sum())
    n_singletons = int((component_sizes == 1).sum())

    for component_id in range(len(component_sizes)):
        component_mask = component_labels == component_id
        tier = np.int8(1) if component_sizes[component_id] > 1 else np.int8(2)
        for original_index in noise_indices[component_mask]:
            resolved_labels[original_index] = next_cluster_id
            label_tiers[original_index] = tier
        next_cluster_id += 1

    return resolved_labels, label_tiers, n_micro_clusters, n_singletons


def compute_embedding_imbalance_report(
    embeddings: np.ndarray,
    random_state: int = 42,
    n_stability_runs: int = IMBALANCE_STABILITY_RUNS,
) -> ImbalanceReport:
    """Measure semantic class imbalance via UMAP + HDBSCAN."""

    matrix = validate_embedding_matrix(embeddings)
    normalized_embeddings = normalize_embeddings(matrix)
    n_samples = len(normalized_embeddings)
    min_cluster_size = max(10, int(0.01 * n_samples))

    clustering_embedding = compute_umap_embedding(
        normalized_embeddings,
        n_components=IMBALANCE_UMAP_COMPONENTS,
        n_neighbors=IMBALANCE_UMAP_NEIGHBORS,
        random_state=random_state,
    )
    raw_labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=IMBALANCE_HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(clustering_embedding)
    n_primary_clusters = int(raw_labels.max()) + 1 if raw_labels.max() >= 0 else 0
    raw_noise_fraction = float((raw_labels == -1).mean())
    labels, label_tiers, n_micro_clusters, n_singletons = resolve_noise_labels(
        clustering_embedding,
        raw_labels,
    )
    cluster_sizes = compute_cluster_sizes(labels)

    visualization_embedding = compute_umap_embedding(
        normalized_embeddings,
        n_components=2,
        n_neighbors=IMBALANCE_UMAP_NEIGHBORS,
        random_state=random_state,
        min_dist=0.1,
    )

    report = ImbalanceReport(
        n_samples=n_samples,
        n_clusters=int(len(cluster_sizes)),
        n_primary_clusters=n_primary_clusters,
        n_micro_clusters=n_micro_clusters,
        n_singletons=n_singletons,
        raw_noise_fraction=raw_noise_fraction,
        cluster_sizes=cluster_sizes,
        labels=labels,
        label_tiers=label_tiers,
        embedding_2d=visualization_embedding,
        gini=float(gini(cluster_sizes)),
        lir=float(cluster_sizes.max() / cluster_sizes.mean()),
        alpha=float(zipf_alpha(cluster_sizes)),
        largest_share=float(cluster_sizes.max() / n_samples),
    )

    for seed in range(random_state + 1, random_state + 1 + n_stability_runs):
        seeded_embedding = compute_umap_embedding(
            normalized_embeddings,
            n_components=IMBALANCE_UMAP_COMPONENTS,
            n_neighbors=IMBALANCE_UMAP_NEIGHBORS,
            random_state=seed,
        )
        seeded_labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=IMBALANCE_HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(seeded_embedding)
        seeded_labels, _, _, _ = resolve_noise_labels(seeded_embedding, seeded_labels)
        seeded_sizes = compute_cluster_sizes(seeded_labels)
        report.gini_per_seed.append(float(gini(seeded_sizes)))
        report.n_clusters_per_seed.append(int(len(seeded_sizes)))

    return report


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


def create_imbalance_plot_figure(
    report: ImbalanceReport,
    output_path: Path,
    title: str,
) -> None:
    """Create the semantic class imbalance figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="white")

    create_cluster_map_plot(report, axes[0])
    create_rank_size_plot(report, axes[1])
    create_lorenz_curve_plot(report, axes[2])

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, format="pdf")
    plt.close(figure)


def create_cluster_map_plot(report: ImbalanceReport, ax: plt.Axes) -> None:
    """Plot a UMAP cluster map with primary, micro, and singleton tiers."""

    embedding_2d = report.embedding_2d
    labels = report.labels
    tiers = report.label_tiers

    singleton_mask = tiers == 2
    if singleton_mask.any():
        ax.scatter(
            embedding_2d[singleton_mask, 0],
            embedding_2d[singleton_mask, 1],
            c="#cccccc",
            s=3,
            alpha=0.35,
            linewidths=0,
            zorder=1,
        )

    micro_cluster_ids = sorted(set(labels[tiers == 1]))
    if micro_cluster_ids:
        micro_palette = plt.cm.Set3(np.linspace(0, 1, max(len(micro_cluster_ids), 1)))
        for rank, cluster_id in enumerate(micro_cluster_ids):
            cluster_mask = labels == cluster_id
            ax.scatter(
                embedding_2d[cluster_mask, 0],
                embedding_2d[cluster_mask, 1],
                c=[micro_palette[rank % 12]],
                s=4,
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )

    primary_cluster_ids = sorted(
        set(labels[tiers == 0]),
        key=lambda cluster_id: int((labels == cluster_id).sum()),
        reverse=True,
    )
    counts = {
        cluster_id: int((labels == cluster_id).sum())
        for cluster_id in primary_cluster_ids
    }
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(primary_cluster_ids), 1)))

    for rank, cluster_id in enumerate(primary_cluster_ids):
        cluster_mask = labels == cluster_id
        color = palette[rank % 20]
        ax.scatter(
            embedding_2d[cluster_mask, 0],
            embedding_2d[cluster_mask, 1],
            c=[color],
            s=4,
            alpha=0.65,
            linewidths=0,
            zorder=2,
        )
        centroid_x = embedding_2d[cluster_mask, 0].mean()
        centroid_y = embedding_2d[cluster_mask, 1].mean()
        ax.text(
            centroid_x,
            centroid_y,
            str(counts[cluster_id]),
            fontsize=6.5,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            zorder=3,
        )

    legend_handles = []
    if singleton_mask.any():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#cccccc",
                markersize=5,
                label=f"singletons ({singleton_mask.sum():,})",
            )
        )
    if micro_cluster_ids:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#aaaaaa",
                markersize=5,
                label=f"micro-clusters ({len(micro_cluster_ids)})",
            )
        )
    if legend_handles:
        ax.legend(legend_handles, [h.get_label() for h in legend_handles], fontsize=7, framealpha=0.6)

    ax.set_title("1 · Cluster map", fontsize=10, fontweight="medium", pad=6)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)


def create_rank_size_plot(report: ImbalanceReport, ax: plt.Axes) -> None:
    """Plot the cluster rank-size curve with a Zipf fit."""

    sizes = report.cluster_sizes
    ranks = np.arange(1, len(sizes) + 1)

    singleton_tail = int((sizes == 1).sum())
    if singleton_tail > 0:
        ax.axvspan(
            len(sizes) - singleton_tail + 0.5,
            len(sizes) + 0.5,
            alpha=0.08,
            color="#888888",
            zorder=0,
            label=f"singletons ({singleton_tail})",
        )

    ax.scatter(ranks, sizes, s=28, color="#378ADD", zorder=3, label="cluster sizes")
    if len(sizes) >= 3:
        log_ranks = np.log(ranks)
        log_sizes = np.log(sizes.astype(np.float64))
        slope, intercept, *_ = linregress(log_ranks, log_sizes)
        fit_y = np.exp(intercept + slope * log_ranks)
        ax.plot(
            ranks,
            fit_y,
            "--",
            color="#D85A30",
            linewidth=1.6,
            label=f"Zipf fit  α = {report.alpha:.2f}",
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rank  (1 = largest cluster)", fontsize=8)
    ax.set_ylabel("cluster size  (# embeddings)", fontsize=8)
    ax.set_title("2 · Rank-size  (log-log)", fontsize=10, fontweight="medium", pad=6)
    ax.legend(fontsize=7.5)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=7)


def create_lorenz_curve_plot(report: ImbalanceReport, ax: plt.Axes) -> None:
    """Plot the Lorenz curve of cluster sizes."""

    cluster_sizes = np.sort(report.cluster_sizes)
    cumulative_sizes = np.concatenate([[0.0], np.cumsum(cluster_sizes) / cluster_sizes.sum()])
    cumulative_groups = np.linspace(0.0, 1.0, len(cluster_sizes) + 1)

    tail_clusters = report.n_singletons + report.n_micro_clusters
    if tail_clusters > 0 and len(cluster_sizes) > 0:
        ax.axvspan(
            0,
            tail_clusters / len(cluster_sizes),
            alpha=0.08,
            color="#888888",
            zorder=0,
            label=f"micro + singletons ({tail_clusters})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="#888888",
        linewidth=1.2,
        label="perfect equality",
    )
    ax.plot(
        cumulative_groups,
        cumulative_sizes,
        color="#534AB7",
        linewidth=2.5,
        zorder=3,
        label="Lorenz curve",
    )
    ax.fill_between(
        cumulative_groups,
        cumulative_sizes,
        cumulative_groups,
        alpha=0.13,
        color="#534AB7",
        zorder=2,
    )

    gini_label = f"Gini = {report.gini:.3f}"
    if report.gini_std > 0:
        gini_label += f"\n±{report.gini_std:.3f} (seeds)"
    ax.text(
        0.68,
        0.14,
        gini_label,
        fontsize=8.5,
        color="#534AB7",
        fontweight="medium",
        transform=ax.transAxes,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("fraction of clusters  (sparse → dense)", fontsize=8)
    ax.set_ylabel("fraction of total embeddings", fontsize=8)
    ax.set_title("3 · Lorenz curve", fontsize=10, fontweight="medium", pad=6)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


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
                total=len(k_values) + 1,
            )
            knn_results = compute_knn_similarity_coefficient(
                embeddings,
                k_values=k_values,
                progress=progress,
                task_id=task_id,
            )
            imbalance_report = compute_embedding_imbalance_report(embeddings)
            progress.advance(task_id)

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
        imbalance_plot_path = PLOT_OUTPUT_DIR / (
            f"{parquet_file.stem}__{sanitize_filename(column)}__imbalance.pdf"
        )
        create_imbalance_plot_figure(
            report=imbalance_report,
            output_path=imbalance_plot_path,
            title=f"{column} ({parquet_file.stem}) semantic imbalance",
        )
        plot_paths["imbalance"] = str(imbalance_plot_path)

        results[column] = {
            "num_rows": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "dropped_null_embeddings": dropped_nulls,
            "knn_density": knn_results["summary"],
            "imbalance": {
                "n_clusters": imbalance_report.n_clusters,
                "n_primary_clusters": imbalance_report.n_primary_clusters,
                "n_micro_clusters": imbalance_report.n_micro_clusters,
                "n_singletons": imbalance_report.n_singletons,
                "raw_noise_fraction": imbalance_report.raw_noise_fraction,
                "gini": imbalance_report.gini,
                "gini_std": imbalance_report.gini_std,
                "lir": imbalance_report.lir,
                "alpha": imbalance_report.alpha,
                "largest_share": imbalance_report.largest_share,
            },
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
        print(f"  imbalance_plot_path: {stats['plot_paths']['imbalance']}")
        print(f"  imbalance_n_clusters: {stats['imbalance']['n_clusters']}")
        print(f"  imbalance_n_primary_clusters: {stats['imbalance']['n_primary_clusters']}")
        print(f"  imbalance_n_micro_clusters: {stats['imbalance']['n_micro_clusters']}")
        print(f"  imbalance_n_singletons: {stats['imbalance']['n_singletons']}")
        print(f"  imbalance_raw_noise_fraction: {stats['imbalance']['raw_noise_fraction']:.6f}")
        print(f"  imbalance_gini: {stats['imbalance']['gini']:.6f}")
        print(f"  imbalance_gini_std: {stats['imbalance']['gini_std']:.6f}")
        print(f"  imbalance_lir: {stats['imbalance']['lir']:.6f}")
        print(f"  imbalance_alpha: {stats['imbalance']['alpha']:.6f}")
        print(f"  imbalance_largest_share: {stats['imbalance']['largest_share']:.6f}")

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
