from __future__ import annotations

# Example usage:
# python analyze_data.py amazon-reviews/processed/Arts_Crafts_and_Sewing__raw_5core/products_filtered_with_embeddings.parquet \
#   --columns=product_title_embeddings_qwen_qwen3_embedding_0_6b,description_json_embeddings_qwen_qwen3_embedding_0_6b
#
# python analyze_data.py amazon-reviews/processed/Arts_Crafts_and_Sewing__raw_5core/reviews_filtered_with_embeddings.parquet \
#   --columns=review_title_embeddings_qwen_qwen3_embedding_0_6b,review_text_embeddings_qwen_qwen3_embedding_0_6b
#
import argparse
from contextlib import contextmanager
import json
from time import perf_counter
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import faiss
    import hdbscan
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
    from scipy.sparse.csgraph import connected_components
    from scipy.stats import kurtosis, linregress, skew, entropy
    from sklearn.decomposition import PCA
    import umap.umap_ as umap
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency: faiss-cpu, hdbscan, matplotlib, numpy, pyarrow, rich, "
        "scipy, scikit-learn, and umap-learn are required to analyze "
        "embedding columns. Install project dependencies again or install "
        "the missing packages manually."
    ) from error


DEFAULT_K_VALUES = (1, 5, 10, 100)
PLOT_OUTPUT_DIR = Path("results") / "plots" / "dataset_analysis"
CACHE_OUTPUT_DIR = Path("data") / "amazon-reviews" / "cache" / "dataset_analysis"
IMBALANCE_UMAP_COMPONENTS = 20
IMBALANCE_UMAP_NEIGHBORS = 15
IMBALANCE_HDBSCAN_MIN_SAMPLES = 5
IMBALANCE_STABILITY_RUNS = 3
IMBALANCE_HALO_EPSILON = 0.15  # Absolute Euclidean distance in UMAP space for halo absorption
console = Console()


@contextmanager
def log_step(message: str):
    """Log the start and end of a potentially expensive step."""

    start = perf_counter()
    console.log(f"[bold cyan]START[/bold cyan] {message}")
    try:
        yield
    except Exception:
        elapsed = perf_counter() - start
        console.log(f"[bold red]FAILED[/bold red] {message} after {elapsed:.1f}s")
        raise
    elapsed = perf_counter() - start
    console.log(f"[bold green]DONE[/bold green] {message} in {elapsed:.1f}s")


def format_shape(array: np.ndarray) -> str:
    """Format an array shape for progress logs."""

    return "x".join(str(dimension) for dimension in array.shape)


def log_array_issue(name: str, values: np.ndarray) -> None:
    """Log compact diagnostics for an array with invalid values."""

    array = np.asarray(values)
    finite_mask = np.isfinite(array)
    finite_values = array[finite_mask]
    nan_count = int(np.isnan(array).sum())
    posinf_count = int(np.isposinf(array).sum())
    neginf_count = int(np.isneginf(array).sum())
    finite_min = float(finite_values.min()) if finite_values.size else None
    finite_max = float(finite_values.max()) if finite_values.size else None
    console.log(
        f"{name}: shape={format_shape(array)} finite={int(finite_mask.sum()):,}/"
        f"{array.size:,} nan={nan_count:,} +inf={posinf_count:,} "
        f"-inf={neginf_count:,} finite_min={finite_min} finite_max={finite_max}"
    )


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
    shannon_entropy: float
    normalized_entropy: float
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
    finite_mask = np.isfinite(adjusted_values)
    if not finite_mask.all():
        log_array_issue("Non-finite values passed to gini; dropping them", adjusted_values)
        adjusted_values = adjusted_values[finite_mask]
    if adjusted_values.size == 0:
        return 0.0

    min_value = adjusted_values.min()
    if min_value < 0:
        with np.errstate(over="ignore", invalid="ignore"):
            adjusted_values = adjusted_values - min_value
        finite_mask = np.isfinite(adjusted_values)
        if not finite_mask.all():
            log_array_issue("Non-finite values after gini shift; dropping them", adjusted_values)
            adjusted_values = adjusted_values[finite_mask]
        if adjusted_values.size == 0:
            return 0.0

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

    matrix = np.asarray(embeddings, dtype=np.float64)
    if not np.isfinite(matrix).all():
        log_array_issue("Non-finite embeddings before normalization", matrix)
        raise ValueError("Embeddings contain NaN/Inf values; cannot normalize safely.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        log_array_issue("Non-finite embedding norms before normalization", norms)
        raise ValueError("Embedding norms contain NaN/Inf values; cannot normalize safely.")

    zero_norm_rows = np.where(norms.reshape(-1) == 0)[0]
    if len(zero_norm_rows) > 0:
        preview = ", ".join(str(index) for index in zero_norm_rows[:10])
        raise ValueError(
            "Embeddings contain zero vectors; cannot normalize. "
            f"zero_vector_rows={len(zero_norm_rows):,} first_rows=[{preview}]"
        )
    return matrix / norms


def prepare_faiss_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 matrix for FAISS."""

    return np.ascontiguousarray(embeddings, dtype=np.float32)


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


import warnings
import numpy as np
import umap.umap_ as umap

try:
    import faiss
except ImportError:
    faiss = None

def compute_umap_embedding(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float = 0.0,
    max_fit_samples: int = 50000,
    sampling_seed: int | None = None,
) -> np.ndarray:
    """
    Compute UMAP embedding, automatically scaling to millions of rows 
    via FAISS Stratified Coreset Sampling if N > max_fit_samples.
    """
    n_samples, n_dims = embeddings.shape

    # ---------------------------------------------------------
    # Base Case: Dataset is small enough for standard UMAP
    # ---------------------------------------------------------
    if n_samples <= max_fit_samples:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            min_dist=min_dist,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return reducer.fit_transform(embeddings)

    # ---------------------------------------------------------
    # Scalable Case: FAISS Stratified Coreset Subsampling
    # ---------------------------------------------------------
    if faiss is None:
        raise ImportError(
            "The 'faiss' library is required to scale UMAP beyond 50,000 points. "
            "Please run `pip install faiss-cpu` or `faiss-gpu`."
        )

    # We want ~50,000 points. K-Means with k=10,000 taking the top 5 points 
    # per centroid is a highly optimal balance of speed vs. topological coverage.
    k_centroids = min(10000, max_fit_samples)
    points_per_centroid = max(1, max_fit_samples // k_centroids)

    # UMAP uses Cosine similarity, so we must L2-normalize the data purely 
    # for the FAISS sampling step to ensure the geometry matches.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Prevent division by zero
    data_normalized = (embeddings / norms).astype(np.float32)

    # 1. Train Spherical K-Means to find topological anchors
    kmeans_kwargs = {
        "d": n_dims,
        "k": k_centroids,
        "niter": 20,
        "verbose": False,
        "spherical": True,  # Optimizes centroids for Cosine/Inner Product space
    }
    if sampling_seed is not None:
        kmeans_kwargs["seed"] = sampling_seed
    kmeans = faiss.Kmeans(**kmeans_kwargs)
    kmeans.train(data_normalized)

    # 2. Find the original embeddings closest to these topological anchors
    index = faiss.IndexFlatIP(n_dims) # Exact Inner Product
    index.add(data_normalized)
    _, indices = index.search(kmeans.centroids, points_per_centroid)

    # Flatten and deduplicate indices (in case centroids overlap neighbors)
    sample_indices = np.unique(indices.flatten())

    # 3. Fit UMAP ONLY on the topologically representative sample
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        min_dist=min_dist,
        verbose=False,
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Learn the manifold skeleton
        reducer.fit(embeddings[sample_indices])
        
        # Transform the full 1M+ dataset into the learned space in O(N) time
        return reducer.transform(embeddings)


def compute_cluster_sizes(labels: np.ndarray) -> np.ndarray:
    """Return descending cluster sizes."""

    if len(labels) == 0 or labels.max() < 0:
        return np.array([1], dtype=np.int64)
    counts = np.bincount(labels[labels >= 0])
    return np.sort(counts[counts > 0])[::-1]


def resolve_noise_labels(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    epsilon: float = IMBALANCE_HALO_EPSILON,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Resolve HDBSCAN noise into halos (absorbed) and singletons."""

    resolved_labels = labels.copy()
    label_tiers = np.where(labels >= 0, np.int8(0), np.int8(-1))

    noise_indices = np.where(resolved_labels == -1)[0]
    clustered_indices = np.where(resolved_labels >= 0)[0]

    n_noise = len(noise_indices)
    if n_noise == 0:
        console.log("No HDBSCAN noise points to resolve")
        return resolved_labels, label_tiers, 0, 0

    console.log(
        "Resolving HDBSCAN noise "
        f"noise_points={n_noise:,} clustered_points={len(clustered_indices):,} "
        f"epsilon={epsilon}"
    )

    # 1. HALO ASSIGNMENT: Map noise to nearest primary cluster
    if len(clustered_indices) > 0:
        with log_step(
            "FAISS L2 halo assignment "
            f"clustered={len(clustered_indices):,} noise={n_noise:,} "
            f"dims={reduced_embeddings.shape[1]}"
        ):
            clustered_embeddings = prepare_faiss_matrix(reduced_embeddings[clustered_indices])
            noise_embeddings = prepare_faiss_matrix(reduced_embeddings[noise_indices])
            index = faiss.IndexFlatL2(clustered_embeddings.shape[1])
            index.add(clustered_embeddings)

            squared_distances, indices = index.search(noise_embeddings, 1)
            distances = np.sqrt(squared_distances)

        for i, (dist, neighbor_idx) in enumerate(zip(distances.flatten(), indices.flatten())):
            if dist <= epsilon:
                # Inherit the label of the closest core point
                target_original_idx = clustered_indices[neighbor_idx]
                resolved_labels[noise_indices[i]] = resolved_labels[target_original_idx]
                label_tiers[noise_indices[i]] = 0  # Absorbed into primary cluster (Tier 0)

    # 2. SINGLETON DECLARATION
    remaining_noise_indices = np.where(resolved_labels == -1)[0]
    n_singletons = len(remaining_noise_indices)
    console.log(
        "Noise resolution summary "
        f"absorbed={n_noise - n_singletons:,} singletons={n_singletons:,}"
    )

    next_cluster_id = int(resolved_labels.max()) + 1 if len(resolved_labels) > 0 else 0
    for idx in remaining_noise_indices:
        resolved_labels[idx] = next_cluster_id
        label_tiers[idx] = 2  # Tier 2 = Singleton
        next_cluster_id += 1

    # In a strict database selectivity model, there are no "micro-clusters" bridging unrelated singletons
    n_micro_clusters = 0

    return resolved_labels, label_tiers, n_micro_clusters, n_singletons


def compute_embedding_imbalance_report(
    embeddings: np.ndarray,
    sampling_seed: int = 42,
    n_stability_runs: int = IMBALANCE_STABILITY_RUNS,
) -> ImbalanceReport:
    """Measure semantic class imbalance via UMAP + HDBSCAN."""

    matrix = validate_embedding_matrix(embeddings)
    n_samples = len(matrix)
    min_cluster_size = max(10, int(0.01 * n_samples))
    console.log(
        "Imbalance analysis setup "
        f"shape={format_shape(matrix)} min_cluster_size={min_cluster_size} "
        f"stability_runs={n_stability_runs}"
    )

    # UMAP natively handles the cosine metric geometry without pre-normalization
    clustering_embedding = compute_umap_embedding(
        matrix,
        n_components=IMBALANCE_UMAP_COMPONENTS,
        n_neighbors=IMBALANCE_UMAP_NEIGHBORS,
        sampling_seed=sampling_seed,
    )

    with log_step(
        "HDBSCAN fit_predict primary "
        f"shape={format_shape(clustering_embedding)} "
        f"min_cluster_size={min_cluster_size} "
        f"min_samples={IMBALANCE_HDBSCAN_MIN_SAMPLES}"
    ):
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

    # Calculate Shannon Entropy for skew tracking
    probabilities = cluster_sizes / cluster_sizes.sum()
    shannon_ent = float(entropy(probabilities, base=2))
    max_ent = np.log2(n_samples) if n_samples > 0 else 0.0
    normalized_ent = shannon_ent / max_ent if max_ent > 0 else 0.0

    visualization_embedding = compute_umap_embedding(
        matrix,
        n_components=2,
        n_neighbors=IMBALANCE_UMAP_NEIGHBORS,
        min_dist=0.1,
        sampling_seed=sampling_seed,
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
        shannon_entropy=shannon_ent,
        normalized_entropy=normalized_ent,
        gini=float(gini(cluster_sizes)),
        lir=float(cluster_sizes.max() / cluster_sizes.mean()) if len(cluster_sizes) > 0 else 0.0,
        alpha=float(zipf_alpha(cluster_sizes)),
        largest_share=float(cluster_sizes.max() / n_samples) if n_samples > 0 else 0.0,
    )

    for seed in range(sampling_seed + 1, sampling_seed + 1 + n_stability_runs):
        console.log(f"Starting imbalance stability run sampling_seed={seed}")
        seeded_embedding = compute_umap_embedding(
            matrix,
            n_components=IMBALANCE_UMAP_COMPONENTS,
            n_neighbors=IMBALANCE_UMAP_NEIGHBORS,
            sampling_seed=seed,
        )
        with log_step(
            "HDBSCAN fit_predict stability "
            f"sampling_seed={seed} shape={format_shape(seeded_embedding)} "
            f"min_cluster_size={min_cluster_size}"
        ):
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

    n_rows, n_dims = embeddings.shape
    if n_rows < 2:
        raise ValueError("At least two embeddings are required for kNN analysis.")

    sorted_k_values = tuple(sorted(k_values))
    for k in sorted_k_values:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")
    max_k = max(sorted_k_values)
    if max_k >= n_rows:
        raise ValueError(
            f"k={max_k} is invalid for {n_rows} embeddings; k must be smaller "
            "than the number of rows."
        )

    console.log(
        "kNN similarity setup "
        f"shape={format_shape(embeddings)} k_values={','.join(str(k) for k in k_values)} "
        f"normalize={normalize}"
    )

    with log_step(f"Validate embedding matrix before FAISS kNN shape={format_shape(embeddings)}"):
        embeddings = validate_embedding_matrix(embeddings)
        n_rows, n_dims = embeddings.shape
        if n_dims == 0:
            raise ValueError("No embedding dimensions remain after validation.")

    if normalize:
        with log_step(f"Normalize embeddings for FAISS kNN shape={format_shape(embeddings)}"):
            embeddings = normalize_embeddings(embeddings)
    with log_step(f"Convert embeddings for FAISS shape={format_shape(embeddings)}"):
        embeddings = prepare_faiss_matrix(embeddings)

    summary: dict[int, dict[str, float]] = {}
    pointwise_min_similarity: dict[int, np.ndarray] = {}

    # 1. Use HNSW for O(N log N) Approximate Search instead of O(N^2) Exact Search
    # 32 represents the number of bi-directional links in the graph. 
    # It provides an excellent balance of >95% recall vs speed.
    ef_search = min(n_rows, max(128, 4 * (max_k + 1)))
    ef_construction = max(40, min(512, ef_search))
    with log_step(
        "Build FAISS IndexHNSWFlat "
        f"rows={n_rows:,} dims={n_dims} efConstruction={ef_construction} "
        f"efSearch={ef_search}"
    ):
        index = faiss.IndexHNSWFlat(n_dims, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(embeddings)

    # 2. SINGLE SEARCH: Search once for the maximum K needed
    with log_step(f"FAISS IndexHNSWFlat search max_k={max_k} rows={n_rows:,}"):
        all_distances, all_indices = index.search(embeddings, max_k + 1)

    neighbor_distances = all_distances[:, 1:]
    neighbor_indices = all_indices[:, 1:]
    invalid_neighbor_mask = (
        (neighbor_indices < 0)
        | ~np.isfinite(neighbor_distances)
        | (neighbor_distances < -1.0001)
        | (neighbor_distances > 1.0001)
    )
    invalid_rows = np.flatnonzero(invalid_neighbor_mask.any(axis=1))
    if len(invalid_rows) > 0:
        sentinel_count = int((neighbor_distances <= -1e20).sum())
        console.log(
            "Invalid HNSW kNN results detected; falling back to exact search "
            f"invalid_rows={len(invalid_rows):,}/{n_rows:,} "
            f"invalid_entries={int(invalid_neighbor_mask.sum()):,} "
            f"sentinel_like_distances={sentinel_count:,}"
        )
        log_array_issue("Invalid HNSW neighbor distances", neighbor_distances[invalid_rows])
        with log_step(
            "Exact FAISS IndexFlatIP fallback for invalid HNSW rows "
            f"rows={len(invalid_rows):,} max_k={max_k}"
        ):
            exact_index = faiss.IndexFlatIP(n_dims)
            exact_index.add(embeddings)
            fallback_distances, fallback_indices = exact_index.search(
                embeddings[invalid_rows],
                max_k + 1,
            )
        all_distances[invalid_rows] = fallback_distances
        all_indices[invalid_rows] = fallback_indices

        neighbor_distances = all_distances[:, 1:]
        neighbor_indices = all_indices[:, 1:]
        invalid_neighbor_mask = (
            (neighbor_indices < 0)
            | ~np.isfinite(neighbor_distances)
            | (neighbor_distances < -1.0001)
            | (neighbor_distances > 1.0001)
        )
        if invalid_neighbor_mask.any():
            log_array_issue("Invalid exact fallback neighbor distances", neighbor_distances)
            raise ValueError(
                "FAISS kNN returned invalid neighbor distances even after exact fallback. "
                "Check the embedding matrix for corrupt rows or incompatible values."
            )

    # 3. Slice the results for specific k values
    for k in sorted_k_values:
        with log_step(f"Compute kNN summary statistics k={k}"):
            # Slice the distance matrix up to the current k (ignoring index 0, which is self)
            k_distances = all_distances[:, 1:k + 1].astype(np.float64)
            if not np.isfinite(k_distances).all():
                log_array_issue(f"Non-finite cosine similarities before summary k={k}", k_distances)
                raise ValueError(f"Non-finite cosine similarities detected for k={k}.")
            if ((k_distances < -1.0001) | (k_distances > 1.0001)).any():
                log_array_issue(f"Out-of-range cosine similarities before summary k={k}", k_distances)
                raise ValueError(f"Out-of-range cosine similarities detected for k={k}.")
            cosine_similarities = np.clip(k_distances, -1.0, 1.0)
            
            pointwise_min_similarity[k] = cosine_similarities.min(axis=1)

            mean_similarity = cosine_similarities.mean(axis=1)
            if not np.isfinite(mean_similarity).all():
                log_array_issue(f"Non-finite mean kNN similarities k={k}", mean_similarity)
                raise ValueError(f"Non-finite mean kNN similarities detected for k={k}.")
            mean_similarity_mean = float(mean_similarity.mean())
            similarity_cv = (
                0.0
                if np.isclose(mean_similarity_mean, 0.0)
                else float(mean_similarity.std() / mean_similarity_mean)
            )

            summary[k] = {
                "similarity_skewness": float(skew(mean_similarity)),
                "similarity_kurtosis": float(kurtosis(mean_similarity)),
                "similarity_cv": similarity_cv,
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

    with log_step(f"Read parquet column column={column} file={parquet_file}"):
        table = pq.read_table(parquet_file, columns=[column])
    values = table.column(column)

    with log_step(f"Materialize Arrow column to Python list column={column}"):
        embeddings = values.to_pylist()
    with log_step(f"Filter null embeddings column={column} rows={len(embeddings):,}"):
        filtered_embeddings = [embedding for embedding in embeddings if embedding is not None]
    dropped_nulls = len(embeddings) - len(filtered_embeddings)

    if not filtered_embeddings:
        raise ValueError(
            f"Column '{column}' contains no non-null embeddings after filtering."
        )

    with log_step(f"Convert embeddings to NumPy column={column} rows={len(filtered_embeddings):,}"):
        matrix = np.asarray(filtered_embeddings, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"Column '{column}' does not form a 2D embedding matrix; got "
            f"shape {matrix.shape}."
        )

    console.log(
        f"Loaded embedding column column={column} shape={format_shape(matrix)} "
        f"dropped_nulls={dropped_nulls:,}"
    )
    return matrix, dropped_nulls


def sanitize_filename(name: str) -> str:
    """Convert a column name into a filesystem-safe filename stem."""

    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def get_cache_filepaths(parquet_file: Path, column: str) -> tuple[Path, Path]:
    """Return metadata and array cache paths for one parquet/column pair."""

    stem = f"{parquet_file.stem}__{sanitize_filename(column)}"
    return (
        CACHE_OUTPUT_DIR / f"{stem}.json",
        CACHE_OUTPUT_DIR / f"{stem}.npz",
    )


def create_umap_scatter_plot_figure(
    umap_projection: np.ndarray,
    point_colors_by_k: dict[int, np.ndarray],
    output_path: Path,
    title: str,
    k_values: tuple[int, ...],
) -> None:
    """Create horizontally packed UMAP scatter subplots for all k values."""

    with log_step(
        "Create UMAP scatter plot "
        f"points={len(umap_projection):,} k_values={','.join(str(k) for k in k_values)} "
        f"output={output_path}"
    ):
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

    with log_step(
        "Create sorted kNN similarity plot "
        f"k_values={','.join(str(k) for k in k_values)} output={output_path}"
    ):
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

    with log_step(
        "Create imbalance plot "
        f"samples={report.n_samples:,} clusters={report.n_clusters:,} output={output_path}"
    ):
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

    # PCA is linear, so pre-normalization is kept here to simulate cosine distances
    with log_step(f"Normalize embeddings before PCA+UMAP shape={format_shape(embeddings)}"):
        normalized_embeddings = normalize_embeddings(embeddings)

    n_rows, n_dims = normalized_embeddings.shape
    n_pca_components = min(50, n_dims, n_rows)
    with log_step(
        "PCA fit_transform before UMAP "
        f"shape={format_shape(normalized_embeddings)} components={n_pca_components}"
    ):
        reduced_embeddings = PCA(n_components=n_pca_components).fit_transform(
            normalized_embeddings
        )

    n_neighbors = min(15, max(2, n_rows - 1))
    with log_step(
        "UMAP fit_transform PCA projection "
        f"shape={format_shape(reduced_embeddings)} components=2 neighbors={n_neighbors}"
    ):
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
        ).fit_transform(reduced_embeddings)


def compute_pca_embedding(embeddings: np.ndarray) -> np.ndarray:
    """Compute the cached PCA representation used before UMAP."""

    with log_step(f"Normalize embeddings before PCA shape={format_shape(embeddings)}"):
        normalized_embeddings = normalize_embeddings(embeddings)
    n_rows, n_dims = normalized_embeddings.shape
    n_pca_components = min(50, n_dims, n_rows)
    with log_step(
        "PCA fit_transform cache projection "
        f"shape={format_shape(normalized_embeddings)} components={n_pca_components}"
    ):
        return PCA(n_components=n_pca_components).fit_transform(normalized_embeddings)


def prepare_column_cache(
    parquet_file: Path,
    column: str,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Compute expensive analysis artifacts once and write them to cache."""

    metadata_path, arrays_path = get_cache_filepaths(parquet_file, column)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    console.log(f"Preparing cache for column={column}")
    embeddings, dropped_nulls = load_embedding_column(parquet_file, column)
    with log_step(f"Compute kNN density column={column}"):
        knn_results = compute_knn_similarity_coefficient(embeddings, k_values=k_values)
    with log_step(f"Compute PCA cache projection column={column}"):
        pca_projection = compute_pca_embedding(embeddings)
    with log_step(
        "UMAP fit_transform cache visualization "
        f"column={column} shape={format_shape(pca_projection)}"
    ):
        umap_projection = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, len(pca_projection) - 1)),
        ).fit_transform(pca_projection)
    with log_step(f"Compute semantic imbalance report column={column}"):
        imbalance_report = compute_embedding_imbalance_report(embeddings)

    with log_step(f"Write compressed NumPy cache arrays output={arrays_path}"):
        np.savez_compressed(
            arrays_path,
            pca_projection=pca_projection.astype(np.float32),
            umap_projection=umap_projection.astype(np.float32),
            labels=imbalance_report.labels.astype(np.int64),
            label_tiers=imbalance_report.label_tiers.astype(np.int8),
            cluster_sizes=imbalance_report.cluster_sizes.astype(np.int64),
            **{
                f"pointwise_min_similarity_k{k}": knn_results["pointwise_min_similarity"][k].astype(np.float32)
                for k in k_values
            },
        )

    metadata = {
        "parquet_file": str(parquet_file),
        "column": column,
        "k_values": list(k_values),
        "num_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "dropped_null_embeddings": int(dropped_nulls),
        "knn_density": knn_results["summary"],
        "imbalance": {
            "n_clusters": imbalance_report.n_clusters,
            "n_primary_clusters": imbalance_report.n_primary_clusters,
            "n_micro_clusters": imbalance_report.n_micro_clusters,
            "n_singletons": imbalance_report.n_singletons,
            "raw_noise_fraction": imbalance_report.raw_noise_fraction,
            "shannon_entropy": imbalance_report.shannon_entropy,
            "normalized_entropy": imbalance_report.normalized_entropy,
            "gini": imbalance_report.gini,
            "gini_std": imbalance_report.gini_std,
            "lir": imbalance_report.lir,
            "alpha": imbalance_report.alpha,
            "largest_share": imbalance_report.largest_share,
        },
    }
    with log_step(f"Write cache metadata output={metadata_path}"):
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return load_cached_column_analysis(parquet_file, column)


def load_cached_column_analysis(parquet_file: Path, column: str) -> dict[str, Any]:
    """Load cached arrays and metadata for one parquet/column pair."""

    metadata_path, arrays_path = get_cache_filepaths(parquet_file, column)
    if not metadata_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(
            f"Missing cache for column '{column}'. Expected {metadata_path} and {arrays_path}."
        )

    with log_step(f"Read cache metadata path={metadata_path}"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with log_step(f"Open cache arrays path={arrays_path}"):
        arrays = np.load(arrays_path)
    k_values = tuple(int(k) for k in metadata["k_values"])
    with log_step(f"Load pointwise kNN arrays k_values={','.join(str(k) for k in k_values)}"):
        pointwise_min_similarity = {
            k: arrays[f"pointwise_min_similarity_k{k}"]
            for k in k_values
        }

    imbalance = metadata["imbalance"]
    cluster_sizes = arrays["cluster_sizes"]
    probabilities = cluster_sizes / cluster_sizes.sum()
    cached_shannon_entropy = float(imbalance.get("shannon_entropy", entropy(probabilities, base=2)))
    cached_normalized_entropy = float(
        imbalance.get(
            "normalized_entropy",
            cached_shannon_entropy / np.log2(int(metadata["num_rows"]))
            if int(metadata["num_rows"]) > 1
            else 0.0,
        )
    )
    imbalance_report = ImbalanceReport(
        n_samples=int(metadata["num_rows"]),
        n_clusters=int(imbalance["n_clusters"]),
        n_primary_clusters=int(imbalance["n_primary_clusters"]),
        n_micro_clusters=int(imbalance["n_micro_clusters"]),
        n_singletons=int(imbalance["n_singletons"]),
        raw_noise_fraction=float(imbalance["raw_noise_fraction"]),
        cluster_sizes=cluster_sizes,
        labels=arrays["labels"],
        label_tiers=arrays["label_tiers"],
        embedding_2d=arrays["umap_projection"],
        shannon_entropy=cached_shannon_entropy,
        normalized_entropy=cached_normalized_entropy,
        gini=float(imbalance["gini"]),
        lir=float(imbalance["lir"]),
        alpha=float(imbalance["alpha"]),
        largest_share=float(imbalance["largest_share"]),
        gini_per_seed=[],
        n_clusters_per_seed=[],
    )

    return {
        "num_rows": int(metadata["num_rows"]),
        "embedding_dim": int(metadata["embedding_dim"]),
        "dropped_null_embeddings": int(metadata["dropped_null_embeddings"]),
        "k_values": k_values,
        "knn_density": {int(k): v for k, v in metadata["knn_density"].items()},
        "pointwise_min_similarity": pointwise_min_similarity,
        "pca_projection": arrays["pca_projection"],
        "umap_projection": arrays["umap_projection"],
        "imbalance_report": imbalance_report,
    }


def analyze_columns(
    parquet_file: Path,
    columns: list[str],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, dict[str, Any]]:
    """Prepare cache if needed, then create plots from cached analysis."""

    console.log(
        f"Starting dataset analysis file={parquet_file} "
        f"columns={','.join(columns)} k_values={','.join(str(k) for k in k_values)}"
    )
    with log_step(f"Validate parquet embedding columns file={parquet_file}"):
        validate_embedding_columns(parquet_file, columns)

    results: dict[str, dict[str, Any]] = {}
    for column_index, column in enumerate(columns, start=1):
        console.log(f"Analyzing column {column_index}/{len(columns)} column={column}")
        metadata_path, arrays_path = get_cache_filepaths(parquet_file, column)
        if metadata_path.exists() and arrays_path.exists():
            console.log(
                f"Cache hit column={column} metadata={metadata_path} arrays={arrays_path}"
            )
            with log_step(f"Load cached analysis column={column}"):
                cached = load_cached_column_analysis(parquet_file, column)
        else:
            console.log(
                f"Cache miss column={column} metadata={metadata_path} arrays={arrays_path}"
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    f"Preparing cache for [cyan]{column}[/cyan]",
                    total=1,
                )
                cached = prepare_column_cache(
                    parquet_file=parquet_file,
                    column=column,
                    k_values=k_values,
                )
                progress.advance(task_id)

        plot_paths: dict[str, str] = {}
        scatter_plot_path = PLOT_OUTPUT_DIR / (
            f"{parquet_file.stem}__{sanitize_filename(column)}__scatter.pdf"
        )
        create_umap_scatter_plot_figure(
            umap_projection=cached["umap_projection"],
            point_colors_by_k=cached["pointwise_min_similarity"],
            output_path=scatter_plot_path,
            title=f"{column} ({parquet_file.stem})",
            k_values=cached["k_values"],
        )
        sorted_plot_path = PLOT_OUTPUT_DIR / (
            f"{parquet_file.stem}__{sanitize_filename(column)}__sorted.pdf"
        )
        create_sorted_knn_similarity_plot_figure(
            point_similarities_by_k=cached["pointwise_min_similarity"],
            output_path=sorted_plot_path,
            title=f"{column} ({parquet_file.stem}) sorted similarities",
            k_values=cached["k_values"],
        )
        plot_paths["scatter"] = str(scatter_plot_path)
        plot_paths["sorted"] = str(sorted_plot_path)
        imbalance_plot_path = PLOT_OUTPUT_DIR / (
            f"{parquet_file.stem}__{sanitize_filename(column)}__imbalance.pdf"
        )
        create_imbalance_plot_figure(
            report=cached["imbalance_report"],
            output_path=imbalance_plot_path,
            title=f"{column} ({parquet_file.stem}) semantic imbalance",
        )
        plot_paths["imbalance"] = str(imbalance_plot_path)

        results[column] = {
            "num_rows": cached["num_rows"],
            "embedding_dim": cached["embedding_dim"],
            "dropped_null_embeddings": cached["dropped_null_embeddings"],
            "cache_metadata_path": str(metadata_path),
            "cache_arrays_path": str(arrays_path),
            "knn_density": cached["knn_density"],
            "imbalance": {
                "n_clusters": cached["imbalance_report"].n_clusters,
                "n_primary_clusters": cached["imbalance_report"].n_primary_clusters,
                "n_micro_clusters": cached["imbalance_report"].n_micro_clusters,
                "n_singletons": cached["imbalance_report"].n_singletons,
                "raw_noise_fraction": cached["imbalance_report"].raw_noise_fraction,
                "shannon_entropy": cached["imbalance_report"].shannon_entropy,
                "normalized_entropy": cached["imbalance_report"].normalized_entropy,
                "gini": cached["imbalance_report"].gini,
                "gini_std": cached["imbalance_report"].gini_std,
                "lir": cached["imbalance_report"].lir,
                "alpha": cached["imbalance_report"].alpha,
                "largest_share": cached["imbalance_report"].largest_share,
            },
            "plot_paths": plot_paths,
        }
        console.log(f"Finished column {column_index}/{len(columns)} column={column}")

    return results


def print_results(results: dict[str, dict[str, Any]]) -> None:
    """Print column analysis results."""

    for column, stats in results.items():
        print(f"Column: {column}")
        print(f"  rows: {stats['num_rows']}")
        print(f"  embedding_dim: {stats['embedding_dim']}")
        print(f"  dropped_null_embeddings: {stats['dropped_null_embeddings']}")
        print(f"  cache_metadata_path: {stats['cache_metadata_path']}")
        print(f"  cache_arrays_path: {stats['cache_arrays_path']}")
        print(f"  scatter_plot_path: {stats['plot_paths']['scatter']}")
        print(f"  sorted_plot_path: {stats['plot_paths']['sorted']}")
        print(f"  imbalance_plot_path: {stats['plot_paths']['imbalance']}")
        print(f"  imbalance_n_clusters: {stats['imbalance']['n_clusters']}")
        print(f"  imbalance_n_primary_clusters: {stats['imbalance']['n_primary_clusters']}")
        print(f"  imbalance_n_micro_clusters: {stats['imbalance']['n_micro_clusters']}")
        print(f"  imbalance_n_singletons: {stats['imbalance']['n_singletons']}")
        print(f"  imbalance_raw_noise_fraction: {stats['imbalance']['raw_noise_fraction']:.6f}")
        print(f"  imbalance_shannon_entropy: {stats['imbalance']['shannon_entropy']:.6f}")
        print(f"  imbalance_normalized_entropy: {stats['imbalance']['normalized_entropy']:.6f}")
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
