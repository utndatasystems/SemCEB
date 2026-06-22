from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import duckdb
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from semceb.reporting.plot_params import apply_plot_params
from semceb.utils.console import console

DEFAULT_K_VALUES = (1, 5, 10, 100)
DATA_SKEW_CACHE_VERSION = 13
DATA_SKEW_UMAP_COMPONENTS = 30
DATA_SKEW_UMAP_NEIGHBORS = 8
DATA_SKEW_HDBSCAN_MIN_CLUSTER_FRACTION = 0.0015
DATA_SKEW_HDBSCAN_MIN_CLUSTER_SIZE = 8
DATA_SKEW_HDBSCAN_MIN_SAMPLES = 3
DATA_SKEW_HDBSCAN_CLUSTER_SELECTION_METHOD = "leaf"
DATA_SKEW_CLUSTER_ASSIGNMENT_SAMPLE_SIZE = 15_000
DATA_SKEW_MICRO_CLUSTER_FRACTION = 0.0003
DATA_SKEW_MICRO_CLUSTER_MIN_SIZE = 5
DATA_SKEW_MICRO_HDBSCAN_MIN_SAMPLES = 2
DATA_SKEW_STABILITY_RUNS = 3
DATA_SKEW_HALO_EPSILON = 0.15
CLUSTER_MAP_RENDER_SAMPLE_SIZE = 10_000


def _import_hdbscan() -> Any:
    """Import HDBSCAN only when data-skew plots are generated."""

    try:
        import hdbscan
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Data skew plots require the optional dependency 'hdbscan'."
        ) from error

    return hdbscan


def _import_faiss() -> Any:
    """Import FAISS only when data-skew plots are generated."""

    try:
        import faiss
    except Exception as error:
        raise ModuleNotFoundError(
            "Data skew plots require the optional dependency 'faiss-cpu'."
        ) from error

    return faiss


def _import_umap() -> Any:
    """Import UMAP only when data-skew plots are generated."""

    try:
        import umap.umap_ as umap
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Data skew plots require the optional dependency 'umap-learn'."
        ) from error

    return umap


def _import_scipy_stats() -> tuple[Any, Any, Any, Any]:
    """Import the SciPy statistics helpers used by the skew pipeline."""

    try:
        from scipy.stats import entropy, kurtosis, linregress, skew
    except Exception as error:
        raise ModuleNotFoundError(
            "Data skew plots require the optional dependency 'scipy'."
        ) from error

    return entropy, kurtosis, linregress, skew


def _import_pca() -> Any:
    """Import PCA only when data-skew plots are generated."""

    try:
        from sklearn.decomposition import PCA
    except Exception as error:
        raise ModuleNotFoundError(
            "Data skew plots require the optional dependency 'scikit-learn'."
        ) from error

    return PCA


@contextmanager
def _log_step(message: str):
    """Log the duration of one expensive data-skew step."""

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


def _format_shape(array: np.ndarray) -> str:
    """Format an array shape for progress logging."""

    return "x".join(str(dimension) for dimension in array.shape)


def _log_array_issue(name: str, values: np.ndarray) -> None:
    """Log compact diagnostics for an array with invalid values."""

    array = np.asarray(values)
    finite_mask = np.isfinite(array)
    finite_values = array[finite_mask]
    finite_min = float(finite_values.min()) if finite_values.size else None
    finite_max = float(finite_values.max()) if finite_values.size else None
    console.log(
        f"{name}: shape={_format_shape(array)} "
        f"finite={int(finite_mask.sum()):,}/{array.size:,} "
        f"nan={int(np.isnan(array).sum()):,} "
        f"+inf={int(np.isposinf(array).sum()):,} "
        f"-inf={int(np.isneginf(array).sum()):,} "
        f"finite_min={finite_min} finite_max={finite_max}"
    )


def _escape_latex_text(value: str) -> str:
    """Escape text for Matplotlib's LaTeX rendering."""

    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _draw_axis_border(axis: Any) -> None:
    """Ensure all spines are visible for a boxed plot style."""

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def _cluster_map_palette(n_colors: int, offset: int = 0) -> list[Any]:
    """Return a broad categorical palette for cluster maps."""

    if n_colors <= 0:
        return []

    colors: list[Any] = []
    for colormap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired", "Dark2"):
        colormap = plt.get_cmap(colormap_name)
        if hasattr(colormap, "colors"):
            colors.extend(colormap.colors)
        else:
            colors.extend(colormap(np.linspace(0, 1, colormap.N)))

    needed_colors = offset + n_colors
    if needed_colors > len(colors):
        extra_count = needed_colors - len(colors)
        colors.extend(plt.get_cmap("hsv")(np.linspace(0, 1, extra_count, endpoint=False)))

    return colors[offset:needed_colors]


def _sanitize_filename(name: str) -> str:
    """Convert a column name into a filesystem-safe filename stem."""

    return "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in name
    )


def _sql_string_literal(value: str) -> str:
    """Return a DuckDB-safe string literal."""

    return "'" + value.replace("'", "''") + "'"


def _sql_identifier(identifier: str) -> str:
    """Return a quoted SQL identifier."""

    return '"' + identifier.replace('"', '""') + '"'


def _is_embedding_type(column_type: str) -> bool:
    """Return whether a DuckDB type stores float embedding arrays."""

    normalized_type = column_type.upper().strip()
    return (
        normalized_type.endswith("[]")
        and (
            normalized_type.startswith("FLOAT")
            or normalized_type.startswith("REAL")
            or normalized_type.startswith("DOUBLE")
        )
    )


def _load_column_schema(parquet_file: Path) -> dict[str, str]:
    """Load a parquet schema as a column-to-type mapping via DuckDB."""

    schema_df = duckdb.sql(
        "DESCRIBE SELECT * FROM read_parquet("
        f"{_sql_string_literal(str(parquet_file))})"
    ).df()

    return {
        str(row["column_name"]): str(row["column_type"])
        for _, row in schema_df.iterrows()
    }


def _embedding_source_column_name(embedding_column: str) -> str:
    """Resolve an embedding column name back to its source column."""

    return embedding_column.split("_embeddings", 1)[0]


def _infer_embedding_columns(parquet_file: Path) -> list[str]:
    """Infer embedding columns from one parquet schema."""

    schema = _load_column_schema(parquet_file)
    return [
        column_name
        for column_name, column_type in schema.items()
        if _is_embedding_type(column_type)
    ]


def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient for a value vector."""

    adjusted_values = values.astype(np.float64, copy=True)
    finite_mask = np.isfinite(adjusted_values)
    if not finite_mask.all():
        _log_array_issue("Non-finite values passed to gini; dropping them", adjusted_values)
        adjusted_values = adjusted_values[finite_mask]

    if adjusted_values.size == 0:
        return 0.0

    min_value = adjusted_values.min()
    if min_value < 0:
        adjusted_values = adjusted_values - min_value
        finite_mask = np.isfinite(adjusted_values)
        if not finite_mask.all():
            _log_array_issue("Non-finite values after gini shift; dropping them", adjusted_values)
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


def _zipf_alpha(cluster_sizes: np.ndarray) -> float:
    """Fit a Zipf exponent on descending cluster sizes."""

    _, _, linregress, _ = _import_scipy_stats()
    sizes = np.sort(cluster_sizes.astype(np.float64))[::-1]
    sizes = sizes[sizes > 0]
    if len(sizes) < 2:
        return 0.0

    log_ranks = np.log(np.arange(1, len(sizes) + 1))
    log_sizes = np.log(sizes)
    slope, *_ = linregress(log_ranks, log_sizes)
    return float(-slope)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length."""

    matrix = np.asarray(embeddings, dtype=np.float64)
    if not np.isfinite(matrix).all():
        _log_array_issue("Non-finite embeddings before normalization", matrix)
        raise ValueError("Embeddings contain NaN or Inf values.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        _log_array_issue("Non-finite embedding norms before normalization", norms)
        raise ValueError("Embedding norms contain NaN or Inf values.")

    zero_norm_rows = np.where(norms.reshape(-1) == 0)[0]
    if len(zero_norm_rows) > 0:
        preview = ", ".join(str(index) for index in zero_norm_rows[:10])
        raise ValueError(
            "Embeddings contain zero vectors; cannot normalize. "
            f"zero_vector_rows={len(zero_norm_rows):,} first_rows=[{preview}]"
        )

    return matrix / norms


def _prepare_faiss_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 matrix for FAISS."""

    return np.ascontiguousarray(embeddings, dtype=np.float32)


def _validate_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Clean one embedding matrix before downstream analysis."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {matrix.shape}.")

    if not np.isfinite(matrix).all():
        warnings.warn("NaN or Inf detected in embeddings; replacing with 0.", stacklevel=2)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    zero_variance_dims = matrix.var(axis=0) == 0
    if zero_variance_dims.any():
        warnings.warn(
            f"Dropping {int(zero_variance_dims.sum())} zero-variance dimensions.",
            stacklevel=2,
        )
        matrix = matrix[:, ~zero_variance_dims]

    if matrix.shape[1] == 0:
        raise ValueError("No embedding dimensions remain after validation.")

    return matrix


def _compute_umap_embedding(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float = 0.0,
    max_fit_samples: int = 50000,
    sampling_seed: int | None = None,
) -> np.ndarray:
    """Compute a UMAP embedding and subsample when the dataset is large."""

    faiss = _import_faiss()
    umap = _import_umap()
    n_samples, n_dims = embeddings.shape

    if n_samples <= max_fit_samples:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            min_dist=min_dist,
            verbose=False,
            random_state=sampling_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return reducer.fit_transform(embeddings)

    k_centroids = min(10000, max_fit_samples)
    points_per_centroid = max(1, max_fit_samples // k_centroids)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = (embeddings / norms).astype(np.float32)

    kmeans_kwargs = {
        "d": n_dims,
        "k": k_centroids,
        "niter": 20,
        "verbose": False,
        "spherical": True,
    }
    if sampling_seed is not None:
        kmeans_kwargs["seed"] = sampling_seed

    with _log_step(
        "FAISS coreset sampling "
        f"shape={_format_shape(normalized_embeddings)} centroids={k_centroids}"
    ):
        kmeans = faiss.Kmeans(**kmeans_kwargs)
        kmeans.train(normalized_embeddings)

        index = faiss.IndexFlatIP(n_dims)
        index.add(normalized_embeddings)
        _, indices = index.search(kmeans.centroids, points_per_centroid)
        sample_indices = np.unique(indices.flatten())

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        min_dist=min_dist,
        verbose=False,
        random_state=sampling_seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer.fit(embeddings[sample_indices])
        return reducer.transform(embeddings)


def _compute_cluster_sizes(labels: np.ndarray) -> np.ndarray:
    """Return descending cluster sizes."""

    if len(labels) == 0 or labels.max() < 0:
        return np.array([1], dtype=np.int64)

    counts = np.bincount(labels[labels >= 0])
    return np.sort(counts[counts > 0])[::-1]


def _compute_primary_cluster_min_size(n_samples: int) -> int:
    """Return a smaller HDBSCAN threshold for finer primary clusters."""

    return max(
        DATA_SKEW_HDBSCAN_MIN_CLUSTER_SIZE,
        int(DATA_SKEW_HDBSCAN_MIN_CLUSTER_FRACTION * n_samples),
    )


def _compute_micro_cluster_min_size(n_samples: int) -> int:
    """Return the HDBSCAN threshold for second-pass tail clusters."""

    return max(
        DATA_SKEW_MICRO_CLUSTER_MIN_SIZE,
        int(DATA_SKEW_MICRO_CLUSTER_FRACTION * n_samples),
    )


def _sample_embeddings_for_data_skew_analysis(
    embeddings: np.ndarray,
    sampling_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample rows before expensive data-skew analysis."""

    if len(embeddings) <= DATA_SKEW_CLUSTER_ASSIGNMENT_SAMPLE_SIZE:
        sample_indices = np.arange(len(embeddings), dtype=np.int64)
        return embeddings, sample_indices

    with _log_step(
        "Sample embeddings for data-skew analysis "
        f"rows={len(embeddings):,} sample_size={DATA_SKEW_CLUSTER_ASSIGNMENT_SAMPLE_SIZE:,}"
    ):
        sample_indices = np.random.default_rng(sampling_seed).choice(
            len(embeddings),
            size=DATA_SKEW_CLUSTER_ASSIGNMENT_SAMPLE_SIZE,
            replace=False,
        )

    sample_indices = np.sort(sample_indices.astype(np.int64, copy=False))
    return embeddings[sample_indices], sample_indices


def _resolve_noise_labels(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    epsilon: float = DATA_SKEW_HALO_EPSILON,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Resolve HDBSCAN noise into absorbed points, micro-clusters, and singletons."""

    faiss = _import_faiss()
    hdbscan = _import_hdbscan()

    resolved_labels = labels.copy()
    label_tiers = np.where(labels >= 0, np.int8(0), np.int8(-1))

    noise_indices = np.where(resolved_labels == -1)[0]
    clustered_indices = np.where(resolved_labels >= 0)[0]
    if len(noise_indices) == 0:
        return resolved_labels, label_tiers, 0, 0

    if len(clustered_indices) > 0:
        with _log_step(
            "Resolve HDBSCAN halos "
            f"clustered={len(clustered_indices):,} noise={len(noise_indices):,}"
        ):
            clustered_embeddings = _prepare_faiss_matrix(reduced_embeddings[clustered_indices])
            noise_embeddings = _prepare_faiss_matrix(reduced_embeddings[noise_indices])
            index = faiss.IndexFlatL2(clustered_embeddings.shape[1])
            index.add(clustered_embeddings)
            squared_distances, indices = index.search(noise_embeddings, 1)
            distances = np.sqrt(squared_distances)

        for offset, (distance, neighbor_idx) in enumerate(
            zip(distances.flatten(), indices.flatten(), strict=False)
        ):
            if distance <= epsilon:
                target_index = clustered_indices[neighbor_idx]
                resolved_labels[noise_indices[offset]] = resolved_labels[target_index]
                label_tiers[noise_indices[offset]] = 0

    remaining_noise_indices = np.where(resolved_labels == -1)[0]
    micro_min_cluster_size = _compute_micro_cluster_min_size(len(resolved_labels))
    n_micro_clusters = 0

    if len(remaining_noise_indices) >= micro_min_cluster_size:
        with _log_step(
            "Resolve HDBSCAN micro-clusters "
            f"noise_points={len(remaining_noise_indices):,} "
            f"min_cluster_size={micro_min_cluster_size}"
        ):
            micro_labels = hdbscan.HDBSCAN(
                min_cluster_size=micro_min_cluster_size,
                min_samples=DATA_SKEW_MICRO_HDBSCAN_MIN_SAMPLES,
                metric="euclidean",
                cluster_selection_method=DATA_SKEW_HDBSCAN_CLUSTER_SELECTION_METHOD,
            ).fit_predict(reduced_embeddings[remaining_noise_indices])

        micro_cluster_ids = sorted(label for label in set(micro_labels) if label >= 0)
        n_micro_clusters = len(micro_cluster_ids)
        if n_micro_clusters > 0:
            next_cluster_id = int(resolved_labels.max()) + 1 if len(resolved_labels) > 0 else 0
            for local_label in micro_cluster_ids:
                micro_mask = micro_labels == local_label
                target_indices = remaining_noise_indices[micro_mask]
                resolved_labels[target_indices] = next_cluster_id
                label_tiers[target_indices] = 1
                next_cluster_id += 1

    remaining_noise_indices = np.where(resolved_labels == -1)[0]
    n_singletons = len(remaining_noise_indices)
    next_cluster_id = int(resolved_labels.max()) + 1 if len(resolved_labels) > 0 else 0
    for index in remaining_noise_indices:
        resolved_labels[index] = next_cluster_id
        label_tiers[index] = 2
        next_cluster_id += 1

    return resolved_labels, label_tiers, n_micro_clusters, n_singletons


@dataclass
class DataSkewReport:
    """Semantic data-skew outputs for one embedding column."""

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
    largest_imbalance_ratio: float
    zipf_alpha: float
    largest_cluster_share: float
    gini_per_seed: list[float] = field(default_factory=list)
    n_clusters_per_seed: list[int] = field(default_factory=list)

    @property
    def gini_std(self) -> float:
        return float(np.std(self.gini_per_seed)) if self.gini_per_seed else 0.0


def _compute_embedding_imbalance_report(
    embeddings: np.ndarray,
    sampling_seed: int = 42,
    n_stability_runs: int = DATA_SKEW_STABILITY_RUNS,
) -> DataSkewReport:
    """Measure semantic data skew via UMAP plus HDBSCAN."""

    hdbscan = _import_hdbscan()
    entropy, _, _, _ = _import_scipy_stats()
    matrix = _validate_embedding_matrix(embeddings)
    n_samples = len(matrix)
    min_cluster_size = _compute_primary_cluster_min_size(n_samples)

    clustering_embedding = _compute_umap_embedding(
        matrix,
        n_components=DATA_SKEW_UMAP_COMPONENTS,
        n_neighbors=DATA_SKEW_UMAP_NEIGHBORS,
        sampling_seed=sampling_seed,
    )

    with _log_step(
        "Primary HDBSCAN clustering "
        f"shape={_format_shape(clustering_embedding)} "
        f"min_cluster_size={min_cluster_size}"
    ):
        raw_labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=DATA_SKEW_HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method=DATA_SKEW_HDBSCAN_CLUSTER_SELECTION_METHOD,
        ).fit_predict(clustering_embedding)

    n_primary_clusters = int(raw_labels.max()) + 1 if raw_labels.max() >= 0 else 0
    raw_noise_fraction = float((raw_labels == -1).mean())

    labels, label_tiers, n_micro_clusters, n_singletons = _resolve_noise_labels(
        clustering_embedding,
        raw_labels,
    )
    cluster_sizes = _compute_cluster_sizes(labels)
    probabilities = cluster_sizes / cluster_sizes.sum()
    shannon_entropy = float(entropy(probabilities, base=2))
    max_entropy = np.log2(n_samples) if n_samples > 0 else 0.0
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0

    visualization_embedding = _compute_umap_embedding(
        matrix,
        n_components=2,
        n_neighbors=DATA_SKEW_UMAP_NEIGHBORS,
        min_dist=0.1,
        sampling_seed=sampling_seed,
    )

    report = DataSkewReport(
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
        shannon_entropy=shannon_entropy,
        normalized_entropy=normalized_entropy,
        gini=float(_gini(cluster_sizes)),
        largest_imbalance_ratio=float(cluster_sizes.max() / cluster_sizes.mean()),
        zipf_alpha=float(_zipf_alpha(cluster_sizes)),
        largest_cluster_share=float(cluster_sizes.max() / n_samples),
    )

    for seed in range(sampling_seed + 1, sampling_seed + 1 + n_stability_runs):
        seeded_embedding = _compute_umap_embedding(
            matrix,
            n_components=DATA_SKEW_UMAP_COMPONENTS,
            n_neighbors=DATA_SKEW_UMAP_NEIGHBORS,
            sampling_seed=seed,
        )
        with _log_step(
            "HDBSCAN stability run "
            f"sampling_seed={seed} shape={_format_shape(seeded_embedding)}"
        ):
            seeded_labels = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=DATA_SKEW_HDBSCAN_MIN_SAMPLES,
                metric="euclidean",
                cluster_selection_method=DATA_SKEW_HDBSCAN_CLUSTER_SELECTION_METHOD,
            ).fit_predict(seeded_embedding)

        seeded_labels, _, _, _ = _resolve_noise_labels(seeded_embedding, seeded_labels)
        seeded_sizes = _compute_cluster_sizes(seeded_labels)
        report.gini_per_seed.append(float(_gini(seeded_sizes)))
        report.n_clusters_per_seed.append(int(len(seeded_sizes)))

    return report


def _compute_knn_similarity_coefficient(
    embeddings: np.ndarray,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    normalize: bool = True,
) -> dict[str, Any]:
    """Compute cosine-similarity-based kNN statistics for multiple k values."""

    faiss = _import_faiss()
    _, kurtosis, _, skew = _import_scipy_stats()
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}.")

    n_rows, n_dims = embeddings.shape
    if n_rows < 2:
        raise ValueError("At least two embeddings are required for kNN analysis.")

    sorted_k_values = tuple(sorted(k_values))
    max_k = max(sorted_k_values)
    if max_k >= n_rows:
        raise ValueError(
            f"k={max_k} is invalid for {n_rows} embeddings; it must be smaller than the number of rows."
        )

    with _log_step(f"Validate embedding matrix before FAISS kNN shape={_format_shape(embeddings)}"):
        validated_embeddings = _validate_embedding_matrix(embeddings)
        n_rows, n_dims = validated_embeddings.shape

    if normalize:
        with _log_step(
            f"Normalize embeddings for FAISS kNN shape={_format_shape(validated_embeddings)}"
        ):
            validated_embeddings = _normalize_embeddings(validated_embeddings)

    with _log_step(
        f"Convert embeddings for FAISS shape={_format_shape(validated_embeddings)}"
    ):
        faiss_embeddings = _prepare_faiss_matrix(validated_embeddings)

    ef_search = min(n_rows, max(128, 4 * (max_k + 1)))
    ef_construction = max(40, min(512, ef_search))
    with _log_step(
        "Build FAISS IndexHNSWFlat "
        f"rows={n_rows:,} dims={n_dims} efConstruction={ef_construction} efSearch={ef_search}"
    ):
        index = faiss.IndexHNSWFlat(n_dims, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(faiss_embeddings)

    with _log_step(f"FAISS IndexHNSWFlat search max_k={max_k} rows={n_rows:,}"):
        all_distances, all_indices = index.search(faiss_embeddings, max_k + 1)

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
        with _log_step(
            "Exact FAISS fallback for invalid HNSW rows "
            f"rows={len(invalid_rows):,} max_k={max_k}"
        ):
            exact_index = faiss.IndexFlatIP(n_dims)
            exact_index.add(faiss_embeddings)
            fallback_distances, fallback_indices = exact_index.search(
                faiss_embeddings[invalid_rows],
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
            _log_array_issue("Invalid exact fallback neighbor distances", neighbor_distances)
            raise ValueError(
                "FAISS kNN returned invalid neighbor distances even after exact fallback."
            )

    summary: dict[int, dict[str, float]] = {}
    pointwise_min_similarity: dict[int, np.ndarray] = {}

    for k in sorted_k_values:
        with _log_step(f"Compute kNN summary statistics k={k}"):
            k_distances = all_distances[:, 1 : k + 1].astype(np.float64)
            cosine_similarities = np.clip(k_distances, -1.0, 1.0)
            pointwise_min_similarity[k] = cosine_similarities.min(axis=1)
            mean_similarity = cosine_similarities.mean(axis=1)
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
                "similarity_gini": float(_gini(mean_similarity)),
                "mean_knn_similarity": float(cosine_similarities.mean()),
            }

    return {
        "summary": summary,
        "pointwise_min_similarity": pointwise_min_similarity,
    }


def _load_embedding_column(parquet_file: Path, column: str) -> tuple[np.ndarray, int]:
    """Load one embedding column from parquet into a dense NumPy matrix."""

    with _log_step(f"Read parquet column column={column} file={parquet_file}"):
        dataframe = duckdb.sql(
            "SELECT "
            f"{_sql_identifier(column)} "
            "FROM read_parquet("
            f"{_sql_string_literal(str(parquet_file))})"
        ).df()

    values = dataframe[column].tolist()
    filtered_embeddings = [embedding for embedding in values if embedding is not None]
    dropped_nulls = len(values) - len(filtered_embeddings)

    if not filtered_embeddings:
        raise ValueError(
            f"Column '{column}' contains no non-null embeddings after filtering."
        )

    with _log_step(f"Convert embeddings to NumPy column={column} rows={len(filtered_embeddings):,}"):
        matrix = np.asarray(filtered_embeddings, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"Column '{column}' does not form a 2D embedding matrix; got shape {matrix.shape}."
        )

    return matrix, dropped_nulls


def _load_filtered_source_values(
    parquet_file: Path,
    source_column: str,
    embedding_column: str,
) -> list[Any]:
    """Load source-column values for rows that keep a non-null embedding."""

    with _log_step(
        "Read source values aligned to embeddings "
        f"source_column={source_column} embedding_column={embedding_column} "
        f"file={parquet_file}"
    ):
        dataframe = duckdb.sql(
            "SELECT "
            f"{_sql_identifier(source_column)}, "
            f"{_sql_identifier(embedding_column)} "
            "FROM read_parquet("
            f"{_sql_string_literal(str(parquet_file))})"
        ).df()

    source_values = dataframe[source_column].tolist()
    embedding_values = dataframe[embedding_column].tolist()
    return [
        source_value
        for source_value, embedding_value in zip(source_values, embedding_values)
        if embedding_value is not None
    ]


def _create_umap_scatter_plot_figure(
    umap_projection: np.ndarray,
    point_colors_by_k: dict[int, np.ndarray],
    output_path: Path,
    title: str,
    k_values: tuple[int, ...],
) -> None:
    """Create horizontally packed UMAP scatter subplots for all k values."""

    apply_plot_params(
        fig_height=2.6,
        scale=max(1.0, 0.85 * len(k_values)),
        double_column=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    color_min = min(point_colors_by_k[k].min() for k in k_values)
    color_max = max(point_colors_by_k[k].max() for k in k_values)

    figure, axes = plt.subplots(1, len(k_values), figsize=(4.4 * len(k_values), 3.9))
    if len(k_values) == 1:
        axes = [axes]

    scatter = None
    for index, k in enumerate(k_values):
        axis = axes[index]
        scatter = axis.scatter(
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
        axis.set_title(rf"$k={k}$")
        axis.set_xlabel("UMAP 1")
        if index == 0:
            axis.set_ylabel("UMAP 2")
        _draw_axis_border(axis)

    assert scatter is not None
    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.9)
    colorbar.set_label("Min kNN Similarity")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _create_sorted_knn_similarity_plot_figure(
    point_similarities_by_k: dict[int, np.ndarray],
    output_path: Path,
    title: str,
    k_values: tuple[int, ...],
) -> None:
    """Create horizontally packed sorted similarity subplots for all k values."""

    apply_plot_params(
        fig_height=2.4,
        scale=max(1.0, 0.85 * len(k_values)),
        double_column=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_min = min(float(np.min(point_similarities_by_k[k])) for k in k_values)
    y_max = max(float(np.max(point_similarities_by_k[k])) for k in k_values)

    figure, axes = plt.subplots(1, len(k_values), figsize=(4.4 * len(k_values), 3.3))
    if len(k_values) == 1:
        axes = [axes]

    for index, k in enumerate(k_values):
        axis = axes[index]
        sorted_similarities = np.sort(point_similarities_by_k[k])
        axis.plot(
            np.arange(len(sorted_similarities)),
            sorted_similarities,
            linewidth=1.5,
            color="#1859FF",
        )
        axis.set_title(rf"$k={k}$")
        axis.set_xlabel("Point Rank")
        if index == 0:
            axis.set_ylabel("Min kNN Similarity")
        axis.set_ylim(y_min, y_max)
        axis.grid(True, alpha=0.35)
        _draw_axis_border(axis)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _create_imbalance_plot_figure(
    report: DataSkewReport,
    output_path: Path,
) -> None:
    """Create the semantic imbalance figure."""

    apply_plot_params(
        fig_height=3.2,
        scale=1,
        double_column=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    _create_cluster_map_plot(report, axes[0])
    _create_rank_size_plot(report, axes[1])
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _create_cluster_map_plot(report: DataSkewReport, axis: Any) -> None:
    """Plot a UMAP cluster map with singleton classes omitted."""

    embedding_2d = report.embedding_2d
    labels = report.labels
    tiers = report.label_tiers
    render_indices = np.where(tiers != 2)[0]
    if len(render_indices) > CLUSTER_MAP_RENDER_SAMPLE_SIZE:
        render_indices = np.random.default_rng(42).choice(
            render_indices,
            size=CLUSTER_MAP_RENDER_SAMPLE_SIZE,
            replace=False,
        )

    render_embedding = embedding_2d[render_indices]
    render_labels = labels[render_indices]

    if len(render_indices) == 0:
        axis.text(
            0.5,
            0.5,
            "No non-singleton clusters",
            ha="center",
            va="center",
            fontsize=8,
            transform=axis.transAxes,
        )
        axis.set_title("Cluster Map", fontsize=10, fontweight="medium", pad=6)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        _draw_axis_border(axis)
        return

    micro_cluster_ids = sorted(set(labels[tiers == 1]))
    if micro_cluster_ids:
        micro_palette = _cluster_map_palette(len(micro_cluster_ids), offset=60)
        for rank, cluster_id in enumerate(micro_cluster_ids):
            cluster_mask = render_labels == cluster_id
            if not cluster_mask.any():
                continue
            axis.scatter(
                render_embedding[cluster_mask, 0],
                render_embedding[cluster_mask, 1],
                c=[micro_palette[rank]],
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
    palette = _cluster_map_palette(len(primary_cluster_ids))

    for rank, cluster_id in enumerate(primary_cluster_ids):
        cluster_mask = render_labels == cluster_id
        if not cluster_mask.any():
            continue
        color = palette[rank]
        axis.scatter(
            render_embedding[cluster_mask, 0],
            render_embedding[cluster_mask, 1],
            c=[color],
            s=4,
            alpha=0.65,
            linewidths=0,
            zorder=2,
        )

    axis.set_title("Cluster Map", fontsize=10, fontweight="medium", pad=6)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    _draw_axis_border(axis)


def _create_rank_size_plot(report: DataSkewReport, axis: Any) -> None:
    """Plot the cluster rank-size curve."""

    sizes = report.cluster_sizes[report.cluster_sizes > 1]
    relative_sizes = sizes / report.n_samples if report.n_samples > 0 else sizes
    ranks = np.arange(1, len(sizes) + 1)

    if len(sizes) == 0:
        axis.text(
            0.5,
            0.5,
            "No non-singleton clusters",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xlabel("Rank (1 = Largest Cluster)")
        axis.set_ylabel(r"Relative Cluster Size (\%)")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        axis.set_title("Cluster Sizes")
        _draw_axis_border(axis)
        return

    axis.plot(
        ranks,
        relative_sizes,
        "-",
        color="#1859FF",
        linewidth=1.7,
        zorder=3,
    )

    axis.set_xlabel("Rank (1 = Largest Cluster)")
    axis.set_ylabel(r"Relative Cluster Size (\%)")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    axis.set_title("Rank-Size")
    axis.grid(True, which="both", alpha=0.2, linewidth=0.5)
    _draw_axis_border(axis)


class DataSkewPlotMixin:
    """Helpers for plotting semantic data-skew diagnostics for embedding columns."""

    AMAZON_REVIEWS_PROCESSED_DIR = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "amazon-reviews"
        / "processed"
    )
    AMAZON_REVIEWS_CACHE_DIR = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "amazon-reviews"
        / "cache"
        / "dataset_analysis"
    )

    def _plot_data_skew(self) -> None:
        """Generate data-skew plots and a summary table for embedded datasets."""

        try:
            self._validate_data_skew_dependencies()
        except ModuleNotFoundError as error:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"Skipping data skew plots: {error}"
            )
            return

        processed_dir = self.AMAZON_REVIEWS_PROCESSED_DIR
        if not processed_dir.exists():
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"Amazon reviews processed directory not found: {processed_dir}"
            )
            return

        embeddings_paths = sorted(processed_dir.rglob("*_with_embeddings.parquet"))
        if not embeddings_paths:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"No embedded parquet files found in {processed_dir}"
            )
            return

        summary_rows: list[dict[str, Any]] = []
        output_dir = self.plot_dir / "data_skew"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.AMAZON_REVIEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for embeddings_path in embeddings_paths:
            summary_rows.extend(
                self._plot_data_skew_for_dataset(
                    embeddings_path=embeddings_path,
                    output_dir=output_dir,
                )
            )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values(
                by=["dataset_name", "embedding_column"]
            )
            summary_path = self.table_dir / "data_skew_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            console.print(
                f"[green]✓[/green] Saved data skew summary to [bold]{summary_path}[/bold]"
            )

    def _validate_data_skew_dependencies(self) -> None:
        """Ensure all optional dependencies required by the skew pipeline are available."""

        _import_hdbscan()
        _import_faiss()
        _import_umap()
        _import_scipy_stats()
        _import_pca()

    def _plot_data_skew_for_dataset(
        self,
        embeddings_path: Path,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Generate data-skew artifacts for one embedded parquet file."""

        dataset_name = str(embeddings_path.parent.name)
        embedding_columns = _infer_embedding_columns(embeddings_path)

        if not embedding_columns:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"No embedding columns found in {embeddings_path.name}"
            )
            return []

        summary_rows: list[dict[str, Any]] = []
        for embedding_column in embedding_columns:
            try:
                summary_rows.append(
                    self._plot_data_skew_for_column(
                        embeddings_path=embeddings_path,
                        dataset_name=dataset_name,
                        embedding_column=embedding_column,
                        output_dir=output_dir,
                    )
                )
            except Exception as error:
                console.print(
                    "[bold yellow]Warning:[/bold yellow] "
                    f"Skipping {embeddings_path.name}:{embedding_column}: {error}"
                )

        return summary_rows

    def _plot_data_skew_for_column(
        self,
        embeddings_path: Path,
        dataset_name: str,
        embedding_column: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Generate data-skew plots for one embedding column."""

        source_column = _embedding_source_column_name(embedding_column)
        cached = self._load_or_prepare_data_skew_cache(
            parquet_file=embeddings_path,
            column=embedding_column,
        )

        dataset_stem = embeddings_path.stem.removesuffix("_with_embeddings")
        safe_column = _sanitize_filename(embedding_column)
        title = (
            f"{_escape_latex_text(dataset_name)} / "
            f"{_escape_latex_text(embedding_column)}"
        )

        scatter_path = output_dir / f"{dataset_stem}__{safe_column}__scatter.pdf"
        _create_umap_scatter_plot_figure(
            umap_projection=cached["umap_projection"],
            point_colors_by_k=cached["pointwise_min_similarity"],
            output_path=scatter_path,
            title=title,
            k_values=cached["k_values"],
        )
        console.print(
            f"[green]✓[/green] Saved data skew scatter plot to [bold]{scatter_path}[/bold]"
        )

        sorted_path = output_dir / f"{dataset_stem}__{safe_column}__sorted.pdf"
        _create_sorted_knn_similarity_plot_figure(
            point_similarities_by_k=cached["pointwise_min_similarity"],
            output_path=sorted_path,
            title=f"{title} sorted similarities",
            k_values=cached["k_values"],
        )
        console.print(
            f"[green]✓[/green] Saved sorted kNN plot to [bold]{sorted_path}[/bold]"
        )

        imbalance_path = output_dir / f"{dataset_stem}__{safe_column}__imbalance.pdf"
        _create_imbalance_plot_figure(
            report=cached["imbalance_report"],
            output_path=imbalance_path,
        )
        console.print(
            f"[green]✓[/green] Saved data skew imbalance plot to [bold]{imbalance_path}[/bold]"
        )

        source_values = _load_filtered_source_values(
            parquet_file=embeddings_path,
            source_column=source_column,
            embedding_column=embedding_column,
        )
        clustered_source_values = np.asarray(source_values, dtype=object)[
            cached["cluster_sample_indices"]
        ]
        cluster_labels = cached["imbalance_report"].labels
        cluster_counts = np.bincount(cluster_labels[cluster_labels >= 0])
        if cluster_counts.size == 0:
            raise ValueError("Cannot export largest-class values without any clusters.")

        largest_cluster_id = int(np.flatnonzero(cluster_counts == cluster_counts.max())[0])
        largest_class_mask = cluster_labels == largest_cluster_id
        largest_class_values = pd.DataFrame(
            {source_column: clustered_source_values[largest_class_mask]}
        )
        largest_class_path = output_dir / f"{dataset_stem}__{safe_column}__largest_class.csv"
        largest_class_values.to_csv(largest_class_path, index=False)
        console.print(
            f"[green]✓[/green] Saved largest-class values to [bold]{largest_class_path}[/bold]"
        )

        result = {
            "dataset_name": dataset_name,
            "parquet_path": str(embeddings_path),
            "source_column": source_column,
            "embedding_column": embedding_column,
            "num_rows": cached["num_rows"],
            "analysis_rows": cached["analysis_rows"],
            "embedding_dim": cached["embedding_dim"],
            "dropped_null_embeddings": cached["dropped_null_embeddings"],
            "scatter_plot_path": str(scatter_path),
            "sorted_plot_path": str(sorted_path),
            "imbalance_plot_path": str(imbalance_path),
        }

        imbalance_report = cached["imbalance_report"]
        result.update(
            {
                "n_clusters": imbalance_report.n_clusters,
                "n_primary_clusters": imbalance_report.n_primary_clusters,
                "n_micro_clusters": imbalance_report.n_micro_clusters,
                "n_singletons": imbalance_report.n_singletons,
                "raw_noise_fraction": imbalance_report.raw_noise_fraction,
                "shannon_entropy": imbalance_report.shannon_entropy,
                "normalized_entropy": imbalance_report.normalized_entropy,
                "gini": imbalance_report.gini,
                "gini_std": imbalance_report.gini_std,
                "largest_imbalance_ratio": imbalance_report.largest_imbalance_ratio,
                "zipf_alpha": imbalance_report.zipf_alpha,
                "largest_cluster_share": imbalance_report.largest_cluster_share,
            }
        )
        for k, metrics in cached["knn_density"].items():
            for metric_name, value in metrics.items():
                result[f"k{k}_{metric_name}"] = value

        return result

    def _get_data_skew_cache_paths(
        self,
        parquet_file: Path,
        column: str,
    ) -> tuple[Path, Path]:
        """Return metadata and array cache paths for one parquet-column pair."""

        stem = f"{parquet_file.stem}__{_sanitize_filename(column)}"
        return (
            self.AMAZON_REVIEWS_CACHE_DIR / f"{stem}.json",
            self.AMAZON_REVIEWS_CACHE_DIR / f"{stem}.npz",
        )

    def _is_data_skew_cache_current(
        self,
        metadata_path: Path,
        arrays_path: Path,
    ) -> bool:
        """Return whether cached analysis artifacts match the current logic."""

        if not metadata_path.exists() or not arrays_path.exists():
            return False

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        return metadata.get("data_skew_cache_version") == DATA_SKEW_CACHE_VERSION

    def _load_or_prepare_data_skew_cache(
        self,
        parquet_file: Path,
        column: str,
    ) -> dict[str, Any]:
        """Load a current skew-analysis cache or build it on demand."""

        metadata_path, arrays_path = self._get_data_skew_cache_paths(parquet_file, column)
        if self._is_data_skew_cache_current(metadata_path, arrays_path):
            console.log(
                f"Data skew cache hit column={column} metadata={metadata_path} arrays={arrays_path}"
            )
            return self._load_cached_data_skew_analysis(parquet_file, column)

        if metadata_path.exists() or arrays_path.exists():
            console.log(
                f"Data skew cache stale column={column} metadata={metadata_path} arrays={arrays_path}"
            )
        else:
            console.log(
                f"Data skew cache miss column={column} metadata={metadata_path} arrays={arrays_path}"
            )

        return self._prepare_data_skew_cache(parquet_file, column)

    def _prepare_data_skew_cache(
        self,
        parquet_file: Path,
        column: str,
        k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    ) -> dict[str, Any]:
        """Compute expensive skew-analysis artifacts once and write them to cache."""

        umap = _import_umap()
        metadata_path, arrays_path = self._get_data_skew_cache_paths(parquet_file, column)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings, dropped_nulls = _load_embedding_column(parquet_file, column)
        analysis_embeddings, analysis_sample_indices = (
            _sample_embeddings_for_data_skew_analysis(embeddings)
        )
        with _log_step(f"Compute kNN density column={column}"):
            knn_results = _compute_knn_similarity_coefficient(
                analysis_embeddings,
                k_values=k_values,
            )

        with _log_step(f"Compute PCA cache projection column={column}"):
            pca_projection = self._compute_pca_embedding(analysis_embeddings)

        with _log_step(
            "UMAP fit_transform cache visualization "
            f"column={column} shape={_format_shape(pca_projection)}"
        ):
            umap_projection = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(2, len(pca_projection) - 1)),
                random_state=42,
            ).fit_transform(pca_projection)

        with _log_step(
            "Compute semantic imbalance report "
            f"column={column} rows={len(analysis_embeddings):,}"
        ):
            imbalance_report = _compute_embedding_imbalance_report(
                analysis_embeddings
            )

        with _log_step(f"Write compressed NumPy cache arrays output={arrays_path}"):
            np.savez_compressed(
                arrays_path,
                pca_projection=pca_projection.astype(np.float32),
                umap_projection=umap_projection.astype(np.float32),
                cluster_sample_indices=analysis_sample_indices.astype(np.int64),
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
            "analysis_rows": int(analysis_embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "dropped_null_embeddings": int(dropped_nulls),
            "cluster_assignment_rows": int(analysis_embeddings.shape[0]),
            "data_skew_cache_version": DATA_SKEW_CACHE_VERSION,
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
                "largest_imbalance_ratio": imbalance_report.largest_imbalance_ratio,
                "zipf_alpha": imbalance_report.zipf_alpha,
                "largest_cluster_share": imbalance_report.largest_cluster_share,
            },
        }
        with _log_step(f"Write cache metadata output={metadata_path}"):
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return self._load_cached_data_skew_analysis(parquet_file, column)

    def _load_cached_data_skew_analysis(
        self,
        parquet_file: Path,
        column: str,
    ) -> dict[str, Any]:
        """Load cached arrays and metadata for one parquet-column pair."""

        entropy, _, _, _ = _import_scipy_stats()
        metadata_path, arrays_path = self._get_data_skew_cache_paths(parquet_file, column)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(arrays_path)
        k_values = tuple(int(k) for k in metadata["k_values"])
        pointwise_min_similarity = {
            k: arrays[f"pointwise_min_similarity_k{k}"]
            for k in k_values
        }

        imbalance = metadata["imbalance"]
        analysis_rows = int(
            metadata.get("analysis_rows", metadata["cluster_assignment_rows"])
        )
        cluster_sizes = arrays["cluster_sizes"]
        probabilities = cluster_sizes / cluster_sizes.sum()
        shannon_entropy = float(
            imbalance.get("shannon_entropy", entropy(probabilities, base=2))
        )
        normalized_entropy = float(
            imbalance.get(
                "normalized_entropy",
                shannon_entropy / np.log2(analysis_rows)
                if analysis_rows > 1
                else 0.0,
            )
        )

        imbalance_report = DataSkewReport(
            n_samples=analysis_rows,
            n_clusters=int(imbalance["n_clusters"]),
            n_primary_clusters=int(imbalance["n_primary_clusters"]),
            n_micro_clusters=int(imbalance["n_micro_clusters"]),
            n_singletons=int(imbalance["n_singletons"]),
            raw_noise_fraction=float(imbalance["raw_noise_fraction"]),
            cluster_sizes=cluster_sizes,
            labels=arrays["labels"],
            label_tiers=arrays["label_tiers"],
            embedding_2d=arrays["umap_projection"],
            shannon_entropy=shannon_entropy,
            normalized_entropy=normalized_entropy,
            gini=float(imbalance["gini"]),
            largest_imbalance_ratio=float(imbalance["largest_imbalance_ratio"]),
            zipf_alpha=float(imbalance["zipf_alpha"]),
            largest_cluster_share=float(imbalance["largest_cluster_share"]),
            gini_per_seed=[],
            n_clusters_per_seed=[],
        )

        return {
            "num_rows": int(metadata["num_rows"]),
            "analysis_rows": analysis_rows,
            "embedding_dim": int(metadata["embedding_dim"]),
            "dropped_null_embeddings": int(metadata["dropped_null_embeddings"]),
            "cluster_assignment_rows": analysis_rows,
            "cluster_sample_indices": arrays["cluster_sample_indices"],
            "k_values": k_values,
            "knn_density": {
                int(k): metric_values
                for k, metric_values in metadata["knn_density"].items()
            },
            "pointwise_min_similarity": pointwise_min_similarity,
            "pca_projection": arrays["pca_projection"],
            "umap_projection": arrays["umap_projection"],
            "imbalance_report": imbalance_report,
        }

    def _compute_pca_embedding(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute the cached PCA representation used before UMAP."""

        PCA = _import_pca()
        with _log_step(f"Normalize embeddings before PCA shape={_format_shape(embeddings)}"):
            normalized_embeddings = _normalize_embeddings(embeddings)

        n_rows, n_dims = normalized_embeddings.shape
        n_components = min(50, n_dims, n_rows)
        with _log_step(
            "PCA fit_transform cache projection "
            f"shape={_format_shape(normalized_embeddings)} components={n_components}"
        ):
            return PCA(n_components=n_components).fit_transform(normalized_embeddings)
