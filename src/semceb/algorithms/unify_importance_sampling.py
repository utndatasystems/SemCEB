from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import lotus.settings

from semceb.algorithms.cardinality_estimate import (
    CardinalityEstimate,
    CardinalityEstimateKind,
)
from semceb.algorithms.interface import AlgorithmInterface
from semceb.queries.query_specification import QuerySpecification
from semceb.queries.template_parser import QueryTemplatePartType


@dataclass(frozen=True)
class PreparedColumnState:
    """Prepared normalized embeddings for one semantic data column."""

    dataset_name: str
    source_column: str
    embedding_column: str
    total_rows: int
    valid_positions: np.ndarray
    normalized_embeddings: np.ndarray

    @property
    def valid_rows(self) -> int:
        return int(self.valid_positions.size)


class UnifyImportanceSampling(AlgorithmInterface):
    """Distance-stratified semantic cardinality estimation."""

    STRATUM_COLUMN = "__unify_stratum"

    def __init__(self, name: str, version: str):
        """Initialize the unify importance sampling algorithm and prepare cost tracking."""
        self.name = name
        self.version = version

        self.model = None
        self.memory_consumption = 0
        self.data_dfs: dict[str, pd.DataFrame] = {}
        self.prepared_states: dict[tuple[str, str], PreparedColumnState] = {}
        self.last_run_details: dict[str, Any] = {}

        self.embedding_model_key: str | None = None
        self.sampling_frac = 0.01
        self.seed = 42
        self.num_bins = 4
        self.distance_edges = np.linspace(0.0, 2.0, 5, dtype=np.float64)
        self.distance_decay_temperature = 1.0

        self.reset_cost_stats()

    def get_memory_consumption(self) -> int:
        """Return retained dataframe, embedding-state, and model bytes."""
        return self.memory_consumption

    def get_cost_stats(self) -> dict:
        """Return accumulated semantic-evaluation cost statistics."""
        return self.cost_stats

    def reset_cost_stats(self) -> None:
        """Reset accumulated LOTUS/LLM cost statistics."""
        self.cost_stats = {"usd": 0.0, "llm_calls": 0, "tokens": 0}

        if self.model is not None:
            self.model.reset_stats()
            # Prepare lotus for next run to determine costs
            lotus.settings.configure(lm=self.model)

    def preparation(
        self,
        data_dfs: dict[str, pd.DataFrame],
        algorithm_kwargs: dict,
    ) -> None:
        """Prepare model configuration, row embeddings, and retained data."""
        self.embedding_model_key = self._require_nonempty_str(
            algorithm_kwargs,
            "embedding_model_key",
        )
        model_name = self._require_nonempty_str(algorithm_kwargs, "model_name")

        self.sampling_frac = float(algorithm_kwargs.get("sampling_frac", 0.01))
        if not 0 < self.sampling_frac <= 1:
            raise ValueError("sampling_frac must be in the interval (0, 1].")

        self.num_bins = int(algorithm_kwargs.get("num_bins", 4))
        if self.num_bins <= 0:
            raise ValueError("num_bins must be a positive integer.")

        self.distance_edges = np.linspace(
            0.0,
            2.0,
            self.num_bins + 1,
            dtype=np.float64,
        )

        self.distance_decay_temperature = float(
            algorithm_kwargs.get("distance_decay_temperature", 1.0)
        )
        if self.distance_decay_temperature < 0:
            raise ValueError("distance_decay_temperature must be non-negative.")

        self._initialize_model(
            model_name=model_name,
            system_prompt=algorithm_kwargs.get("system_prompt"),
            max_batch_size=int(algorithm_kwargs.get("max_batch_size", 64)),
            cache_size=int(algorithm_kwargs.get("cache_size", 1000)),
        )

        # Discover and normalize every relevant embedding column automatically.
        # Retain only the corresponding semantic source columns afterwards.
        self.prepared_states = self._prepare_embedding_states(data_dfs)
        self.data_dfs = self._retain_semantic_source_columns(data_dfs)

        self.memory_consumption = self._estimate_memory_consumption()
        self.reset_cost_stats()

    def run(self, query_spec: QuerySpecification) -> CardinalityEstimate:
        """Estimate one single-table, single-column semantic-filter cardinality only."""
        unsupported_reason = self._unsupported_reason(query_spec)
        if unsupported_reason is not None:
            return CardinalityEstimate(
                kind=CardinalityEstimateKind.UNSUPPORTED,
                reason=unsupported_reason,
            )

        dataset_name = query_spec.datasets[0].table_ref
        source_column = self._referenced_columns(query_spec)[0]
        state = self.prepared_states[(dataset_name, source_column)]

        query_embedding = self._get_query_embedding(query_spec)
        embedding_dimension = state.normalized_embeddings.shape[1]
        if query_embedding.size != embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for query {query_spec.id}: "
                f"query has {query_embedding.size}, but "
                f"{dataset_name}.{source_column} has {embedding_dimension}."
            )

        # Reuse the matrix-vector result as the distance buffer to avoid
        # retaining separate full-length similarity and distance arrays.
        distances = state.normalized_embeddings @ query_embedding
        np.subtract(1.0, distances, out=distances)
        np.clip(distances, 0.0, 2.0, out=distances)

        stratum_ids = self._assign_strata(distances)
        stratum_sizes = np.bincount(
            stratum_ids,
            minlength=self.num_bins,
        ).astype(np.int64)

        weights = self._distance_decay_weights(stratum_sizes)
        requested_budget = self._sample_budget(state.valid_rows)
        sample_counts = self._allocate_samples(
            stratum_sizes=stratum_sizes,
            weights=weights,
            requested_budget=requested_budget,
        )

        sampled_df, total_sampled = self._build_combined_sample(
            dataset_name=dataset_name,
            state=state,
            stratum_ids=stratum_ids,
            sample_counts=sample_counts,
            query_id=int(query_spec.id),
        )

        query_str = self._build_filter_query_str(query_spec)
        usage_before = self._model_usage_snapshot()

        filtered_df = sampled_df.sem_filter(user_instruction=query_str)

        self._record_model_usage_delta(
            before=usage_before,
            llm_calls=total_sampled,
        )

        positive_counts = (
            filtered_df.groupby(self.STRATUM_COLUMN).size().to_dict()
            if not filtered_df.empty
            else {}
        )

        estimate = 0.0
        stratum_details: list[dict[str, Any]] = []

        for stratum_id, (population_size, sample_count) in enumerate(
            zip(stratum_sizes.tolist(), sample_counts.tolist())
        ):
            if population_size == 0:
                continue
            if sample_count <= 0:
                raise RuntimeError("A non-empty stratum received zero samples.")

            positives = int(positive_counts.get(stratum_id, 0))
            stratum_estimate = population_size * (positives / sample_count)
            estimate += stratum_estimate

            stratum_details.append(
                {
                    "stratum": stratum_id,
                    "distance_min": float(self.distance_edges[stratum_id]),
                    "distance_max": float(self.distance_edges[stratum_id + 1]),
                    "population": population_size,
                    "importance_weight": float(weights[stratum_id]),
                    "sampled": sample_count,
                    "positives": positives,
                    "estimate": float(stratum_estimate),
                }
            )

        rounded_estimate = int(round(estimate))

        # Preparation guarantees complete embeddings, so valid_rows and
        # total_rows must describe the same population. Use valid_rows as the
        # estimator's explicit feasibility bound because strata are built from
        # exactly this population.
        if state.valid_rows != state.total_rows:
            raise RuntimeError(
                f"Incomplete prepared embedding state for "
                f"{dataset_name}.{source_column}: "
                f"{state.valid_rows} valid rows out of {state.total_rows}."
            )

        rounded_estimate = max(
            0,
            min(rounded_estimate, state.valid_rows),
        )

        self.last_run_details = {
            "query_id": int(query_spec.id),
            "dataset": dataset_name,
            "column": source_column,
            "requested_sample_budget": requested_budget,
            "actual_sample_budget": total_sampled,
            "estimated_cardinality": rounded_estimate,
            "sampling_mode": "distance_decay_nested",
            "nested_sampling": True,
            "strata": stratum_details,
        }

        return CardinalityEstimate(
            kind=CardinalityEstimateKind.INT,
            value=rounded_estimate,
        )

    def _initialize_model(
        self,
        model_name: str,
        system_prompt: str | None,
        max_batch_size: int,
        cache_size: int,
    ) -> None:
        """Initialize LOTUS via LiteLLM proxy or direct OpenAI credentials."""
        from lotus.cache import CacheConfig, CacheFactory, CacheType
        from lotus.models.lm import LM

        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive.")

        load_dotenv()

        litellm_endpoint = os.getenv("LITELLM_ENDPOINT")
        litellm_api_key = os.getenv("LITELLM_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if bool(litellm_endpoint) != bool(litellm_api_key):
            raise ValueError(
                "LiteLLM proxy configuration is incomplete. "
                "Set both LITELLM_ENDPOINT and LITELLM_API_KEY, "
                "or remove both to use OPENAI_API_KEY."
            )

        if not litellm_endpoint and not openai_api_key:
            raise ValueError(
                "No API credentials found. Set either "
                "LITELLM_ENDPOINT and LITELLM_API_KEY, "
                "or OPENAI_API_KEY."
            )

        cache = CacheFactory.create_cache(
            CacheConfig(
                cache_type=CacheType.IN_MEMORY,
                max_size=cache_size,
            )
        )

        lm_kwargs: dict[str, Any] = {
            "model": model_name,
            "rate_limit": None,
            "max_batch_size": max_batch_size,
            "cache": cache,
        }

        if litellm_endpoint and litellm_api_key:
            lm_kwargs["api_base"] = litellm_endpoint
            lm_kwargs["api_key"] = litellm_api_key

        self.model = LM(**lm_kwargs)
        self.model.system_prompt = system_prompt

        lotus.settings.configure(
            lm=self.model,
            enable_cache=True,
        )

    @staticmethod
    def _require_nonempty_str(algorithm_kwargs: dict, key: str) -> str:
        value = algorithm_kwargs.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"algorithm_kwargs['{key}'] must be provided.")
        return str(value).strip()

    @staticmethod
    def sanitize_embedding_model_key(embedding_model_key: str) -> str:
        normalized = embedding_model_key.lower().replace("/", "_")
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        return normalized.strip("_")

    @classmethod
    def embedding_column_name(
        cls,
        source_column: str,
        embedding_model_key: str,
    ) -> str:
        return (
            f"{source_column}_embeddings_"
            f"{cls.sanitize_embedding_model_key(embedding_model_key)}"
        )

    def _prepare_embedding_states(
        self,
        data_dfs: dict[str, pd.DataFrame],
    ) -> dict[tuple[str, str], PreparedColumnState]:
        assert self.embedding_model_key is not None

        suffix = "_embeddings_" + self.sanitize_embedding_model_key(
            self.embedding_model_key
        )
        states: dict[tuple[str, str], PreparedColumnState] = {}

        for dataset_name, data_df in data_dfs.items():
            for embedding_column in data_df.columns:
                if not embedding_column.endswith(suffix):
                    continue

                source_column = embedding_column[: -len(suffix)]
                if not source_column or source_column not in data_df.columns:
                    continue

                state = self._extract_embedding_state(
                    dataset_name=dataset_name,
                    data_df=data_df,
                    source_column=source_column,
                    embedding_column=embedding_column,
                )
                if state is not None:
                    states[(dataset_name, source_column)] = state

        if not states:
            raise ValueError(
                "No usable data embedding columns were found for embedding key "
                f"'{self.embedding_model_key}'."
            )

        return states

    def _extract_embedding_state(
        self,
        dataset_name: str,
        data_df: pd.DataFrame,
        source_column: str,
        embedding_column: str,
    ) -> PreparedColumnState | None:
        vectors: list[np.ndarray] = []
        positions: list[int] = []
        expected_dimension: int | None = None
        source_non_null = data_df[source_column].notna().to_numpy()

        for position, embedding in enumerate(data_df[embedding_column].tolist()):
            if not source_non_null[position] or self._is_missing_embedding(embedding):
                continue

            vector = np.asarray(embedding, dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0:
                continue
            if not np.isfinite(vector).all():
                continue

            if expected_dimension is None:
                expected_dimension = int(vector.size)
            elif vector.size != expected_dimension:
                raise ValueError(
                    f"Inconsistent embedding dimensions in "
                    f"{dataset_name}.{embedding_column}: expected "
                    f"{expected_dimension}, found {vector.size}."
                )

            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                continue

            vectors.append(vector / norm)
            positions.append(position)

        if not vectors:
            return None

        if len(positions) != len(data_df):
            missing = len(data_df) - len(positions)
            raise ValueError(
                f"{dataset_name}.{source_column} has {missing} row(s) without "
                "a usable source value and embedding. "
                "UnifyImportanceSampling requires complete embeddings."
            )

        return PreparedColumnState(
            dataset_name=dataset_name,
            source_column=source_column,
            embedding_column=embedding_column,
            total_rows=int(len(data_df)),
            valid_positions=np.asarray(positions, dtype=np.int64),
            normalized_embeddings=np.stack(vectors).astype(
                np.float32,
                copy=False,
            ),
        )

    @staticmethod
    def _is_missing_embedding(value: Any) -> bool:
        return value is None or (isinstance(value, float) and math.isnan(value))

    def _retain_semantic_source_columns(
        self,
        data_dfs: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Retain only source columns backed by discovered embedding states.

        Embedding columns and unrelated dataset columns are discarded after
        normalized embedding matrices have been prepared. Positional row order
        and the original index are preserved so PreparedColumnState positions
        remain valid for ``iloc`` sampling.
        """
        source_columns_by_dataset: dict[str, set[str]] = {}

        for state in self.prepared_states.values():
            source_columns_by_dataset.setdefault(
                state.dataset_name,
                set(),
            ).add(state.source_column)

        retained: dict[str, pd.DataFrame] = {}

        for dataset_name, source_columns in source_columns_by_dataset.items():
            data_df = data_dfs.get(dataset_name)
            if data_df is None:
                raise ValueError(
                    f"Prepared embedding state references missing dataset "
                    f"{dataset_name!r}."
                )

            ordered_columns = [
                column for column in data_df.columns if column in source_columns
            ]

            if len(ordered_columns) != len(source_columns):
                missing_columns = sorted(source_columns.difference(ordered_columns))
                raise ValueError(
                    f"Dataset {dataset_name!r} is missing semantic source "
                    f"column(s): {missing_columns}."
                )

            retained[dataset_name] = data_df.loc[
                :,
                ordered_columns,
            ].copy()

        return retained

    def _unsupported_reason(
        self,
        query_spec: QuerySpecification,
    ) -> str | None:
        if len(query_spec.datasets) != 1:
            return (
                "UnifyImportanceSampling supports only single-table "
                "semantic filters."
            )

        columns = self._referenced_columns(query_spec)
        if len(columns) != 1:
            return (
                "UnifyImportanceSampling requires exactly one distinct "
                "referenced column."
            )

        dataset_name = query_spec.datasets[0].table_ref
        state_key = (dataset_name, columns[0])

        if state_key not in self.prepared_states:
            expected_column = self.embedding_column_name(
                columns[0],
                self.embedding_model_key or "",
            )
            return (
                f"No prepared embedding state for "
                f"{dataset_name}.{columns[0]}. Expected "
                f"'{expected_column}'."
            )

        return None

    @staticmethod
    def _referenced_columns(
        query_spec: QuerySpecification,
    ) -> list[str]:
        columns: list[str] = []

        for part in query_spec.filter_parsed.parts:
            if part.type != QueryTemplatePartType.COLUMN_REF:
                continue

            column_name = part.value.column_name
            if column_name not in columns:
                columns.append(column_name)

        return columns

    def _get_query_embedding(
        self,
        query_spec: QuerySpecification,
    ) -> np.ndarray:
        if self.embedding_model_key is None:
            raise RuntimeError(
                "UnifyImportanceSampling.preparation() must be called "
                "before query embeddings can be accessed."
            )

        embedding = query_spec.embeddings.get(self.embedding_model_key)
        if embedding is None:
            raise ValueError(
                f"Query {query_spec.id} does not provide embedding key "
                f"'{self.embedding_model_key}'."
            )

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(
                f"Embedding for query {query_spec.id} must be a non-empty vector."
            )
        if not np.isfinite(vector).all():
            raise ValueError(
                f"Embedding for query {query_spec.id} contains non-finite values."
            )

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError(f"Embedding for query {query_spec.id} has zero norm.")

        return vector / norm

    @staticmethod
    def _build_filter_query_str(
        query_spec: QuerySpecification,
    ) -> str:
        parts: list[str] = []

        for part in query_spec.filter_parsed.parts:
            if part.type == QueryTemplatePartType.TEXT:
                parts.append(str(part.value))
            elif part.type == QueryTemplatePartType.COLUMN_REF:
                parts.append(f"{{{part.value.column_name}}}")

        return "".join(parts)

    def _assign_strata(self, distances: np.ndarray) -> np.ndarray:
        stratum_dtype = np.uint8 if self.num_bins <= 256 else np.int32
        return np.digitize(
            distances,
            self.distance_edges[1:-1],
            right=False,
        ).astype(stratum_dtype)

    def _distance_decay_weights(
        self,
        stratum_sizes: np.ndarray,
    ) -> np.ndarray:
        centers = (self.distance_edges[:-1] + self.distance_edges[1:]) / 2.0

        weights = np.exp(-self.distance_decay_temperature * centers)

        active = stratum_sizes > 0
        weights = np.where(active, weights, 0.0)

        total = float(weights.sum())
        if total <= 0:
            raise ValueError("No non-empty distance stratum is available.")

        return weights / total

    def _sample_budget(self, valid_rows: int) -> int:
        if valid_rows <= 0:
            return 0

        return min(
            valid_rows,
            max(1, int(round(valid_rows * self.sampling_frac))),
        )

    @staticmethod
    def _allocate_samples(
        stratum_sizes: np.ndarray,
        weights: np.ndarray,
        requested_budget: int,
    ) -> np.ndarray:
        """Allocate samples with deterministic, nested weighted prefixes.

        For fixed strata and weights, increasing the requested budget can only
        add samples; it never removes samples from a stratum. This makes a
        smaller sampling fraction directly comparable to a larger one.
        """
        sizes = np.asarray(stratum_sizes, dtype=np.int64)
        probabilities = np.asarray(weights, dtype=np.float64)

        if sizes.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("stratum_sizes and weights must be vectors.")
        if sizes.size != probabilities.size:
            raise ValueError("stratum_sizes and weights must be aligned.")
        if requested_budget <= 0:
            return np.zeros_like(sizes)

        active = np.flatnonzero(sizes > 0)
        if active.size == 0:
            return np.zeros_like(sizes)

        total_population = int(sizes.sum())
        budget = min(
            total_population,
            max(int(requested_budget), int(active.size)),
        )

        # Guarantee that every non-empty stratum is estimable.
        counts = np.zeros_like(sizes)
        counts[active] = 1

        remaining = budget - int(active.size)
        if remaining <= 0:
            return counts

        # Allocate every additional sample through one deterministic weighted
        # sequence. Re-running this procedure with a larger budget extends the
        # same sequence, so allocations are nested across sampling fractions.
        extra_counts = np.zeros_like(sizes)

        for _ in range(remaining):
            eligible = np.flatnonzero(counts < sizes)
            if eligible.size == 0:
                break

            eligible_probabilities = probabilities[eligible]
            if float(eligible_probabilities.sum()) <= 0:
                eligible_probabilities = (sizes[eligible] - counts[eligible]).astype(
                    np.float64
                )

            positive = eligible_probabilities > 0
            if not positive.any():
                break

            eligible = eligible[positive]
            eligible_probabilities = eligible_probabilities[positive]

            # Weighted fair ordering: the next slot goes to the stratum with
            # the smallest next-allocation threshold. Stable index ordering
            # gives deterministic tie breaking.
            thresholds = (extra_counts[eligible] + 1) / eligible_probabilities
            chosen_relative = int(np.argmin(thresholds))
            chosen_stratum = int(eligible[chosen_relative])

            counts[chosen_stratum] += 1
            extra_counts[chosen_stratum] += 1

        return counts

    def _build_combined_sample(
        self,
        dataset_name: str,
        state: PreparedColumnState,
        stratum_ids: np.ndarray,
        sample_counts: np.ndarray,
        query_id: int,
    ) -> tuple[pd.DataFrame, int]:
        """Sample each stratum and combine rows for one semantic-filter call."""
        samples: list[pd.DataFrame] = []
        total_sampled = 0

        for stratum_id, sample_count in enumerate(sample_counts.tolist()):
            if sample_count <= 0:
                continue

            local_candidates = np.flatnonzero(stratum_ids == stratum_id)

            # Build one deterministic ordering per query and stratum. Sampling
            # fractions take prefixes of this ordering, so the 1% rows are
            # guaranteed to be contained in the 5% rows for the same query.
            stratum_seed = np.random.SeedSequence([self.seed, query_id, stratum_id])
            stratum_rng = np.random.default_rng(stratum_seed)
            candidate_order = stratum_rng.permutation(local_candidates)
            chosen_local = candidate_order[:sample_count]

            original_positions = state.valid_positions[chosen_local]

            # The semantic predicate references exactly one column, so avoid
            # copying other retained semantic columns into this query sample.
            sample_df = (
                self.data_dfs[dataset_name]
                .iloc[original_positions][[state.source_column]]
                .copy()
            )
            sample_df[self.STRATUM_COLUMN] = stratum_id

            samples.append(sample_df)
            total_sampled += sample_count

        if not samples:
            raise RuntimeError("The allocated sample is empty.")

        return pd.concat(samples, ignore_index=True), total_sampled

    def _model_usage_snapshot(self) -> tuple[float, int]:
        if self.model is None:
            return 0.0, 0

        usage = getattr(
            getattr(self.model, "stats", None),
            "virtual_usage",
            None,
        )
        if usage is None:
            return 0.0, 0

        return (
            float(getattr(usage, "total_cost", 0.0) or 0.0),
            int(getattr(usage, "total_tokens", 0) or 0),
        )

    def _record_model_usage_delta(
        self,
        before: tuple[float, int],
        llm_calls: int,
    ) -> None:
        after = self._model_usage_snapshot()

        self.cost_stats["usd"] += max(0.0, after[0] - before[0])
        self.cost_stats["tokens"] += max(0, after[1] - before[1])
        self.cost_stats["llm_calls"] += int(llm_calls)

    def _estimate_memory_consumption(self) -> int:
        size = sys.getsizeof(self.data_dfs)

        for dataset_name, data_df in self.data_dfs.items():
            size += sys.getsizeof(dataset_name)
            size += int(data_df.memory_usage(index=True, deep=True).sum())

        size += sys.getsizeof(self.prepared_states)

        for key, state in self.prepared_states.items():
            size += sys.getsizeof(key)
            size += int(state.valid_positions.nbytes)
            size += int(state.normalized_embeddings.nbytes)

        size += sys.getsizeof(self.model)
        return int(size)
