from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from semceb.algorithms.cardinality_estimate import CardinalityEstimate
from semceb.algorithms.helpers import get_dict_memory_usage
from semceb.algorithms.interface import AlgorithmInterface
from semceb.queries.query_specification import QuerySpecification
from semceb.queries.template_parser import QueryTemplatePartType


SUPPORTED_SOURCE_COLUMNS = (
    "product_title",
    "review_title",
    "review_text",
    "description_json",
    "details_json",
    "features_json",
)


@dataclass
class ImportanceSamplingState:
    dataset_name: str
    source_column: str
    data_df: pd.DataFrame
    row_embeddings: np.ndarray
    total_rows: int
    memory_bytes: int


class ImportanceSampling(AlgorithmInterface):
    """Unify-style semantic cardinality estimation with importance sampling."""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.model = None
        self.embedding_model_key: str | None = None
        self.num_samples = 100
        self.num_distance_groups = 10
        self.importance_temperature = 4.0
        self.seed = 42
        self.importance_values: list[float] | None = None
        self.prepared_states: dict[tuple[str, str], ImportanceSamplingState] = {}
        self.memory_consumption = 0
        self.reset_cost_stats()

    def get_memory_consumption(self) -> int:
        return self.memory_consumption

    def get_cost_stats(self) -> dict:
        return self.cost_stats

    def reset_cost_stats(self) -> None:
        self.cost_stats = {
            "usd": 0,
            "llm_calls": 0,
            "tokens": 0,
        }

        if self.model is not None:
            self.model.reset_stats()
            import lotus.settings

            lotus.settings.configure(lm=self.model)

    def preparation(
        self,
        data_dfs: dict[str, pd.DataFrame],
        algorithm_kwargs: dict,
    ) -> None:
        self.embedding_model_key = self._require_algorithm_kwarg(
            algorithm_kwargs,
            "embedding_model_key",
        )
        self.num_samples = int(algorithm_kwargs.get("num_samples", self.num_samples))
        self.num_distance_groups = int(
            algorithm_kwargs.get("num_distance_groups", self.num_distance_groups)
        )
        self.importance_temperature = float(
            algorithm_kwargs.get(
                "importance_temperature",
                self.importance_temperature,
            )
        )
        self.seed = int(algorithm_kwargs.get("seed", self.seed))
        self.importance_values = self._parse_importance_values(
            algorithm_kwargs.get("importance_values")
        )

        self._validate_positive_int(self.num_samples, "num_samples")
        self._validate_positive_int(self.num_distance_groups, "num_distance_groups")
        if self.importance_temperature < 0:
            raise ValueError("importance_temperature must be non-negative.")

        model_name = algorithm_kwargs.get("model_name")
        self._validate_model_name(model_name)
        self._initialize_model(
            model_name=model_name,
            system_prompt=algorithm_kwargs.get("system_prompt"),
        )

        self.data_rows = {
            name: df.shape[0]
            for name, df in data_dfs.items()
        }
        self.prepared_states = {}
        self.memory_consumption = get_dict_memory_usage(self.data_rows) + sys.getsizeof(
            self.model
        )

        for dataset_name, data_df in data_dfs.items():
            for source_column in SUPPORTED_SOURCE_COLUMNS:
                if source_column not in data_df.columns:
                    continue

                embedding_column = self.embedding_column_name(
                    source_column=source_column,
                    embedding_model_key=self.embedding_model_key,
                )
                if embedding_column not in data_df.columns:
                    continue

                state = self._build_state(
                    dataset_name=dataset_name,
                    source_column=source_column,
                    embedding_column=embedding_column,
                    data_df=data_df,
                )
                if state is None:
                    continue

                self.prepared_states[(dataset_name, source_column)] = state
                self.memory_consumption += state.memory_bytes

    def run(self, query_spec: QuerySpecification) -> CardinalityEstimate:
        if len(query_spec.datasets) != 1:
            return CardinalityEstimate.unsupported(
                "ImportanceSampling supports only single-table filter queries."
            )

        column_refs = [
            part.value
            for part in query_spec.filter_parsed.parts
            if part.type == QueryTemplatePartType.COLUMN_REF
        ]
        if len(column_refs) != 1:
            return CardinalityEstimate.unsupported(
                "ImportanceSampling supports only predicates with exactly one column reference."
            )

        if self.embedding_model_key is None:
            raise RuntimeError("ImportanceSampling must be prepared before running.")

        dataset_name = query_spec.datasets[0].table_ref
        source_column = column_refs[0].column_name
        state = self.prepared_states.get((dataset_name, source_column))
        if state is None:
            return CardinalityEstimate.unsupported(
                f"ImportanceSampling does not support dataset '{dataset_name}' and column '{source_column}'."
            )

        query_embedding_values = query_spec.embeddings.get(self.embedding_model_key)
        if query_embedding_values is None:
            raise ValueError(
                f"Query {query_spec.id} does not provide embedding key '{self.embedding_model_key}'."
            )

        query_embedding = self._normalize_vector(
            np.asarray(query_embedding_values, dtype=np.float32)
        )
        distances = 1.0 - np.clip(state.row_embeddings @ query_embedding, -1.0, 1.0)
        groups = self._build_distance_groups(distances)
        importance = self._importance_values_for_groups(groups, distances)
        sample_counts = self._allocate_samples(
            groups=groups,
            importance=importance,
            sample_budget=min(self.num_samples, state.row_embeddings.shape[0]),
        )

        cardinality = self._estimate_filter_cardinality(
            query_spec=query_spec,
            state=state,
            groups=groups,
            sample_counts=sample_counts,
        )
        return CardinalityEstimate.int(cardinality)

    def _build_state(
        self,
        dataset_name: str,
        source_column: str,
        embedding_column: str,
        data_df: pd.DataFrame,
    ) -> ImportanceSamplingState | None:
        valid_mask = data_df[source_column].notna() & data_df[embedding_column].notna()
        subset = data_df.loc[valid_mask].copy()
        if subset.empty:
            return None

        row_embeddings = []
        valid_positions = []
        for position, embedding in enumerate(subset[embedding_column].to_list()):
            embedding_array = np.asarray(embedding, dtype=np.float32)
            if embedding_array.ndim != 1 or embedding_array.size == 0:
                continue
            if not np.isfinite(embedding_array).all():
                continue

            try:
                normalized_embedding = self._normalize_vector(embedding_array)
            except ValueError:
                continue

            row_embeddings.append(normalized_embedding)
            valid_positions.append(position)

        if not row_embeddings:
            return None

        subset = subset.iloc[valid_positions].reset_index(drop=True)
        embedding_matrix = np.stack(row_embeddings).astype(np.float32, copy=False)
        memory_bytes = (
            int(subset.memory_usage(index=True, deep=True).sum())
            + int(embedding_matrix.nbytes)
            + sys.getsizeof(dataset_name)
            + sys.getsizeof(source_column)
        )

        return ImportanceSamplingState(
            dataset_name=dataset_name,
            source_column=source_column,
            data_df=subset,
            row_embeddings=embedding_matrix,
            total_rows=int(data_df.shape[0]),
            memory_bytes=memory_bytes,
        )

    def _build_distance_groups(self, distances: np.ndarray) -> list[np.ndarray]:
        effective_group_count = min(
            self.num_distance_groups,
            self.num_samples,
            int(distances.shape[0]),
        )
        sorted_indices = np.argsort(distances, kind="stable")
        return [
            group.astype(np.int64, copy=False)
            for group in np.array_split(sorted_indices, effective_group_count)
            if group.size > 0
        ]

    def _importance_values_for_groups(
        self,
        groups: list[np.ndarray],
        distances: np.ndarray,
    ) -> np.ndarray:
        if self.importance_values is not None:
            if len(self.importance_values) != len(groups):
                raise ValueError(
                    "importance_values must have the same length as the effective number "
                    f"of distance groups ({len(groups)})."
                )
            return self._normalize_weights(np.asarray(self.importance_values, dtype=np.float64))

        total_rows = sum(group.size for group in groups)
        base_mass = np.asarray(
            [group.size / total_rows for group in groups],
            dtype=np.float64,
        )
        group_centers = np.asarray(
            [float(np.mean(distances[group])) for group in groups],
            dtype=np.float64,
        )
        closeness_boost = np.exp(-self.importance_temperature * group_centers)
        return self._normalize_weights(base_mass * closeness_boost)

    def _allocate_samples(
        self,
        groups: list[np.ndarray],
        importance: np.ndarray,
        sample_budget: int,
    ) -> np.ndarray:
        if sample_budget <= 0:
            raise ValueError("sample_budget must be positive.")

        capacities = np.asarray([group.size for group in groups], dtype=np.int64)
        raw_counts = importance * sample_budget
        sample_counts = np.floor(raw_counts).astype(np.int64)
        sample_counts = np.minimum(sample_counts, capacities)

        for group_idx in np.argsort(-importance):
            if sample_counts.sum() >= sample_budget:
                break
            if sample_counts[group_idx] == 0 and capacities[group_idx] > 0:
                sample_counts[group_idx] = 1

        while sample_counts.sum() < sample_budget:
            remaining_capacity = capacities - sample_counts
            candidates = np.flatnonzero(remaining_capacity > 0)
            if candidates.size == 0:
                break

            deficits = raw_counts[candidates] - sample_counts[candidates]
            best_candidate = candidates[int(np.argmax(deficits))]
            sample_counts[best_candidate] += 1

        while sample_counts.sum() > sample_budget:
            candidates = np.flatnonzero(sample_counts > 1)
            if candidates.size == 0:
                candidates = np.flatnonzero(sample_counts > 0)
            worst_candidate = candidates[int(np.argmin(importance[candidates]))]
            sample_counts[worst_candidate] -= 1

        return sample_counts

    def _estimate_filter_cardinality(
        self,
        query_spec: QuerySpecification,
        state: ImportanceSamplingState,
        groups: list[np.ndarray],
        sample_counts: np.ndarray,
    ) -> int:
        sampled_parts = []
        rng = np.random.default_rng(self.seed + int(query_spec.id))
        group_column = self._unused_column_name(state.data_df, "__semceb_importance_group")

        for group_idx, (group, sample_count) in enumerate(zip(groups, sample_counts)):
            if sample_count <= 0:
                continue

            sampled_indices = rng.choice(
                group,
                size=int(sample_count),
                replace=False,
            )
            sampled_part = state.data_df.iloc[sampled_indices].copy()
            sampled_part[group_column] = group_idx
            sampled_parts.append(sampled_part)

        if not sampled_parts:
            return 0

        sampled_df = pd.concat(sampled_parts, axis=0, ignore_index=True)

        result_df = sampled_df.sem_filter(
            user_instruction=self._build_filter_query_str(query_spec),
        )

        positives_by_group = (
            result_df[group_column].value_counts().to_dict()
            if group_column in result_df.columns
            else {}
        )

        estimate = 0.0
        for group_idx, (group, sample_count) in enumerate(zip(groups, sample_counts)):
            if sample_count <= 0:
                continue

            positives = int(positives_by_group.get(group_idx, 0))
            estimate += group.size * (positives / sample_count)

        self._track_costs(sampled_df)
        return max(0, min(round(estimate), state.total_rows))

    def _initialize_model(self, model_name: str, system_prompt: str | None) -> None:
        from lotus.cache import CacheConfig, CacheFactory, CacheType
        from lotus.models.lm import LM
        import lotus.settings

        cache_config = CacheConfig(
            cache_type=CacheType.IN_MEMORY,
            max_size=1000,
        )
        cache = CacheFactory.create_cache(cache_config)

        self.model = LM(
            model=model_name,
            rate_limit=None,
            max_batch_size=64,
            cache=cache,
        )
        self.model.system_prompt = system_prompt

        lotus.settings.configure(
            lm=self.model,
            enable_cache=True,
        )

    def _build_filter_query_str(self, query_spec: QuerySpecification) -> str:
        query_str = ""

        for part in query_spec.filter_parsed.parts:
            if part.type == QueryTemplatePartType.TEXT:
                query_str += part.value
            elif part.type == QueryTemplatePartType.COLUMN_REF:
                query_str += f"{{{part.value.column_name}}}"

        return query_str

    def _track_costs(self, data: pd.DataFrame) -> None:
        self._add_cost(self._get_costs(data))

    def _get_costs(self, data: pd.DataFrame) -> dict:
        return {
            "usd": self.model.stats.virtual_usage.total_cost,
            "llm_calls": data.shape[0],
            "tokens": self.model.stats.virtual_usage.total_tokens,
        }

    def _add_cost(self, cost_stats: dict) -> None:
        self.cost_stats["usd"] += cost_stats["usd"]
        self.cost_stats["llm_calls"] += cost_stats["llm_calls"]
        self.cost_stats["tokens"] += cost_stats["tokens"]

    def _validate_model_name(self, model_name: str | None) -> None:
        if not model_name:
            raise ValueError("model_name must be a valid name of a model.")

    @staticmethod
    def _require_algorithm_kwarg(
        algorithm_kwargs: dict[str, Any],
        key: str,
    ) -> str:
        value = algorithm_kwargs.get(key)
        if value in (None, ""):
            raise ValueError(f"ImportanceSampling requires algorithm_kwargs['{key}'].")
        return str(value)

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _parse_importance_values(value: Any) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("importance_values must be a list of numbers.")
        if not value:
            raise ValueError("importance_values must not be empty.")
        return [float(item) for item in value]

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("Embedding vectors must be finite and non-zero.")
        return (vector / norm).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("Importance weights must be a non-empty vector.")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Importance weights must be finite and non-negative.")

        total = float(weights.sum())
        if total <= 0:
            raise ValueError("At least one importance weight must be positive.")
        return weights / total

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
        return f"{source_column}_embeddings_{cls.sanitize_embedding_model_key(embedding_model_key)}"

    @staticmethod
    def _unused_column_name(data_df: pd.DataFrame, base_name: str) -> str:
        column_name = base_name
        counter = 1
        while column_name in data_df.columns:
            column_name = f"{base_name}_{counter}"
            counter += 1
        return column_name
