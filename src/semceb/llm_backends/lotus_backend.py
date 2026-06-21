from collections.abc import Callable
from duckdb import df
import pandas as pd

import json
from pathlib import Path

import lotus.settings
from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models.lm import LM
from lotus.dtype_extensions import ImageArray
from semceb.queries.template_parser import QueryTemplatePartType
from semceb.queries.query_specification import QuerySpecification
from semceb.queries.template_parser import ColumnRef


class LotusBackend():
    """LOTUS backend for ground-truth cardinality estimation and caching."""

    DATASETS_IMAGE_COLUMNS = {"main_image_local"}
    LOTUS_CACHE_MAX_SIZE = 10_000_000

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        scale_factor: int | None,
        *,
        enable_lotus_cache: bool = True,
        repeated_runs: int = 1,
    ):
        """Initialize the LOTUS backend, cache, and model instance."""
        self.safe_model_name = self._safe_cache_name(model_name)
        self.safe_scale_factor = self._safe_scale_factor_name(scale_factor)
        self.enable_lotus_cache = enable_lotus_cache
        self.repeated_runs = repeated_runs

        self.cache_path = (
            Path("benchmark_queries")
            / f"ground_truth_cache_{self.safe_model_name}_{self.safe_scale_factor}.json"
        )
        self.query_results_dir = Path("results") / "raw" / "query_results"
        self.cache = self._load_cache()

        self.name = model_name
        self.system_prompt = system_prompt
        self.scale_factor = scale_factor
        self._warm_litellm_pydantic_models()
        cache = None
        if self.enable_lotus_cache:
            cache_config = CacheConfig(
                cache_type=CacheType.IN_MEMORY,
                max_size=self.LOTUS_CACHE_MAX_SIZE,
            )
            cache = CacheFactory.create_cache(cache_config)
        self.lm = LM(
            model=self.name,
            rate_limit=None,
            max_batch_size=256,
            cache=cache,
        )
        self.lm.system_prompt = self.system_prompt
        lotus.settings.configure(
            lm=self.lm,
            enable_cache=self.enable_lotus_cache,
        )

    def _warm_litellm_pydantic_models(self) -> None:
        """Eagerly rebuild LiteLLM Pydantic models before threaded batch use.

        LiteLLM lazily rebuilds some response models on first access. Under
        threaded batch completion, that can race and surface as missing
        ``__pydantic_core_schema__`` on ``Message``. Rebuilding these upfront
        keeps initialization single-threaded.
        """
        try:
            from litellm.types import utils as litellm_utils
        except Exception:
            return

        for class_name in [
            "Message",
            "Choices",
            "StreamingChoices",
            "Delta",
            "ModelResponse",
        ]:
            model_class = getattr(litellm_utils, class_name, None)
            model_rebuild = getattr(model_class, "model_rebuild", None)

            if model_rebuild is None:
                continue

            model_rebuild(force=True)

    def _safe_cache_name(self, model_name: str) -> str:
        """Return a filesystem-safe name for the model cache file."""
        return (
            model_name
            .replace("/", "__")
            .replace(":", "_")
            .replace(" ", "_")
        )

    def _safe_scale_factor_name(self, scale_factor: int | None) -> str:
        """Return a filesystem-safe name for the scale-factor cache file suffix."""
        if scale_factor is None:
            return "sf_full"

        return f"sf{scale_factor}"

    def _format_scale_factor_for_filename(self) -> str:
        """Format the scale factor for query result filenames."""
        if self.scale_factor is None:
            return "full"

        return str(self.scale_factor)

    def _validate_repeated_runs(self) -> None:
        """Ensure the configured repeated-run count is valid."""

        if self.repeated_runs < 1:
            raise ValueError("repeated_runs must be at least 1.")

    def _make_cache_key(
        self,
        query_type: str,
        query_spec: QuerySpecification,
        query_str: str,
    ) -> str:
        """Create a stable cache key of the query."""
        datasets = query_spec.datasets

        if isinstance(datasets, str):
            datasets = [datasets]

        key_data = {
            "model_name": self.name,
            "system_prompt": self.system_prompt,
            "query_type": query_type,
            "datasets": ", ".join([f"{dataset_spec.alias}:{dataset_spec.table_ref}" for dataset_spec in datasets]),
            "scale_factor": self.scale_factor,
            "query": query_str,
        }

        return " | ".join(
            f"{key}='{value}'"
            for key, value in key_data.items()
        )

    def _load_cache(self) -> dict:
        """Load cardinality cache from disk."""
        if self.cache_path is None:
            return {}

        if not self.cache_path.exists():
            return {}

        with self.cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self) -> None:
        """Persist cardinality cache to disk."""
        if self.cache_path is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.cache_path.with_suffix(".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

        tmp_path.replace(self.cache_path)

    def _get_query_result_path(self, query_spec: QuerySpecification) -> Path:
        """Return the final parquet path for one cached query result."""

        return (
            self.query_results_dir
            / f"{self.safe_model_name}_sf{self._format_scale_factor_for_filename()}_q{query_spec.id}.parquet"
        )

    def _get_repeated_query_result_path(
        self,
        query_spec: QuerySpecification,
        run_index: int,
    ) -> Path:
        """Return a temporary parquet path for one repeated run."""

        result_path = self._get_query_result_path(query_spec)
        return result_path.with_name(f"{result_path.stem}__run{run_index}{result_path.suffix}")

    def _persist_query_result(self, result_df: pd.DataFrame, output_path: Path) -> None:
        """Persist one query result dataframe to parquet atomically."""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        columns_to_drop = [
            column_name
            for column_name in result_df.columns
            if column_name == "main_image_local" or column_name.endswith(".main_image_local")
        ]
        persisted_df = result_df.drop(columns=columns_to_drop, errors="ignore")
        persisted_df.to_parquet(tmp_path, index=False)
        tmp_path.replace(output_path)

    def _save_query_result(self, query_spec: QuerySpecification, result_df: pd.DataFrame) -> None:
        """Persist the full ground-truth match set for a query to Parquet."""

        self._persist_query_result(result_df, self._get_query_result_path(query_spec))

    def _prefix_join_output_columns(
        self,
        result_df: pd.DataFrame,
        data_left_df: pd.DataFrame,
        data_right_df: pd.DataFrame,
        left_prefix: str,
        right_prefix: str,
    ) -> pd.DataFrame:
        """Prefix join output columns by dataset to make their origin explicit."""

        left_columns = set(data_left_df.columns)
        right_columns = set(data_right_df.columns)
        rename_map: dict[str, str] = {}

        for column_name in result_df.columns:
            if column_name.endswith(":left"):
                base_name = column_name.removesuffix(":left")
                rename_map[column_name] = f"{left_prefix}.{base_name}"
                continue

            if column_name.endswith(":right"):
                base_name = column_name.removesuffix(":right")
                rename_map[column_name] = f"{right_prefix}.{base_name}"
                continue

            if column_name in left_columns and column_name not in right_columns:
                rename_map[column_name] = f"{left_prefix}.{column_name}"
                continue

            if column_name in right_columns and column_name not in left_columns:
                rename_map[column_name] = f"{right_prefix}.{column_name}"

        if not rename_map:
            return result_df

        return result_df.rename(columns=rename_map)

    def _get_cached_cardinality(self, cache_key: str) -> int | None:
        """Return cached cardinality if available."""
        value = self.cache.get(cache_key)

        if value is None or not isinstance(value, dict):
            return None

        try:
            cardinality = int(value["cardinality"])
        except (KeyError, TypeError, ValueError):
            return None

        if self.repeated_runs <= 1:
            return cardinality

        repeated = value.get("cardinalities_repeated")
        if not isinstance(repeated, list) or len(repeated) != self.repeated_runs:
            return None

        try:
            repeated_cardinalities = [int(entry) for entry in repeated]
        except (TypeError, ValueError):
            return None

        if cardinality != self._select_median_cardinality(repeated_cardinalities):
            return None

        return cardinality

    def _set_cached_cardinality(
        self,
        cache_key: str,
        cardinality: int,
        selectivity: float,
        cardinalities_repeated: list[int],
    ) -> None:
        """Store cardinality in the cache."""
        self.cache[cache_key] = {
            "cardinality": int(cardinality),
            "selectivity": float(selectivity),
            "cardinalities_repeated": [int(value) for value in cardinalities_repeated],
        }
        self._save_cache()

    def _select_median_run_index(self, cardinalities: list[int]) -> int:
        """Return the index of the run whose cardinality is the median order statistic."""

        if not cardinalities:
            raise ValueError("At least one repeated cardinality is required.")

        ranked_indices = sorted(
            range(len(cardinalities)),
            key=lambda index: (cardinalities[index], index),
        )
        return ranked_indices[len(ranked_indices) // 2]

    def _select_median_cardinality(self, cardinalities: list[int]) -> int:
        """Return the cardinality of the median repeated run."""

        return int(cardinalities[self._select_median_run_index(cardinalities)])

    def _execute_query_with_repeats(
        self,
        *,
        cache_key: str,
        query_spec: QuerySpecification,
        run_query: Callable[[], pd.DataFrame],
        selectivity_denominator: int,
    ) -> int:
        """Execute one semantic query repeatedly and keep the median run."""

        self._validate_repeated_runs()

        cached = self._get_cached_cardinality(cache_key)
        if cached is not None:
            return cached

        cardinalities_repeated: list[int] = []
        repeated_result_paths: list[Path] = []
        median_run_index: int | None = None

        try:
            for run_index in range(self.repeated_runs):
                result_df = run_query()
                cardinalities_repeated.append(int(result_df.shape[0]))

                repeated_result_path = self._get_repeated_query_result_path(
                    query_spec=query_spec,
                    run_index=run_index,
                )
                self._persist_query_result(result_df, repeated_result_path)
                repeated_result_paths.append(repeated_result_path)

            median_run_index = self._select_median_run_index(cardinalities_repeated)
            median_cardinality = int(cardinalities_repeated[median_run_index])
            repeated_result_paths[median_run_index].replace(
                self._get_query_result_path(query_spec)
            )

            self._set_cached_cardinality(
                cache_key,
                cardinality=median_cardinality,
                selectivity=median_cardinality / selectivity_denominator,
                cardinalities_repeated=cardinalities_repeated,
            )
            return median_cardinality
        finally:
            for run_index, repeated_result_path in enumerate(repeated_result_paths):
                if median_run_index == run_index and not repeated_result_path.exists():
                    continue
                if repeated_result_path.exists():
                    repeated_result_path.unlink()

    def _load_image_column(
        self,
        df: pd.DataFrame,
        image_column: str,
        image_root: Path,
    ) -> pd.DataFrame:
        """Return a copy of the dataframe with an image column loaded as an ImageArray."""
        df = df.copy()

        image_paths = [
            str(image_root / image_path) if pd.notna(image_path) else None
            for image_path in df[image_column]
        ]

        df[image_column] = ImageArray(image_paths)
        return df

    def _resolve_column_dataset_ref(
        self,
        query_spec: QuerySpecification,
        column_ref: ColumnRef,
    ) -> str:
        """Return the dataset alias for a column reference."""

        if column_ref.dataset_ref is not None:
            return column_ref.dataset_ref

        if len(query_spec.datasets) == 1:
            return query_spec.datasets[0].alias

        raise ValueError(
            f"Column '{column_ref.column_name}' has no dataset reference. "
            "Dataset reference is required when querying multiple datasets."
        )

    def _load_referenced_image_columns(
        self,
        query_spec: QuerySpecification,
        dataframes: list[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        """Load referenced image columns for LOTUS semantic operations."""

        image_columns_by_dataset: dict[str, set[str]] = {}

        for part in query_spec.filter_parsed.parts:
            if part.type != QueryTemplatePartType.COLUMN_REF:
                continue

            column_ref = part.value

            if column_ref.column_name not in self.DATASETS_IMAGE_COLUMNS:
                continue

            dataset_ref = self._resolve_column_dataset_ref(
                query_spec=query_spec,
                column_ref=column_ref,
            )

            image_columns_by_dataset.setdefault(
                dataset_ref,
                set(),
            ).add(column_ref.column_name)

        if not image_columns_by_dataset:
            return dataframes

        prepared_dataframes = list(dataframes)

        for idx, dataset_spec in enumerate(query_spec.datasets):
            image_columns = image_columns_by_dataset.get(dataset_spec.alias)

            if not image_columns:
                continue

            dataset_root = (
                Path("data")
                / "datasets"
                / dataset_spec.table_ref.split("/")[0]
            )
            image_root = dataset_root / "images"

            for image_column in image_columns:
                prepared_dataframes[idx] = self._load_image_column(
                    df=prepared_dataframes[idx],
                    image_column=image_column,
                    image_root=image_root,
                )

        return prepared_dataframes

    def filtering_query(self, query_spec: QuerySpecification, df: pd.DataFrame) -> int:
        """Run a semantic filter query and return the number of matching rows."""
        query_str = self._format_filtering_query(query_spec, df)

        cache_key = self._make_cache_key(
            query_type="filter",
            query_spec=query_spec,
            query_str=query_str,
        )

        [df] = self._load_referenced_image_columns(
            query_spec=query_spec,
            dataframes=[df],
        )

        return self._execute_query_with_repeats(
            cache_key=cache_key,
            query_spec=query_spec,
            run_query=lambda: df.sem_filter(
                user_instruction=query_str,
            ),
            selectivity_denominator=df.shape[0],
        )

    def _format_filtering_query(self, query_spec: QuerySpecification, df: pd.DataFrame) -> str:
        """Format LOTUS query string for filtering."""
        self._validate_filtering_query(query_spec, df)

        # Supporting '<alias>.<column>' as well as just '<column>' when just filtering on one dataset
        query_str = ""
        for part in query_spec.filter_parsed.parts:
            if part.type == QueryTemplatePartType.TEXT:
                query_str += part.value
            if part.type == QueryTemplatePartType.COLUMN_REF:
                query_str += f"{{{part.value.column_name}}}"
        return query_str
    
    def _validate_filtering_query(self, query_spec: QuerySpecification, df: pd.DataFrame) -> None:
        """Validate that the filter query is well-formed for a single dataset."""

        if len(query_spec.datasets) != 1:
            raise ValueError("Filtering query must contain exactly one dataset.")

        columns_refs = [part.value for part in query_spec.filter_parsed.parts if part.type == QueryTemplatePartType.COLUMN_REF]
        
        if not columns_refs:
            raise ValueError("Filtering query requires at least one column reference.")

        for column_ref in columns_refs:
            if column_ref.column_name not in df.columns:
                raise ValueError(f"Column '{column_ref.column_name}' does not exist in data.")

    def joining_query(
        self,
        query_spec: QuerySpecification,
        data_left_df: pd.DataFrame,
        data_right_df: pd.DataFrame,
        data_left_ref: str,
        data_right_ref: str,
    ) -> int:
        """Run a semantic join query and return the number of matching rows."""
        query_str = self._format_joining_query(query_spec, data_left_df, data_right_df)
        
        cache_key = self._make_cache_key(
            query_type="join",
            query_spec=query_spec,
            query_str=query_str,
        )

        data_left_df, data_right_df = self._load_referenced_image_columns(
            query_spec=query_spec,
            dataframes=[data_left_df, data_right_df],
        )

        return self._execute_query_with_repeats(
            cache_key=cache_key,
            query_spec=query_spec,
            run_query=lambda: self._prefix_join_output_columns(
                result_df=data_left_df.sem_join(
                    data_right_df,
                    query_str,
                ),
                data_left_df=data_left_df,
                data_right_df=data_right_df,
                left_prefix=data_left_ref,
                right_prefix=data_right_ref,
            ),
            selectivity_denominator=data_left_df.shape[0] * data_right_df.shape[0],
        )
    
    def _format_joining_query(
        self,
        query_spec: QuerySpecification,
        data_left_df: pd.DataFrame,
        data_right_df: pd.DataFrame,
    ) -> str:
        """Format LOTUS query string for joining.

        Currently supports exactly two datasets.
        """

        self._validate_joining_query(query_spec, data_left_df, data_right_df)

        columns_by_dataset = self._get_columns_by_dataset(query_spec)

        if len(columns_by_dataset) != 2:
            raise ValueError("Joining query must contain exactly two datasets.")

        dataset_side = {
            columns_by_dataset[0][0]: "left",
            columns_by_dataset[1][0]: "right",
        }

        query_parts: list[str] = []

        for part in query_spec.filter_parsed.parts:
            if part.type == QueryTemplatePartType.TEXT:
                query_parts.append(part.value)
            elif part.type == QueryTemplatePartType.COLUMN_REF:
                query_parts.append(
                    self._format_lotus_join_column(part, dataset_side)
                )
            else:
                raise ValueError(f"Unknown query template part type: {part.type}")

        return "".join(query_parts)
    

    def _format_lotus_join_column(self, column_ref: ColumnRef, dataset_side: dict[str, str]) -> str:
        """Format a Lotus join column reference using the dataset side mapping."""

        if column_ref.value.dataset_ref not in dataset_side:
            raise ValueError(
                f"Unknown dataset '{column_ref.value.dataset_ref}' in column reference '{column_ref}'. "
                f"Expected one of: {list(dataset_side.keys())}."
            )

        return f"{{{column_ref.value.column_name}:{dataset_side[column_ref.value.dataset_ref]}}}"

    def _validate_joining_query(
        self,
        query_spec: QuerySpecification,
        data_left_df: pd.DataFrame,
        data_right_df: pd.DataFrame,
    ) -> None:
        """Validate that a joining query references both datasets and valid columns."""

        if len(query_spec.datasets) != 2:
            raise ValueError("Joining query must contain exactly two datasets.")

        dataset_spec_left, dataset_spec_right = query_spec.datasets

        dataframes_by_dataset_name = {
            dataset_spec_left.alias: data_left_df,
            dataset_spec_right.alias: data_right_df,
        }

        columns_by_dataset = self._get_columns_by_dataset(query_spec)

        for dataset_name, columns in columns_by_dataset:
            if not columns:
                raise ValueError(
                    f"Dataset '{dataset_name}' is not used in the query. "
                    "A joining query must use at least one column from each dataset."
                )

        for dataset_name, columns in columns_by_dataset:
            df = dataframes_by_dataset_name[dataset_name]

            for column in columns:
                if column not in df.columns:
                    raise ValueError(
                        f"Column '{column}' does not exist in dataset "
                        f"'{dataset_name}'. Available columns are: {list(df.columns)}."
                    )

    def _get_columns_by_dataset(self, query_spec: QuerySpecification) -> list[tuple[str, list[str]]]:
        """Return column references grouped by dataset name.

        The returned list follows the order of query_spec.datasets.

        Example 1:
            query_spec.datasets = ["products", "reviews"]

            {reviews.review_text} and {products.title}

        becomes:
            [
                ("products", ["title"]),
                ("reviews", ["review_text"]),
            ]

        Example 2:
            query_spec.datasets = ["products as p1", "products as p1"]

            {p1.title} and {p2.title}

        becomes:
            [
                ("p", ["title"]),
                ("r", ["review_text"]),
            ]
        """

        columns_by_dataset: dict[str, list[str]] = {
            dataset_spec.alias: []
            for dataset_spec in query_spec.datasets
        }

        for part in query_spec.filter_parsed.parts:
            if part.type != QueryTemplatePartType.COLUMN_REF:
                continue

            if part.value.dataset_ref not in columns_by_dataset:
                raise ValueError(
                    f"Unknown dataset '{part.value.dataset_ref}' in column reference "
                    f"'{part.value}'. Expected one of: {query_spec.datasets}."
                )

            columns_by_dataset[part.value.dataset_ref].append(part.value.column_name)

        return [
            (dataset_alias, columns_by_dataset)
            for dataset_alias, columns_by_dataset in columns_by_dataset.items()
        ]
