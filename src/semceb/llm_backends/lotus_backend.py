import pandas as pd

import json
from pathlib import Path

import lotus.settings
from lotus.models.lm import LM
from semceb.queries.template_parser import QueryTemplatePartType
from semceb.queries.query_specification import QuerySpecification
from semceb.queries.template_parser import ColumnRef


class LotusBackend():
    """LOTUS backend for ground-truth cardinality estimation and caching."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        scale_factor: int | None,
    ):
        """Initialize the LOTUS backend, cache, and model instance."""
        self.safe_model_name = self._safe_cache_name(model_name)
        self.safe_scale_factor = self._safe_scale_factor_name(scale_factor)

        self.cache_path = (
            Path("benchmark_queries")
            / f"ground_truth_cache_{self.safe_model_name}_{self.safe_scale_factor}.json"
        )
        self.query_results_dir = Path("results") / "raw" / "query_results"
        self.cache = self._load_cache()

        self.name = model_name
        self.system_prompt = system_prompt
        self.scale_factor = scale_factor
        self.lm = LM(
            model=self.name,
            rate_limit=None,
            max_batch_size=256,
        )
        self.lm.system_prompt = self.system_prompt
        lotus.settings.configure(lm=self.lm)

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

    def _save_query_result(self, query_spec: QuerySpecification, result_df: pd.DataFrame) -> None:
        """Persist the full ground-truth match set for a query to CSV."""

        self.query_results_dir.mkdir(parents=True, exist_ok=True)

        result_path = (
            self.query_results_dir
            / f"{self.safe_model_name}_sf{self._format_scale_factor_for_filename()}_q{query_spec.id}.csv"
        )

        tmp_path = result_path.with_suffix(".tmp")
        result_df.to_csv(tmp_path, index=False)
        tmp_path.replace(result_path)

    def _get_cached_cardinality(self, cache_key: str) -> int | None:
        """Return cached cardinality if available."""
        value = self.cache.get(cache_key)

        if value is None:
            return None

        return int(value["cardinality"])

    def _set_cached_cardinality(self, cache_key: str, cardinality: int, selectivity: float) -> None:
        """Store cardinality in the cache."""
        self.cache[cache_key] = {
            "cardinality": int(cardinality),
            "selectivity": float(selectivity),
        }
        self._save_cache()

    def filtering_query(self, query_spec: QuerySpecification, df: pd.DataFrame) -> int:
        """Run a semantic filter query and return the number of matching rows."""
        query_str = self._format_filtering_query(query_spec, df)

        cache_key = self._make_cache_key(
            query_type="filter",
            query_spec=query_spec,
            query_str=query_str,
        )

        cached = self._get_cached_cardinality(cache_key)
        if cached is not None:
            return cached

        result_df = df.sem_filter(
            user_instruction=query_str,
        )
        cardinality = result_df.shape[0]

        self._save_query_result(query_spec, result_df)

        self._set_cached_cardinality(cache_key, cardinality, selectivity=cardinality / df.shape[0])
        return cardinality  

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

    def joining_query(self, query_spec: QuerySpecification, data_left_df: pd.DataFrame, data_right_df: pd.DataFrame) -> int:
        """Run a semantic join query and return the number of matching rows."""
        query_str = self._format_joining_query(query_spec, data_left_df, data_right_df)
        
        cache_key = self._make_cache_key(
            query_type="join",
            query_spec=query_spec,
            query_str=query_str,
        )

        cached = self._get_cached_cardinality(cache_key)
        if cached is not None:
            return cached
        
        result_df = data_left_df.sem_join(
            data_right_df,
            query_str,
        )
        cardinality = result_df.shape[0]

        self._save_query_result(query_spec, result_df)

        self._set_cached_cardinality(cache_key, cardinality, selectivity=cardinality / (data_left_df.shape[0] * data_right_df.shape[0]))
        return cardinality
    
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
