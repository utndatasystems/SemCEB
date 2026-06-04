import pandas as pd

import json
from pathlib import Path

import lotus.settings
from lotus.models.lm import LM
from queries.template_parser import QueryTemplatePartType
from queries.query_specification import QuerySpecification
from queries.template_parser import ColumnRef


class LotusBackend():
    """Model wrapper using LOTUS for ground-truth selectivity."""

    def __init__(self, model_name: str, system_prompt: str, scale_factor: int):

        self.cache_path = Path("queries") / "ground_truth_cache.json"
        self.cache = self._load_cache()

        self.name = model_name
        self.system_prompt = system_prompt
        self.scale_factor = scale_factor
        self.lm = LM(
            model=self.name,
            rate_limit=None,
            max_batch_size=64,
        )
        self.lm.system_prompt = self.system_prompt
        lotus.settings.configure(lm=self.lm)

    def _make_cache_key(
        self,
        query_type: str,
        query_spec: QuerySpecification,
        query_str: str,
    ) -> str:
        """Create a stable cache key for a LOTUS selectivity query."""
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
        """Load selectivity cache from disk."""
        if self.cache_path is None:
            return {}

        if not self.cache_path.exists():
            return {}

        with self.cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self) -> None:
        """Persist selectivity cache to disk."""
        if self.cache_path is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.cache_path.with_suffix(".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

        tmp_path.replace(self.cache_path)

    def _get_cached_selectivity(self, cache_key: str) -> int | None:
        """Return cached selectivity if available."""
        value = self.cache.get(cache_key)

        if value is None:
            return None

        return int(value["selectivity"])

    def _set_cached_selectivity(self, cache_key: str, selectivity: int) -> None:
        """Store selectivity in the cache."""
        self.cache[cache_key] = {
            "selectivity": int(selectivity),
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

        cached = self._get_cached_selectivity(cache_key)
        if cached is not None:
            return cached

        selectivity = df.sem_filter(
            user_instruction=query_str,
        ).shape[0]

        self._set_cached_selectivity(cache_key, selectivity)
        return selectivity  

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

        cached = self._get_cached_selectivity(cache_key)
        if cached is not None:
            return cached
        
        selectivity = data_left_df.sem_join(
            data_right_df,
            query_str,
        ).shape[0]

        self._set_cached_selectivity(cache_key, selectivity)
        return selectivity
    
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