import pandas as pd

import json
from pathlib import Path

import lotus.settings
from lotus.models.lm import LM


class LotusBackend():
    """Model wrapper using LOTUS for ground-truth selectivity."""

    def __init__(self, model_name: str, system_prompt: str):

        self.cache_path = Path("queries") / "ground_truth_cache.json"
        self.cache = self._load_cache()

        self.name = model_name
        self.system_prompt = system_prompt
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
        query_dict: dict,
        query_str: str,
    ) -> str:
        """Create a stable cache key for a LOTUS selectivity query."""
        datasets = query_dict.get("datasets", [])
        columns = query_dict.get("columns", [])

        if isinstance(datasets, str):
            datasets = [datasets]

        if isinstance(columns, str):
            columns = [columns]

        key_data = {
            "model_name": self.name,
            "system_prompt": self.system_prompt,
            "query_type": query_type,
            "datasets": ", ".join(datasets),
            "columns": ", ".join(columns),
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

    def filtering_query(self, query_dict: dict, df: pd.DataFrame) -> int:
        """Run a semantic filter query and return the number of matching rows."""
        query_str = self._format_filtering_query(query_dict, df)

        cache_key = self._make_cache_key(
            query_type="filter",
            query_dict=query_dict,
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

    def _format_filtering_query(self, query_dict: dict, df: pd.DataFrame) -> str:
        """Format LOTUS query string for filtering."""
        self._validate_filtering_query(query_dict, df)
        return f"{query_dict['filter']} {' '.join(f'{{{col}}}' for col in query_dict['columns'])}"
    
    def _validate_filtering_query(self, query_dict: dict, df: pd.DataFrame) -> None:
        if "filter" not in query_dict or "columns" not in query_dict or "datasets" not in query_dict:
            raise ValueError("query must contain 'filter', 'columns', and 'datasets'.")

        if len(query_dict["datasets"]) != 1:
            raise ValueError("Filtering query must contain exactly one dataset.")

        for column in query_dict["columns"]:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in data.")

    def joining_query(self, query_dict: dict, data_left_df: pd.DataFrame, data_right_df: pd.DataFrame) -> int:
        """Run a semantic join query and return the number of matching rows."""
        query_str = self._format_joining_query(query_dict, data_left_df, data_right_df)
        
        cache_key = self._make_cache_key(
            query_type="join",
            query_dict=query_dict,
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
    
    def _format_joining_query(self, query_dict: dict, data_left_df: pd.DataFrame, data_right_df: pd.DataFrame) -> str:
        """Format LOTUS query string for joining."""
        self._validate_joining_query(query_dict, data_left_df, data_right_df)
        column_left, column_right = query_dict["columns"]
        return f"{query_dict['filter']} {{{column_left}:left}} {{{column_right}:right}}"

    def _validate_joining_query(
        self,
        query_dict: dict,
        data_left_df: pd.DataFrame,
        data_right_df: pd.DataFrame,
    ) -> None:
        if "filter" not in query_dict or "columns" not in query_dict or "datasets" not in query_dict:
            raise ValueError("query must contain 'filter', 'columns', and 'datasets'.")

        if len(query_dict["datasets"]) != 2:
            raise ValueError("Joining query must contain exactly two datasets.")

        if len(query_dict["columns"]) != 2:
            raise TypeError(
                f"Key column should yield two columns for a joining query, "
                f"not '{len(query_dict['columns'])}'."
            )

        column_left, column_right = query_dict["columns"]

        if column_left not in data_left_df.columns:
            raise ValueError(f"Column '{column_left}' does not exist in the left dataframe.")

        if column_right not in data_right_df.columns:
            raise ValueError(f"Column '{column_right}' does not exist in the right dataframe.")