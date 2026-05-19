import pandas as pd
import lotus.settings

from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models.lm import LM


class LotusBackend():
    """Model wrapper using LOTUS for ground-truth selectivity."""

    def __init__(self, model_name: str, system_prompt: str):
        self.name = model_name
        self.system_prompt = system_prompt
        self.lm = LM(
            model=self.name,
            rate_limit=None,
            max_batch_size=64,
        )
        self.lm.system_prompt = self.system_prompt
        lotus.settings.configure(lm=self.lm)

    def filtering_query(self, query_dict: dict, df: pd.DataFrame) -> int:
        """Run a semantic filter query and return the number of matching rows."""
        query_str = self._format_query(query_dict, df)
        return df.sem_filter(
            user_instruction=query_str,
        ).shape[0]

    def _format_query(self, query_dict: dict, df: pd.DataFrame) -> str:
        """Format LOTUS query string."""
        self._validate_query(query_dict, df)
        return f"{query_dict['query']} {{{query_dict['column']}}}"
    
    def _validate_query(self, query_dict: dict, df: pd.DataFrame) -> None:
        if "query" not in query_dict or "column" not in query_dict:
            raise ValueError("query must contain 'query' and 'column'.")

        if df is not None and query_dict["column"] not in df.columns:
            raise ValueError(
                f"Column '{query_dict['column']}' does not exist in data."
            )