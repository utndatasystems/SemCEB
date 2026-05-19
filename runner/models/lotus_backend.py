import pandas as pd
import lotus.settings

from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models.lm import LM


class LotusBackend:
    """Model wrapper using LOTUS / LiteLLM."""
    # TODO - define interface to allow other backends than only lotus as well

    def __init__(self, model_name: str, system_prompt: str, using_cache: bool):
        self.name = model_name
        self.system_prompt = system_prompt
        self.using_cache = using_cache

        cache_config = CacheConfig(
            cache_type=CacheType.IN_MEMORY,
            max_size=1000,
        )
        cache = CacheFactory.create_cache(cache_config)

        self.lm = LM(
            model=self.name,
            rate_limit=None,
            max_batch_size=64,
            cache=cache,
        )

        self.lm.system_prompt = self.system_prompt

        lotus.settings.configure(
            lm=self.lm,
            enable_cache=self.using_cache,
        )

    def filtering_query(self, query: str, df: pd.DataFrame) -> int:
        """Run a semantic filter query and return the number of matching rows."""
        query = self._format_query(query, df)
        return df.sem_filter(
            user_instruction=query,
        ).shape[0]

    def _format_query(self, query: dict, df: pd.DataFrame) -> str:
        """Format LOTUS query string."""
        self._validate_query(query, df)
        return f"{query['query']} {{{query['column']}}}"
    
    def _validate_query(self, query: dict, df: pd.DataFrame) -> None:
        if "query" not in query or "column" not in query:
            raise ValueError("query must contain 'query' and 'column'.")

        if df is not None and query["column"] not in df.columns:
            raise ValueError(
                f"Column '{query['column']}' does not exist in data."
            )
        

    def get_costs(self, data: pd.DataFrame) -> dict:
        """Return virtual and physical cost stats."""
        virtual_cost_stats = {
            "usd": self.lm.stats.virtual_usage.total_cost,
            "llm_calls": data.shape[0], # Calculation possible because no cascade
            "tokens": self.lm.stats.virtual_usage.total_tokens
        }
        physical_cost_stats = {
            "usd": self. lm.stats.physical_usage.total_cost,
            "llm_calls": data.shape[0] - self.lm.stats.cache_hits, # Calculation possible because no cascade
            "tokens": self.lm.stats.physical_usage.total_tokens
        }
        return {"virtual": virtual_cost_stats, "physical": physical_cost_stats}