import pandas as pd
import lotus.settings

from lotus.cache import CacheConfig, CacheFactory, CacheType
from lotus.models.lm import LM


class LotusBackend:
    """Model wrapper using LOTUS / LiteLLM."""

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
        return df.sem_filter(
            user_instruction=query,
        ).shape[0]

    def get_stats(self) -> dict[str, int | float | str]:
        """Return virtual model usage stats, excluding cache savings."""
        usage = self.lm.stats.virtual_usage

        return {
            "model": self.lm.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "costs": usage.total_cost,
        }
