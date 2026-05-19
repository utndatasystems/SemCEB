import sys
import pandas as pd

from runner.algorithms.interface import AlgorithmInterface


class ExtrapolatedSampling(AlgorithmInterface):
    """Algorithm based on extrapolation sampling."""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

        self.model = None
        self.reset_cost_stats()

    def get_memory_consumption(self) -> int:
        """Return tracked memory consumption."""
        return self.memory_consumption

    def get_cost_stats(self) -> dict:
        """Return tracked algorithm cost stats."""
        return self.cost_stats

    def reset_cost_stats(self) -> None:
        """Reset tracked algorithm cost."""
        self.cost_stats = {
            "usd": 0,
            "llm_calls": 0,
            "tokens": 0
            }

        if self.model is not None:
            self.model.reset_stats()

    def preparation(self, data: pd.DataFrame, algorithm_kwargs: dict) -> None:
        """Prepare extrapolation sampling."""
        
        # Data set to evaluate
        self.data_rows = data.shape[0]


        # Algorithm specific arguments
        sampling_frac = algorithm_kwargs.get("sampling_frac", -1)
        if not 0 < sampling_frac <= 1:
            raise ValueError("sampling_frac must be in the interval (0, 1].")
        
        model_name = algorithm_kwargs.get("model_name", None)
        if not model_name:
            raise ValueError("model_name must be a valid name of a model.")

        import lotus.settings
        from lotus.cache import CacheConfig, CacheFactory, CacheType
        from lotus.models.lm import LM
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
        self.model.system_prompt = algorithm_kwargs.get("system_prompt", None)
        lotus.settings.configure(
            lm=self.model,
            enable_cache=True,
        )        

        self.data_sample = data.sample(frac=sampling_frac, random_state=42)
        if self.data_sample.empty:
            raise ValueError(
                "Sample is empty. Increase sampling_frac or provide more data."
            )

        self.memory_consumption = (
            sys.getsizeof(self.data_rows)
            + sys.getsizeof(self.data_sample)
            + sys.getsizeof(self.model)
        )

    def run(self, query_dict: dict) -> int:
        """Run extrapolation sampling and return estimated selectivity."""

        # Evaluate sample
        sample_estimation = self.data_sample.sem_filter(
            user_instruction=f"{query_dict['query']} {{{query_dict['column']}}}",
        ).shape[0]

        # Extrapolate to estimate
        selectivity_estimation = (
            sample_estimation / self.data_sample.shape[0] * self.data_rows
        )

        # Track costs
        llm_cost_stats = self._get_costs(self.data_sample)
        self._add_cost(llm_cost_stats)

        return max(0, min(round(selectivity_estimation), self.data_rows))
    

    def _get_costs(self, data: pd.DataFrame) -> dict:
        """Return virtual (= without caching) cost stats."""
        return  {
            "usd": self.model.stats.virtual_usage.total_cost,
            "llm_calls": data.shape[0], # Calculation possible because no cascade
            "tokens": self.model.stats.virtual_usage.total_tokens
        }


    def _add_cost(self, cost_stats: dict) -> None:
        """Add cost amount to tracked algorithm cost."""
        self.cost_stats["usd"] += cost_stats["usd"]
        self.cost_stats["llm_calls"] += cost_stats["llm_calls"]
        self.cost_stats["tokens"] += cost_stats["tokens"]
