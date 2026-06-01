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

    def preparation(self, data_dfs: dict[str, pd.DataFrame], algorithm_kwargs: dict) -> None:
        """Prepare the algorithm before execution.

        This method should collect and store all information required for
        selectivity estimation.

        During execution, the algorithm will only receive the dataset name(s)
        and column name(s) needed to perform the selectivity estimate.
        """
        
        # Data set to evaluate
        self.data_rows = {}
        for name, df in data_dfs.items():
            self.data_rows[name] = df.shape[0]

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

        self.data_sample = {}
        for name, df in data_dfs.items():
            self.data_sample[name] = df.sample(frac=sampling_frac, random_state=42)
            if self.data_sample[name].empty:
                raise ValueError(
                    f"Sample of dataframe '{name}' is empty. Increase sampling_frac or provide more data."
                )

        self.memory_consumption = (
            sys.getsizeof(self.data_rows)
            + sys.getsizeof(self.data_sample)
            + sys.getsizeof(self.model)
        )

    def run(self, query_dict: dict) -> int:
        """Run extrapolation sampling and return estimated selectivity."""

        # Filtering
        if len(query_dict["datasets"]) == 1:

            # Evaluate sample
            name = query_dict["datasets"][0]
            sample_estimation = self.data_sample[name].sem_filter(
                user_instruction=f"{query_dict['filter']} {' '.join(f'{{{col}}}' for col in query_dict['columns'])}",
            ).shape[0]

            # Extrapolate to estimate
            selectivity_estimation = (
                sample_estimation / self.data_sample[name].shape[0] * self.data_rows[name]
            )

            # Track costs
            llm_cost_stats = self._get_costs(self.data_sample[name])
            self._add_cost(llm_cost_stats)
            
            selectivity_estimation = max(0, min(round(selectivity_estimation), self.data_rows[name]))

        # Joining
        elif len(query_dict["datasets"]) > 1:

            # Evaluate sample
            name_left, name_right = query_dict["datasets"]
            data_left = self.data_sample[name_left]
            data_right = self.data_sample[name_right]
            
            column_left, column_right = query_dict["columns"]
            query_str = f"{query_dict['filter']} {{{column_left}:left}} {{{column_right}:right}}"

            sample_estimation = data_left.sem_join(
                data_right,
                query_str,
            ).shape[0]

            # Extrapolate to estimate
            selectivity_estimation = (
                0.5 * sample_estimation / self.data_sample[name_left].shape[0] * self.data_rows[name_left] 
                + 0.5 * sample_estimation / self.data_sample[name_right].shape[0] * self.data_rows[name_right] 
            )

            # Track costs
            llm_cost_stats_left = self._get_costs(self.data_sample[name_left])
            llm_cost_stats_right = self._get_costs(self.data_sample[name_right]) # to get llm calls
            llm_cost_stats = llm_cost_stats_left
            llm_cost_stats["llm_calls"] *= llm_cost_stats_right["llm_calls"]
            self._add_cost(llm_cost_stats)

        return selectivity_estimation


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
