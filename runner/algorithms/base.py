from typing import Any

import pandas as pd


class AlgorithmBase:
    """Base class for benchmark algorithm implementations.

    Required implementation contract:

    - preparation(...) prepares the algorithm for one query and returns the
      selectivity ground truth.
    - run(...) executes the algorithm for one query and returns the estimated
      selectivity.
    """

    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
        self.cost_stats = {
            "virtual": {"usd": 0, "llm_calls": 0, "tokens": 0},
            "physical": {"usd": 0, "llm_calls": 0, "tokens": 0},
            }

    def reset_cost_stats(self) -> None:
        """Reset tracked algorithm cost."""
        self.cost_stats = {
            "virtual": {"usd": 0, "llm_calls": 0, "tokens": 0},
            "physical": {"usd": 0, "llm_calls": 0, "tokens": 0},
            }

    def add_cost(self, virtual_cost_stats: dict, physical_cost_stats: dict) -> None:
        """Add cost amount to tracked algorithm cost."""
        self.cost_stats["virtual"]["usd"] += virtual_cost_stats["usd"]
        self.cost_stats["virtual"]["llm_calls"] += virtual_cost_stats["llm_calls"]
        self.cost_stats["virtual"]["tokens"] += virtual_cost_stats["tokens"]

        self.cost_stats["physical"]["usd"] += physical_cost_stats["usd"]
        self.cost_stats["physical"]["llm_calls"] += physical_cost_stats["llm_calls"]
        self.cost_stats["physical"]["tokens"] += physical_cost_stats["tokens"]

    def preparation(
        self,
        query: dict[str, Any],
        data: pd.DataFrame,
        model_name: str,
        system_prompt: str,
        using_cache_for_LLM: bool,
        algorithm_kwargs: dict[str, Any],
    ) -> int:
        """Prepare the algorithm for one query.

        Args:
            query: Query config from the benchmark query file.
            data: Loaded dataset for the query.
            model_name: Model name selected by the benchmark runner.
            system_prompt: System prompt selected by the benchmark runner.
            using_cache_for_LLM: Enables caching of llm results.
            algorithm_kwargs: Algorithm-specific configuration.

        Returns:
            The selectivity ground truth for the query.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement preparation()."
        )

    def run(self, query: dict[str, Any]) -> int:
        """Run the algorithm for one query.

        Args:
            query: Query config from the benchmark query file.

        Returns:
            Estimated selectivity.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run()."
        )
