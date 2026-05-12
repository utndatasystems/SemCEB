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
        self.cost_usd = 0.0

    def reset_cost(self) -> None:
        """Reset tracked algorithm cost."""
        self.cost_usd = 0.0

    def add_cost(self, cost_usd: float) -> None:
        """Add cost amount to tracked algorithm cost."""
        self.cost_usd += cost_usd

    def preparation(
        self,
        query: dict[str, Any],
        data: pd.DataFrame,
        model: str,
        system_prompt: str,
        algorithm_kwargs: dict[str, Any],
    ) -> int:
        """Prepare the algorithm for one query.

        Args:
            query: Query config from the benchmark query file.
            data: Loaded dataset for the query.
            model: Model name selected by the benchmark runner.
            system_prompt: System prompt selected by the benchmark runner.
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
