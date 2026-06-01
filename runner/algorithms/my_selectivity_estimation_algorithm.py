from typing import Any

import pandas as pd

from runner.algorithms.interface import AlgorithmInterface


class MySelectivityEstimationAlgorithm(AlgorithmInterface):

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

        self.memory_consumption = -1
        self.cost_stats = {
            "usd": -1,
            "llm_calls": -1,
            "tokens": -1
            }

    def get_memory_consumption(self) -> int:
        """Return tracked memory consumption."""
        return self.memory_consumption

    def get_cost_stats(self) -> dict:
        """Return tracked algorithm cost stats."""
        return self.cost_stats

    def reset_cost_stats(self) -> None:
        """Reset tracked algorithm cost."""
        ...

    def preparation(self, data_dfs: dict[str, pd.DataFrame], algorithm_kwargs: dict) -> None:
        """Prepare the algorithm before execution.

        This method should collect and store all information required for
        selectivity estimation.

        During execution, the algorithm will only receive the dataset name(s)
        and column name(s) needed to perform the selectivity estimate.
        """
        ...

    def run(self, query: dict) -> int:
        """Run the algorithm and return the estimated result."""
        ...
        return 1