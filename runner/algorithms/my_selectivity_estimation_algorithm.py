from typing import Any

import pandas as pd

from runner.algorithms.interface import AlgorithmInterface
from queries.query_specification import QuerySpecification


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
        """
        Return the memory storage footprint of metadata structures used by the cardinality estimation algorithm in bytes.

        For example, if the algorithm holds a sample of the data, this method should return the size of that sample in bytes.
        """
        return self.memory_consumption

    def get_cost_stats(self) -> dict:
        """
        Return the accumulated cost statistics for cardinality estimation.
        """
        return self.cost_stats

    def reset_cost_stats(self) -> None:
        """Reset tracked algorithm cost."""
        ...

    def preparation(self, data_dfs: dict[str, pd.DataFrame], algorithm_kwargs: dict) -> None:
        """Prepare the algorithm before execution.

        This method should collect and store all information required for selectivity estimation.
        Specifially, implementations are expected to take ownership of the provided dataframes.

        During execution, the algorithm will only receive the dataset name(s)
        and column name(s) needed to perform the selectivity estimate.
        """
        ...

    def run(self, query_spec: QuerySpecification) -> int:
        """Run the algorithm and return the estimated output cardinality for the given query."""
        ...
        return 1