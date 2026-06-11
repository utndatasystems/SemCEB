from abc import ABC, abstractmethod

import pandas as pd
from src.semceb.queries.query_specification import QuerySpecification

class AlgorithmInterface(ABC):
    """Abstract interface for algorithms."""

    @abstractmethod
    def get_memory_consumption(self) -> int:
        """
        Return the memory storage footprint of metadata structures used by the cardinality estimation algorithm in bytes.

        For example, if the algorithm holds a sample of the data, this method should return the size of that sample in bytes.
        """
        pass

    @abstractmethod
    def get_cost_stats(self) -> dict:
        """
        Return the accumulated cost statistics for cardinality estimation.
        """
        pass

    @abstractmethod
    def reset_cost_stats(self) -> None:
        """Reset tracked algorithm cost."""
        pass

    @abstractmethod
    def preparation(self, data_dfs: dict[str, pd.DataFrame], algorithm_kwargs: dict) -> None:
        """Prepare the algorithm before execution.

        This method should collect and store all information required for selectivity estimation.
        Specifially, implementations are expected to take ownership of the provided dataframes.

        During execution, the algorithm will only receive the dataset name(s)
        and column name(s) needed to perform the selectivity estimate.
        """
        pass

    @abstractmethod
    def run(self, query_spec: QuerySpecification) -> int:
        """Run the algorithm and return the estimated output cardinality for the given query."""
        pass