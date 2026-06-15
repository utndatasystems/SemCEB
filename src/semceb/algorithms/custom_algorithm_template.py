from typing import Any

import pandas as pd

from semceb.algorithms.interface import AlgorithmInterface
from semceb.queries.query_specification import QuerySpecification
from semceb.algorithms.cardinality_estimate import CardinalityEstimate, CardinalityEstimateKind


class CustomAlgorithmTemplate(AlgorithmInterface):
    """Template algorithm implementation for development and prototyping.

    This class provides a minimal algorithm stub that satisfies the
    benchmark algorithm interface. It should be replaced with a real
    cardinality estimation algorithm implementation.
    """

    def __init__(self, name: str, version: str):
        """Initialize a prototype algorithm with default metadata values."""
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
        """Reset internal algorithm cost statistics to default values."""
        ...

    def preparation(self, data_dfs: dict[str, pd.DataFrame], algorithm_kwargs: dict) -> None:
        """Prepare the algorithm before execution.

        This method should collect and store all information required for selectivity estimation.
        Specifially, implementations are expected to take ownership of the provided dataframes.

        During execution, the algorithm will only receive the dataset name(s)
        and column name(s) needed to perform the selectivity estimate.
        """
        ...

    def run(self, query_spec: QuerySpecification) -> CardinalityEstimate:
        """Estimate the output cardinality for a single query.

        Use the algorithm state prepared in `preparation` to approximate how
        many rows should be produced by `query_spec` without loading the full
        dataset again.
        """
        ...
        import random
        return random.choice([
            CardinalityEstimate(kind=CardinalityEstimateKind.INT, value=1),
            CardinalityEstimate(kind=CardinalityEstimateKind.UNSUPPORTED, reason="This is a placeholder estimate indicating that the algorithm does not support this query type."),
        ])