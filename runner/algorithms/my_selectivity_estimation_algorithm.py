from typing import Any

import pandas as pd

from runner.algorithms.base import AlgorithmBase


class MySelectivityEstimationAlgorithm(AlgorithmBase):
    """Template for a semantic selectivity estimation algorithm.

    Replace this class with your own algorithm implementation.

    The benchmark runner calls two methods:

    1. preparation(...)
       Called once before each query run. Use this to store data, initialize
       models, build indexes, compute embeddings, or read algorithm-specific
       settings.

    2. run(...)
       Called after preparation(...). Use this to estimate the selectivity
       for the given query.

    The return value of run(...) must be an integer representing the estimated
    number of matching rows.
    """

    def __init__(self, name: str, version: str) -> None:
        super().__init__(name, version)

        self.data: pd.DataFrame | None = None
        self.model: Any | None = None
        self.algorithm_kwargs: dict[str, Any] = {}

    def preparation(
        self,
        query: dict[str, Any],
        data: pd.DataFrame,
        model_name: str,
        system_prompt: str,
        using_cache_for_LLM: bool,
        algorithm_kwargs: dict[str, Any],
    ) -> int:
        """Prepare the algorithm for one benchmark query.

        Args:
            query: Query dictionary from the JSONL query file.
                   Usually contains fields such as:
                   - "id"
                   - "dataset"
                   - "query"
                   - "column"

            data: Dataset loaded by the benchmark runner.

            model_name: Model name selected by the benchmark configuration.

            system_prompt: System prompt selected by the benchmark configuration.

            using_cache_for_LLM: Enables caching of llm results.

            algorithm_kwargs: Algorithm-specific settings from the benchmark
                              config file.

        Returns:
            Selectivity ground truth as an integer.

        Notes:
            The current benchmark runner expects preparation(...) to return
            the selectivity ground truth. If your algorithm does not compute
            ground truth itself, replace this section with the project's
            preferred ground-truth method.
        """
        self.data = data
        self.algorithm_kwargs = algorithm_kwargs

        # Example: read custom algorithm parameters.
        sample_size = algorithm_kwargs.get("sample_size", 10)

        # TODO: initialize your model, index, cache, embeddings, etc.
        # Example:
        # self.model = MyModel(model=model, system_prompt=system_prompt)

        # TODO: compute or retrieve the ground truth for this query.
        # This placeholder uses 0 so the template is runnable.
        selectivity_ground_truth = 0

        return selectivity_ground_truth

    def run(self, query: dict[str, Any]) -> int:
        """Estimate selectivity for one query.

        Args:
            query: Query dictionary from the JSONL query file.

        Returns:
            Estimated selectivity as an integer.
        """
        if self.data is None:
            raise RuntimeError("Algorithm has not been prepared yet.")

        # Typical fields:
        query_text = query["query"]
        column_name = query["column"]

        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' does not exist in data.")

        # TODO: replace this with your selectivity estimation logic.
        #
        # Example ideas:
        # - sample rows and extrapolate
        # - use embeddings
        # - use a classifier
        # - use metadata/statistics
        # - call an LLM on selected rows
        #
        # The result should be the estimated number of matching rows.
        selectivity_estimation = 0

        # Optional: track algorithm/model cost.
        # self.add_cost(0.001)

        # Keep result within valid bounds.
        selectivity_estimation = max(
            0,
            min(round(selectivity_estimation), len(self.data)),
        )

        return selectivity_estimation
