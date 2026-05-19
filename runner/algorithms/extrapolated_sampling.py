import sys
import pandas as pd

from runner.algorithms.base import AlgorithmBase
from runner.models.lotus_backend import LotusBackend


class ExtrapolatedSampling(AlgorithmBase):
    """Algorithm based on extrapolation sampling."""

    def __init__(self, name: str, version: str):
        super().__init__(name, version)
        self.data: pd.DataFrame | None = None
        self.data_sample: pd.DataFrame | None = None
        self.model: LotusBackend | None = None

    def reset_cost_stats(self) -> None:
        """Reset tracked algorithm cost."""
        self.cost_stats = {
            "virtual": {"usd": 0, "llm_calls": 0, "tokens": 0},
            "physical": {"usd": 0, "llm_calls": 0, "tokens": 0},
            }

        if self.model is not None:
            self.model.lm.reset_stats()

    def preparation(self, data: pd.DataFrame, algorithm_kwargs: dict) -> int:
        """Prepare extrapolation sampling."""
        self.data = data

        sampling_frac = algorithm_kwargs.get("sampling_frac", -1)

        if not 0 < sampling_frac <= 1:
            raise ValueError("sampling_frac must be in the interval (0, 1].")

        self.data_sample = self.data.sample(frac=sampling_frac, random_state=42)

        if self.data_sample.empty:
            raise ValueError(
                "Sample is empty. Increase sampling_frac or provide more data."
            )

        # Track additional memory consumption of algorithm
        self.memory_consumption = sys.getsizeof(self.data_sample)

    def run(self, query: dict, backend: LotusBackend) -> int:
        """Run extrapolation sampling and return estimated selectivity."""

        # Evaluate sample
        sample_estimation = backend.filtering_query(query, self.data_sample)

        # Extrapolate to estimate
        selectivity_estimation = (
            sample_estimation / self.data_sample.shape[0] * self.data.shape[0]
        )

        # Track costs
        llm_cost_stats = backend.get_costs(self.data)
        self.add_cost(llm_cost_stats["virtual"], llm_cost_stats["physical"])

        return max(0, min(round(selectivity_estimation), self.data.shape[0]))
