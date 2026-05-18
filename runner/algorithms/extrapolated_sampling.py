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

    def preparation(
        self,
        query: dict,
        data: pd.DataFrame,
        model_name: str,
        system_prompt: str,
        using_cache_for_LLM: bool,
        algorithm_kwargs: dict,
    ) -> int:
        """Prepare extrapolation sampling and return the model-based ground truth."""
        self.data = data

        sampling_frac = algorithm_kwargs.get("sampling_frac", 0.1)

        if not 0 < sampling_frac <= 1:
            raise ValueError("sampling_frac must be in the interval (0, 1].")

        self.data_sample = self.data.sample(frac=sampling_frac, random_state=42)

        if self.data_sample.empty:
            raise ValueError(
                "Sample is empty. Increase sampling_frac or provide more data."
            )

        if (
            self.model is None
            or self.model.name != model_name
            or self.model.system_prompt != system_prompt
        ):
            self.model = LotusBackend(model_name=model_name, system_prompt=system_prompt, using_cache=using_cache_for_LLM)

        selectivity_ground_truth = self._obtain_ground_truth(query)
        return selectivity_ground_truth

    def _validate_query(self, query: dict) -> None:
        if "query" not in query or "column" not in query:
            raise ValueError("query must contain 'query' and 'column'.")

        if self.data is not None and query["column"] not in self.data.columns:
            raise ValueError(
                f"Column '{query['column']}' does not exist in data."
            )

    def _format_query(self, query: dict) -> str:
        """Format LOTUS query string."""
        self._validate_query(query)
        return f"{query['query']} {{{query['column']}}}"

    def _obtain_ground_truth(self, query: dict) -> int:
        """Obtain model-based selectivity ground truth."""
        if self.data is None:
            raise RuntimeError("Algorithm has not been prepared yet.")

        if self.model is None:
            raise RuntimeError("Model has not been initialized.")

        selectivity_ground_truth = self.model.filtering_query(
            self._format_query(query),
            self.data,
        )

        self.reset_cost_stats()
        return selectivity_ground_truth

    def run(self, query: dict) -> int:
        """Run extrapolation sampling and return estimated selectivity."""
        if self.data is None or self.data_sample is None:
            raise RuntimeError("Algorithm has not been prepared yet.")

        if self.model is None:
            raise RuntimeError("Model has not been initialized.")

        sample_estimation = self.model.filtering_query(
            self._format_query(query),
            self.data_sample,
        )

        selectivity_estimation = (
            sample_estimation / self.data_sample.shape[0] * self.data.shape[0]
        )

        virtual_cost_stats = {
            "usd": self.model.lm.stats.virtual_usage.total_cost,
            "llm_calls": self.data_sample.shape[0], # Calculation possible because no cascade
            "tokens": self.model.lm.stats.virtual_usage.total_tokens
        }
    
        physical_cost_stats = {
            "usd": self.model.lm.stats.physical_usage.total_cost,
            "llm_calls": self.data_sample.shape[0] - self.model.lm.stats.cache_hits, # Calculation possible because no cascade
            "tokens": self.model.lm.stats.physical_usage.total_tokens
        }

        self.add_cost(virtual_cost_stats, physical_cost_stats)

        return max(0, min(round(selectivity_estimation), self.data.shape[0]))
