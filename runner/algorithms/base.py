import pandas as pd
from pandas import DataFrame


class AlgorithmBase:
    """Base class for algorithm implementation"""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.cost_usd = 0
        self.data = None  # fill in 'self.preparation()'
        self.system_prompt = None  # fill in 'self.preparation()'

    def reset_cost(self):
        """Resets tracked algorithm costs"""
        self.cost_usd = 0

    def add_cost(self, cost_usd: float) -> None:
        """Add cost amount to algorim costs"""
        self.cost_usd += cost_usd

    def preparation(self, data: pd.DataFrame, system_prompt: str) -> None:
        """Preparation phase of algorithm - e.g. computation of embeddings"""
        self.data = ...
        self.system_prompt = ...
        raise NotImplementedError(
            "Implement the preparation phase of your algorithm!"
        )

    def run(self, query: str) -> int:
        """Run semantic selectivity algorithm. Returns estimated selectivity (int)"""
        raise NotImplementedError(
            "Implement the main function of your algorithm!"
        )
