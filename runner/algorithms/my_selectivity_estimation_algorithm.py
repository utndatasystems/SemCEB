import pandas as pd
from pandas import DataFrame

from runner.algorithms.base import AlgorithmBase


class MySelectivityEstimationAlgorithm(AlgorithmBase):
    """Design algorithm based on your idea here"""

    def __init__(self, name: str, version: str):
        super().__init__(name, version)

    def preparation(self, data: pd.DataFrame, system_prompt: str) -> None:
        """Preparation phase of your algorithm - e.g. computation of embeddings"""
        self.data = data
        self.system_prompt = system_prompt
        ...

    def run(self, query: str) -> int:
        """Run semantic selectivity algorithm using your algorithm. Returns estimated selectivity (int)"""
        relevant_data = self.data.get(query, None)
        ...

        import random

        selectivity_estimation = max(0, round(random.gauss(mu=123, sigma=10)))
        return selectivity_estimation
