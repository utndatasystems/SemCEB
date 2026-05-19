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
        return self.memory_consumption

    def get_cost_stats(self) -> dict:
        return self.cost_stats

    def reset_cost_stats(self) -> None:
        ...

    def preparation(self, data: pd.DataFrame, algorithm_kwargs: dict) -> None:
        ...

    def run(self, query: dict) -> int:
        ...
        return 1