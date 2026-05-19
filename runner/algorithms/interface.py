from abc import ABC, abstractmethod

import pandas as pd


class AlgorithmInterface(ABC):
    """Abstract interface for algorithms."""

    @abstractmethod
    def get_memory_consumption(self) -> int:
        """Return tracked memory consumption."""
        pass

    @abstractmethod
    def get_cost_stats(self) -> dict:
        """Return tracked algorithm cost stats."""
        pass

    @abstractmethod
    def reset_cost_stats(self) -> None:
        """Reset tracked algorithm cost."""
        pass

    @abstractmethod
    def preparation(self, data: pd.DataFrame, algorithm_kwargs: dict) -> None:
        """Prepare the algorithm before running."""
        pass

    @abstractmethod
    def run(self, query: dict) -> int:
        """Run the algorithm and return the estimated result."""
        pass