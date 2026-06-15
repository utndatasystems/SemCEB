import sys

import pandas as pd


def get_dict_memory_usage(data: dict) -> int:
    """Estimate memory usage of a flat dictionary including keys and values."""
    return sys.getsizeof(data) + sum(
        sys.getsizeof(key) + sys.getsizeof(value)
        for key, value in data.items()
    )


def get_sample_memory_usage(data_sample: dict[str, pd.DataFrame]) -> int:
    """Estimate memory usage of sampled dataframes including dict metadata."""
    return sys.getsizeof(data_sample) + sum(
        sys.getsizeof(name) + int(sample_df.memory_usage(index=True, deep=True).sum())
        for name, sample_df in data_sample.items()
    )
