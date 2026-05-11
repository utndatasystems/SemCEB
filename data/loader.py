from pathlib import Path

import pandas as pd


class DataLoader:
    """Loads locally stored data."""

    def __init__(self):
        self.folderpath_raw_data = Path(r"data\.raw")

    def load(self, dataset: str, scale_factor: int) -> pd.DataFrame:
        """Load raw data into pandas dataframe."""

        filepath = (
            self.folderpath_raw_data
            / dataset
            / f"scale_factor_{scale_factor}"
            / "table.csv"
        )

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        return pd.read_csv(filepath)
