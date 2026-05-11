import pandas as pd


class DataLoader:
    """Loads locally stored data."""

    def __init__(self):
        self.folderpath_raw_data = r"data\.raw"

    def load(self, dataset: str, scale_factor: int) -> pd.DataFrame:
        """Load raw data into pandas dataframe"""
        return pd.DataFrame()
