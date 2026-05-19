from pathlib import Path
import pandas as pd
from data.downloader import DataDownloader


class DataLoader:
    """Loads locally stored data."""

    def __init__(self):
        self.folderpath_raw_data = Path("data") / ".raw"

    def load(self, dataset: str, scale_factor: int) -> pd.DataFrame:
        """Load raw data into pandas dataframe.

        scale_factor:
            1 = original dataset
            2 = 50%
            3 = 25%
            4 = 10%
            5 = 5%
        """

        path = self.folderpath_raw_data / f"{dataset}.csv"
        df = pd.read_csv(path)

        scale_factors = {
            1: 1.00,
            2: 0.50,
            3: 0.25,
            4: 0.10,
            5: 0.05,
        }

        if scale_factor not in scale_factors:
            raise ValueError(
                f"Invalid scale_factor={scale_factor}. "
                f"Allowed values are: {list(scale_factors.keys())}."
            )

        fraction = scale_factors[scale_factor]

        if fraction == 1.00:
            return df

        return (
            df.sample(frac=fraction, random_state=42)
            .sort_index()
            .reset_index(drop=True)
        )