from pathlib import Path

import pandas as pd


class DataLoader:
    """Loads locally stored data."""

    def __init__(self):
        self.folderpath_datasets_data = Path("data") / "datasets"

    def _load(self, dataset: str, scale_factor: int) -> pd.DataFrame:
        """Load raw data into pandas dataframe.

        scale_factor:
            1 = 5% of rows, shuffled
            2 = 10% of rows, shuffled
            3 = 25% of rows, shuffled
            4 = 50% of rows, shuffled
            5 = 100% of rows, shuffled
        """

        df = self._load_dataset(dataset)

        scale_factors = {
            1: 0.05,
            2: 0.10,
            3: 0.25,
            4: 0.50,
            5: 1.00,
        }

        if scale_factor not in scale_factors:
            raise ValueError(
                f"Invalid scale_factor={scale_factor}. "
                f"Allowed values are: {list(scale_factors.keys())}."
            )

        fraction = scale_factors[scale_factor]
        
        # Deterministic shuffle
        shuffled_df = df.sample(frac=1.0, replace=False, random_state=42)

        if fraction == 1.00:
            return shuffled_df.reset_index(drop=True)

        scaled_df = (
            shuffled_df
            .head(int(fraction * len(df)))
            .reset_index(drop=True)
        )

        return scaled_df

    def _load_dataset(self, dataset: str) -> pd.DataFrame:
        csv_path = self.folderpath_datasets_data / f"{dataset}.csv"
        parquet_path = self.folderpath_datasets_data / f"{dataset}.parquet"

        if csv_path.exists():
            return pd.read_csv(csv_path)

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)

        raise FileNotFoundError(
            f"Could not find dataset '{dataset}' as CSV or Parquet in "
            f"{self.folderpath_datasets_data}."
        )    

    def load(self, datasets: list[str], scale_factor: int) -> dict[str, pd.DataFrame]:
        """Load datasets into pandas dataframe."""
        dataset_df = {}
        for dataset in datasets:
            dataset_df[dataset] = self._load(dataset=dataset, scale_factor=scale_factor)
        return dataset_df