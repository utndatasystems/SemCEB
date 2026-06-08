from pathlib import Path
from rich.prompt import Confirm
import pandas as pd
from src.semceb.utils.console import console


class DataLoader:
    """Loads locally stored data."""

    def __init__(self):
        self.folderpath_datasets_data = Path("data") / "datasets"

    def load(self, datasets: list[str], scale_factor: int | None = None) -> dict[str, pd.DataFrame]:
        """
        Load datasets into pandas DataFrames.

        scale_factor:
            Number of rows to load per table.
            If None, the full table is loaded.
        """

        datasets_df: dict[str, pd.DataFrame] = {}

        for dataset in datasets:
            if dataset not in datasets_df.keys():
                if dataset.startswith("amazon-reviews"):
                    datasets_df = self._load_amazon_reviews_dataset(
                        datasets_df=datasets_df,
                        dataset=dataset,
                        scale_factor=scale_factor,
                    )
                else:
                    raise NotImplementedError(
                        f"The dataset '{dataset}' can not be loaded!"
                    )

        return datasets_df

    def _load_amazon_reviews_dataset(
        self,
        datasets_df: dict[str, pd.DataFrame],
        dataset: str,
        scale_factor: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load Amazon Reviews dataset tables."""

        products_df = self._load_dataset(dataset="amazon-reviews/products_filtered")
        reviews_df = self._load_dataset(dataset="amazon-reviews/reviews_filtered")

        # Shuffle main dataset table
        products_df = products_df.sample(frac=1.0, replace=False, random_state=42).reset_index(drop=True)

        # Apply scale factor on main dataset table
        if scale_factor is not None:
            if scale_factor <= 0:
                raise ValueError(
                    f"Invalid scale_factor={scale_factor}. "
                    "It must be a positive integer or None."
                )

            products_df = products_df.head(scale_factor).reset_index(drop=True)
        else:
            console.print(
                "[bold yellow]WARNING:[/bold yellow] "
                "[yellow]No scale_factor was provided. Loading the full dataset. "
                "This may cause high computational demand and many LLM calls, which can increase costs.[/yellow]"
            )

            continue_loading = Confirm.ask(
                "[bold yellow]Do you want to continue loading the full dataset?[/bold yellow]",
                default=False,
            )

            if not continue_loading:
                raise RuntimeError(
                    "Aborted because no scale_factor was provided and the user declined "
                    "to load the full dataset."
                )

        # Keep only reviews related to selected products
        selected_parent_asins = products_df["parent_asin"].dropna().unique()

        # Join other dataset tabels on main dataset table
        reviews_df = reviews_df[
            reviews_df["asin"].isin(selected_parent_asins)
        ].reset_index(drop=True)

        # Save dataset dfs
        datasets_df["amazon-reviews/products_filtered"] = products_df
        datasets_df["amazon-reviews/reviews_filtered"] = reviews_df

        return datasets_df

    def _load_dataset(self, dataset: str) -> pd.DataFrame:
        """Load raw dataset file as a pandas DataFrame."""

        csv_path = self.folderpath_datasets_data / f"{dataset}.csv"
        parquet_path = self.folderpath_datasets_data / f"{dataset}.parquet"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        elif parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(
                f"Could not find dataset '{dataset}' as CSV or Parquet in "
                f"{self.folderpath_datasets_data}."
            )

        return df