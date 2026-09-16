from pathlib import Path
from rich.prompt import Confirm
import pandas as pd
import duckdb
from semceb.utils.console import console


class DataLoader:
    """Loads locally stored benchmark dataset files into memory."""

    def __init__(self):
        """Initialize the dataset loader with the local dataset root path."""
        self.folderpath_datasets_data = Path("data") / "datasets"

    def load(
        self, datasets: list[str], scale_factor: int | None = None
    ) -> dict[str, pd.DataFrame]:
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

        products_dataset = "amazon-reviews/products_filtered_with_embeddings"
        reviews_dataset = "amazon-reviews/reviews_filtered_with_embeddings"

        products_parquet = self.folderpath_datasets_data / f"{products_dataset}.parquet"
        reviews_parquet = self.folderpath_datasets_data / f"{reviews_dataset}.parquet"

        products_df = self._load_sampled_products_from_parquet(
            parquet_path=products_parquet,
            scale_factor=scale_factor,
        )
        reviews_df = self._load_filtered_reviews_from_parquet(
            parquet_path=reviews_parquet,
            products_df=products_df,
        )

        datasets_df[products_dataset] = products_df
        datasets_df[reviews_dataset] = reviews_df

        return datasets_df

    def _load_sampled_products_from_parquet(
        self,
        parquet_path: Path,
        scale_factor: int | None,
    ) -> pd.DataFrame:
        """Select product rows before loading their embedding-heavy columns."""

        with duckdb.connect() as con:
            product_rows = con.execute(
                """
                SELECT file_row_number
                FROM read_parquet(?, file_row_number=true)
                ORDER BY file_row_number
                """,
                [str(parquet_path)],
            ).df()

        product_rows = self._shuffle_products(product_rows)
        product_rows = self._apply_scale_factor(product_rows, scale_factor)
        product_rows["sample_order"] = range(len(product_rows))

        with duckdb.connect() as con:
            con.register("selected_product_rows", product_rows)
            products_df = con.execute(
                """
                SELECT p.* EXCLUDE (file_row_number)
                FROM read_parquet(?, file_row_number=true) AS p
                JOIN selected_product_rows AS s
                  ON p.file_row_number = s.file_row_number
                ORDER BY s.sample_order
                """,
                [str(parquet_path)],
            ).df()

        return products_df.reset_index(drop=True)

    def _load_filtered_reviews_from_parquet(
        self,
        parquet_path: Path,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter reviews in DuckDB before converting them to pandas."""

        selected_asins = pd.DataFrame(
            {"asin": products_df["parent_asin"].dropna().unique()}
        )

        with duckdb.connect() as con:
            con.register("selected_asins", selected_asins)
            reviews_df = con.execute(
                """
                SELECT r.* EXCLUDE (file_row_number)
                FROM read_parquet(?, file_row_number=true) AS r
                WHERE r.asin IN (SELECT asin FROM selected_asins)
                ORDER BY r.file_row_number
                """,
                [str(parquet_path)],
            ).df()

        return reviews_df.reset_index(drop=True)

    def _shuffle_products(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Randomly shuffle product records in a stable way for sampling."""
        return products_df.sample(frac=1.0, replace=False, random_state=42).reset_index(
            drop=True
        )

    def _apply_scale_factor(
        self,
        products_df: pd.DataFrame,
        scale_factor: int | None = None,
    ) -> pd.DataFrame:
        """Trim the products dataframe according to the requested scale factor."""
        if scale_factor is None:
            self._confirm_full_dataset_load()
            return products_df

        if scale_factor <= 0:
            raise ValueError(
                f"Invalid scale_factor={scale_factor}. "
                "It must be a positive integer or None."
            )

        return products_df.head(scale_factor).reset_index(drop=True)

    def _confirm_full_dataset_load(self) -> None:
        """Warn the user before loading the full dataset and require confirmation."""
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
