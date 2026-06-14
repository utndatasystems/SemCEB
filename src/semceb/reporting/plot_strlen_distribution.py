from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from semceb.reporting.plot_params import apply_plot_params
from semceb.utils.console import console


def _sql_string_literal(value: str) -> str:
    """Return a DuckDB-safe string literal."""

    return "'" + value.replace("'", "''") + "'"


def _sql_identifier(identifier: str) -> str:
    """Return a quoted SQL identifier."""

    return '"' + identifier.replace('"', '""') + '"'


def _escape_latex_text(value: str) -> str:
    """Escape text for Matplotlib's LaTeX rendering."""

    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


class StringLengthDistributionPlotMixin:
    """Helpers for plotting string-length distributions for embedded text columns."""

    AMAZON_REVIEWS_DATASET_DIR = (
        Path(__file__).resolve().parents[2]
        / "semceb"
        / "data"
        / "amazon-reviews"
        / "processed"
        / "Arts_Crafts_and_Sewing__raw_5core"
    )

    STRING_TYPES = {"VARCHAR", "STRING", "TEXT"}

    def _plot_string_length_distributions(self) -> None:
        """Plot string-length histograms for embedded text columns."""

        dataset_dir = self.AMAZON_REVIEWS_DATASET_DIR

        if not dataset_dir.exists():
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"Amazon reviews dataset directory not found: {dataset_dir}"
            )
            return

        embeddings_paths = sorted(dataset_dir.glob("*_with_embeddings.parquet"))

        if not embeddings_paths:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"No embedded parquet files found in {dataset_dir}"
            )
            return

        self.plot_dir.mkdir(parents=True, exist_ok=True)

        for embeddings_path in embeddings_paths:
            self._plot_string_length_distribution_for_dataset(
                embeddings_path=embeddings_path,
            )

    def _plot_string_length_distribution_for_dataset(
        self,
        embeddings_path: Path,
    ) -> None:
        """Plot text-length histograms for all embedded string columns in one dataset."""

        dataset_name = embeddings_path.stem.removesuffix("_with_embeddings")
        base_path = embeddings_path.with_name(f"{dataset_name}.parquet")

        if not base_path.exists():
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"Base parquet file not found for {embeddings_path.name}: {base_path}"
            )
            return

        con = duckdb.connect()

        try:
            source_columns = self._discover_embedding_source_columns(
                con=con,
                embeddings_path=embeddings_path,
            )
            column_types = self._load_column_types(con=con, parquet_path=base_path)
            text_columns = [
                column_name
                for column_name in source_columns
                if column_types.get(column_name, "").upper() in self.STRING_TYPES
            ]

            if not text_columns:
                console.print(
                    "[bold yellow]Warning:[/bold yellow] "
                    f"No string columns with embeddings found in {base_path.name}"
                )
                return

            con.execute(
                "CREATE OR REPLACE TEMP VIEW source_data AS "
                f"SELECT * FROM read_parquet({_sql_string_literal(str(base_path))})"
            )

            for column_name in text_columns:
                histogram = self._fetch_string_length_histogram(
                    con=con,
                    column_name=column_name,
                )

                if histogram.empty:
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        f"No non-null values found for {column_name} in {base_path.name}"
                    )
                    continue

                self._save_string_length_distribution_plot(
                    dataset_name=dataset_name,
                    column_name=column_name,
                    histogram=histogram,
                )
        finally:
            con.close()

    def _discover_embedding_source_columns(
        self,
        con: duckdb.DuckDBPyConnection,
        embeddings_path: Path,
    ) -> list[str]:
        """Return the unique source columns that have embedding columns."""

        schema = self._load_column_schema(con=con, parquet_path=embeddings_path)
        source_columns: list[str] = []

        for column_name in schema:
            if "_embeddings" not in column_name:
                continue

            source_column_name = column_name.partition("_embeddings")[0]
            if source_column_name:
                source_columns.append(source_column_name)

        return list(dict.fromkeys(source_columns))

    def _load_column_schema(
        self,
        con: duckdb.DuckDBPyConnection,
        parquet_path: Path,
    ) -> dict[str, str]:
        """Load a parquet schema as a column-to-type mapping."""

        schema_df = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({_sql_string_literal(str(parquet_path))})"
        ).fetchdf()

        return {
            str(row["column_name"]): str(row["column_type"])
            for _, row in schema_df.iterrows()
        }

    def _load_column_types(
        self,
        con: duckdb.DuckDBPyConnection,
        parquet_path: Path,
    ) -> dict[str, str]:
        """Load the source parquet schema as a column-to-type mapping."""

        return self._load_column_schema(con=con, parquet_path=parquet_path)

    def _fetch_string_length_histogram(
        self,
        con: duckdb.DuckDBPyConnection,
        column_name: str,
    ) -> pd.DataFrame:
        """Compute a fine-grained histogram of string lengths using DuckDB."""

        query = f"""
            SELECT
                length({_sql_identifier(column_name)}) AS string_length,
                COUNT(*) AS count
            FROM source_data
            WHERE {_sql_identifier(column_name)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """

        return con.execute(query).fetchdf()

    def _save_string_length_distribution_plot(
        self,
        dataset_name: str,
        column_name: str,
        histogram: pd.DataFrame,
    ) -> None:
        """Render and save one string-length histogram as a PDF."""

        apply_plot_params(
            fig_height=2.8,
            scale=1,
            double_column=False,
        )

        fig, axis = plt.subplots()

        lengths = histogram["string_length"].astype(int).tolist()
        counts = histogram["count"].astype(int).tolist()

        axis.bar(
            lengths,
            counts,
            width=0.9,
            color="#1859FF",
            edgecolor="#1859FF",
            linewidth=0.8,
        )

        axis.set_title(_escape_latex_text(column_name))
        axis.set_xlabel(r"\#Characters")
        axis.set_ylabel("Number of Strings")
        axis.set_xlim(left=min(lengths) - 0.5, right=max(lengths) + 0.5)
        axis.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=7))
        axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        axis.grid(axis="y", alpha=0.55)
        axis.grid(axis="x", visible=False)

        axis.text(
            0.98,
            0.98,
            f"Max: {max(lengths)}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize="medium",
            bbox={
                "facecolor": "white",
                "edgecolor": "#666666",
                "boxstyle": "round,pad=0.25",
                "alpha": 0.9,
            },
        )

        sns.despine(
            ax=axis,
            top=True,
            right=True,
            left=False,
            bottom=False,
        )

        fig.tight_layout()

        pdf_path = (
            self.plot_dir
            / f"{dataset_name}__{column_name}_text_length_dist.pdf"
        )

        fig.savefig(pdf_path, bbox_inches="tight")
        console.print(
            f"[green]✓[/green] Saved string length distribution plot to [bold]{pdf_path}[/bold]"
        )

        plt.close(fig)
