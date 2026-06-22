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


def _format_k_tick(value: float, _position: int) -> str:
    """Format large tick values in compact thousands notation."""

    absolute_value = abs(value)
    if absolute_value >= 1000:
        scaled_value = absolute_value / 1000
        prefix = "-" if value < 0 else ""
        return f"{prefix}{scaled_value:g}k"

    return f"{int(value)}"


class StringLengthDistributionPlotMixin:
    """Helpers for plotting string-length distributions for embedded text columns."""

    AMAZON_REVIEWS_DATASET_DIR = (
        Path(__file__).resolve().parents[3] / "data" / "datasets" / "amazon-reviews"
    )

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

        con = duckdb.connect()

        try:
            column_types = self._load_column_schema(
                con=con,
                parquet_path=embeddings_path,
            )
            source_columns = self._discover_embedding_source_columns(column_types)

            if not source_columns:
                console.print(
                    "[bold yellow]Warning:[/bold yellow] "
                    f"No columns with embeddings found in {embeddings_path.name}"
                )
                return

            con.execute(
                "CREATE OR REPLACE TEMP VIEW source_data AS "
                f"SELECT * FROM read_parquet({_sql_string_literal(str(embeddings_path))})"
            )

            for column_name in source_columns:
                histogram = self._fetch_string_length_histogram(
                    con=con,
                    column_name=column_name,
                )

                if histogram.empty:
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        f"No non-null values found for {column_name} in {embeddings_path.name}"
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
        schema: dict[str, str],
    ) -> list[str]:
        """Return the unique source columns that have embedding columns."""

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
                length(CAST({_sql_identifier(column_name)} AS VARCHAR)) AS string_length,
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
            fig_height=2.4,
            scale=0.8,
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
        axis.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
        axis.xaxis.set_major_formatter(mticker.FuncFormatter(_format_k_tick))
        axis.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(_format_k_tick))
        axis.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        axis.tick_params(
            axis="both",
            which="major",
            bottom=True,
            left=True,
            top=False,
            right=False,
            length=5,
            width=0.8,
        )
        axis.tick_params(
            axis="both",
            which="minor",
            bottom=True,
            left=True,
            top=False,
            right=False,
            length=3,
            width=0.7,
        )
        axis.grid(axis="both", which="major", alpha=0.55)
        axis.grid(axis="both", which="minor", alpha=0.22)

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

        pdf_path = self.plot_dir / f"{dataset_name}__{column_name}_text_length_dist.pdf"

        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
        console.print(
            f"[green]✓[/green] Saved string length distribution plot to [bold]{pdf_path}[/bold]"
        )

        plt.close(fig)
