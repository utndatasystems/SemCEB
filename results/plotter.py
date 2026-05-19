from pathlib import Path
from typing import Any
import json
import math
import io

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from rich.console import Console
from utils.console import console
from rich.table import Table


class ResultsPlotter:
    """Plots benchmark run results."""

    def __init__(self):
        self.raw_results_path = Path("results") / "raw" / "result.jsonl"
        self.plot_dir = Path("results") / "plots"
        self.table_dir = Path("results") / "tables"

        self.single_bar_width = 0.10
        self.grouped_bar_width = 0.10

    def plot(self) -> None:
        """Create benchmark run plots and summary tables."""

        results = self._load_results()
        df = self._to_dataframe(results)

        if df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No results to plot."
            )
            return

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

        summary = self._summarize_by_algorithm(df)
        table = self._create_summary_table(summary)

        console.print()
        console.print(table)
        console.print()

        self._save_summary_table(table)
        self._plot_algorithm_comparison(summary)

    def _load_results(self) -> list[dict[str, Any]]:
        """Load raw benchmark results from JSONL."""

        path = Path(self.raw_results_path)

        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        results: list[dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {path}: {error}"
                    ) from error

        return results

    def _to_dataframe(self, results: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert nested JSONL results into a flat DataFrame."""

        rows: list[dict[str, Any]] = []

        for result in results:
            query = result["query"]
            algorithm = result["algorithm"]
            cost_stats = algorithm["cost_stats"]
            virtual_cost = cost_stats
            physical_cost = cost_stats

            rows.append(
                {
                    "query_id": query["id"],
                    "query_name": query["name"],
                    "query_version": query["version"],
                    "dataset": query["dataset"],
                    "column": query["column"],
                    "query": query["query"],
                    "algorithm_name": algorithm["name"],
                    "algorithm_version": algorithm["version"],
                    "selectivity_ground_truth": algorithm[
                        "selectivity_ground_truth"
                    ],
                    "selectivity_estimation": algorithm[
                        "selectivity_estimation"
                    ],
                    "q_error": algorithm["q_error"],
                    "time_ms": algorithm["time_ms"],
                    "virtual_cost_usd": virtual_cost["usd"],
                    "virtual_llm_calls": virtual_cost["llm_calls"],
                    "virtual_tokens": virtual_cost["tokens"],
                    "physical_cost_usd": physical_cost["usd"],
                    "physical_llm_calls": physical_cost["llm_calls"],
                    "physical_tokens": physical_cost["tokens"],
                }
            )

        df = pd.DataFrame(rows)

        numeric_columns = [
            "selectivity_ground_truth",
            "selectivity_estimation",
            "q_error",
            "time_ms",
            "virtual_cost_usd",
            "virtual_llm_calls",
            "virtual_tokens",
            "physical_cost_usd",
            "physical_llm_calls",
            "physical_tokens",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

        return df

    def _summarize_by_algorithm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate values per algorithm."""

        summary = (
            df.groupby("algorithm_name")
            .agg(
                query_count=("query_id", "count"),
                q_error_avg=("q_error", "mean"),
                q_error_min=("q_error", "min"),
                q_error_max=("q_error", "max"),
                time_ms_avg=("time_ms", "mean"),
                time_ms_min=("time_ms", "min"),
                time_ms_max=("time_ms", "max"),

                virtual_cost_usd_total=("virtual_cost_usd", "sum"),
                virtual_cost_usd_min=("virtual_cost_usd", "min"),
                virtual_cost_usd_max=("virtual_cost_usd", "max"),
                physical_cost_usd_total=("physical_cost_usd", "sum"),
                physical_cost_usd_min=("physical_cost_usd", "min"),
                physical_cost_usd_max=("physical_cost_usd", "max"),

                virtual_llm_calls_total=("virtual_llm_calls", "sum"),
                virtual_llm_calls_min=("virtual_llm_calls", "min"),
                virtual_llm_calls_max=("virtual_llm_calls", "max"),
                physical_llm_calls_total=("physical_llm_calls", "sum"),
                physical_llm_calls_min=("physical_llm_calls", "min"),
                physical_llm_calls_max=("physical_llm_calls", "max"),

                virtual_tokens_total=("virtual_tokens", "sum"),
                virtual_tokens_min=("virtual_tokens", "min"),
                virtual_tokens_max=("virtual_tokens", "max"),
                physical_tokens_total=("physical_tokens", "sum"),
                physical_tokens_min=("physical_tokens", "min"),
                physical_tokens_max=("physical_tokens", "max"),
            )
            .reset_index()
        )

        summary = summary.sort_values(
            by=["q_error_avg", "time_ms_avg"],
            ascending=[True, True],
        ).reset_index(drop=True)

        return summary

    def _create_summary_table(self, summary: pd.DataFrame) -> Table:
        """Create benchmark summary table."""

        table = Table(title="Algorithm benchmark summary")

        table.add_column("Algorithm", style="cyan")
        table.add_column("Queries", justify="right")
        table.add_column("Avg q-error", justify="right")
        table.add_column("Min q-error", justify="right")
        table.add_column("Max q-error", justify="right")
        table.add_column("Avg time ms", justify="right")
        table.add_column("Min time ms", justify="right")
        table.add_column("Max time ms", justify="right")
        table.add_column("Virtual cost USD", justify="right")
        table.add_column("Physical cost USD", justify="right")
        table.add_column("Virtual LLM calls", justify="right")
        table.add_column("Physical LLM calls", justify="right")

        for _, row in summary.iterrows():
            table.add_row(
                str(row["algorithm_name"]),
                str(int(row["query_count"])),
                self._format_float(row["q_error_avg"], decimals=4),
                self._format_float(row["q_error_min"], decimals=4),
                self._format_float(row["q_error_max"], decimals=4),
                self._format_float(row["time_ms_avg"], decimals=2),
                self._format_float(row["time_ms_min"], decimals=2),
                self._format_float(row["time_ms_max"], decimals=2),
                self._format_float(row["virtual_cost_usd_total"], decimals=8),
                self._format_float(row["physical_cost_usd_total"], decimals=8),
                str(int(row["virtual_llm_calls_total"])),
                str(int(row["physical_llm_calls_total"])),
            )

        return table
    
    def _get_algorithm_colors(
        self,
        algorithms: list[str],
    ) -> dict[str, Any]:
        """Assign one consistent color to each algorithm."""

        cmap = plt.get_cmap("tab10")

        return {
            algorithm: cmap(index % 10)
            for index, algorithm in enumerate(algorithms)
        }

    def _plot_algorithm_comparison(self, summary: pd.DataFrame) -> None:
        """
        Create one plot with:
        - q-error comparison
        - runtime comparison
        - cost comparison: virtual vs physical
        - token comparison: virtual vs physical
        - LLM-call comparison: virtual vs physical
        """

        algorithms = summary["algorithm_name"].tolist()
        algorithm_colors = self._get_algorithm_colors(algorithms)

        fig = plt.figure(figsize=(18, 10))
        grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

        q_error_axis = fig.add_subplot(grid[0, 0])
        time_axis = fig.add_subplot(grid[0, 1])
        cost_axis = fig.add_subplot(grid[1, 0])
        tokens_axis = fig.add_subplot(grid[1, 1])
        llm_calls_axis = fig.add_subplot(grid[1, 2])

        self._plot_single_metric_bars(
            axis=q_error_axis,
            algorithms=algorithms,
            values=summary["q_error_avg"].tolist(),
            min_values=summary["q_error_min"].tolist(),
            max_values=summary["q_error_max"].tolist(),
            title="Average q-error",
            ylabel="Average q-error",
            algorithm_colors=algorithm_colors,
        )

        self._plot_single_metric_bars(
            axis=time_axis,
            algorithms=algorithms,
            values=summary["time_ms_avg"].tolist(),
            min_values=summary["time_ms_min"].tolist(),
            max_values=summary["time_ms_max"].tolist(),
            title="Average runtime",
            ylabel="Average time in ms",
            algorithm_colors=algorithm_colors,
        )

        self._plot_virtual_physical_bars(
            axis=cost_axis,
            algorithms=algorithms,
            virtual_values=summary["virtual_cost_usd_total"].tolist(),
            physical_values=summary["physical_cost_usd_total"].tolist(),
            title="Total cost",
            ylabel="Total cost in USD",
            algorithm_colors=algorithm_colors,
            use_scientific_notation_for_small_values=True,
        )

        self._plot_virtual_physical_bars(
            axis=tokens_axis,
            algorithms=algorithms,
            virtual_values=summary["virtual_tokens_total"].tolist(),
            physical_values=summary["physical_tokens_total"].tolist(),
            title="Total tokens",
            ylabel="Total tokens",
            algorithm_colors=algorithm_colors,
        )

        self._plot_virtual_physical_bars(
            axis=llm_calls_axis,
            algorithms=algorithms,
            virtual_values=summary["virtual_llm_calls_total"].tolist(),
            physical_values=summary["physical_llm_calls_total"].tolist(),
            title="Total LLM calls",
            ylabel="Total LLM calls",
            algorithm_colors=algorithm_colors,
        )

        legend_axis = fig.add_subplot(grid[0, 2])
        self._plot_shared_legend(
            axis=legend_axis,
            algorithms=algorithms,
            algorithm_colors=algorithm_colors,
        )

        fig.suptitle("Algorithm performance comparison", fontsize=14)
        fig.tight_layout()

        output_path = self.plot_dir / "algorithm_comparison.png"

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        console.print(
            f"[green]✓[/green] Saved algorithm comparison plot to [bold]{output_path}[/bold]"
        )

    def _plot_single_metric_bars(
        self,
        axis: Any,
        algorithms: list[str],
        values: list[float],
        min_values: list[float],
        max_values: list[float],
        title: str,
        ylabel: str,
        algorithm_colors: dict[str, Any],
    ) -> None:
        """Plot one metric as vertical bars with min/max range lines."""

        x_positions = list(range(len(algorithms)))

        for index, algorithm in enumerate(algorithms):
            axis.bar(
                index,
                values[index],
                label=algorithm,
                width=self.single_bar_width,
                color=algorithm_colors[algorithm],
                edgecolor="black",
                linewidth=0.8,
            )

            axis.vlines(
                x=index,
                ymin=min_values[index],
                ymax=max_values[index],
                color="black",
                linewidth=1.4,
                zorder=3,
            )

            axis.hlines(
                y=min_values[index],
                xmin=index - self.single_bar_width,
                xmax=index + self.single_bar_width,
                color="black",
                linewidth=1.2,
                zorder=3,
            )

            axis.hlines(
                y=max_values[index],
                xmin=index - self.single_bar_width,
                xmax=index + self.single_bar_width,
                color="black",
                linewidth=1.2,
                zorder=3,
            )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([""] * len(algorithms))
        axis.grid(axis="y", alpha=0.3)

    def _plot_virtual_physical_bars(
        self,
        axis: Any,
        algorithms: list[str],
        virtual_values: list[float],
        physical_values: list[float],
        title: str,
        ylabel: str,
        algorithm_colors: dict[str, Any],
        use_scientific_notation_for_small_values: bool = False,
    ) -> None:
        """Plot virtual and physical values as grouped bars.

        Fill color:
        - virtual: light grey
        - physical: dark grey

        Border color:
        - algorithm-specific color
        """

        x_positions = list(range(len(algorithms)))
        bar_width = self.grouped_bar_width

        virtual_positions = [
            position - bar_width / 2 for position in x_positions
        ]
        physical_positions = [
            position + bar_width / 2 for position in x_positions
        ]

        for index, algorithm in enumerate(algorithms):
            axis.bar(
                virtual_positions[index],
                virtual_values[index],
                width=bar_width,
                color="lightgrey",
                edgecolor=algorithm_colors[algorithm],
                linewidth=2.0,
            )

            axis.bar(
                physical_positions[index],
                physical_values[index],
                width=bar_width,
                color="dimgray",
                edgecolor=algorithm_colors[algorithm],
                linewidth=2.0,
            )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(algorithms, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.3)

        max_value = max(virtual_values + physical_values)

        if (
            use_scientific_notation_for_small_values
            and 0 < max_value < 0.01
        ):
            axis.ticklabel_format(
                axis="y",
                style="scientific",
                scilimits=(0, 0),
            )

    def _draw_min_max_marker(
        self,
        axis: Any,
        x_position: float,
        min_value: float,
        max_value: float,
        marker_width: float,
    ) -> None:
        """Draw a min/max marker over a bar."""

        axis.vlines(
            x=x_position,
            ymin=min_value,
            ymax=max_value,
            color="black",
            linewidth=1.2,
            zorder=4,
        )

        axis.hlines(
            y=min_value,
            xmin=x_position - marker_width,
            xmax=x_position + marker_width,
            color="black",
            linewidth=1.0,
            zorder=4,
        )

        axis.hlines(
            y=max_value,
            xmin=x_position - marker_width,
            xmax=x_position + marker_width,
            color="black",
            linewidth=1.0,
            zorder=4,
        )

    def _plot_shared_legend(
        self,
        axis: Any,
        algorithms: list[str],
        algorithm_colors: dict[str, Any],
    ) -> None:
        """Use the free top-right plot area as a full-width shared legend."""

        axis.axis("off")

        legend_box = FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.02",
            transform=axis.transAxes,
            facecolor="white",
            edgecolor="lightgrey",
            linewidth=1.0,
            clip_on=False,
        )
        axis.add_patch(legend_box)

        # -----------------------------
        # Algorithm legend section
        # -----------------------------
        algorithm_title_y = 0.92
        algorithm_start_y = 0.80

        # This is the lowest y-position the algorithm rows may use.
        # Everything below this stays reserved for the cost/stat legend.
        algorithm_min_y = 0.34

        max_algorithm_row_distance = 0.12
        algorithm_count = len(algorithms)

        if algorithm_count <= 1:
            algorithm_row_distance = 0.0
        else:
            available_algorithm_height = algorithm_start_y - algorithm_min_y
            algorithm_row_distance = min(
                max_algorithm_row_distance,
                available_algorithm_height / (algorithm_count - 1),
            )

        axis.text(
            0.06,
            algorithm_title_y,
            "Algorithms",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

        y_position = algorithm_start_y

        for algorithm in algorithms:
            axis.add_patch(
                FancyBboxPatch(
                    (0.06, y_position - 0.035),
                    0.05,
                    0.05,
                    boxstyle="square,pad=0.0",
                    transform=axis.transAxes,
                    facecolor="white",
                    edgecolor=algorithm_colors[algorithm],
                    linewidth=2.0,
                    clip_on=False,
                )
            )

            axis.text(
                0.14,
                y_position,
                algorithm,
                transform=axis.transAxes,
                fontsize=9,
                va="center",
                ha="left",
            )

            y_position -= algorithm_row_distance

        # -----------------------------
        # Cost/stat type legend section
        # -----------------------------
        # Move this down by lowering these y-values.
        cost_title_y = 0.22
        cost_item_y = 0.10

        axis.text(
            0.06,
            cost_title_y,
            "Cost/stat type",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # Virtual item
        axis.add_patch(
            FancyBboxPatch(
                (0.06, cost_item_y - 0.025),
                0.05,
                0.05,
                boxstyle="square,pad=0.0",
                transform=axis.transAxes,
                facecolor="lightgrey",
                edgecolor="black",
                linewidth=1.0,
                clip_on=False,
            )
        )
        axis.text(
            0.14,
            cost_item_y,
            "Virtual",
            transform=axis.transAxes,
            fontsize=9,
            va="center",
            ha="left",
        )

        # Physical item
        axis.add_patch(
            FancyBboxPatch(
                (0.42, cost_item_y - 0.025),
                0.05,
                0.05,
                boxstyle="square,pad=0.0",
                transform=axis.transAxes,
                facecolor="dimgray",
                edgecolor="black",
                linewidth=1.0,
                clip_on=False,
            )
        )
        axis.text(
            0.50,
            cost_item_y,
            "Physical",
            transform=axis.transAxes,
            fontsize=9,
            va="center",
            ha="left",
        )


    def _save_summary_table(self, table: Table) -> None:
        """Save Rich summary table to text and HTML files without printing it again."""

        table_txt_path = self.table_dir / "algorithm_summary.txt"
        table_html_path = self.table_dir / "algorithm_summary.html"

        text_buffer = io.StringIO()
        text_console = Console(
            file=text_buffer,
            record=True,
            width=180,
            color_system=None,
        )
        text_console.print(table)

        with open(table_txt_path, "w", encoding="utf-8") as file:
            file.write(text_console.export_text())

        console.print(
            f"[green]✓[/green] Saved algorithm comparison table in *.txt to [bold]{table_txt_path}[/bold]"
        )

        html_buffer = io.StringIO()
        html_console = Console(
            file=html_buffer,
            record=True,
            width=180,
        )
        html_console.print(table)
        html_console.save_html(str(table_html_path))

        console.print(
            f"[green]✓[/green] Saved algorithm comparison table in *.html to [bold]{table_html_path}[/bold]"
        )

    def _format_float(self, value: Any, decimals: int) -> str:
        """Format floats safely for tables."""

        numeric_value = float(value)

        if math.isnan(numeric_value):
            return "n/a"

        return f"{numeric_value:.{decimals}f}"