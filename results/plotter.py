from pathlib import Path
from typing import Any
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from rich.console import Console
from utils.console import console
from rich.table import Table
import seaborn as sns
from weasyprint import HTML


class ResultsPlotter:
    """Plots benchmark run results."""

    def __init__(self):
        self.raw_results_path = Path("results") / "raw" / "result.jsonl"
        self.plot_dir = Path("results") / "plots"
        self.table_dir = Path("results") / "tables"

        # Optional: force algorithms to always appear, even when a metric has no valid data.
        # If None, this is inferred from all algorithms present in the result file.
        self.algorithm_order: list[str] | None = None

        # Maximum element width inside one algorithm slot.
        self.metric_element_width = 0.75

        # Keep at least this many visual slots in every subplot.
        # Example: with 2 algorithms and this set to 6,
        # the 2 algorithms appear on the left and the remaining space stays empty.
        self.minimum_visible_algorithm_slots = 8

        self.plot_font_size = 12
        self.main_title_font_size = 18


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

        self._save_summary_table(summary)
        self._save_algorithm_summary_csv(summary)
        self._plot_algorithm_comparison(df)
        self._save_per_query_report(df)
        self._save_per_query_statistics_csv(df)

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

            rows.append(
                {
                    "query_id": query["id"],
                    "query_name": query.get("name", ""),
                    "query_category": query.get("category", ""),
                    "datasets": query["datasets"],
                    "filter": query["filter"],
                    "algorithm_name": algorithm["name"],
                    "algorithm_version": algorithm["version"],
                    "memory_consumption": algorithm["memory_consumption"],
                    "selectivity_ground_truth": algorithm["selectivity_ground_truth"],
                    "selectivity_estimation": algorithm["selectivity_estimation"],
                    "q_error": algorithm["q_error"],
                    "time_ms": algorithm["time_ms"],
                    "cost_usd": cost_stats["usd"],
                    "llm_calls": cost_stats["llm_calls"],
                    "tokens": cost_stats["tokens"],
                }
            )

        df = pd.DataFrame(rows)

        numeric_columns = [
            "selectivity_ground_truth",
            "selectivity_estimation",
            "q_error",
            "time_ms",
            "memory_consumption",
            "cost_usd",
            "llm_calls",
            "tokens",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

        return df

    def _summarize_by_algorithm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate values per algorithm."""

        valid_df = df.copy()

        # Treat -1 as missing for metrics where -1 means "not available".
        nullable_columns = [
            "memory_consumption",
            "cost_usd",
            "llm_calls",
            "tokens",
        ]

        for column in nullable_columns:
            valid_df.loc[valid_df[column] < 0, column] = pd.NA

        summary = (
            valid_df.groupby("algorithm_name")
            .agg(
                query_count=("query_id", "count"),

                q_error_avg=("q_error", "mean"),
                q_error_min=("q_error", "min"),
                q_error_max=("q_error", "max"),

                time_ms_avg=("time_ms", "mean"),
                time_ms_min=("time_ms", "min"),
                time_ms_max=("time_ms", "max"),

                cost_usd_total=("cost_usd", "sum"),

                llm_calls_total=("llm_calls", "sum"),

                tokens_total=("tokens", "sum"),

                memory_consumption_total=("memory_consumption", "sum"),
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
        table.add_column("Total cost USD", justify="right")
        table.add_column("Total LLM calls", justify="right")
        table.add_column("Total tokens", justify="right")
        table.add_column("Total memory", justify="right")

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
                self._format_float(row["cost_usd_total"], decimals=8),
                self._format_optional_int(row["llm_calls_total"]),
                self._format_optional_int(row["tokens_total"]),
                self._format_float(row["memory_consumption_total"], decimals=2),
            )

        return table

    def _get_algorithm_styles(
        self,
        algorithms: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Assign grayscale scientific styles to algorithms."""

        facecolors = [
            "#ffffff",  # white
            "#e6e6e6",  # light grey
            "#cfcfcf",  # medium-light grey
            "#b3b3b3",  # medium grey
            "#8c8c8c",  # dark grey
            "#666666",  # darker grey
            "#f2f2f2",  # very light grey
            "#d9d9d9",  # soft grey
        ]

        hatches = [
            "",
            "///",
            "\\\\\\",
            "...",
            "xxx",
            "---",
            "++",
            "oo",
        ]

        styles: dict[str, dict[str, Any]] = {}

        for index, algorithm in enumerate(algorithms):
            styles[algorithm] = {
                "facecolor": facecolors[index % len(facecolors)],
                "edgecolor": "#222222",
                "hatch": hatches[index % len(hatches)],
            }

        return styles

    def _save_algorithm_summary_csv(self, summary: pd.DataFrame) -> None:
        """Save algorithm-level aggregate statistics as CSV."""

        csv_path = self.table_dir / "algorithm_summary.csv"

        summary.to_csv(
            csv_path,
            index=False,
            encoding="utf-8",
        )

        console.print(
            f"[green]✓[/green] Saved algorithm summary CSV to [bold]{csv_path}[/bold]"
        )

    def _plot_algorithm_comparison(self, df: pd.DataFrame) -> None:
        """Plot one metric as total bars while preserving empty algorithm slots."""
        
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["DejaVu Serif"]
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        sns.set_theme(
            context="paper",
            style="whitegrid",
            font_scale=1.0,
            rc={
                "font.size": self.plot_font_size,
                "axes.titlesize": self.plot_font_size,
                "axes.labelsize": self.plot_font_size,
                "xtick.labelsize": self.plot_font_size,
                "ytick.labelsize": self.plot_font_size,
                "legend.fontsize": self.plot_font_size,
                "legend.title_fontsize": self.plot_font_size,
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "grid.color": "#d0d0d0",
                "grid.linewidth": 0.8,
                "axes.edgecolor": "#666666",
                "axes.linewidth": 0.8,
                "axes.titleweight": "bold",
                "axes.labelcolor": "#222222",
                "xtick.color": "#222222",
                "ytick.color": "#222222",
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],  
              },
        )

        if self.algorithm_order is None:
            algorithms = df["algorithm_name"].drop_duplicates().tolist()
        else:
            algorithms = self.algorithm_order

        algorithm_styles = self._get_algorithm_styles(algorithms)

        palette = {
            algorithm: algorithm_styles[algorithm]["facecolor"]
            for algorithm in algorithms
        }

        fig, axes = plt.subplots(2, 3, figsize=(20, 11))
        axes = axes.flatten()

        distribution_plot_specs = [
            ("q_error", "Q-error", "Q-error"),
        ]

        resource_plot_specs = [
            ("time_ms", "Total Runtime", "Runtime in ms"),
            ("cost_usd", "Total Cost", "Cost in USD"),
            ("tokens", "Total Tokens", "Tokens"),
            ("llm_calls", "Total LLM-calls", "LLM-calls"),
            ("memory_consumption", "Total Memory", "Memory in bytes"),
        ]

        all_plot_specs = distribution_plot_specs + resource_plot_specs

        for axis, (column, title, ylabel) in zip(axes, all_plot_specs):
            plot_df = df[df[column] >= 0].copy()

            if column == "q_error":
                self._plot_box_metric(
                    axis=axis,
                    plot_df=plot_df,
                    column=column,
                    title=title,
                    ylabel=ylabel,
                    algorithms=algorithms,
                    palette=palette,
                    algorithm_styles=algorithm_styles,
                )
            else:
                self._plot_total_metric(
                    axis=axis,
                    plot_df=plot_df,
                    column=column,
                    title=title,
                    ylabel=ylabel,
                    algorithms=algorithms,
                    algorithm_styles=algorithm_styles,
                )

            if column == "cost_usd":
                axis.ticklabel_format(
                    axis="y",
                    style="scientific",
                    scilimits=(0, 0),
                )

        legend_handles = [
            Patch(
                facecolor=algorithm_styles[algorithm]["facecolor"],
                edgecolor=algorithm_styles[algorithm]["edgecolor"],
                hatch=algorithm_styles[algorithm]["hatch"],
                linewidth=1.1,
                label=algorithm,
            )
            for algorithm in algorithms
        ]

        fig.legend(
            handles=legend_handles,
            labels=algorithms,
            loc="upper center",
            bbox_to_anchor=(0.04, 0.955, 0.92, 0.05),
            ncol=len(algorithms),
            frameon=False,
            columnspacing=1.6,
            handletextpad=0.3,
            borderaxespad=0.0,
        )

        fig.suptitle(
            "Algorithm Performance Comparison",
            fontsize=self.main_title_font_size,
            fontweight="bold",
            color="#111111",
            y=1.06,
        )

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))

        png_path = self.plot_dir / "algorithm_comparison.png"
        pdf_path = self.plot_dir / "algorithm_comparison.pdf"

        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        console.print(
            f"[green]✓[/green] Saved algorithm comparison plot to [bold]{png_path}[/bold]"
        )

        fig.savefig(pdf_path, bbox_inches="tight")
        console.print(
            f"[green]✓[/green] Saved algorithm comparison plot to [bold]{pdf_path}[/bold]"
        )

        plt.close(fig)


    def _plot_box_metric(
        self,
        axis: Any,
        plot_df: pd.DataFrame,
        column: str,
        title: str,
        ylabel: str,
        algorithms: list[str],
        palette: dict[str, Any],
        algorithm_styles: dict[str, dict[str, Any]]
    ) -> None:
        """Plot one metric as a boxplot while preserving empty algorithm slots."""

        sns.boxplot(
            data=plot_df,
            x="algorithm_name",
            y=column,
            hue="algorithm_name",
            order=algorithms,
            hue_order=algorithms,
            palette=palette,
            width=self.metric_element_width,
            linewidth=1.2,
            fliersize=4,
            ax=axis,
            legend=False,
            boxprops={"alpha": 0.75},
            medianprops={"color": "#222222", "linewidth": 1.5},
            whiskerprops={"color": "#444444", "linewidth": 1.1},
            capprops={"color": "#444444", "linewidth": 1.1},
        )

        for patch, algorithm in zip(axis.patches, algorithms):
            style = algorithm_styles[algorithm]
            patch.set_facecolor(style["facecolor"])
            patch.set_edgecolor(style["edgecolor"])
            patch.set_hatch(style["hatch"])
            patch.set_linewidth(1.1)

        sns.stripplot(
            data=plot_df,
            x="algorithm_name",
            y=column,
            order=algorithms,
            color="#222222",
            dodge=False,
            jitter=0.18,
            size=3,
            alpha=0.35,
            ax=axis,
        )

        self._style_metric_axis(
            axis=axis,
            title=title,
            ylabel=ylabel,
            algorithms=algorithms,
        )

    def _plot_total_metric(
        self,
        axis: Any,
        plot_df: pd.DataFrame,
        column: str,
        title: str,
        ylabel: str,
        algorithms: list[str],
        algorithm_styles: dict[str, dict[str, Any]],
    ) -> None:
        """Plot one resource metric as total bars while preserving empty algorithm slots."""

        totals = (
            plot_df.groupby("algorithm_name")[column]
            .sum()
            .reindex(algorithms)
        )

        x_positions = list(range(len(algorithms)))

        for index, algorithm in enumerate(algorithms):
            value = totals.loc[algorithm]

            if pd.isna(value):
                continue

            style = algorithm_styles[algorithm]

            axis.bar(
                index,
                value,
                width=self.metric_element_width,
                color=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=1.1,
                alpha=1.0,
            )

        axis.set_xticks(x_positions)

        self._style_metric_axis(
            axis=axis,
            title=title,
            ylabel=ylabel,
            algorithms=algorithms,
        )

    def _style_metric_axis(
        self,
        axis: Any,
        title: str,
        ylabel: str,
        algorithms: list[str],
    ) -> None:
        """Apply consistent scientific-grey styling to one metric axis.

        The x-axis is left-aligned and keeps a minimum number of visual slots.
        This prevents boxes/bars from becoming visually too wide when only a few
        algorithms are plotted.
        """

        axis.set_title(title, fontsize=self.plot_font_size)
        axis.set_xlabel("", fontsize=self.plot_font_size)
        axis.set_ylabel(ylabel, fontsize=self.plot_font_size)

        axis.tick_params(
            axis="both",
            which="major",
            labelsize=self.plot_font_size,
        )

        algorithm_count = len(algorithms)
        visible_slot_count = max(
            algorithm_count,
            self.minimum_visible_algorithm_slots,
        )

        axis.set_xticks(list(range(algorithm_count)))
        axis.set_xticklabels([""] * algorithm_count)

        # Left-align algorithms and leave empty space on the right if there are
        # fewer algorithms than minimum_visible_algorithm_slots.
        axis.set_xlim(-0.5, visible_slot_count - 0.5)

        axis.grid(axis="y", alpha=0.55)
        axis.grid(axis="x", visible=False)

        sns.despine(
            ax=axis,
            top=True,
            right=True,
            left=False,
            bottom=False,
        )

    def _save_summary_table(self, summary: pd.DataFrame) -> None:
        """Save Rich summary table to text and HTML files."""

        table_html_path = self.table_dir / "algorithm_summary.html"

        summary_html = summary.copy()

        summary_html = summary_html.rename(
            columns={
                "algorithm_name": "Algorithm",
                "query_count": "Queries",
                "q_error_avg": "Avg q-error",
                "q_error_min": "Min q-error",
                "q_error_max": "Max q-error",
                "time_ms_avg": "Avg time ms",
                "time_ms_min": "Min time ms",
                "time_ms_max": "Max time ms",
                "cost_usd_total": "Total cost USD",
                "llm_calls_total": "Total LLM calls",
                "tokens_total": "Total tokens",
                "memory_consumption_total": "Total memory",
            }
        )

        html = f"""
        <html>
        <head>
        <meta charset="utf-8">
        <style>
        @page {{
            size: A3 landscape;
            margin: 10mm;
        }}

        body {{
            font-family: "DejaVu Serif", serif;
            color: #222222;
            margin: 20px;
        }}

        h1 {{
            font-size: 18px;
            margin-bottom: 14px;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            table-layout: auto;
            font-size: 10px;
        }}

        th {{
            background: #f2f2f2;
            border: 1px solid #bbbbbb;
            padding: 5px 6px;
            text-align: left;
            white-space: nowrap;
        }}

        td {{
            border: 1px solid #d0d0d0;
            padding: 5px 6px;
            white-space: nowrap;
            text-align: right;
        }}

        td:first-child,
        th:first-child {{
            text-align: left;
            max-width: 220px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        </style>
        </head>
        <body>
        <h1>Algorithm benchmark summary</h1>
        {summary_html.to_html(index=False, escape=True)}
        </body>
        </html>
        """

        with open(table_html_path, "w", encoding="utf-8") as file:
            file.write(html)

        console.print(
            f"[green]✓[/green] Saved algorithm comparison table in *.html to [bold]{table_html_path}[/bold]"
        )


    def _format_float(self, value: Any, decimals: int) -> str:
        """Format floats safely for tables."""

        if pd.isna(value):
            return "n/a"

        numeric_value = float(value)

        if math.isnan(numeric_value):
            return "n/a"

        return f"{numeric_value:.{decimals}f}"
    
    def _format_optional_int(self, value: Any) -> str:
        """Format optional integer values safely for tables."""

        if pd.isna(value):
            return "n/a"

        return str(int(value))
    
    def _save_per_query_report(self, df: pd.DataFrame) -> None:
        """Save a per-query HTML report with absolute values and normalized bars."""

        if self.algorithm_order is None:
            algorithms = df["algorithm_name"].drop_duplicates().tolist()
        else:
            algorithms = self.algorithm_order

        algorithm_styles = self._get_algorithm_styles(algorithms)

        report_df = df.copy()

        nullable_columns = [
            "memory_consumption",
            "cost_usd",
            "llm_calls",
            "tokens",
        ]

        for column in nullable_columns:
            report_df.loc[report_df[column] < 0, column] = pd.NA

        report_df = report_df.sort_values(
            by=["query_id", "algorithm_name"],
            kind="stable",
        )

        metric_specs = [
            ("q_error", "Q-error", "{:.2f}", "lower"),
            ("time_ms", "Runtime ms", "{:.2f}", "lower"),
            ("cost_usd", "Cost USD", "{:.4f}", "lower"),
            ("tokens", "Tokens", "{:.0f}", "lower"),
            ("llm_calls", "LLM calls", "{:.0f}", "lower"),
            ("memory_consumption", "Memory bytes", "{:.0f}", "lower"),
        ]

        html_parts = [
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<style>",
            """
            @page {
                size: A4 landscape;
                margin: 8mm;
            }

            @media print {
                
                body {
                    margin: 0;
                }

                .query-block {
                    break-inside: avoid;
                    page-break-inside: avoid;
                }

                .query-block h2 {
                    break-after: avoid;
                    page-break-after: avoid;
                }

                .query-block table {
                    break-inside: avoid;
                    page-break-inside: avoid;
                }

                .query-meta {
                    font-size: 9px;
                    margin-bottom: 6px;
                }

                .query-meta-topline {
                    gap: 16px;
                    margin-bottom: 3px;
                }

                .query-meta-query {
                    line-height: 1.25;
                }

                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 24px;
                    table-layout: fixed;
                }

                th,
                td {
                    overflow: hidden;
                    box-sizing: border-box;
                }

                h1 {
                    font-size: 16px;
                }

                h2 {
                    font-size: 13px;
                    page-break-after: avoid;
                    break-after: avoid;
                }

                .metric-cell {
                    display: grid;
                    grid-template-columns: 40% auto 50%;
                    align-items: center;
                    width: 100%;
                    min-width: 0;
                    overflow: hidden;
                    white-space: nowrap;
                }

                .metric-value {
                    grid-column: 1;
                    font-variant-numeric: tabular-nums;
                    text-align: right;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }

                .bar-track {
                    grid-column: 3;
                    width: 100%;
                    min-width: 0;
                    height: 11px;
                    background: #eeeeee;
                    border: 1px solid #cccccc;
                    box-sizing: border-box;
                    display: block;
                    overflow: hidden;
                }

                .bar-fill {
                    display: block;
                    height: 100%;
                    background: #777777;
                }

                .query-text {
                    font-size: 9px;
                }
            }

            body {
                font-family: DejaVu Serif, Arial, serif;
                margin: 24px;
                color: #222222;
                background: white;
            }

            h1 {
                font-size: 22px;
                margin-bottom: 24px;
            }

            h2 {
                font-size: 16px;
                margin-top: 28px;
                margin-bottom: 8px;
                border-bottom: 1px solid #cccccc;
                padding-bottom: 4px;
            }

            .query-block {
                break-inside: avoid;
                page-break-inside: avoid;
            }

            .query-meta {
                font-size: 12px;
                margin-bottom: 10px;
                color: #444444;
            }

            .query-meta-topline {
                display: flex;
                gap: 24px;
                flex-wrap: wrap;
                margin-bottom: 4px;
            }

            .query-meta-topline span {
                white-space: nowrap;
            }

            .query-meta-query {
                white-space: normal;
                overflow-wrap: anywhere;
                line-height: 1.35;
            }

            .query-meta strong {
                color: #222222;
            }

            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 24px;
                table-layout: fixed;
            }

            th {
                background: #f2f2f2;
                border: 1px solid #cccccc;
                padding: 6px 8px;
                font-size: 12px;
                text-align: left;
            }

            td {
                border: 1px solid #d6d6d6;
                padding: 6px 8px;
                font-size: 12px;
                vertical-align: middle;
            }

            .algorithm-header {
                width: 170px;
                max-width: 170px;
            }

            .algorithm-cell {
                width: 170px;
                max-width: 170px;
                white-space: nowrap;
                font-weight: 500;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .algorithm-name {
                display: inline-block;
                max-width: 135px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                vertical-align: middle;
            }

            .style-swatch {
                display: inline-block;
                width: 14px;
                min-width: 14px;
                height: 14px;
                border: 1px solid #222222;
                margin-right: 6px;
                vertical-align: middle;
            }

            .metric-cell {
                display: flex;
                align-items: center;
                gap: 6px;
                white-space: nowrap;
                width: 100%;
                min-width: 0;
                overflow: hidden;
            }

            .metric-value {
                flex: 0 0 64px;
                max-width: 64px;
                font-variant-numeric: tabular-nums;
                text-align: right;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .bar-track {
                flex: 1 1 auto;
                min-width: 18px;
                height: 11px;
                background: #eeeeee;
                border: 1px solid #cccccc;
                box-sizing: border-box;
                display: inline-block;
                overflow: hidden;
            }

            .bar-fill {
                display: block;
                height: 100%;
                background: #777777;
            }

            .missing {
                color: #999999;
                font-style: italic;
            }

            .query-text {
                font-size: 12px;
                margin-bottom: 8px;
                color: #444444;
            }
            """,
            "</style>",
            "</head>",
            "<body>",
            "<h1>Per-query benchmark report</h1>",
        ]

        for query_id, query_df in report_df.groupby("query_id", sort=True):
            query_df = query_df.set_index("algorithm_name").reindex(algorithms).reset_index()
            
            query_meta = query_df.dropna(subset=["filter"]).iloc[0]
            query_category = str(query_meta.get("query_category", ""))
            query_datasets = [str(s) for s in query_meta.get("datasets", "")]
            query_text = str(query_meta.get("filter", ""))

            query_text = ""
            if "filter" in query_df.columns:
                non_empty_queries = query_df["filter"].dropna()
                if not non_empty_queries.empty:
                    query_text = str(non_empty_queries.iloc[0])

            html_parts.append("<section class='query-block'>")
            html_parts.append(f"<h2>Query ID: {query_id}</h2>")

            html_parts.append("<div class='query-meta'>")
            datasets_html = ", ".join(
                self._escape_html(dataset)
                for dataset in query_datasets
            )
            html_parts.append(
                "<div class='query-meta-topline'>"
                f"<span><strong>Category:</strong> {self._escape_html(query_category)}</span>"
                f"<span><strong>Datasets:</strong> {datasets_html}</span>"
                "</div>"
            )
            html_parts.append(
                "<div class='query-meta-query'>"
                f"<strong>Query:</strong> {self._escape_html(query_text)}"
                "</div>"
            )
            html_parts.append("</div>")

            html_parts.append("<table>")
            html_parts.append("<thead>")
            html_parts.append("<tr>")
            html_parts.append("<th class='algorithm-header'>Algorithm</th>")

            for _, label, _, _ in metric_specs:
                html_parts.append(f"<th>{label}</th>")

            html_parts.append("</tr>")
            html_parts.append("</thead>")
            html_parts.append("<tbody>")

            metric_max_values = {}

            for column, _, _, _ in metric_specs:
                values = pd.to_numeric(query_df[column], errors="coerce")
                valid_values = values[values >= 0]
                metric_max_values[column] = valid_values.max() if not valid_values.empty else pd.NA

            for _, row in query_df.iterrows():
                algorithm = row["algorithm_name"]
                style = algorithm_styles.get(
                    algorithm,
                    {
                        "facecolor": "#ffffff",
                        "edgecolor": "#222222",
                        "hatch": "",
                    },
                )

                html_parts.append("<tr>")

                html_parts.append(
                    "<td class='algorithm-cell'>"
                    f"<span class='style-swatch' style='background:{style['facecolor']};'></span>"
                    f"<span class='algorithm-name' title='{self._escape_html(str(algorithm))}'>"
                    f"{self._escape_html(str(algorithm))}"
                    "</span>"
                    "</td>"
                )

                for column, _, format_string, _ in metric_specs:
                    value = row.get(column, pd.NA)
                    max_value = metric_max_values[column]

                    html_parts.append(
                        "<td>"
                        + self._format_metric_cell_html(
                            value=value,
                            max_value=max_value,
                            format_string=format_string,
                        )
                        + "</td>"
                    )

                html_parts.append("</tr>")

            html_parts.append("</tbody>")
            html_parts.append("</table>")
            html_parts.append("</section>")
        
        html_parts.extend(["</body>", "</html>"])

        html_path = self.table_dir / "per_query_benchmark_report.html"

        with open(html_path, "w", encoding="utf-8") as file:
            file.write("\n".join(html_parts))

        console.print(
            f"[green]✓[/green] Saved per-query benchmark report to [bold]{html_path}[/bold]"
        )

    def _save_per_query_statistics_csv(self, df: pd.DataFrame) -> None:
        """Save per-query, per-algorithm statistics as CSV."""

        csv_path = self.table_dir / "per_query_statistics.csv"

        export_df = df.copy()

        nullable_columns = [
            "memory_consumption",
            "cost_usd",
            "llm_calls",
            "tokens",
        ]

        for column in nullable_columns:
            export_df.loc[export_df[column] < 0, column] = pd.NA

        export_df = export_df.sort_values(
            by=["query_id", "algorithm_name"],
            kind="stable",
        )

        export_df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8",
        )

        console.print(
            f"[green]✓[/green] Saved per-query statistics CSV to [bold]{csv_path}[/bold]"
        )

    def _format_metric_cell_html(
        self,
        value: Any,
        max_value: Any,
        format_string: str,
    ) -> str:
        """Format one metric cell with absolute value and normalized bar."""

        if pd.isna(value):
            return "<span class='missing'>n/a</span>"

        numeric_value = float(value)

        if math.isnan(numeric_value) or numeric_value < 0:
            return "<span class='missing'>n/a</span>"

        if pd.isna(max_value) or float(max_value) <= 0:
            width_percent = 0.0
        else:
            width_percent = min(100.0, 100.0 * numeric_value / float(max_value))

        if format_string.endswith("f}") and format_string.startswith("{:.0"):
            value_text = str(int(round(numeric_value)))
        else:
            value_text = format_string.format(numeric_value)

        return (
            "<div class='metric-cell'>"
            f"<span class='metric-value'>{value_text}</span>"
            "<span class='bar-track'>"
            f"<span class='bar-fill' style='display:block; width:{width_percent:.1f}%;'></span>"
            "</span>"
            "</div>"
        )


    def _escape_html(self, value: str) -> str:
        """Escape text for simple HTML output."""

        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )