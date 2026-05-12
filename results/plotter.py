from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table


class ResultsPlotter:
    """Plots benchmark run result."""

    def __init__(self, console: Console):
        self.console = console

        self.raw_results_path = r"results\raw\result.jsonl"
        self.plot_dir = Path(r"results\plots")
        self.table_dir = Path(r"results\tables")

    def plot(self) -> None:
        """Create benchmark run plot."""

        results = self._load_results()
        df = self._to_dataframe(results)

        if df.empty:
            self.console.print(
                "[bold yellow]Warning:[/bold yellow] No results to plot."
            )
            return

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

        summary = self._summarize_by_algorithm(df)

        table = self._create_summary_table(summary)

        self.console.print()
        self.console.print(table)
        self.console.print()

        self._save_summary_table(table)
        self._plot_algorithm_comparison(summary)

    def _load_results(self) -> list[dict[str, Any]]:
        """Load raw benchmark results from JSONL."""

        path = Path(self.raw_results_path)

        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        results = []

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                results.append(json.loads(line))

        return results

    def _to_dataframe(self, results: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert nested JSONL results into a flat DataFrame."""

        rows = []

        for result in results:
            query = result["query"]
            algorithm = result["algorithm"]

            rows.append(
                {
                    "query_id": query["id"],
                    "query_name": query["name"],
                    "dataset": query["dataset"],
                    "query": query["query"],
                    "algorithm_name": algorithm["name"],
                    "algorithm_version": algorithm["version"],
                    "cost_usd": algorithm["cost_usd"],
                    "selectivity_ground_truth": algorithm[
                        "selectivity_ground_truth"
                    ],
                    "selectivity_estimation": algorithm[
                        "selectivity_estimation"
                    ],
                    "q_error": algorithm["q_error"],
                    "time_ms": algorithm["time_ms"],
                }
            )

        return pd.DataFrame(rows)

    def _summarize_by_algorithm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate min, average, and max values per algorithm."""

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
                cost_usd_total=("cost_usd", "sum"),
            )
            .reset_index()
        )

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

        for _, row in summary.iterrows():
            table.add_row(
                str(row["algorithm_name"]),
                str(int(row["query_count"])),
                f"{row['q_error_avg']:.4f}",
                f"{row['q_error_min']:.4f}",
                f"{row['q_error_max']:.4f}",
                f"{row['time_ms_avg']:.4f}",
                f"{row['time_ms_min']:.4f}",
                f"{row['time_ms_max']:.4f}",
                f"{row['cost_usd_total']:.6f}",
            )

        return table

    def _plot_algorithm_comparison(
        self,
        summary: pd.DataFrame,
    ) -> None:
        """Create one plot comparing algorithms by q-error and runtime."""

        algorithms = summary["algorithm_name"].tolist()
        x_positions = range(len(algorithms))

        q_error_avg = summary["q_error_avg"]
        q_error_min = summary["q_error_min"]
        q_error_max = summary["q_error_max"]

        time_avg = summary["time_ms_avg"]
        time_min = summary["time_ms_min"]
        time_max = summary["time_ms_max"]

        q_error_lower = q_error_avg - q_error_min
        q_error_upper = q_error_max - q_error_avg

        time_lower = time_avg - time_min
        time_upper = time_max - time_avg

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].errorbar(
            x_positions,
            q_error_avg,
            yerr=[q_error_lower, q_error_upper],
            fmt="o",
            capsize=6,
        )
        axes[0].set_title("Q-error by algorithm")
        axes[0].set_ylabel("Q-error")
        axes[0].set_xticks(list(x_positions))
        axes[0].set_xticklabels(algorithms, rotation=30, ha="right")
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].errorbar(
            x_positions,
            time_avg,
            yerr=[time_lower, time_upper],
            fmt="o",
            capsize=6,
        )
        axes[1].set_title("Runtime by algorithm")
        axes[1].set_ylabel("Time in ms")
        axes[1].set_xticks(list(x_positions))
        axes[1].set_xticklabels(algorithms, rotation=30, ha="right")
        axes[1].grid(axis="y", alpha=0.3)

        for index, row in summary.iterrows():
            axes[0].annotate(
                f"avg={row['q_error_avg']:.2f}\n"
                f"min={row['q_error_min']:.2f}\n"
                f"max={row['q_error_max']:.2f}",
                xy=(index, row["q_error_avg"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

            axes[1].annotate(
                f"avg={row['time_ms_avg']:.2f}\n"
                f"min={row['time_ms_min']:.2f}\n"
                f"max={row['time_ms_max']:.2f}",
                xy=(index, row["time_ms_avg"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        fig.suptitle("Algorithm performance comparison")
        fig.tight_layout()

        output_path = self.plot_dir / "algorithm_comparison.png"

        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        self.console.print(
            f"[green]✓[/green] Saved algorithm comparison plot to [bold]{output_path}[/bold]"
        )

    def _save_summary_table(self, table: Table) -> None:
        """Save Rich summary table to text and HTML files."""

        table_txt_path = self.table_dir / "algorithm_summary.txt"
        table_html_path = self.table_dir / "algorithm_summary.html"

        text_console = Console(
            record=True,
            width=140,
            color_system=None,
        )

        self.console.print(
            f"[green]✓[/green] Saved algorithm comparison table in *.txt to [bold]{table_txt_path}[/bold]"
        )

        with open(table_txt_path, "w", encoding="utf-8") as file:
            file.write(text_console.export_text())

        html_console = Console(
            record=True,
            width=140,
        )

        html_console.save_html(str(table_html_path))

        self.console.print(
            f"[green]✓[/green] Saved algorithm comparison table in *.html to [bold]{table_html_path}[/bold]"
        )
