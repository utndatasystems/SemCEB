from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter
import pandas as pd
import seaborn as sns

from semceb.reporting.plot_params import apply_plot_params
from semceb.utils.console import console


class QuerySelectivityPlotMixin:
    """Helpers for plotting query selectivity distributions."""

    def _plot_ground_truth_selectivity_distributions(self) -> None:
        """Plot filter and join selectivity distributions from ground-truth caches."""

        benchmark_queries_dir = Path("benchmark_queries")

        if not benchmark_queries_dir.exists():
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"Benchmark query directory not found: {benchmark_queries_dir}"
            )
            return

        cache_paths = self._list_ground_truth_cache_paths(benchmark_queries_dir)

        if not cache_paths:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"No ground-truth cache files found in {benchmark_queries_dir}"
            )
            return

        ground_truth_caches = [
            (cache_path, self._load_ground_truth_cache(cache_path))
            for cache_path in cache_paths
            if cache_path.is_file()
        ]

        apply_plot_params(
            fig_height=1.8,
            scale=1,
            double_column=False,
        )

        for cache_path, cache_object in ground_truth_caches:
            selectivities_by_query_type = (
                self._collect_ground_truth_selectivities_by_query_type(cache_object)
            )

            for query_type, selectivities in selectivities_by_query_type.items():
                self._plot_ground_truth_selectivity_distribution(
                    cache_path=cache_path,
                    query_type=query_type,
                    selectivities=selectivities,
                )

    def _list_ground_truth_cache_paths(self, benchmark_queries_dir: Path) -> list[Path]:
        """Return all ground-truth cache files sorted by filename."""
        return sorted(benchmark_queries_dir.glob("ground_truth_cache_*.json"))

    def _load_ground_truth_cache(self, cache_path: Path) -> dict[str, Any]:
        """Load a single ground-truth cache JSON object."""

        with open(cache_path, "r", encoding="utf-8") as file:
            cache_object = json.load(file)

        if not isinstance(cache_object, dict):
            raise ValueError(f"Expected JSON object in {cache_path}")

        return cache_object

    def _collect_ground_truth_selectivities_by_query_type(
        self,
        cache_object: dict[str, Any],
    ) -> dict[str, list[float]]:
        """Collect selectivity values for filter and join entries."""

        selectivities_by_query_type = {
            "filter": [],
            "join": [],
        }

        for cache_key, cache_value in cache_object.items():
            query_type = self._extract_ground_truth_query_type(cache_key)

            if query_type is None:
                continue

            if not isinstance(cache_value, dict) or "selectivity" not in cache_value:
                continue

            selectivity = pd.to_numeric(cache_value["selectivity"], errors="coerce")

            if pd.isna(selectivity):
                continue

            selectivities_by_query_type[query_type].append(float(selectivity))

        return selectivities_by_query_type

    def _extract_ground_truth_query_type(self, cache_key: str) -> str | None:
        """Extract the query type encoded in a ground-truth cache key."""

        if "query_type='filter'" in cache_key:
            return "filter"

        if "query_type='join'" in cache_key:
            return "join"

        return None

    def _plot_ground_truth_selectivity_distribution(
        self,
        cache_path: Path,
        query_type: str,
        selectivities: list[float],
    ) -> None:
        """Plot an empirical CDF of workload selectivities for one query type."""

        if not selectivities:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"No {query_type} selectivities found in {cache_path}"
            )
            return

        sorted_selectivities = sorted(selectivities)
        plot_selectivities, _ = self._prepare_selectivities_for_log_plot(
            sorted_selectivities
        )
        cumulative_probabilities = [
            index / len(plot_selectivities)
            for index in range(1, len(plot_selectivities) + 1)
        ]
        lower_x_limit = 0.0001
        line_color = "#D67D00"
        fill_color = "#DEA555"

        fig, axis = plt.subplots()
        axis.step(
            plot_selectivities,
            cumulative_probabilities,
            where="post",
            color=line_color,
            linewidth=1.6,
            zorder=3,
        )
        axis.fill_between(
            plot_selectivities,
            cumulative_probabilities,
            y2=0,
            step="post",
            color=fill_color,
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )

        axis.set_xscale("log")
        axis.set_xlim(left=lower_x_limit, right=1)
        axis.set_ylim(bottom=0, top=1)
        axis.set_title(f"{query_type.capitalize()} Queries")

        axis.set_xlabel("Selectivity [log-scale]")
        axis.set_ylabel("CDF")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.minorticks_on()
        axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10), numticks=100))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(
            axis="both",
            which="major",
            bottom=True,
            left=True,
            top=False,
            right=False,
            length=6,
            width=1.0,
            color="#222222",
            direction="out",
            labelbottom=True,
            labelleft=True,
        )
        axis.tick_params(
            axis="x",
            which="minor",
            bottom=True,
            top=False,
            length=5,
            width=0.9,
            color="#666666",
            direction="out",
        )
        axis.grid(axis="x", alpha=0.55)
        axis.grid(axis="x", which="minor", alpha=0.22)
        axis.grid(axis="y", alpha=0.35)

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
            / (
                "query_selectivity_cdf_"
                f"{cache_path.stem}_{query_type}.pdf"
            )
        )

        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
        console.print(
            f"[green]✓[/green] Saved {query_type} selectivity CDF plot "
            f"to [bold]{pdf_path}[/bold]"
        )

        plt.close(fig)

    def _prepare_selectivities_for_log_plot(
        self,
        selectivities: list[float],
    ) -> tuple[list[float], float | None]:
        """Replace non-positive values with a positive floor for log-scale plotting."""

        nonpositive_floor = 0.0001

        plot_selectivities = [
            value if value > 0 else nonpositive_floor
            for value in selectivities
        ]

        if all(value > 0 for value in selectivities):
            return plot_selectivities, None

        return plot_selectivities, nonpositive_floor
