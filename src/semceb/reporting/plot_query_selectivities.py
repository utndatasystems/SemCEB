from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.prompt import Prompt

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

        selected_cache_paths = self._prompt_for_ground_truth_cache_paths(cache_paths)

        if not selected_cache_paths:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                "No ground-truth cache files were selected; skipping selectivity plots."
            )
            return

        ground_truth_caches = [
            (cache_path, self._load_ground_truth_cache(cache_path))
            for cache_path in selected_cache_paths
            if cache_path.is_file()
        ]

        apply_plot_params(
            fig_height=2.8,
            scale=1.2,
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

    def _prompt_for_ground_truth_cache_paths(
        self,
        cache_paths: list[Path],
    ) -> list[Path]:
        """Ask the user which cache files should be used for plotting."""

        console.print()
        console.print("[bold cyan]Ground-truth cache files[/bold cyan]")

        for index, cache_path in enumerate(cache_paths, start=1):
            console.print(f"  [cyan]{index}[/cyan]: {cache_path.name}")

        console.print()

        while True:
            try:
                selection = Prompt.ask(
                    "Select cache files to plot (comma-separated numbers, ranges like 1-3, or 'all')",
                    default="all",
                ).strip()
            except EOFError:
                console.print(
                    "[bold yellow]Warning:[/bold yellow] "
                    "No interactive input available; using all ground-truth cache files."
                )
                return cache_paths

            if not selection or selection.lower() in {"all", "a", "*"}:
                return cache_paths

            try:
                selected_indices = self._parse_ground_truth_cache_selection(
                    selection,
                    len(cache_paths),
                )
            except ValueError as error:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] {error}"
                )
                continue

            return [cache_paths[index - 1] for index in selected_indices]

    def _parse_ground_truth_cache_selection(
        self,
        selection: str,
        num_cache_paths: int,
    ) -> list[int]:
        """Parse a user selection into 1-based cache indices."""

        selected_indices: set[int] = set()

        for raw_part in selection.split(","):
            part = raw_part.strip()

            if not part:
                continue

            if "-" in part:
                start_str, end_str = [value.strip() for value in part.split("-", maxsplit=1)]

                if not start_str or not end_str:
                    raise ValueError(
                        f"Invalid range '{part}'. Use the form 'start-end'."
                    )

                start_index = int(start_str)
                end_index = int(end_str)

                if start_index > end_index:
                    raise ValueError(
                        f"Invalid range '{part}'. Start must not be greater than end."
                    )

                for index in range(start_index, end_index + 1):
                    self._validate_ground_truth_cache_index(index, num_cache_paths)
                    selected_indices.add(index)
                continue

            index = int(part)
            self._validate_ground_truth_cache_index(index, num_cache_paths)
            selected_indices.add(index)

        if not selected_indices:
            raise ValueError("No cache files were selected.")

        return sorted(selected_indices)

    def _validate_ground_truth_cache_index(
        self,
        index: int,
        num_cache_paths: int,
    ) -> None:
        """Validate a 1-based cache index."""

        if index < 1 or index > num_cache_paths:
            raise ValueError(
                f"Selection index {index} is out of range. "
                f"Valid values are 1 through {num_cache_paths}."
            )

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
        """Plot a sorted workload selectivity distribution for one query type."""

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

        x_values = list(range(1, len(plot_selectivities) + 1))
        lower_y_limit = 0.0001

        fig, axis = plt.subplots()
        axis.plot(
            x_values,
            plot_selectivities,
            color="#222222",
            marker="x",
            markersize=4,
            linewidth=1.5,
        )
        axis.fill_between(
            x_values,
            plot_selectivities,
            y2=lower_y_limit,
            color="#777777",
            alpha=0.25,
            linewidth=0,
        )

        axis.set_yscale("log")
        axis.set_ylim(bottom=lower_y_limit, top=1)
        axis.set_title(f"{query_type.capitalize()} query selectivity distribution")

        axis.set_xlabel(r"\#Predicates")
        axis.set_ylabel("Selectivity")
        axis.set_xticks(x_values)
        axis.grid(axis="y", alpha=0.55)
        axis.grid(axis="x", visible=False)

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
                "query_selectivity_distribution_"
                f"{cache_path.stem}_{query_type}.pdf"
            )
        )

        fig.savefig(pdf_path, bbox_inches="tight")
        console.print(
            f"[green]✓[/green] Saved {query_type} selectivity distribution plot "
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
