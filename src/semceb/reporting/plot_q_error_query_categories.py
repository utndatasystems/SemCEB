from __future__ import annotations

import math
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import pandas as pd
import seaborn as sns

from semceb.reporting.plot_params import apply_plot_params
from semceb.utils.console import console


class QErrorQueryCategoriesPlotMixin:
    """Helpers for per-algorithm q-error drill-down plots by query category."""

    Q_ERROR_QUERY_CATEGORY_PLOT_LIMIT = 4
    Q_ERROR_QUERY_CATEGORY_COMPARISON_ALGORITHMS = (
        ("Extrapolation Sampling 5%", "5\\%-Sample"),
        ("Semantic Histogram", "Semantic Histogram"),
    )
    QUERY_CATEGORY_ORDER = (
        "equality",
        "inequality",
        "lexical",
        "negation",
        "ordinal",
        "spatial",
        "temporal",
        "multiple",
    )

    def _plot_q_error_query_categories(self, df: pd.DataFrame) -> None:
        """Create one q-error drill-down plot per algorithm and query type."""

        if df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No results available for q-error category analysis."
            )
            return

        apply_plot_params(
            fig_height=1.2,
            scale=1.8,
            double_column=False,
        )

        analysis_df = self._prepare_q_error_query_category_dataframe(df)

        if analysis_df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No finite q-error values found for category drill-down plots."
            )
            return

        self.plot_dir.mkdir(parents=True, exist_ok=True)

        for algorithm_name in analysis_df["algorithm_name"].drop_duplicates():
            algorithm_df = analysis_df[
                analysis_df["algorithm_name"] == algorithm_name
            ].copy()

            self._plot_q_error_query_categories_for_algorithm(
                algorithm_name=algorithm_name,
                algorithm_df=algorithm_df,
            )

    def _plot_q_error_query_categories_comparison(self, df: pd.DataFrame) -> None:
        """Create a stacked filter-query comparison across three algorithms."""

        if df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No results available for q-error category comparison plot."
            )
            return

        apply_plot_params(
            fig_height=2.2,
            scale=1.8,
            double_column=False,
        )

        analysis_df = self._prepare_q_error_query_category_dataframe(df)

        if analysis_df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No finite q-error values found for the comparison plot."
            )
            return

        algorithm_names = [
            algorithm_name
            for algorithm_name, _display_name in (
                self.Q_ERROR_QUERY_CATEGORY_COMPARISON_ALGORITHMS
            )
        ]
        filter_df = analysis_df[
            (analysis_df["query_type"] == "filter")
            & (analysis_df["algorithm_name"].isin(algorithm_names))
        ].copy()

        if filter_df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No filter-query results found for the comparison plot."
            )
            return

        common_query_ids = self._get_common_supported_query_ids(
            data=filter_df,
            algorithm_names=algorithm_names,
        )
        if not common_query_ids:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No filter queries are supported by all comparison algorithms."
            )
            return

        comparison_df = filter_df[filter_df["query_id"].isin(common_query_ids)].copy()
        categories = self._get_query_category_plot_order(
            comparison_df["query_category_group"].drop_duplicates().tolist()
        )
        category_labels = self._build_query_category_count_labels(
            data=comparison_df,
            categories=categories,
        )

        self.plot_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            nrows=len(self.Q_ERROR_QUERY_CATEGORY_COMPARISON_ALGORITHMS),
            ncols=1,
            sharex=True,
        )

        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for index, ((algorithm_name, display_name), axis) in enumerate(
            zip(self.Q_ERROR_QUERY_CATEGORY_COMPARISON_ALGORITHMS, axes, strict=False)
        ):
            algorithm_df = comparison_df[
                comparison_df["algorithm_name"] == algorithm_name
            ].copy()
            self._plot_q_error_query_category_subfigure(
                axis=axis,
                data=algorithm_df,
                title=display_name,
                categories=categories,
                show_ylabel=True,
                show_xticklabels=(
                    index == len(self.Q_ERROR_QUERY_CATEGORY_COMPARISON_ALGORITHMS) - 1
                ),
                mark_empty_categories=True,
                custom_xticklabels=category_labels,
            )
            axis.set_title(display_name)

        fig.tight_layout()
        for axis in axes:
            self._add_q_error_direction_labels(fig=fig, axis=axis)

        pdf_path = self.plot_dir / "q_error_query_categories_comparison.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
        console.print(
            f"[green]✓[/green] Saved q-error category comparison plot to [bold]{pdf_path}[/bold]"
        )

        plt.close(fig)

    def _prepare_q_error_query_category_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize q-error data and assign category buckets for plotting."""

        analysis_df = df[
            [
                "query_id",
                "algorithm_name",
                "datasets",
                "query_category",
                "q_error",
            ]
        ].copy()
        analysis_df["query_type"] = analysis_df["datasets"].apply(
            self._classify_query_type
        )
        analysis_df["q_error"] = pd.to_numeric(
            analysis_df["q_error"],
            errors="coerce",
        )
        analysis_df = analysis_df.dropna(subset=["query_type", "q_error"]).copy()
        analysis_df = analysis_df[analysis_df["q_error"].apply(math.isfinite)].copy()

        if analysis_df.empty:
            return analysis_df

        analysis_df["query_category_group"] = analysis_df["query_category"].apply(
            self._group_query_categories
        )
        analysis_df["q_error_plot"] = analysis_df["q_error"].apply(
            self._transform_q_error_for_plot
        )

        return analysis_df

    def _plot_q_error_query_categories_for_algorithm(
        self,
        algorithm_name: str,
        algorithm_df: pd.DataFrame,
    ) -> None:
        """Render one filter plot and one join plot for one algorithm."""

        categories = self._get_query_category_plot_order(
            algorithm_df["query_category_group"].drop_duplicates().tolist()
        )
        plot_specs = (
            ("filter", "Filter Queries"),
            ("join", "Join Queries"),
        )

        for query_type, title in plot_specs:
            query_type_df = algorithm_df[
                algorithm_df["query_type"] == query_type
            ].copy()
            fig, axis = plt.subplots()
            self._plot_q_error_query_category_subfigure(
                axis=axis,
                data=query_type_df,
                title=title,
                categories=categories,
                show_ylabel=True,
            )
            fig.tight_layout()

            if not query_type_df.empty:
                self._add_q_error_direction_labels(fig=fig, axis=axis)

            pdf_path = self.plot_dir / (
                "q_error_query_categories_"
                f"{self._sanitize_filename_component(algorithm_name)}_{query_type}.pdf"
            )
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
            console.print(
                f"[green]✓[/green] Saved q-error category drill-down plot to [bold]{pdf_path}[/bold]"
            )

            plt.close(fig)

    def _plot_q_error_query_category_subfigure(
        self,
        axis: Any,
        data: pd.DataFrame,
        title: str,
        categories: list[str],
        show_ylabel: bool,
        show_xticklabels: bool = True,
        mark_empty_categories: bool = False,
        custom_xticklabels: list[str] | None = None,
    ) -> None:
        """Plot one q-error category drill-down subfigure."""

        if data.empty:
            self._show_empty_q_error_query_category_axis(
                axis=axis,
                title=title,
                message=f"No {title.lower()}",
                show_ylabel=show_ylabel,
            )
            return

        sns.boxplot(
            data=data,
            x="query_category_group",
            y="q_error_plot",
            order=categories,
            color="#ffffff",
            width=0.65,
            linewidth=1.1,
            boxprops={
                "edgecolor": "#000000",
                "linewidth": 1.1,
            },
            whiskerprops={
                "color": "#000000",
                "linewidth": 1.1,
            },
            capprops={
                "color": "#000000",
                "linewidth": 1.1,
            },
            flierprops={
                "marker": "x",
                "markersize": 4.0,
                "markeredgecolor": "#222222",
                "markeredgewidth": 0.9,
                "markerfacecolor": "none",
                "linestyle": "none",
            },
            medianprops={
                "color": "#000000",
                "linewidth": 2.0,
            },
            ax=axis,
        )

        axis.set_title("")
        axis.set_xlabel("")
        if show_ylabel:
            axis.set_ylabel("Q-Error", labelpad=self.Q_ERROR_YLABEL_PAD)
        else:
            axis.set_ylabel("")
        self._apply_fixed_q_error_query_category_ticks(axis)
        self._style_axis_frame(axis)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(
            FixedLocator(self._get_fixed_q_error_query_category_minor_tick_positions())
        )
        axis.tick_params(axis="y", which="minor", length=2.5, width=0.7)
        self._style_shared_x_axis(axis, show_ticklabels=show_xticklabels)
        if custom_xticklabels is not None:
            axis.set_xticks(range(len(custom_xticklabels)))
            axis.set_xticklabels(custom_xticklabels)
        axis.axhline(
            0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.7, zorder=0
        )
        axis.grid(axis="y", alpha=0.6)
        axis.grid(axis="x", visible=False)
        if mark_empty_categories:
            self._mark_empty_q_error_query_categories(
                axis=axis,
                data=data,
                categories=categories,
            )

    def _show_empty_q_error_query_category_axis(
        self,
        axis: Any,
        title: str,
        message: str,
        show_ylabel: bool,
    ) -> None:
        """Render an empty q-error subplot without clearing the shared y-axis ticks."""

        axis.set_title("")
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_xlabel("")
        if show_ylabel:
            axis.set_ylabel("Q-Error", labelpad=self.Q_ERROR_YLABEL_PAD)
        else:
            axis.set_ylabel("")
        self._apply_fixed_q_error_query_category_ticks(axis)
        self._style_axis_frame(axis)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(
            FixedLocator(self._get_fixed_q_error_query_category_minor_tick_positions())
        )
        axis.tick_params(axis="y", which="minor", length=2.5, width=0.7)
        self._style_shared_x_axis(axis, show_ticklabels=False)
        axis.axhline(
            0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.7, zorder=0
        )
        axis.grid(axis="y", alpha=0.6)
        axis.grid(axis="x", visible=False)

    def _apply_fixed_q_error_query_category_ticks(self, axis: Any) -> None:
        """Pin the drill-down q-error axis to a symmetric +/- 10^4 range."""

        plot_limit = self.Q_ERROR_QUERY_CATEGORY_PLOT_LIMIT
        axis.set_ylim(-plot_limit, plot_limit)

        major_tick_values = self._get_q_error_major_tick_values(plot_limit)
        axis.set_yticks(
            [self._transform_q_error_for_plot(value) for value in major_tick_values]
        )
        axis.set_yticklabels(
            [self._format_scientific_tick(value) for value in major_tick_values]
        )

    def _get_fixed_q_error_query_category_minor_tick_positions(self) -> list[float]:
        """Return minor q-error tick positions for the fixed +/- 10^4 plot range."""

        plot_limit = self.Q_ERROR_QUERY_CATEGORY_PLOT_LIMIT
        minor_positions: list[float] = []

        for exponent in range(plot_limit):
            for factor in range(2, 10):
                raw_value = factor * (10**exponent)
                if raw_value >= 10**plot_limit:
                    continue

                position = self._transform_q_error_for_plot(float(raw_value))
                minor_positions.append(position)
                minor_positions.append(-position)

        return sorted(set(minor_positions))

    def _group_query_categories(self, query_categories: Any) -> str:
        """Map raw query categories to one of the requested plotting buckets."""

        normalized_categories = self._normalize_query_categories(query_categories)

        if not normalized_categories:
            raise ValueError(
                f"Encountered query without categories: {query_categories!r}"
            )

        unique_categories = sorted(set(normalized_categories))

        if len(unique_categories) == 1:
            return unique_categories[0]

        if set(unique_categories) == {"equality", "negation"}:
            return "inequality"

        return "multiple"

    def _normalize_query_categories(self, query_categories: Any) -> list[str]:
        """Normalize a raw category value into a cleaned list of category names."""

        if isinstance(query_categories, str):
            candidates = [query_categories]
        elif isinstance(query_categories, list):
            candidates = query_categories
        else:
            return []

        normalized_categories: list[str] = []
        for category in candidates:
            if category is None:
                continue

            normalized_category = str(category).strip().lower()
            if normalized_category:
                normalized_categories.append(normalized_category)

        return normalized_categories

    def _get_query_category_plot_order(self, categories: list[str]) -> list[str]:
        """Return a stable display order that always includes the standard buckets."""

        observed_categories = set(categories)
        ordered_categories = list(self.QUERY_CATEGORY_ORDER)
        remaining_categories = sorted(observed_categories - set(ordered_categories))

        return ordered_categories + remaining_categories

    def _sanitize_filename_component(self, value: str) -> str:
        """Convert a plot label into a filesystem-safe filename component."""

        sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
        return sanitized.strip("._") or "value"

    def _get_common_supported_query_ids(
        self,
        data: pd.DataFrame,
        algorithm_names: list[str],
    ) -> set[Any]:
        """Return query ids that have plottable q-error values for every algorithm."""

        common_query_ids: set[Any] | None = None

        for algorithm_name in algorithm_names:
            algorithm_query_ids = set(
                data.loc[data["algorithm_name"] == algorithm_name, "query_id"]
                .dropna()
                .tolist()
            )
            if common_query_ids is None:
                common_query_ids = algorithm_query_ids
            else:
                common_query_ids &= algorithm_query_ids

        return common_query_ids or set()

    def _mark_empty_q_error_query_categories(
        self,
        axis: Any,
        data: pd.DataFrame,
        categories: list[str],
    ) -> None:
        """Mark empty category slots with a red x on the q-error baseline."""

        category_counts = (
            data["query_category_group"].value_counts()
            if not data.empty
            else pd.Series(dtype=int)
        )

        empty_positions = [
            index
            for index, category in enumerate(categories)
            if int(category_counts.get(category, 0)) == 0
        ]
        if not empty_positions:
            return

        axis.scatter(
            empty_positions,
            [0.0] * len(empty_positions),
            marker="x",
            s=80,
            linewidths=1.8,
            color="#cc0000",
            zorder=4,
        )

    def _build_query_category_count_labels(
        self,
        data: pd.DataFrame,
        categories: list[str],
    ) -> list[str]:
        """Format category tick labels with shared-subset query counts."""

        category_counts = (
            data[["query_id", "query_category_group"]]
            .drop_duplicates()["query_category_group"]
            .value_counts()
        )

        return [
            f"{category}\n($|q|={int(category_counts.get(category, 0))}$)"
            for category in categories
        ]
