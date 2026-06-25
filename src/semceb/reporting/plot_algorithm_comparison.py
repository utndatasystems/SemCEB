from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import pandas as pd
import seaborn as sns

from semceb.reporting.plot_params import apply_plot_params
from semceb.utils.console import console


class AlgorithmComparisonPaperPlotMixin:
    """Helpers for the paper-oriented algorithm comparison figure."""

    Q_ERROR_YLABEL_PAD = 18
    Q_ERROR_DIRECTION_LABEL_X_OFFSET = 0.075
    Q_ERROR_DIRECTION_LABEL_MIN_X = 0.01
    Q_ERROR_DIRECTION_LABEL_FONT_SIZE = "x-small"
    Q_ERROR_DIRECTION_LABEL_POSITIONS = {
        "over-\nestimation": 0.78,
        "under-\nestimation": 0.22,
    }
    LEFT_COLUMN_YLABEL_X = -0.28
    ALGORITHM_BAR_HATCH_PATTERNS = ("////", "\\\\\\\\", "xxxx", "....", "++", "oo")
    COST_YLABEL = r"\shortstack{Cost [US-\$] \\ (per sub-plan)}"
    TIME_YLABEL = r"Time [s]\\(per sub-plan)"
    MEMORY_CONSUMPTION_YLABEL = "Memory [Bytes]"
    SUPPORTED_QUERY_REFERENCE_ALGORITHM = "Semantic Histogram"
    SUPPORT_SCOPE_ALL = "All SemCEB Queries"
    SUPPORT_SCOPE_REFERENCE = "Queries supported by Semantic Histograms"
    SUPPORT_SCOPE_ORDER = (SUPPORT_SCOPE_ALL, SUPPORT_SCOPE_REFERENCE)
    SUPPORT_SCOPE_HATCHES = {
        SUPPORT_SCOPE_ALL: "",
        SUPPORT_SCOPE_REFERENCE: "xxxx",
    }

    ALGORITHM_LABELS: dict[str, str] = {
        # "Custom Algorithm Template": "Template",
        "Extrapolation Sampling 1%": "Sample 1\\%",
        "Extrapolation Sampling 5%": "Sample 5\\%",
        "Extrapolation Sampling 10%": "Sample 10\\%",
        "Extrapolation Sampling 20%": "Sample 20\\%",
        "Semantic Histogram": "SemHist",
    }

    def _plot_algorithm_comparison_paper(self, df: pd.DataFrame) -> None:
        """Plot the paper-oriented algorithm comparison figure."""

        if df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No results available for q-error analysis."
            )
            return

        apply_plot_params(
            fig_height=4.0,
            scale=1.8,
            double_column=False,
        )

        analysis_df = self._prepare_analysis_dataframe(df)
        q_error_df = self._filter_finite_q_error_rows(analysis_df)

        if q_error_df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                "No finite q-error values found for the configured algorithms."
            )
            return

        algorithm_labels = analysis_df["algorithm_label"].cat.categories.tolist()
        palette = {algorithm: "#ffffff" for algorithm in algorithm_labels}

        self.plot_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(4, 2, sharex=True)

        filter_q_error_data = q_error_df[q_error_df["query_type"] == "filter"]
        join_q_error_data = q_error_df[q_error_df["query_type"] == "join"]
        filter_cost_data = analysis_df[analysis_df["query_type"] == "filter"]
        join_cost_data = analysis_df[analysis_df["query_type"] == "join"]
        filter_time_data = analysis_df[analysis_df["query_type"] == "filter"]
        join_time_data = analysis_df[analysis_df["query_type"] == "join"]
        filter_memory_data = analysis_df[analysis_df["query_type"] == "filter"]
        join_memory_data = analysis_df[analysis_df["query_type"] == "join"]

        self._plot_q_error_subfigure(
            axis=axes[0, 0],
            data=filter_q_error_data,
            title="Filter Queries",
            algorithms=algorithm_labels,
            palette=palette,
            show_ylabel=True,
            compare_supported_subset=True,
            fixed_plot_limit=3,
        )
        self._plot_q_error_subfigure(
            axis=axes[0, 1],
            data=join_q_error_data,
            title="Join Queries",
            algorithms=algorithm_labels,
            palette=palette,
            show_ylabel=False,
            compare_supported_subset=False,
            fixed_plot_limit=None,
        )
        self._plot_cost_subfigure(
            axis=axes[1, 0],
            data=filter_cost_data,
            algorithms=algorithm_labels,
            palette=palette,
            show_ylabel=True,
            compare_supported_subset=True,
        )
        self._plot_cost_subfigure(
            axis=axes[1, 1],
            data=join_cost_data,
            algorithms=algorithm_labels,
            palette=palette,
            show_ylabel=False,
            compare_supported_subset=False,
        )
        self._plot_time_subfigure(
            axis=axes[2, 0],
            data=filter_time_data,
            algorithms=algorithm_labels,
            palette=palette,
            show_ylabel=True,
            compare_supported_subset=True,
        )
        self._plot_time_subfigure(
            axis=axes[2, 1],
            data=join_time_data,
            algorithms=algorithm_labels,
            palette=palette,
            show_ylabel=False,
            compare_supported_subset=False,
        )
        self._plot_memory_consumption_subfigure(
            axis=axes[3, 0],
            data=filter_memory_data,
            algorithms=algorithm_labels,
            palette=palette,
            show_ylabel=True,
            compare_supported_subset=True,
        )
        self._plot_memory_consumption_subfigure(
            axis=axes[3, 1],
            data=join_memory_data,
            algorithms=algorithm_labels,
            palette=palette,
            show_ylabel=False,
            compare_supported_subset=False,
        )

        fig.legend(
            handles=self._build_support_scope_legend_handles(),
            labels=list(self.SUPPORT_SCOPE_ORDER),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
            columnspacing=1.4,
            handletextpad=0.5,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        self._align_left_column_ylabels(axes[:, 0])
        if not filter_q_error_data.empty:
            self._add_q_error_direction_labels(fig=fig, axis=axes[0, 0])

        pdf_path = self.plot_dir / "algorithm_comparison_paper.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
        console.print(
            f"[green]✓[/green] Saved algorithm comparison paper plot to [bold]{pdf_path}[/bold]"
        )

        plt.close(fig)

    def _prepare_analysis_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the input data for q-error and cost plotting."""

        analysis_df = df[
            [
                "query_id",
                "algorithm_name",
                "datasets",
                "q_error",
                "cost_usd",
                "time_ms",
                "memory_consumption",
            ]
        ].copy()
        available_algorithm_names = set(analysis_df["algorithm_name"])
        available_algorithms = [
            algorithm_name
            for algorithm_name in self.ALGORITHM_LABELS
            if algorithm_name in available_algorithm_names
        ]
        analysis_df = analysis_df[
            analysis_df["algorithm_name"].isin(available_algorithms)
        ].copy()
        analysis_df["query_type"] = analysis_df["datasets"].apply(
            self._classify_query_type
        )
        analysis_df["q_error"] = pd.to_numeric(
            analysis_df["q_error"],
            errors="coerce",
        )
        analysis_df["cost_usd"] = pd.to_numeric(
            analysis_df["cost_usd"],
            errors="coerce",
        )
        analysis_df["time_ms"] = pd.to_numeric(
            analysis_df["time_ms"],
            errors="coerce",
        )
        analysis_df["memory_consumption"] = pd.to_numeric(
            analysis_df["memory_consumption"],
            errors="coerce",
        )
        analysis_df = analysis_df.dropna(subset=["query_type"])
        analysis_df["algorithm_label"] = analysis_df["algorithm_name"].map(
            self.ALGORITHM_LABELS
        )
        algorithm_labels = [
            self.ALGORITHM_LABELS[name] for name in available_algorithms
        ]
        analysis_df["algorithm_label"] = pd.Categorical(
            analysis_df["algorithm_label"],
            categories=algorithm_labels,
            ordered=True,
        )

        analysis_df = analysis_df.dropna(subset=["algorithm_label"])
        supported_query_ids = self._get_supported_query_ids(analysis_df)
        analysis_df = self._expand_analysis_dataframe_by_support_scope(
            analysis_df=analysis_df,
            supported_query_ids=supported_query_ids,
        )

        return analysis_df

    def _get_supported_query_ids(self, analysis_df: pd.DataFrame) -> set[Any]:
        """Return the query IDs covered by the reference algorithm."""

        supported_query_ids = set(
            analysis_df.loc[
                analysis_df["algorithm_name"]
                == self.SUPPORTED_QUERY_REFERENCE_ALGORITHM,
                "query_id",
            ].tolist()
        )

        if not supported_query_ids:
            raise ValueError(
                f"No queries found for reference algorithm {self.SUPPORTED_QUERY_REFERENCE_ALGORITHM!r}."
            )

        return supported_query_ids

    def _expand_analysis_dataframe_by_support_scope(
        self,
        analysis_df: pd.DataFrame,
        supported_query_ids: set[Any],
    ) -> pd.DataFrame:
        """Duplicate rows to compare all queries against the reference-supported subset."""

        all_queries_df = analysis_df[
            analysis_df["algorithm_name"] != self.SUPPORTED_QUERY_REFERENCE_ALGORITHM
        ].copy()
        all_queries_df["support_scope"] = self.SUPPORT_SCOPE_ALL

        supported_queries_df = analysis_df[
            analysis_df["query_id"].isin(supported_query_ids)
        ].copy()
        supported_queries_df["support_scope"] = self.SUPPORT_SCOPE_REFERENCE

        expanded_df = pd.concat(
            [all_queries_df, supported_queries_df],
            ignore_index=True,
        )
        expanded_df["support_scope"] = pd.Categorical(
            expanded_df["support_scope"],
            categories=list(self.SUPPORT_SCOPE_ORDER),
            ordered=True,
        )

        return expanded_df

    def _filter_finite_q_error_rows(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """Return only rows with finite q-error values and transformed plotting coordinates."""

        q_error_df = analysis_df.dropna(subset=["q_error"]).copy()
        q_error_df = q_error_df[q_error_df["q_error"].apply(math.isfinite)]
        q_error_df["q_error_plot"] = q_error_df["q_error"].apply(
            self._transform_q_error_for_plot
        )

        return q_error_df

    def _classify_query_type(self, datasets: Any) -> str | None:
        """Classify a query as filter or join based on the datasets field."""

        if not isinstance(datasets, list):
            return None

        if len(datasets) == 1:
            return "filter"

        if len(datasets) > 1:
            return "join"

        return None

    def _plot_q_error_subfigure(
        self,
        axis: Any,
        data: pd.DataFrame,
        title: str,
        algorithms: list[str],
        palette: dict[str, Any],
        show_ylabel: bool,
        compare_supported_subset: bool,
        fixed_plot_limit: int | None,
    ) -> None:
        """Plot one q-error boxplot for a single query type."""

        if data.empty:
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                f"No {title.lower()}",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
            return

        boxplot_kwargs = {
            "data": data,
            "x": "algorithm_label",
            "y": "q_error_plot",
            "order": algorithms,
            "width": 0.72 if compare_supported_subset else 0.65,
            "linewidth": 1.1,
            "boxprops": {
                "edgecolor": "#000000",
                "linewidth": 1.1,
            },
            "whiskerprops": {
                "color": "#000000",
                "linewidth": 1.1,
            },
            "capprops": {
                "color": "#000000",
                "linewidth": 1.1,
            },
            "flierprops": {
                "marker": ".",
                "markersize": 3.5,
                "markeredgecolor": "#222222",
                "markeredgewidth": 0.6,
                "markerfacecolor": "#222222",
                "linestyle": "none",
            },
            "medianprops": {
                "color": "#000000",
                "linewidth": 2.0,
            },
            "ax": axis,
            "legend": False,
        }
        if compare_supported_subset:
            boxplot_kwargs.update(
                {
                    "hue": "support_scope",
                    "hue_order": list(self.SUPPORT_SCOPE_ORDER),
                    "palette": {scope: "#ffffff" for scope in self.SUPPORT_SCOPE_ORDER},
                    "dodge": True,
                }
            )
        else:
            boxplot_kwargs.update(
                {
                    "color": "#ffffff",
                    "dodge": False,
                }
            )

        sns.boxplot(**boxplot_kwargs)
        if compare_supported_subset:
            self._apply_support_scope_box_hatches(axis=axis, data=data)

        q_error_values = data["q_error"].tolist()

        axis.set_title(title)
        axis.set_xlabel("")
        if show_ylabel:
            axis.set_ylabel("Q-Error", labelpad=self.Q_ERROR_YLABEL_PAD)
        else:
            axis.set_ylabel("")
        self._apply_q_error_ticks(
            axis=axis,
            q_error_values=q_error_values,
            fixed_plot_limit=fixed_plot_limit,
        )
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(
            FixedLocator(
                self._get_q_error_minor_tick_positions(
                    q_error_values=q_error_values,
                    fixed_plot_limit=fixed_plot_limit,
                )
            )
        )
        self._style_axis_frame(axis)
        axis.tick_params(axis="y", which="minor", length=2.5, width=0.7)
        self._style_shared_x_axis(axis, show_ticklabels=False)

        axis.axhline(
            0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.7, zorder=0
        )
        axis.grid(axis="y", alpha=0.6)
        self._mark_missing_q_error_slots(
            axis=axis,
            data=data,
            algorithms=algorithms,
            compare_supported_subset=compare_supported_subset,
        )

    def _plot_cost_subfigure(
        self,
        axis: Any,
        data: pd.DataFrame,
        algorithms: list[str],
        palette: dict[str, Any],
        show_ylabel: bool,
        compare_supported_subset: bool,
    ) -> None:
        """Plot per-query costs for one query type."""

        cost_df = self._prepare_boxplot_metric_dataframe(
            data=data,
            metric_column="cost_usd",
            require_non_negative=True,
        )

        if cost_df.empty:
            self._show_empty_axis(axis, title="", message="No cost data")
            return

        boxplot_kwargs = {
            "data": cost_df,
            "x": "algorithm_label",
            "y": "cost_usd",
            "order": algorithms,
            "width": 0.72 if compare_supported_subset else 0.65,
            "linewidth": 1.1,
            "boxprops": {
                "edgecolor": "#000000",
                "linewidth": 1.1,
            },
            "whiskerprops": {
                "color": "#000000",
                "linewidth": 1.1,
            },
            "capprops": {
                "color": "#000000",
                "linewidth": 1.1,
            },
            "flierprops": {
                "marker": ".",
                "markersize": 3.5,
                "markeredgecolor": "#222222",
                "markeredgewidth": 0.6,
                "markerfacecolor": "#222222",
                "linestyle": "none",
            },
            "medianprops": {
                "color": "#000000",
                "linewidth": 2.0,
            },
            "ax": axis,
            "legend": False,
        }
        if compare_supported_subset:
            boxplot_kwargs.update(
                {
                    "hue": "support_scope",
                    "hue_order": list(self.SUPPORT_SCOPE_ORDER),
                    "palette": {scope: "#ffffff" for scope in self.SUPPORT_SCOPE_ORDER},
                    "dodge": True,
                }
            )
        else:
            boxplot_kwargs.update(
                {
                    "color": "#ffffff",
                    "dodge": False,
                }
            )

        sns.boxplot(**boxplot_kwargs)
        if compare_supported_subset:
            self._apply_support_scope_box_hatches(axis=axis, data=cost_df)

        axis.set_title("")
        axis.set_xlabel("")
        if show_ylabel:
            axis.set_ylabel(self.COST_YLABEL)
        else:
            axis.set_ylabel("")
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(FuncFormatter(self._format_usd_tick))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(NullLocator())
        self._style_axis_frame(axis)
        self._style_shared_x_axis(axis, show_ticklabels=False)
        axis.grid(axis="y", alpha=0.6)
        self._mark_missing_metric_slots_at_zero(
            axis=axis,
            data=cost_df,
            algorithms=algorithms,
            compare_supported_subset=compare_supported_subset,
        )

    def _plot_time_subfigure(
        self,
        axis: Any,
        data: pd.DataFrame,
        algorithms: list[str],
        palette: dict[str, Any],
        show_ylabel: bool,
        compare_supported_subset: bool,
    ) -> None:
        """Plot per-query runtimes for one query type."""

        time_df = self._prepare_boxplot_metric_dataframe(
            data=data,
            metric_column="time_ms",
            require_non_negative=True,
        )
        time_df["time_s"] = time_df["time_ms"] / 1000.0

        if time_df.empty:
            self._show_empty_axis(axis, title="", message="No runtime data")
            return

        boxplot_kwargs = {
            "data": time_df,
            "x": "algorithm_label",
            "y": "time_s",
            "order": algorithms,
            "width": 0.72 if compare_supported_subset else 0.65,
            "linewidth": 1.1,
            "boxprops": {
                "edgecolor": "#000000",
                "linewidth": 1.1,
            },
            "whiskerprops": {
                "color": "#000000",
                "linewidth": 1.1,
            },
            "capprops": {
                "color": "#000000",
                "linewidth": 1.1,
            },
            "flierprops": {
                "marker": ".",
                "markersize": 3.5,
                "markeredgecolor": "#222222",
                "markeredgewidth": 0.6,
                "markerfacecolor": "#222222",
                "linestyle": "none",
            },
            "medianprops": {
                "color": "#000000",
                "linewidth": 2.0,
            },
            "ax": axis,
            "legend": False,
        }
        if compare_supported_subset:
            boxplot_kwargs.update(
                {
                    "hue": "support_scope",
                    "hue_order": list(self.SUPPORT_SCOPE_ORDER),
                    "palette": {scope: "#ffffff" for scope in self.SUPPORT_SCOPE_ORDER},
                    "dodge": True,
                }
            )
        else:
            boxplot_kwargs.update(
                {
                    "color": "#ffffff",
                    "dodge": False,
                }
            )

        sns.boxplot(**boxplot_kwargs)
        if compare_supported_subset:
            self._apply_support_scope_box_hatches(axis=axis, data=time_df)

        axis.set_title("")
        axis.set_xlabel("")
        if show_ylabel:
            axis.set_ylabel(self.TIME_YLABEL)
        else:
            axis.set_ylabel("")
        axis.set_ylim(bottom=0)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(NullLocator())
        self._style_axis_frame(axis)
        self._style_shared_x_axis(axis, show_ticklabels=False)
        axis.grid(axis="y", alpha=0.6)
        self._mark_missing_metric_slots_at_zero(
            axis=axis,
            data=time_df,
            algorithms=algorithms,
            compare_supported_subset=compare_supported_subset,
        )

    def _plot_memory_consumption_subfigure(
        self,
        axis: Any,
        data: pd.DataFrame,
        algorithms: list[str],
        palette: dict[str, Any],
        show_ylabel: bool,
        compare_supported_subset: bool,
    ) -> None:
        """Plot one constant memory-consumption value per algorithm for one query type."""

        memory_df = self._build_memory_consumption_dataframe(data, algorithms)

        if memory_df.empty:
            self._show_empty_axis(axis, title="", message="No memory data")
            return

        barplot_kwargs = {
            "data": memory_df,
            "x": "algorithm_label",
            "y": "memory_consumption",
            "order": algorithms,
            "width": 0.72 if compare_supported_subset else 0.65,
            "edgecolor": "#000000",
            "linewidth": 1.1,
            "ax": axis,
            "legend": False,
        }
        if compare_supported_subset:
            barplot_kwargs.update(
                {
                    "hue": "support_scope",
                    "hue_order": list(self.SUPPORT_SCOPE_ORDER),
                    "palette": {scope: "#ffffff" for scope in self.SUPPORT_SCOPE_ORDER},
                }
            )
        else:
            barplot_kwargs.update({"color": "#ffffff"})

        sns.barplot(**barplot_kwargs)
        if compare_supported_subset:
            self._apply_support_scope_bar_hatches(axis)

        axis.set_title("")
        axis.set_xlabel("")
        if show_ylabel:
            axis.set_ylabel(self.MEMORY_CONSUMPTION_YLABEL)
        else:
            axis.set_ylabel("")
        axis.set_yscale("symlog", linthresh=1)
        axis.set_ylim(bottom=0)
        self._apply_memory_ticks(
            axis=axis,
            memory_values=memory_df["memory_consumption"].tolist(),
        )
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(NullLocator())
        self._style_axis_frame(axis)
        self._style_shared_x_axis(axis, show_ticklabels=True)
        axis.grid(axis="y", alpha=0.6)
        self._mark_missing_metric_slots_at_zero(
            axis=axis,
            data=memory_df,
            algorithms=algorithms,
            compare_supported_subset=compare_supported_subset,
        )

    def _prepare_boxplot_metric_dataframe(
        self,
        data: pd.DataFrame,
        metric_column: str,
        require_non_negative: bool,
    ) -> pd.DataFrame:
        """Prepare one per-query metric for boxplot rendering."""

        valid_metric_df = data.dropna(subset=[metric_column]).copy()
        valid_metric_df = valid_metric_df[
            valid_metric_df[metric_column].apply(math.isfinite)
        ]
        if require_non_negative:
            valid_metric_df = valid_metric_df[valid_metric_df[metric_column] >= 0]

        return valid_metric_df

    def _build_memory_consumption_dataframe(
        self,
        data: pd.DataFrame,
        algorithms: list[str],
    ) -> pd.DataFrame:
        """Validate and extract one memory-consumption value per algorithm and support scope."""

        if data.empty:
            return pd.DataFrame(
                columns=["algorithm_label", "support_scope", "memory_consumption"]
            )

        valid_memory_df = data.dropna(subset=["memory_consumption"]).copy()
        valid_memory_df = valid_memory_df[
            valid_memory_df["memory_consumption"].apply(math.isfinite)
        ]
        if valid_memory_df.empty:
            return pd.DataFrame(
                columns=["algorithm_label", "support_scope", "memory_consumption"]
            )

        grouped_values = valid_memory_df.groupby(
            ["algorithm_label", "support_scope"],
            observed=False,
        )["memory_consumption"]
        distinct_values = grouped_values.nunique(dropna=True)
        inconsistent_algorithms = distinct_values[distinct_values > 1].index.tolist()
        if inconsistent_algorithms:
            raise ValueError(
                "Inconsistent memory_consumption values found for algorithm/support-scope combinations: "
                + ", ".join(
                    f"{algorithm} [{support_scope}]"
                    for algorithm, support_scope in inconsistent_algorithms
                )
            )

        memory_by_algorithm = grouped_values.first().reset_index()
        memory_by_algorithm = memory_by_algorithm[
            memory_by_algorithm["memory_consumption"] >= 0
        ]
        memory_by_algorithm["algorithm_label"] = pd.Categorical(
            memory_by_algorithm["algorithm_label"],
            categories=algorithms,
            ordered=True,
        )
        memory_by_algorithm["support_scope"] = pd.Categorical(
            memory_by_algorithm["support_scope"],
            categories=list(self.SUPPORT_SCOPE_ORDER),
            ordered=True,
        )
        memory_by_algorithm = memory_by_algorithm.sort_values(
            ["algorithm_label", "support_scope"]
        )

        return memory_by_algorithm

    def _show_empty_axis(self, axis: Any, title: str, message: str) -> None:
        """Render a consistent placeholder for empty subplots."""

        axis.set_title(title)
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    def _style_axis_frame(self, axis: Any) -> None:
        """Apply the shared axis frame and major tick styling."""

        axis.tick_params(
            axis="both",
            which="both",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="out",
        )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#666666")
            spine.set_linewidth(0.8)

    def _style_shared_x_axis(self, axis: Any, show_ticklabels: bool) -> None:
        """Configure the shared categorical x-axis formatting."""

        axis.tick_params(axis="x", labelrotation=30, labelbottom=show_ticklabels)
        for label in axis.get_xticklabels():
            label.set_ha("right")

    def _apply_support_scope_box_hatches(
        self,
        axis: Any,
        data: pd.DataFrame,
    ) -> None:
        """Apply support-scope hatch patterns to dodged boxplots."""

        plotted_boxes = [patch for patch in axis.patches if patch.get_fill()]
        plotted_scopes_df = data[["algorithm_label", "support_scope"]].drop_duplicates()
        plotted_scopes: list[str] = []
        for support_scope in self.SUPPORT_SCOPE_ORDER:
            scope_rows = plotted_scopes_df[
                plotted_scopes_df["support_scope"] == support_scope
            ].sort_values("algorithm_label")
            plotted_scopes.extend([str(support_scope)] * len(scope_rows))

        if len(plotted_boxes) != len(plotted_scopes):
            raise ValueError(
                "Mismatch between plotted box count and support-scope combinations."
            )

        for patch, support_scope in zip(
            plotted_boxes,
            plotted_scopes,
            strict=False,
        ):
            patch.set_hatch(self.SUPPORT_SCOPE_HATCHES[support_scope])

    def _apply_support_scope_bar_hatches(self, axis: Any) -> None:
        """Apply support-scope hatch patterns to dodged bars."""

        for container, support_scope in zip(
            axis.containers,
            self.SUPPORT_SCOPE_ORDER,
            strict=False,
        ):
            for patch in container.patches:
                patch.set_hatch(self.SUPPORT_SCOPE_HATCHES[support_scope])

    def _build_support_scope_legend_handles(self) -> list[Patch]:
        """Create legend handles for the support-scope comparison."""

        return [
            Patch(
                facecolor="#ffffff",
                edgecolor="#000000",
                hatch=self.SUPPORT_SCOPE_HATCHES[support_scope],
                linewidth=1.1,
                label=support_scope,
            )
            for support_scope in self.SUPPORT_SCOPE_ORDER
        ]

    def _mark_missing_q_error_slots(
        self,
        axis: Any,
        data: pd.DataFrame,
        algorithms: list[str],
        compare_supported_subset: bool,
    ) -> None:
        """Mark algorithm slots without q-error observations using a red x."""

        x_positions = self._get_missing_metric_slot_positions(
            data=data,
            algorithms=algorithms,
            compare_supported_subset=compare_supported_subset,
        )
        if not x_positions:
            return

        axis.scatter(
            x_positions,
            [0.0] * len(x_positions),
            marker="x",
            s=55,
            linewidths=1.8,
            color="#cc0000",
            zorder=4,
        )

    def _mark_missing_metric_slots_on_x_axis(
        self,
        axis: Any,
        data: pd.DataFrame,
        algorithms: list[str],
        compare_supported_subset: bool,
    ) -> None:
        """Mark metric slots without observations using a red x near the x-axis."""

        x_positions = self._get_missing_metric_slot_positions(
            data=data,
            algorithms=algorithms,
            compare_supported_subset=compare_supported_subset,
        )
        if not x_positions:
            return

        axis.scatter(
            x_positions,
            [0.035] * len(x_positions),
            marker="x",
            s=55,
            linewidths=1.8,
            color="#cc0000",
            transform=transforms.blended_transform_factory(
                axis.transData,
                axis.transAxes,
            ),
            clip_on=False,
            zorder=4,
        )

    def _mark_missing_metric_slots_at_zero(
        self,
        axis: Any,
        data: pd.DataFrame,
        algorithms: list[str],
        compare_supported_subset: bool,
    ) -> None:
        """Mark metric slots without observations using a red x at y=0."""

        x_positions = self._get_missing_metric_slot_positions(
            data=data,
            algorithms=algorithms,
            compare_supported_subset=compare_supported_subset,
        )
        if not x_positions:
            return

        axis.scatter(
            x_positions,
            [0.0] * len(x_positions),
            marker="x",
            s=55,
            linewidths=1.8,
            color="#cc0000",
            clip_on=False,
            zorder=4,
        )

    def _get_missing_metric_slot_positions(
        self,
        data: pd.DataFrame,
        algorithms: list[str],
        compare_supported_subset: bool,
    ) -> list[float]:
        """Return x positions for missing algorithm/support-scope metric slots."""

        x_positions: list[float] = []

        if compare_supported_subset:
            box_width = 0.72
            scope_count = len(self.SUPPORT_SCOPE_ORDER)
            offsets = [
                (-box_width / 2.0) + ((scope_index + 0.5) * box_width / scope_count)
                for scope_index in range(scope_count)
            ]
            for algorithm_index, algorithm_label in enumerate(algorithms):
                for scope_index, support_scope in enumerate(self.SUPPORT_SCOPE_ORDER):
                    has_data = not data[
                        (data["algorithm_label"] == algorithm_label)
                        & (data["support_scope"] == support_scope)
                    ].empty
                    if not has_data:
                        x_positions.append(algorithm_index + offsets[scope_index])
        else:
            for algorithm_index, algorithm_label in enumerate(algorithms):
                has_data = not data[data["algorithm_label"] == algorithm_label].empty
                if not has_data:
                    x_positions.append(float(algorithm_index))

        return x_positions

    def _apply_bar_hatches(
        self,
        axis: Any,
        plotted_algorithms: list[str],
        all_algorithms: list[str],
    ) -> None:
        """Apply a stable per-algorithm hatch pattern to a bar plot."""

        hatch_map = {
            algorithm: self.ALGORITHM_BAR_HATCH_PATTERNS[
                index % len(self.ALGORITHM_BAR_HATCH_PATTERNS)
            ]
            for index, algorithm in enumerate(all_algorithms)
        }

        for patch, algorithm in zip(axis.patches, plotted_algorithms, strict=False):
            patch.set_hatch(hatch_map[algorithm])

    def _align_left_column_ylabels(self, axes: Any) -> None:
        """Align all left-column y-axis labels to the same x-position."""

        for axis in axes:
            if axis.get_ylabel():
                axis.yaxis.set_label_coords(self.LEFT_COLUMN_YLABEL_X, 0.5)

    def _add_q_error_direction_labels(self, fig: Any, axis: Any) -> None:
        """Add separate direction labels for the upper and lower plot halves."""

        axis_position = axis.get_position()
        label_x = max(
            self.Q_ERROR_DIRECTION_LABEL_MIN_X,
            axis_position.x0 - self.Q_ERROR_DIRECTION_LABEL_X_OFFSET,
        )

        for label_text, relative_y in self.Q_ERROR_DIRECTION_LABEL_POSITIONS.items():
            fig.text(
                label_x,
                axis_position.y0 + (relative_y * axis_position.height),
                label_text,
                rotation=90,
                ha="center",
                va="center",
                fontsize=self.Q_ERROR_DIRECTION_LABEL_FONT_SIZE,
            )

    def _transform_q_error_for_plot(self, q_error: float) -> float:
        """Map signed q-error values onto a log-like plotting coordinate."""

        if q_error == 0:
            return float("nan")

        if q_error == 1:
            return 0.0

        magnitude = abs(q_error)

        return math.copysign(math.log10(magnitude), q_error)

    def _apply_q_error_ticks(
        self,
        axis: Any,
        q_error_values: list[float],
        fixed_plot_limit: int | None = None,
    ) -> None:
        """Set log-like q-error ticks with 1 in the center."""

        if not q_error_values:
            axis.set_yticks([0.0])
            axis.set_yticklabels(["1"])
            return

        plot_limit = self._get_q_error_plot_limit(
            q_error_values=q_error_values,
            fixed_plot_limit=fixed_plot_limit,
        )
        axis.set_ylim(-plot_limit, plot_limit)

        major_tick_values = self._get_q_error_major_tick_values(plot_limit)

        axis.set_yticks(
            [self._transform_q_error_for_plot(value) for value in major_tick_values]
        )
        axis.set_yticklabels(
            [self._format_scientific_tick(value) for value in major_tick_values]
        )

    def _get_q_error_plot_limit(
        self,
        q_error_values: list[float],
        fixed_plot_limit: int | None = None,
    ) -> int:
        """Return the symmetric log-scale limit used for q-error plotting."""

        if fixed_plot_limit is not None:
            return fixed_plot_limit

        max_abs_value = max(
            abs(value) for value in q_error_values if math.isfinite(value)
        )
        return max(1, math.ceil(math.log10(max_abs_value)) + 1)

    def _get_q_error_major_tick_values(self, plot_limit: int) -> list[float]:
        """Select a reduced set of major q-error ticks to avoid label crowding."""

        if plot_limit == 1:
            exponents = [1]
        elif plot_limit == 2:
            exponents = [2]
        elif plot_limit <= 6:
            exponents = list(range(2, plot_limit + 1, 2))
        else:
            exponents = list(range(3, plot_limit + 1, 3))

        if not exponents and plot_limit >= 1:
            exponents = list(range(1, plot_limit + 1))

        tick_values = [10**exponent for exponent in exponents]

        return sorted({-value for value in tick_values} | {1.0} | set(tick_values))

    def _get_q_error_minor_tick_positions(
        self,
        q_error_values: list[float],
        fixed_plot_limit: int | None = None,
    ) -> list[float]:
        """Build minor tick positions that follow log spacing on the transformed axis."""

        if not q_error_values:
            return []

        plot_limit = self._get_q_error_plot_limit(
            q_error_values=q_error_values,
            fixed_plot_limit=fixed_plot_limit,
        )
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

    def _format_scientific_tick(self, value: float) -> str:
        """Format tick labels as powers of ten."""

        if value == 1:
            return "1"

        exponent = int(round(math.log10(abs(value))))

        return rf"$10^{{{exponent}}}$"

    def _format_usd_tick(self, value: float, _position: float) -> str:
        """Format aggregated cost ticks in US dollars."""

        if value == 0:
            return r"\$0"
        if abs(value) >= 1:
            return rf"\${value:,.2f}"
        if abs(value) >= 0.01:
            return rf"\${value:,.2f}"

        return rf"\${value:,.3f}"

    def _apply_memory_ticks(self, axis: Any, memory_values: list[float]) -> None:
        """Set a sparse set of major ticks for the memory subplot."""

        positive_values = [
            value for value in memory_values if math.isfinite(value) and value > 0
        ]
        if not positive_values:
            axis.set_yticks([0])
            axis.set_yticklabels(["0"])
            return

        max_exponent = max(
            0,
            math.ceil(math.log10(max(positive_values))),
        )
        exponent_step = 2 if max_exponent >= 4 else 1
        exponents = list(range(0, max_exponent + 1, exponent_step))
        if exponents[-1] != max_exponent:
            exponents.append(max_exponent)

        tick_values = [0] + [10**exponent for exponent in exponents]
        axis.yaxis.set_major_locator(FixedLocator(tick_values))
        axis.set_yticklabels(
            [
                "0" if value == 0 else self._format_scientific_tick(float(value))
                for value in tick_values
            ]
        )
