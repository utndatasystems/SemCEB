from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import pandas as pd
import seaborn as sns

from semceb.reporting.plot_params import apply_plot_params
from semceb.utils.console import console


class QErrorAnalysisPlotMixin:
    """Helpers for plotting q-error distributions by query type and algorithm."""

    ALGORITHM_LABELS: dict[str, str] = {
        #"Custom Algorithm Template": "Template",
        "Extrapolation Sampling 1%": "ES 1%",
        "Extrapolation Sampling 5%": "ES 5%",
        "Extrapolation Sampling 10%": "ES 10%",
        "Extrapolation Sampling 20%": "ES 20%",
    }

    def _plot_q_error_analysis(self, df: pd.DataFrame) -> None:
        """Plot q-error boxplots for filter and join queries."""

        if df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] No results available for q-error analysis."
            )
            return

        apply_plot_params(
            fig_height=2.0,
            scale=2,
            double_column=False,
        )

        analysis_df = self._prepare_q_error_analysis_dataframe(df)

        if analysis_df.empty:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                "No finite q-error values found for the configured algorithms."
            )
            return

        algorithms = analysis_df["algorithm_label"].cat.categories.tolist()
        algorithm_styles = self._get_algorithm_styles(algorithms)
        palette = {
            algorithm: algorithm_styles[algorithm]["facecolor"]
            for algorithm in algorithms
        }

        self.plot_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2)

        self._plot_q_error_subfigure(
            axis=axes[0],
            data=analysis_df[analysis_df["query_type"] == "filter"],
            title="Filter queries",
            algorithms=algorithms,
            palette=palette,
            show_ylabel=True,
        )
        self._plot_q_error_subfigure(
            axis=axes[1],
            data=analysis_df[analysis_df["query_type"] == "join"],
            title="Join queries",
            algorithms=algorithms,
            palette=palette,
            show_ylabel=False,
        )

        fig.tight_layout()

        pdf_path = self.plot_dir / "q_error_analysis.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
        console.print(
            f"[green]✓[/green] Saved q-error analysis plot to [bold]{pdf_path}[/bold]"
        )

        plt.close(fig)

    def _prepare_q_error_analysis_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the input data for q-error plotting."""

        analysis_df = df[["algorithm_name", "datasets", "q_error"]].copy()
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
        analysis_df = analysis_df.dropna(subset=["query_type", "q_error"])
        analysis_df = analysis_df[analysis_df["q_error"].apply(math.isfinite)]
        analysis_df["algorithm_label"] = analysis_df["algorithm_name"].map(
            self.ALGORITHM_LABELS
        )
        algorithm_labels = [self.ALGORITHM_LABELS[name] for name in available_algorithms]
        analysis_df["algorithm_label"] = pd.Categorical(
            analysis_df["algorithm_label"],
            categories=algorithm_labels,
            ordered=True,
        )
        analysis_df["q_error_plot"] = analysis_df["q_error"].apply(
            self._transform_q_error_for_plot
        )

        analysis_df = analysis_df.dropna(subset=["algorithm_label"])

        return analysis_df

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

        sns.boxplot(
            data=data,
            x="algorithm_label",
            y="q_error_plot",
            hue="algorithm_label",
            order=algorithms,
            hue_order=algorithms,
            palette=palette,
            width=0.65,
            linewidth=1.1,
            flierprops={
                "marker": "x",
                "markersize": 4.0,
                "markeredgecolor": "#222222",
                "markeredgewidth": 0.9,
                "markerfacecolor": "none",
                "linestyle": "none",
            },
            ax=axis,
            dodge=False,
            legend=False,
        )

        axis.set_title(title)
        axis.set_xlabel("")
        if show_ylabel:
            axis.set_ylabel(
                r"q-error"
                "\n"
                r"{\small $\leftarrow$ underestimation \quad overestimation $\rightarrow$}",
                labelpad=6,
            )
        else:
            axis.set_ylabel("")
        self._apply_q_error_ticks(axis=axis, data=data["q_error"].tolist())
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(
            FixedLocator(
                self._build_q_error_minor_tick_positions(
                    data["q_error"].tolist()
                )
            )
        )
        axis.tick_params(
            axis="both",
            which="both",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="out",
        )
        axis.tick_params(axis="y", which="minor", length=2.5, width=0.7)
        axis.tick_params(axis="x", labelrotation=30)
        for label in axis.get_xticklabels():
            label.set_ha("right")

        axis.axhline(0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.7, zorder=0)
        axis.grid(axis="y", alpha=0.6)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#666666")
            spine.set_linewidth(0.8)

    def _transform_q_error_for_plot(self, q_error: float) -> float:
        """Map signed q-error values onto a log-like plotting coordinate."""

        if q_error == 0:
            return float("nan")

        if q_error == 1:
            return 0.0

        magnitude = abs(q_error)

        return math.copysign(math.log10(magnitude), q_error)

    def _apply_q_error_ticks(self, axis: Any, data: list[float]) -> None:
        """Set log-like q-error ticks with 1 in the center."""

        if not data:
            axis.set_yticks([0.0])
            axis.set_yticklabels(["1"])
            return

        max_abs_value = max(abs(value) for value in data if math.isfinite(value))
        plot_limit = max(1, math.ceil(math.log10(max_abs_value)) + 1)
        axis.set_ylim(-plot_limit, plot_limit)

        tick_values: list[float] = [10**exponent for exponent in range(plot_limit + 1)]

        unique_values = sorted(
            {
                -value for value in tick_values if value > 1
            }
            | {1.0}
            | {value for value in tick_values if value > 1}
        )

        axis.set_yticks(
            [self._transform_q_error_for_plot(value) for value in unique_values]
        )
        axis.set_yticklabels(
            [
                self._format_scientific_tick(value)
                for value in unique_values
            ]
        )

    def _build_q_error_minor_tick_positions(self, data: list[float]) -> list[float]:
        """Build minor tick positions that follow log spacing on the transformed axis."""

        finite_values = [abs(value) for value in data if math.isfinite(value)]
        if not finite_values:
            return []

        max_abs_value = max(finite_values)
        plot_limit = max(1, math.ceil(math.log10(max_abs_value)) + 1)

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
