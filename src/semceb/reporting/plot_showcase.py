from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from semceb.reporting.plot_params import apply_plot_params
from semceb.utils.console import console


class ShowcasePlotMixin:
    """Helpers for plotting showcase execution results."""

    def _plot_showcase_cost_histogram(self) -> None:
        """Plot a histogram of showcase plan costs from cached JSONL results."""

        showcase_results_path = Path(self.showcase_results_path)

        if not showcase_results_path.exists():
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"Showcase results file not found: {showcase_results_path}"
            )
            return

        showcase_plan_results = self._load_showcase_plan_results(showcase_results_path)
        costs = [entry["cost"] for entry in showcase_plan_results]

        if not costs:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"No showcase plan costs found in {showcase_results_path}"
            )
            return

        self._print_cheapest_showcase_plans(showcase_plan_results)

        self.plot_dir.mkdir(parents=True, exist_ok=True)

        apply_plot_params(
            fig_height=2.4,
            scale=0.8,
            double_column=False,
        )

        fig, axis = plt.subplots()
        bin_count = self._determine_showcase_histogram_bin_count(costs)

        axis.hist(
            costs,
            bins=bin_count,
            color="#D67D00",
            edgecolor="#8A4F00",
            linewidth=0.8,
            alpha=0.9,
        )

        max_cost = max(costs)
        right_limit = max_cost * 1.05 if max_cost > 0 else 0.01

        axis.set_xlim(left=0, right=right_limit)
        axis.set_xlabel("Total Cost (USD)")
        axis.set_ylabel("Number of Plans")
        axis.set_title("Showcase Plan Cost Distribution")
        axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        axis.xaxis.set_minor_locator(mticker.AutoMinorLocator())
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
        axis.grid(axis="y", alpha=0.35)

        sns.despine(
            ax=axis,
            top=True,
            right=True,
            left=False,
            bottom=False,
        )

        fig.tight_layout()

        pdf_path = self.plot_dir / "showcase_cost_histogram.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
        console.print(
            "[green]✓[/green] Saved showcase cost histogram to "
            f"[bold]{pdf_path}[/bold]"
        )

        plt.close(fig)

    def _determine_showcase_histogram_bin_count(self, costs: list[float]) -> int:
        """Choose a fine-grained histogram bin count for showcase plan costs."""

        if not costs:
            return 1

        unique_cost_count = len({round(cost, 12) for cost in costs})

        if unique_cost_count <= 1:
            return 1

        return min(100, max(20, unique_cost_count))

    def _load_showcase_plan_results(
        self,
        showcase_results_path: Path,
    ) -> list[dict[str, Any]]:
        """Load showcase plan rows together with their extracted plan cost."""

        showcase_plan_results: list[dict[str, Any]] = []
        skipped_rows = 0

        with showcase_results_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    result = json.loads(stripped_line)
                except json.JSONDecodeError:
                    skipped_rows += 1
                    console.print(
                        "[bold yellow]Warning:[/bold yellow] "
                        f"Skipping malformed showcase JSON on line {line_number} "
                        f"in {showcase_results_path}"
                    )
                    continue

                cost = self._extract_showcase_plan_cost(result)

                if cost is None:
                    skipped_rows += 1
                    continue

                showcase_plan_results.append(
                    {
                        "cost": cost,
                        "result": result,
                    }
                )

        if skipped_rows > 0:
            console.print(
                "[bold yellow]Warning:[/bold yellow] "
                f"Skipped {skipped_rows} showcase rows without a valid plan cost."
            )

        return showcase_plan_results

    def _load_showcase_plan_costs(self, showcase_results_path: Path) -> list[float]:
        """Extract all showcase plan costs from the JSONL results file."""
        return [
            entry["cost"]
            for entry in self._load_showcase_plan_results(showcase_results_path)
        ]

    def _print_cheapest_showcase_plans(
        self,
        showcase_plan_results: list[dict[str, Any]],
    ) -> None:
        """Print the cheapest showcase plan or all tied cheapest plans."""

        if not showcase_plan_results:
            return

        cheapest_cost = min(entry["cost"] for entry in showcase_plan_results)
        cheapest_entries = [
            entry
            for entry in showcase_plan_results
            if entry["cost"] == cheapest_cost
        ]

        console.print(
            "[cyan]Cheapest showcase plan cost[/cyan]: "
            f"[bold]{cheapest_cost:.6f}[/bold] USD "
            f"across [bold]{len(cheapest_entries)}[/bold] plan(s)"
        )

        for entry in cheapest_entries:
            console.print(json.dumps(entry["result"], indent=2))

    def _extract_showcase_plan_cost(self, result: dict[str, Any]) -> float | None:
        """Return one showcase plan cost if present and numeric."""

        try:
            cost_value = result["plan_cost"]["virtual_usage"]["total_cost"]
        except (KeyError, TypeError):
            return None

        try:
            return float(cost_value)
        except (TypeError, ValueError):
            return None
