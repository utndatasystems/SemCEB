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

    def _plot_showcase_cost_ranks(self) -> None:
        """Plot showcase plan costs ordered by rank from cached JSONL results."""

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
            scale=1.0,
            double_column=False,
        )

        fig, axis = plt.subplots()
        ranked_costs = sorted(costs)
        ranks = list(range(1, len(ranked_costs) + 1))

        axis.plot(
            ranks,
            ranked_costs,
            color="#D67D00",
            linewidth=1.5,
            zorder=2,
        )

        min_cost = min(costs)
        max_cost = max(costs)
        bottom_limit = min_cost / 1.1 if min_cost > 0 else 0.001
        top_limit = max_cost * 1.1 if max_cost > 0 else 0.01
        rank_count = len(ranked_costs)
        x_left, x_right = (1, rank_count) if rank_count > 1 else (0.5, 1.5)

        axis.set_xlim(left=x_left, right=x_right)
        axis.set_ylim(bottom=bottom_limit, top=top_limit)
        axis.set_yscale("log")
        axis.set_xlabel("Plan Rank (1 = Cheapest)")
        axis.set_ylabel("Cost (USD)")
        axis.set_title("Plan Cost by Rank")
        axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_locator(
            mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        )
        axis.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:g}")
        )
        axis.yaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0))
        )
        axis.yaxis.set_minor_formatter(mticker.NullFormatter())
        axis.xaxis.set_minor_locator(mticker.AutoMinorLocator())
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

        pdf_path = self.plot_dir / "showcase_cost_rank.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        console.print(
            "[green]✓[/green] Saved showcase cost rank plot to "
            f"[bold]{pdf_path}[/bold]"
        )

        plt.close(fig)

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
