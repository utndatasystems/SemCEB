from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from semceb.data.downloader import DataDownloader
from semceb.data.loader import DataLoader
from semceb.utils.console import console

try:
    from semceb.llm_backends.lotus_backend import LotusBackend
except Exception:
    LotusBackend = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from semceb.llm_backends.lotus_backend import LotusBackend as LotusBackendType
else:
    LotusBackendType = Any


@dataclass(frozen=True)
class ShowcaseFilter:
    """A named semantic filter predicate for one aliased base relation."""

    name: str
    instruction: str


@dataclass(frozen=True)
class ShowcaseJoinStep:
    """One binary semantic join step in a showcase plan."""

    kind: str
    left_relation: str
    right_relation: str
    output_relation: str
    left_entity_alias: str
    right_entity_alias: str


@dataclass(frozen=True)
class ShowcasePlan:
    """A fully specified showcase execution plan."""

    plan_id: int
    filter_orders: dict[str, tuple[ShowcaseFilter, ...]]
    join_steps: tuple[ShowcaseJoinStep, ShowcaseJoinStep]


class CardEstShowcaseRunner:
    """Run a fixed showcase query under many filter and join permutations."""

    SHOWCASE_DATASETS = [
        "amazon-reviews/products_filtered_with_embeddings",
        "amazon-reviews/reviews_filtered_with_embeddings",
    ]

    FILTERS_BY_RELATION: dict[str, tuple[ShowcaseFilter, ...]] = {
        "p1": (
            ShowcaseFilter(
                name="refills_not_mentioned",
                instruction=(
                    "The features do not mention whether refills are included: "
                    "{p1.features_json}"
                ),
            ),
            ShowcaseFilter(
                name="size_mentioned",
                instruction="Product details mention its size: {p1.details_json}",
            ),
        ),
        "p2": (
            ShowcaseFilter(
                name="size_mentioned",
                instruction="Product details mention its size: {p2.details_json}",
            ),
            ShowcaseFilter(
                name="intended_use_mentioned",
                instruction=(
                    "The following product description does say what the product "
                    "is intended for: {p2.description_json}"
                ),
            ),
        ),
        "r": (
            ShowcaseFilter(
                name="english",
                instruction="Review is written in english: {r.review_text}",
            ),
            ShowcaseFilter(
                name="mentions_shipping",
                instruction="Review mentions shipping: {r.review_text}",
            ),
        ),
    }

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        join_scale_factor: int | None,
    ) -> None:
        """Store showcase configuration and placeholders for loaded state."""
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.join_scale_factor = join_scale_factor
        self.execution_join_scale_factor = self.join_scale_factor

        self.data_dfs: dict[str, pd.DataFrame] = {}
        self.backend: LotusBackendType | None = None
        self.base_relations: dict[str, pd.DataFrame] = {}
        self.plans: list[ShowcasePlan] = []
        self.plan_results: list[dict[str, Any]] = []
        self.result_dir = Path("results") / "raw" / "showcase"
        self.result_jsonl_path = self.result_dir / "showcase_results.jsonl"
        self.metadata_path = self.result_dir / "showcase_metadata.json"

    def run(self, enumerate_only: bool = False) -> None:
        """Prepare the showcase, generate plans, and optionally execute them."""
        self.plans = self._generate_plans()

        if enumerate_only:
            self._print_enumerated_plans()
            return

        self._ensure_data_available()
        self.data_dfs = self._load_showcase_datasets()
        self.backend = self._initialize_backend()
        self._configure_lotus_runtime()
        self.base_relations = self._prepare_base_relations()
        self._initialize_result_files()
        self._print_setup_summary()
        self.plan_results = self._execute_plans()
        self._finalize_result_files()

    def _ensure_data_available(self) -> None:
        """Download benchmark data when it is not available locally."""
        data_complete = DataDownloader().ensure_files_available()
        if not data_complete:
            console.print(
                "[bold red]Showcase aborted.[/bold red]\n"
                "[yellow]Required benchmark data is missing, and the download was skipped.[/yellow]"
            )
            raise SystemExit(1)

    def _load_showcase_datasets(self) -> dict[str, pd.DataFrame]:
        """Load showcase datasets using the benchmark's join sampling path."""
        with console.status(
            "Loading showcase datasets with "
            f"join_scale_factor={self.execution_join_scale_factor} ...",
            spinner="dots",
        ):
            return DataLoader().load(
                datasets=self.SHOWCASE_DATASETS,
                scale_factor=self.execution_join_scale_factor,
            )

    def _initialize_backend(self) -> LotusBackendType:
        """Initialize the LOTUS backend with the configured model and prompt."""
        backend_class = LotusBackend
        if backend_class is None:
            raise RuntimeError(
                "LOTUS backend could not be imported. Verify the showcase runtime "
                "dependencies are installed correctly."
            )

        return backend_class(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            scale_factor=self.execution_join_scale_factor,
        )

    def _configure_lotus_runtime(self) -> None:
        """Enable LOTUS caching so repeated operators are reused across plans."""
        if self.backend is None:
            raise RuntimeError("LOTUS backend must exist before runtime configuration.")

        import lotus.settings

        lotus.settings.configure(
            lm=self.backend.lm,
            enable_cache=True,
        )

    def _prepare_base_relations(self) -> dict[str, pd.DataFrame]:
        """Create aliased base relations for the fixed showcase query."""
        products_df = self.data_dfs["amazon-reviews/products_filtered_with_embeddings"]
        reviews_df = self.data_dfs["amazon-reviews/reviews_filtered_with_embeddings"]

        return {
            "p1": self._prefix_relation_columns(products_df, "p1"),
            "p2": self._prefix_relation_columns(products_df, "p2"),
            "r": self._prefix_relation_columns(reviews_df, "r"),
        }

    def _prefix_relation_columns(
        self,
        df: pd.DataFrame,
        alias: str,
    ) -> pd.DataFrame:
        """Return a copy whose column names are prefixed with the given alias."""
        return df.rename(columns={column: f"{alias}.{column}" for column in df.columns}).copy()

    def _generate_plans(self) -> list[ShowcasePlan]:
        """Enumerate all filter-order permutations combined with distinct join orders."""
        filter_orders_by_relation = {
            relation: tuple(permutations(filters))
            for relation, filters in self.FILTERS_BY_RELATION.items()
        }
        join_plans = self._build_join_plans()

        plans: list[ShowcasePlan] = []

        for plan_id, (p1_order, p2_order, r_order, join_steps) in enumerate(
            product(
                filter_orders_by_relation["p1"],
                filter_orders_by_relation["p2"],
                filter_orders_by_relation["r"],
                join_plans,
            ),
            start=1,
        ):
            plans.append(
                ShowcasePlan(
                    plan_id=plan_id,
                    filter_orders={
                        "p1": p1_order,
                        "p2": p2_order,
                        "r": r_order,
                    },
                    join_steps=join_steps,
                )
            )

        return plans

    def _build_join_plans(self) -> tuple[tuple[ShowcaseJoinStep, ShowcaseJoinStep], ...]:
        """Return the distinct binary join orders for the showcase query."""
        return (
            (
                ShowcaseJoinStep(
                    kind="product_product",
                    left_relation="p1",
                    right_relation="p2",
                    output_relation="p1_p2",
                    left_entity_alias="p1",
                    right_entity_alias="p2",
                ),
                ShowcaseJoinStep(
                    kind="product_review",
                    left_relation="p1_p2",
                    right_relation="r",
                    output_relation="result",
                    left_entity_alias="p1",
                    right_entity_alias="r",
                ),
            ),
            (
                ShowcaseJoinStep(
                    kind="product_product",
                    left_relation="p1",
                    right_relation="p2",
                    output_relation="p1_p2",
                    left_entity_alias="p1",
                    right_entity_alias="p2",
                ),
                ShowcaseJoinStep(
                    kind="product_review",
                    left_relation="p1_p2",
                    right_relation="r",
                    output_relation="result",
                    left_entity_alias="p2",
                    right_entity_alias="r",
                ),
            ),
            (
                ShowcaseJoinStep(
                    kind="product_review",
                    left_relation="p1",
                    right_relation="r",
                    output_relation="p1_r",
                    left_entity_alias="p1",
                    right_entity_alias="r",
                ),
                ShowcaseJoinStep(
                    kind="product_product",
                    left_relation="p1_r",
                    right_relation="p2",
                    output_relation="result",
                    left_entity_alias="p1",
                    right_entity_alias="p2",
                ),
            ),
            (
                ShowcaseJoinStep(
                    kind="product_review",
                    left_relation="p2",
                    right_relation="r",
                    output_relation="p2_r",
                    left_entity_alias="p2",
                    right_entity_alias="r",
                ),
                ShowcaseJoinStep(
                    kind="product_product",
                    left_relation="p2_r",
                    right_relation="p1",
                    output_relation="result",
                    left_entity_alias="p2",
                    right_entity_alias="p1",
                ),
            ),
        )

    def _execute_plans(self) -> list[dict[str, Any]]:
        """Execute every showcase plan and return the collected statistics."""
        results: list[dict[str, Any]] = []
        total_plans = len(self.plans)

        for plan in self.plans:
            self._print_plan_configuration(plan, total_plans)
            result = self._execute_plan(plan)
            results.append(result)
            self._append_plan_result(result)
            self._print_plan_result(result)

        return results

    def _execute_plan(self, plan: ShowcasePlan) -> dict[str, Any]:
        """Execute one plan and collect per-step filter/join cardinality stats."""
        if self.backend is None:
            raise RuntimeError("LOTUS backend must be initialized before plan execution.")

        if hasattr(self.backend.lm, "reset_stats"):
            self.backend.lm.reset_stats()

        plan_usage_before = self._capture_usage_snapshot()
        relations = {
            relation_name: relation_df.copy()
            for relation_name, relation_df in self.base_relations.items()
        }
        filter_stats: list[dict[str, Any]] = []
        join_stats: list[dict[str, Any]] = []

        for relation_name in ("p1", "p2", "r"):
            current_df = relations[relation_name]

            for filter_step in plan.filter_orders[relation_name]:
                input_rows = len(current_df)
                step_usage_before = self._capture_usage_snapshot()
                current_df = self._apply_filter(
                    df=current_df,
                    relation_name=relation_name,
                    filter_step=filter_step,
                )
                step_usage_after = self._capture_usage_snapshot()
                filter_stats.append(
                    {
                        "relation": relation_name,
                        "filter_name": filter_step.name,
                        "instruction": filter_step.instruction,
                        "input_rows": input_rows,
                        "output_rows": len(current_df),
                        "cost": self._usage_delta_dict(
                            before=step_usage_before,
                            after=step_usage_after,
                        ),
                    }
                )

            relations[relation_name] = current_df

        for join_index, join_step in enumerate(plan.join_steps, start=1):
            left_df = relations[join_step.left_relation]
            right_df = relations[join_step.right_relation]
            instruction = self._build_join_instruction(join_step)
            step_usage_before = self._capture_usage_snapshot()

            result_df = self._apply_join(
                left_df=left_df,
                right_df=right_df,
                join_step=join_step,
                join_index=join_index,
            )
            step_usage_after = self._capture_usage_snapshot()

            join_stats.append(
                {
                    "step": join_index,
                    "kind": join_step.kind,
                    "left_relation": join_step.left_relation,
                    "right_relation": join_step.right_relation,
                    "output_relation": join_step.output_relation,
                    "instruction": instruction,
                    "left_rows": len(left_df),
                    "right_rows": len(right_df),
                    "input_pairs": len(left_df) * len(right_df),
                    "output_rows": len(result_df),
                    "cost": self._usage_delta_dict(
                        before=step_usage_before,
                        after=step_usage_after,
                    ),
                }
            )

            relations[join_step.output_relation] = result_df

        final_df = relations["result"]
        plan_usage_after = self._capture_usage_snapshot()

        return {
            "plan_id": plan.plan_id,
            "filter_orders": {
                relation_name: [filter_step.name for filter_step in ordered_filters]
                for relation_name, ordered_filters in plan.filter_orders.items()
            },
            "join_plan": [
                {
                    "kind": join_step.kind,
                    "left_relation": join_step.left_relation,
                    "right_relation": join_step.right_relation,
                    "output_relation": join_step.output_relation,
                    "left_entity_alias": join_step.left_entity_alias,
                    "right_entity_alias": join_step.right_entity_alias,
                }
                for join_step in plan.join_steps
            ],
            "filter_stats": filter_stats,
            "join_stats": join_stats,
            "final_rows": len(final_df),
            "plan_cost": self._usage_delta_dict(
                before=plan_usage_before,
                after=plan_usage_after,
            ),
            "lm_stats": self._capture_lm_stats(),
        }

    def _apply_filter(
        self,
        df: pd.DataFrame,
        relation_name: str,
        filter_step: ShowcaseFilter,
    ) -> pd.DataFrame:
        """Run one semantic filter unless the input is already empty."""
        if df.empty:
            return df.copy()

        prepared_df, restore_columns = self._prepare_filter_dataframe(
            df=df,
            relation_name=relation_name,
        )

        filtered_df = prepared_df.sem_filter(
            user_instruction=self._format_filter_instruction(
                relation_name=relation_name,
                instruction=filter_step.instruction,
            ),
            progress_bar_desc=(
                f"Plan filter {relation_name}.{filter_step.name}"
            ),
        )

        return filtered_df.rename(columns=restore_columns)

    def _prepare_filter_dataframe(
        self,
        df: pd.DataFrame,
        relation_name: str,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Create a single-relation filter view with unqualified LOTUS column names."""
        prefix = f"{relation_name}."
        renamed_columns = {
            column_name: column_name.removeprefix(prefix)
            for column_name in df.columns
        }
        restore_columns = {
            stripped_name: original_name
            for original_name, stripped_name in renamed_columns.items()
        }

        return df.rename(columns=renamed_columns), restore_columns

    def _format_filter_instruction(
        self,
        relation_name: str,
        instruction: str,
    ) -> str:
        """Convert single-relation filter placeholders into plain LOTUS column refs."""
        return instruction.replace(f"{{{relation_name}.", "{")

    def _apply_join(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        join_step: ShowcaseJoinStep,
        join_index: int,
    ) -> pd.DataFrame:
        """Run one semantic join unless one side is already empty."""
        if left_df.empty or right_df.empty:
            return pd.DataFrame(columns=[*left_df.columns, *right_df.columns])

        return left_df.sem_join(
            right_df,
            self._build_join_instruction(join_step),
            progress_bar_desc=f"Plan join {join_index}",
        )

    def _build_join_instruction(self, join_step: ShowcaseJoinStep) -> str:
        """Return the LOTUS join predicate for one join step."""
        if join_step.kind == "product_product":
            return (
                "The descriptions suggest the same use, but the listed dimensions "
                "are very different. First description: "
                f"{{{join_step.left_entity_alias}.description_json:left}}. "
                "First details: "
                f"{{{join_step.left_entity_alias}.details_json:left}}. "
                "Second description: "
                f"{{{join_step.right_entity_alias}.description_json:right}}. "
                "Second details: "
                f"{{{join_step.right_entity_alias}.details_json:right}}."
            )

        if join_step.kind == "product_review":
            return (
                "The review praises a specific feature of the product. Features: "
                f"{{{join_step.left_entity_alias}.features_json:left}}. "
                "Description: "
                f"{{{join_step.left_entity_alias}.description_json:left}}. "
                "Details: "
                f"{{{join_step.left_entity_alias}.details_json:left}}. "
                "Review: "
                f"{{{join_step.right_entity_alias}.review_text:right}}."
            )

        raise ValueError(f"Unsupported join kind: {join_step.kind}")

    def _capture_lm_stats(self) -> dict[str, Any]:
        """Snapshot the current LOTUS model usage and cache counters."""
        if self.backend is None:
            return {}

        stats = self.backend.lm.stats

        return {
            "virtual_usage": {
                "prompt_tokens": stats.virtual_usage.prompt_tokens,
                "completion_tokens": stats.virtual_usage.completion_tokens,
                "total_tokens": stats.virtual_usage.total_tokens,
                "total_cost": stats.virtual_usage.total_cost,
            },
            "physical_usage": {
                "prompt_tokens": stats.physical_usage.prompt_tokens,
                "completion_tokens": stats.physical_usage.completion_tokens,
                "total_tokens": stats.physical_usage.total_tokens,
                "total_cost": stats.physical_usage.total_cost,
            },
            "cache_hits": stats.cache_hits,
            "operator_cache_hits": stats.operator_cache_hits,
        }

    def _capture_usage_snapshot(self) -> dict[str, Any]:
        """Capture the current LM usage counters for later delta computation."""
        if self.backend is None:
            return self._empty_usage_snapshot()

        stats = self.backend.lm.stats

        return {
            "virtual_usage": self._usage_to_dict(stats.virtual_usage),
            "physical_usage": self._usage_to_dict(stats.physical_usage),
            "cache_hits": int(stats.cache_hits),
            "operator_cache_hits": int(stats.operator_cache_hits),
        }

    def _empty_usage_snapshot(self) -> dict[str, Any]:
        """Return a zeroed usage snapshot."""
        return {
            "virtual_usage": self._usage_zero_dict(),
            "physical_usage": self._usage_zero_dict(),
            "cache_hits": 0,
            "operator_cache_hits": 0,
        }

    def _usage_zero_dict(self) -> dict[str, Any]:
        """Return a zeroed usage dictionary."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

    def _usage_to_dict(self, usage: Any) -> dict[str, Any]:
        """Convert a LOTUS usage object into a JSON-serializable dictionary."""
        return {
            "prompt_tokens": int(usage.prompt_tokens),
            "completion_tokens": int(usage.completion_tokens),
            "total_tokens": int(usage.total_tokens),
            "total_cost": float(usage.total_cost),
        }

    def _usage_delta_dict(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute the usage delta between two LM usage snapshots.

        Virtual usage is the cache-independent cost signal and should be used
        for analysis. Physical usage is persisted alongside it to expose the
        actual cached runtime spend.
        """
        return {
            "virtual_usage": self._subtract_usage_dict(
                before["virtual_usage"],
                after["virtual_usage"],
            ),
            "physical_usage": self._subtract_usage_dict(
                before["physical_usage"],
                after["physical_usage"],
            ),
            "cache_hits": int(after["cache_hits"] - before["cache_hits"]),
            "operator_cache_hits": int(
                after["operator_cache_hits"] - before["operator_cache_hits"]
            ),
        }

    def _subtract_usage_dict(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """Subtract one usage dictionary from another."""
        return {
            "prompt_tokens": int(after["prompt_tokens"] - before["prompt_tokens"]),
            "completion_tokens": int(
                after["completion_tokens"] - before["completion_tokens"]
            ),
            "total_tokens": int(after["total_tokens"] - before["total_tokens"]),
            "total_cost": float(after["total_cost"] - before["total_cost"]),
        }

    def _initialize_result_files(self) -> None:
        """Reset showcase output files and write run metadata."""
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.result_jsonl_path.write_text("", encoding="utf-8")
        self._write_metadata(
            {
                "model_name": self.model_name,
                "system_prompt": self.system_prompt,
                "configured_join_scale_factor": self.join_scale_factor,
                "effective_join_scale_factor": self.execution_join_scale_factor,
                "total_plans": len(self.plans),
                "enumerate_only": False,
                "result_jsonl_path": str(self.result_jsonl_path),
                "executed_plans": 0,
            }
        )

    def _append_plan_result(self, result: dict[str, Any]) -> None:
        """Append one showcase plan result as JSONL."""
        with self.result_jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")

    def _finalize_result_files(self) -> None:
        """Update metadata after showcase execution completes."""
        self._write_metadata(
            {
                "model_name": self.model_name,
                "system_prompt": self.system_prompt,
                "configured_join_scale_factor": self.join_scale_factor,
                "effective_join_scale_factor": self.execution_join_scale_factor,
                "total_plans": len(self.plans),
                "enumerate_only": False,
                "result_jsonl_path": str(self.result_jsonl_path),
                "executed_plans": len(self.plan_results),
            }
        )

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        """Persist showcase metadata as JSON."""
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def _print_setup_summary(self) -> None:
        """Report the prepared showcase setup and planned permutation count."""
        console.print("[green]✓[/green] Showcase setup is ready.")
        console.print(f"  Model: [cyan]{self.model_name}[/cyan]")
        console.print(
            "  Join scale factor: " f"[bold]{self.execution_join_scale_factor}[/bold]"
        )

        for dataset_name, data_df in self.data_dfs.items():
            console.print(
                f"  Dataset [cyan]{dataset_name}[/cyan]: [bold]{len(data_df):,}[/bold] rows"
            )

        console.print(f"  Plans to execute: [bold]{len(self.plans):,}[/bold]")
        console.print(f"  Results file: [bold]{self.result_jsonl_path}[/bold]")

    def _print_enumerated_plans(self) -> None:
        """Print all enumerated plans and the final plan count without execution."""
        total_plans = len(self.plans)

        for plan in self.plans:
            self._print_plan_configuration(plan, total_plans)

        console.print()
        console.print(
            "[green]✓[/green] Enumerated "
            f"[bold]{total_plans:,}[/bold] showcase plans."
        )

    def _print_plan_configuration(self, plan: ShowcasePlan, total_plans: int) -> None:
        """Print one plan configuration before it is executed."""
        console.print()
        console.rule(f"[bold cyan]Showcase plan {plan.plan_id}/{total_plans}[/bold cyan]")
        console.print(
            "Filter order p1: "
            + " -> ".join(filter_step.name for filter_step in plan.filter_orders["p1"])
        )
        console.print(
            "Filter order p2: "
            + " -> ".join(filter_step.name for filter_step in plan.filter_orders["p2"])
        )
        console.print(
            "Filter order r: "
            + " -> ".join(filter_step.name for filter_step in plan.filter_orders["r"])
        )

        for join_index, join_step in enumerate(plan.join_steps, start=1):
            console.print(
                f"Join step {join_index}: "
                f"{join_step.left_relation} x {join_step.right_relation} "
                f"-> {join_step.output_relation} ({join_step.kind}; "
                f"left entity={join_step.left_entity_alias}, "
                f"right entity={join_step.right_entity_alias})"
            )

    def _print_plan_result(self, result: dict[str, Any]) -> None:
        """Print the main recorded cardinalities for one executed plan."""
        console.print(f"Final rows: [bold]{result['final_rows']:,}[/bold]")

        for filter_stat in result["filter_stats"]:
            console.print(
                "  Filter "
                f"{filter_stat['relation']}.{filter_stat['filter_name']}: "
                f"{filter_stat['input_rows']:,} -> {filter_stat['output_rows']:,}"
            )

        for join_stat in result["join_stats"]:
            console.print(
                f"  Join {join_stat['step']} "
                f"{join_stat['left_relation']} x {join_stat['right_relation']}: "
                f"{join_stat['input_pairs']:,} pairs -> {join_stat['output_rows']:,}"
            )

