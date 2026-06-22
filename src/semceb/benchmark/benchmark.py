import sys
import importlib
from pathlib import Path
import json
import math
from enum import Enum
import time
import pandas as pd
from typing import Any
from rich.prompt import Confirm
from semceb.utils.progress import create_benchmark_progress, suspend_progress
from semceb.utils.console import console
from semceb.data.downloader import DataDownloader
from semceb.data.loader import DataLoader
from semceb.algorithms.interface import AlgorithmInterface
from semceb.llm_backends.lotus_backend import LotusBackend
from semceb.queries.query_specification import QuerySpecification
from semceb.algorithms.cardinality_estimate import CardinalityEstimateKind


class BenchmarkRunner:
    """Run benchmark queries and collect algorithm evaluation results."""

    def __init__(
        self,
        algorithms: list[dict[str, Any]],
        default_ground_truth_model_name: str,
        default_ground_truth_system_prompt: str,
        scale_factor: int,
        join_scale_factor: int,
        categories: list[str] | None,
        types: list[str] | None,
    ):
        """Initialize benchmark runner settings, query selection, and local storage paths."""
        self.algorithms = algorithms
        self.default_ground_truth_model_name = default_ground_truth_model_name
        self.default_ground_truth_system_prompt = default_ground_truth_system_prompt
        self.scale_factor = scale_factor
        self.query_categories = set(category.lower() for category in (categories or []))
        self.query_types = set(type.lower() for type in (types or []))
        self.join_scale_factor = join_scale_factor

        self.result_filepath = Path("results") / "raw" / "result.jsonl"
        self.query_filepath = Path("benchmark_queries") / "queries.jsonl"

        self._dataset_cache: dict[
            tuple[frozenset[str], int | None],
            dict[str, pd.DataFrame],
        ] = {}

        self.queries_specs = self._load_queries_specs(self.query_filepath)

        self._handle_cloud_data()

    def _load_queries_specs(self, file_path: str) -> list[QuerySpecification]:
        """Load query specifications from a JSONL file and apply configured filters."""

        queries_specs = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                query_spec = QuerySpecification.from_dict(json.loads(line))
                if self._query_matches_selection(query_spec):
                    queries_specs.append(query_spec)

        return queries_specs

    def _query_matches_selection(self, query_spec: QuerySpecification) -> bool:
        """Return whether a query passes configured category and type filters."""

        category_matches = not self.query_categories or any(
            category.lower() in self.query_categories
            for category in query_spec.category
        )
        type_matches = (
            not self.query_types or query_spec.type.lower() in self.query_types
        )

        return category_matches and type_matches

    def _handle_cloud_data(self) -> None:
        """Download data if it is not available locally."""

        downloader = DataDownloader()

        data_complete = downloader.ensure_files_available()
        if not data_complete:
            console.print(
                "[bold red]Benchmark aborted.[/bold red]\n"
                "[yellow]Required benchmark data is missing, and the download was skipped.[/yellow]"
            )
            raise SystemExit(1)

    def _load_algorithm_from_file(
        self, algorithm_config: dict[str, Any]
    ) -> AlgorithmInterface:
        """Load algorithm class from runner.algorithms."""

        project_root = Path(__file__).resolve().parent.parent

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        filename = algorithm_config["filename"]
        module_stem = Path(filename).stem
        module_name = f"semceb.algorithms.{module_stem}"

        module = importlib.import_module(module_name)

        class_name = algorithm_config["class"]

        if not hasattr(module, class_name):
            raise AttributeError(
                f"Algorithm class '{class_name}' not found in {module_name}"
            )

        algorithm_class = getattr(module, class_name)

        if not issubclass(algorithm_class, AlgorithmInterface):
            raise TypeError(
                f"Algorithm class '{class_name}' must match "
                f"{AlgorithmInterface.__module__}.{AlgorithmInterface.__name__}."
            )

        if algorithm_class.__abstractmethods__:
            raise TypeError(
                f"Algorithm class '{class_name}' is still abstract. "
                f"Missing implementations: "
                f"{', '.join(sorted(algorithm_class.__abstractmethods__))}"
            )

        return algorithm_class(
            algorithm_config["name"],
            algorithm_config["version"],
        )

    def run(self) -> None:
        """Measure, run, and store results of benchmark queries."""

        self._clear_result_file()
        query_groups = self._build_query_groups()

        self._confirm_join_benchmark_run_if_required(query_groups)

        total_runs = sum(
            len(self.algorithms) * len(group["query_specs"]) for group in query_groups
        )

        with create_benchmark_progress() as progress:
            task = progress.add_task(
                "Running benchmark...",
                total=total_runs,
            )

            for query_group in query_groups:
                self._run_query_group(query_group, progress, task)

        console.print(
            f"[green]✓[/green] Results written to [bold]{self.result_filepath}[/bold]"
        )

    def _clear_result_file(self) -> None:
        """Reset the result file before writing new benchmark output."""

        with open(self.result_filepath, "w"):
            pass

    def _build_query_groups(self) -> list[dict[str, Any]]:
        """Group query specifications by query type and scale factor."""

        filter_query_specs, join_query_specs = self._split_query_specs_by_type()

        return [
            {
                "name": "filter",
                "query_specs": filter_query_specs,
                "scale_factor": self.scale_factor,
            },
            {
                "name": "join",
                "query_specs": join_query_specs,
                "scale_factor": self.join_scale_factor,
            },
        ]

    def _confirm_join_benchmark_run_if_required(
        self,
        query_groups: list[dict[str, Any]],
    ) -> None:
        """Ask for confirmation before any benchmark queries run if join queries are selected."""

        join_group = next(
            (
                query_group
                for query_group in query_groups
                if query_group["name"] == "join" and query_group["query_specs"]
            ),
            None,
        )

        if join_group is None:
            return

        data_dfs = self._load_required_datasets(
            query_specs=join_group["query_specs"],
            scale_factor=join_group["scale_factor"],
        )

        self._confirm_join_benchmark_run(data_dfs)

    def _run_query_group(
        self,
        query_group: dict[str, Any],
        progress,
        task,
    ) -> None:
        """Execute benchmark runs for one query group."""

        query_specs = query_group["query_specs"]
        if not query_specs:
            return

        with suspend_progress(progress):
            data_dfs = self._load_required_datasets(
                query_specs=query_specs,
                scale_factor=query_group["scale_factor"],
            )

        for algorithm_config in self.algorithms:
            self._run_algorithm_for_query_group(
                algorithm_config=algorithm_config,
                query_specs=query_specs,
                data_dfs=data_dfs,
                progress=progress,
                task=task,
                group_name=query_group["name"],
                scale_factor=query_group["scale_factor"],
            )

    def _get_ground_truth_params(
        self, algorithm_config: dict[str, Any]
    ) -> tuple[str, str]:
        """Resolve the ground-truth model name and system prompt for an algorithm."""

        ground_truth = algorithm_config.get("ground_truth", {})

        return (
            ground_truth.get("model_name") or self.default_ground_truth_model_name,
            ground_truth.get("system_prompt")
            or self.default_ground_truth_system_prompt,
        )

    def _run_algorithm_for_query_group(
        self,
        algorithm_config: dict[str, Any],
        query_specs: list[QuerySpecification],
        data_dfs: dict[str, pd.DataFrame],
        progress,
        task,
        group_name: str,
        scale_factor: int | None,
    ) -> None:
        """Run one algorithm on a group of queries."""

        algorithm = self._load_algorithm_from_file(algorithm_config)
        ground_truth_model_name, ground_truth_system_prompt = (
            self._get_ground_truth_params(
                algorithm_config,
            )
        )
        algorithm_kwargs = algorithm_config.get("algorithm_kwargs", {})
        algorithm.preparation(data_dfs, algorithm_kwargs)

        for query_spec in query_specs:
            self._run_single_query(
                algorithm=algorithm,
                algorithm_config=algorithm_config,
                query_spec=query_spec,
                data_dfs=data_dfs,
                progress=progress,
                task=task,
                group_name=group_name,
                scale_factor=scale_factor,
                ground_truth_model_name=ground_truth_model_name,
                ground_truth_system_prompt=ground_truth_system_prompt,
            )

        progress.console.print(
            f"[green]✓[/green] Finished algorithm "
            f"[bold cyan]{algorithm.name}[/bold cyan] "
            f"on [bold]{len(query_specs)}[/bold] {group_name} queries."
        )

    def _run_single_query(
        self,
        algorithm: AlgorithmInterface,
        algorithm_config: dict[str, Any],
        query_spec: QuerySpecification,
        data_dfs: dict[str, pd.DataFrame],
        progress,
        task,
        group_name: str,
        scale_factor: int | None,
        ground_truth_model_name: str,
        ground_truth_system_prompt: str,
    ) -> None:
        """Run one query for a single algorithm and write the result."""

        progress.update(
            task,
            description=(
                f"Group: [magenta]{group_name}[/magenta] "
                f"| Algorithm: [cyan]{algorithm_config['name']}[/cyan] "
                f"| Query ID: [yellow]{query_spec.id}[/yellow]"
            ),
        )

        with suspend_progress(progress):
            cardinality_ground_truth = self._get_cardinality_ground_truth(
                ground_truth_model_name,
                ground_truth_system_prompt,
                scale_factor,
                query_spec,
                data_dfs,
            )

        algorithm.reset_cost_stats()

        with suspend_progress(progress):
            start = time.perf_counter()
            cardinality_estimation = algorithm.run(query_spec)
            time_ms = (time.perf_counter() - start) * 1000

        if cardinality_estimation.kind == CardinalityEstimateKind.INT:
            q_error = self._calculate_q_error(
                cardinality_estimation.value,
                cardinality_ground_truth,
            )

            selectivity_estimation = self._calculate_selectivity(
                cardinality_estimation.value,
                query_spec,
                data_dfs,
            )

            self._save_result(
                query_spec=query_spec,
                algorithm_name=algorithm_config["name"],
                algorithm_version=algorithm_config["version"],
                algorithm_memory_consumption=algorithm.get_memory_consumption(),
                algorithm_cost_stats=algorithm.get_cost_stats(),
                cardinality_ground_truth=cardinality_ground_truth,
                cardinality_estimation=cardinality_estimation.value,
                selectivity_estimation=selectivity_estimation,
                q_error=q_error,
                time_ms=time_ms,
            )

        elif cardinality_estimation.kind == CardinalityEstimateKind.UNSUPPORTED:
            console.print(
                "[yellow]Algorithm returned unsupported estimate. Skipping q-error and selectivity calculations.[/yellow]"
            )

        progress.advance(task)

    def _split_query_specs_by_type(self):
        """Split loaded queries into filter and join groups based on dataset count."""
        filter_query_specs = []
        join_query_specs = []

        for query_spec in self.queries_specs:
            dataset_count = len(query_spec.datasets)

            if dataset_count == 1:
                filter_query_specs.append(query_spec)
            elif dataset_count == 2:
                join_query_specs.append(query_spec)
            else:
                raise NotImplementedError(
                    f"Invalid dataset amount for query {query_spec.id}: "
                    f"expected 1 or 2 datasets, got {dataset_count}"
                )

        return filter_query_specs, join_query_specs

    def _load_required_datasets(self, query_specs, scale_factor):
        """Load all datasets required by a set of query specifications."""
        datasets = {
            dataset.table_ref
            for query_spec in query_specs
            for dataset in query_spec.datasets
        }

        cache_key = (frozenset(datasets), scale_factor)

        if cache_key in self._dataset_cache:
            console.print("[green]✓[/green] Required datasets already loaded.")
            return self._dataset_cache[cache_key]

        with console.status(
            f"Loading required datasets with scale_factor={scale_factor} ...",
            spinner="dots",
        ):
            data_dfs = DataLoader().load(
                datasets=datasets,
                scale_factor=scale_factor,
            )

        self._dataset_cache[cache_key] = data_dfs

        console.print("[green]✓[/green] Required datasets loaded.")

        return data_dfs

    def _confirm_join_benchmark_run(self, data_dfs: dict[str, pd.DataFrame]) -> None:
        """Ask the user to confirm running join benchmarks after showing input sizes."""

        row_counts = {
            table_ref: len(data_df) for table_ref, data_df in data_dfs.items()
        }

        largest_table_ref, largest_row_count = max(
            row_counts.items(),
            key=lambda item: item[1],
        )

        biggest_join_combination = largest_row_count * largest_row_count
        biggest_join_pair = (largest_table_ref, largest_table_ref)

        console.print()
        console.print("[bold yellow]Join benchmark input size[/bold yellow]")

        for table_ref, row_count in row_counts.items():
            console.print(
                f"  [cyan]{table_ref}[/cyan]: [bold]{row_count:,}[/bold] rows"
            )

        console.print(
            "  [magenta]Biggest possible pairwise join combination[/magenta]: "
            f"[cyan]{biggest_join_pair[0]}[/cyan] × [cyan]{biggest_join_pair[1]}[/cyan] = "
            f"[bold]{biggest_join_combination:,}[/bold]"
        )
        console.print()
        console.print(
            "  [yellow]Warning[/yellow]: uncached ground-truth cardinality or pairwise "
            "join algorithms may trigger many LLM calls."
        )
        console.print(
            f"  Worst case for this input size: [bold]{biggest_join_combination:,}[/bold] "
            "LLM calls."
        )
        console.print()

        should_continue = Confirm.ask(
            "Continue with join benchmark?",
            default=False,
        )

        if not should_continue:
            console.print("[yellow]Benchmark aborted by user.[/yellow]")
            raise SystemExit(0)

    def _get_cardinality_ground_truth(
        self,
        model_name: str,
        system_prompt: str,
        scale_factor: int | None,
        query_spec: QuerySpecification,
        data_dfs: dict[str, pd.DataFrame],
    ) -> int:
        """Obtain model-based selectivity ground truth."""
        backend = LotusBackend(
            model_name=model_name,
            system_prompt=system_prompt,
            scale_factor=scale_factor,
        )

        if len(query_spec.datasets) == 1:
            # Filtering
            dataset_spec = query_spec.datasets[0]
            data = data_dfs[dataset_spec.table_ref]
            cardinality_ground_truth = backend.filtering_query(query_spec, data)
        elif len(query_spec.datasets) > 1:
            # Joining
            dataset_spec_left, dataset_spec_right = query_spec.datasets
            data_left = data_dfs[dataset_spec_left.table_ref]
            data_right = data_dfs[dataset_spec_right.table_ref]
            cardinality_ground_truth = backend.joining_query(
                query_spec,
                data_left,
                data_right,
                dataset_spec_left.alias,
                dataset_spec_right.alias,
            )

        return cardinality_ground_truth

    def _calculate_q_error(
        self, cardinality_estimation: int, cardinality_ground_truth: int
    ) -> float:
        """Calcualte q error. Higher is worse."""
        safe_cardinality_estimation = max(1, cardinality_estimation)
        safe_cardinality_ground_truth = max(1, cardinality_ground_truth)

        if cardinality_estimation == cardinality_ground_truth:
            return 1.0
        if cardinality_estimation < cardinality_ground_truth:
            # Return a negative q-error for underestimation to distinguish it from overestimation in the results.
            return -(safe_cardinality_ground_truth / safe_cardinality_estimation)
        return safe_cardinality_estimation / safe_cardinality_ground_truth

    def _calculate_selectivity(
        self,
        cardinality_estimation: int,
        query_spec: QuerySpecification,
        data_dfs: dict[str, pd.DataFrame],
    ) -> float:
        """Calculate selectivity based on cardinality estimation and number of input rows."""
        if len(query_spec.datasets) == 1:  # Filter query
            table_ref = query_spec.datasets[0].table_ref
            input_rows = data_dfs[table_ref].shape[0]

        elif len(query_spec.datasets) > 1:  # Join query
            input_rows = 1
            for dataset in query_spec.datasets:
                table_ref = dataset.table_ref
                input_rows *= data_dfs[table_ref].shape[0]
        else:
            raise ValueError("Used dataset of query can not be empty!")

        selectivity = cardinality_estimation / input_rows
        return selectivity

    def _save_result(
        self,
        query_spec: QuerySpecification,
        algorithm_name: str,
        algorithm_version: str,
        algorithm_memory_consumption: int,
        algorithm_cost_stats: dict,
        cardinality_ground_truth: int,
        cardinality_estimation: int,
        selectivity_estimation: float,
        q_error: float,
        time_ms: float,
    ) -> None:
        """Save query result as JSONL."""

        algorithm_data = {
            "name": algorithm_name,
            "version": algorithm_version,
            "memory_consumption": algorithm_memory_consumption,
            "cost_stats": algorithm_cost_stats,
            "cardinality_ground_truth": cardinality_ground_truth,
            "cardinality_estimation": cardinality_estimation,
            "selectivity_estimation": selectivity_estimation,
            "q_error": q_error,
            "time_ms": time_ms,
        }
        result = {"query": query_spec.to_dict(), "algorithm": algorithm_data}

        with open(self.result_filepath, "a", encoding="utf-8") as file:
            file.write(json.dumps(result, default=self.json_default) + "\n")

    def json_default(self, obj):
        """JSON serializer helper for objects that provide to_dict or Enum values."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        if isinstance(obj, Enum):
            return obj.value

        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
