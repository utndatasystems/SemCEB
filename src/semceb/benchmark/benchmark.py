import sys
import importlib
from pathlib import Path
import json
from enum import Enum
import time
import pandas as pd
from typing import Any
from src.semceb.utils.progress import create_benchmark_progress, suspend_progress
from src.semceb.utils.console import console
from src.semceb.data.downloader import DataDownloader
from src.semceb.data.loader import DataLoader
from src.semceb.algorithms.interface import AlgorithmInterface
from src.semceb.llm_backends.lotus_backend import LotusBackend
from src.semceb.queries.query_specification import QuerySpecification

class BenchmarkRunner:
    """Runs benchmark queries."""

    def __init__(
        self,
        algorithms: list[dict[str, Any]],
        default_ground_truth_model_name: str,
        default_ground_truth_system_prompt: str,
        scale_factor: int,
        join_scale_factor: int,
        categories: list[str],
    ):
        self.algorithms = algorithms
        self.default_ground_truth_model_name = default_ground_truth_model_name
        self.default_ground_truth_system_prompt = default_ground_truth_system_prompt
        self.scale_factor = scale_factor
        self.join_scale_factor = join_scale_factor
        self.categories = categories

        self.result_filepath = Path("results") / "raw" / "result.jsonl"
        self.query_filepath = Path("benchmark_queries") / "queries.jsonl"

        self.queries_specs = self._load_queries_specs(self.query_filepath)

        self._handle_cloud_data()

    def _load_queries_specs(self, file_path: str) -> list[QuerySpecification]:
        """Load queries specifications from a JSONL file."""

        queries_specs = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                query_spec = QuerySpecification.from_dict(json.loads(line))
                if query_spec.category in self.categories:
                    queries_specs.append(query_spec)

        return queries_specs

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
        module_name = f"src.semceb.algorithms.{module_stem}"

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

        # Clear result file
        with open(self.result_filepath, "w"):
            pass

        filter_query_specs, join_query_specs = self._split_query_specs_by_type()

        query_groups = [
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

        total_runs = sum(
            len(self.algorithms) * len(group["query_specs"])
            for group in query_groups
        )

        with create_benchmark_progress() as progress:
            task = progress.add_task(
                "Running benchmark...",
                total=total_runs,
            )

            for query_group in query_groups:
                query_specs = query_group["query_specs"]

                if not query_specs:
                    continue

                scale_factor = query_group["scale_factor"]
                group_name = query_group["name"]

                with suspend_progress(progress):
                    data_dfs = self._load_required_datasets(
                        query_specs=query_specs,
                        scale_factor=scale_factor,
                    )

                for algorithm_config in self.algorithms:
                    algorithm = self._load_algorithm_from_file(algorithm_config)

                    ground_truth_model_name = (
                        algorithm_config
                        .get("ground_truth", {})
                        .get("model_name")
                        or self.default_ground_truth_model_name
                    )

                    ground_truth_system_prompt = (
                        algorithm_config
                        .get("ground_truth", {})
                        .get("system_prompt")
                        or self.default_ground_truth_system_prompt
                    )

                    algorithm_kwargs = algorithm_config.get("algorithm_kwargs", {})
                    algorithm.preparation(data_dfs, algorithm_kwargs)

                    for query_spec in query_specs:
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
                                query_spec,
                                data_dfs,
                            )

                        algorithm.reset_cost_stats()

                        with suspend_progress(progress):
                            start = time.perf_counter()
                            cardinality_estimation = algorithm.run(query_spec)
                            time_ms = (time.perf_counter() - start) * 1000

                        q_error = self._calculate_q_error(
                            cardinality_estimation,
                            cardinality_ground_truth,
                        )

                        selectivity_estimation = self._calculate_selectivity(
                            cardinality_estimation,
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
                            cardinality_estimation=cardinality_estimation,
                            selectivity_estimation=selectivity_estimation,
                            q_error=q_error,
                            time_ms=time_ms,
                        )

                        progress.advance(task)

                    progress.console.print(
                        f"[green]✓[/green] Finished algorithm "
                        f"[bold cyan]{algorithm.name}[/bold cyan] "
                        f"on [bold]{len(query_specs)}[/bold] {group_name} queries."
                    )

        console.print(
            f"[green]✓[/green] Results written to [bold]{self.result_filepath}[/bold]"
        )

    def _split_query_specs_by_type(self):
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
        datasets = {
            dataset.table_ref
            for query_spec in query_specs
            for dataset in query_spec.datasets
        }

        return DataLoader().load(
            datasets=datasets,
            scale_factor=scale_factor,
        )

    def _get_cardinality_ground_truth(self, model_name: str, system_prompt: str, query_spec: QuerySpecification, data_dfs: dict[str, pd.DataFrame]) -> int:
        """Obtain model-based selectivity ground truth."""
        backend = LotusBackend(model_name=model_name, system_prompt=system_prompt, scale_factor=self.scale_factor)

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
            cardinality_ground_truth = backend.joining_query(query_spec, data_left, data_right)

        return cardinality_ground_truth

    def _calculate_q_error(
        self, cardinality_estimation: int, cardinality_ground_truth: int
    ) -> float:
        """Calcualte q error. Higher is worse."""
        if cardinality_estimation == cardinality_ground_truth:
            return 1.0
        elif cardinality_estimation == 0 or cardinality_ground_truth == 0:
            return sys.float_info.max
        else:
            return max(
                cardinality_estimation / cardinality_ground_truth,
                cardinality_ground_truth / cardinality_estimation,
            )
    
    def _calculate_selectivity(self, cardinality_estimation: int, query_spec: QuerySpecification, data_dfs: dict[str, pd.DataFrame]) -> float:
        """Calculate selectivity based on cardinality estimation and number of input rows."""
        if len(query_spec.datasets) == 1: # Filter query
            table_ref = query_spec.datasets[0].table_ref
            input_rows = data_dfs[table_ref].shape[0]
        
        elif len(query_spec.datasets) > 1: # Join query
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
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        if isinstance(obj, Enum):
            return obj.value

        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")