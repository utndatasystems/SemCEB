import sys
import importlib
from pathlib import Path
import json
from enum import Enum
import time
import pandas as pd
from typing import Any
from utils.progress import create_benchmark_progress, suspend_progress
from utils.console import console
from data.downloader import DataDownloader
from data.loader import DataLoader
from runner.algorithms.interface import AlgorithmInterface
from runner.llm_backends.lotus_backend import LotusBackend
from queries.template_parser import QueryTemplateParser


class BenchmarkRunner:
    """Runs benchmark queries."""

    def __init__(
        self,
        algorithms: list[dict[str, Any]],
        default_ground_truth_model_name: str,
        default_ground_truth_system_prompt: str,
        scale_factor: int,
        categories: list[str],
    ):
        self.algorithms = algorithms
        self.default_ground_truth_model_name = default_ground_truth_model_name
        self.default_ground_truth_system_prompt = default_ground_truth_system_prompt
        self.scale_factor = scale_factor
        self.categories = categories

        self.result_filepath = Path("results") / "raw" / "result.jsonl"
        self.query_filepath = Path("queries") / "queries.jsonl"

        self.queries = self._load_queries(self.query_filepath)

        self._handle_cloud_data()

    def _load_queries(self, file_path: str) -> list[dict[str, Any]]:
        """Load queries from a JSONL file."""

        queries = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                query_data = json.loads(line)
                if query_data["category"] in self.categories:
                    queries.append(query_data)

        return queries

    def _handle_cloud_data(self) -> bool:
        """Download data if it is not available locally."""

        downloader = DataDownloader()

        files_complete = downloader.ensure_files_available()
        if not files_complete:
            console.print(
                "[bold red]Benchmark aborted.[/bold red]\n"
                "[yellow]Required benchmark data is missing, and the download was skipped.[/yellow]"
            )
            raise SystemExit(1)
            
        images_complete = downloader.ensure_images_available()
        if not images_complete:
            console.print(
                    "[bold red]Benchmark aborted.[/bold red]\n"
                    "[yellow]Required benchmark image data is missing, and the download or extraction was skipped.[/yellow]"
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
        module_name = f"runner.algorithms.{module_stem}"

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
        """Measure, run and store result of benchmark queries."""

        total_runs = len(self.algorithms) * len(self.queries)

        with create_benchmark_progress() as progress:
            task = progress.add_task(
                "Running benchmark...",
                total=total_runs,
            )

            # Clear file
            with open(self.result_filepath, "w"):
                pass

            # Load the required datasets
            datasets = {
                dataset
                for query_dict in self.queries
                for dataset in query_dict.get("datasets", [])
            }
            data_dfs = DataLoader().load(datasets=datasets, scale_factor=self.scale_factor)
            # TODO - DEBUG - Manually shortend
            data_dfs = {k: df.head(10) for k, df in data_dfs.items()}

            for algorithm_config in self.algorithms:
                algorithm = self._load_algorithm_from_file(algorithm_config)

                # Check if algorithm configuration demands non default ground truth procedure
                ground_truth_model_name = algorithm_config.get("ground_truth", {}).get("model_name", None)
                if not ground_truth_model_name:
                    ground_truth_model_name = self.default_ground_truth_model_name
                ground_truth_system_prompt = algorithm_config.get("ground_truth", {}).get("system_prompt", None)
                if not ground_truth_system_prompt:
                    ground_truth_system_prompt = self.default_ground_truth_system_prompt

                algorithm_kwargs = algorithm_config.get("algorithm_kwargs", {})
                algorithm.preparation(data_dfs, algorithm_kwargs)

                for query_dict in self.queries:
                    
                    query_dict["filter_parsed"] = QueryTemplateParser.parse(query_dict["filter"])

                    progress.update(
                        task,
                        description=(
                            f"Algorithm: [cyan]{algorithm_config['name']}[/cyan] "
                            f"| Query ID: [yellow]{query_dict['id']}[/yellow]"
                        ),
                    )

                    with suspend_progress(progress):
                        selectivity_ground_truth = self._get_selectivity_ground_truth(ground_truth_model_name, ground_truth_system_prompt, query_dict, data_dfs)

                    algorithm.reset_cost_stats()

                    with suspend_progress(progress):
                        start = time.perf_counter()
                        selectivity_estimation = algorithm.run(query_dict)
                        time_ms = (time.perf_counter() - start) * 1000

                    q_error = self._calculate_q_error(
                        selectivity_estimation, selectivity_ground_truth
                    )

                    self._save_result(
                        query_dict=query_dict,
                        algorithm_name=algorithm_config["name"],
                        algorithm_version=algorithm_config["version"],
                        algorithm_memory_consumption=algorithm.get_memory_consumption(),
                        algorithm_cost_stats=algorithm.get_cost_stats(),
                        selectivity_ground_truth=selectivity_ground_truth,
                        selectivity_estimation=selectivity_estimation,
                        q_error=q_error,
                        time_ms=time_ms,
                    )

                    progress.advance(task)

                progress.console.print(
                    f"[green]✓[/green] Finished algorithm "
                    f"[bold cyan]{algorithm.name}[/bold cyan] "
                    f"on [bold]{len(self.queries)}[/bold] queries."
                )

        console.print(
            f"[green]✓[/green] Results written to [bold]{self.result_filepath}[/bold]"
        )

    def _get_selectivity_ground_truth(self, model_name: str, system_prompt: str, query_dict: dict, data_dfs: dict[str, pd.DataFrame]) -> int:
        """Obtain model-based selectivity ground truth."""
        backend = LotusBackend(model_name=model_name, system_prompt=system_prompt, scale_factor=self.scale_factor)

        if len(query_dict["datasets"]) == 1:
            # Filtering
            name = query_dict["datasets"][0]
            data = data_dfs[name]
            selectivity_ground_truth = backend.filtering_query(query_dict, data)
        elif len(query_dict["datasets"]) > 1:
            # Joining
            name_left, name_right = query_dict["datasets"]
            data_left = data_dfs[name_left]
            data_right = data_dfs[name_right]
            selectivity_ground_truth = backend.joining_query(query_dict, data_left, data_right)

        return selectivity_ground_truth

    def _calculate_q_error(
        self, selectivity_estimation: int, selectivity_ground_truth: int
    ) -> float:
        """Calcualte q error. Higher is worse."""
        if selectivity_estimation == selectivity_ground_truth:
            return 1.0
        elif selectivity_estimation == 0 or selectivity_ground_truth == 0:
            return sys.float_info.max
        else:
            return max(
                selectivity_estimation / selectivity_ground_truth,
                selectivity_ground_truth / selectivity_estimation,
            )

    def _save_result(
        self,
        query_dict: dict,
        algorithm_name: str,
        algorithm_version: str,
        algorithm_memory_consumption: int,
        algorithm_cost_stats: dict,
        selectivity_ground_truth: int,
        selectivity_estimation: int,
        q_error: float,
        time_ms: float,
    ) -> None:
        """Save query result as JSONL."""

        algorithm_data = {
            "name": algorithm_name,
            "version": algorithm_version,
            "memory_consumption": algorithm_memory_consumption,
            "cost_stats": algorithm_cost_stats,
            "selectivity_ground_truth": selectivity_ground_truth,
            "selectivity_estimation": selectivity_estimation,
            "q_error": q_error,
            "time_ms": time_ms,
        }
        result = {"query": query_dict, "algorithm": algorithm_data}

        with open(self.result_filepath, "a", encoding="utf-8") as file:
            file.write(json.dumps(result, default=self.json_default) + "\n")

    def json_default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        if isinstance(obj, Enum):
            return obj.value

        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")