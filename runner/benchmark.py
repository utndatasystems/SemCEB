import sys
import importlib
from pathlib import Path
import json
import time
import pandas as pd
from typing import Any
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from data.loader import DataLoader
from runner.algorithms.interface import AlgorithmInterface
from runner.llm_backends.lotus_backend import LotusBackend


class BenchmarkRunner:
    """Runs benchmark queries."""

    def __init__(
        self,
        algorithms: list[dict[str, Any]],
        default_ground_truth_model_name: str,
        default_ground_truth_system_prompt: str,
        scale_factor: int,
        categories: list[str],
        console: Console,
    ):
        self.algorithms = algorithms
        self.default_ground_truth_model_name = default_ground_truth_model_name
        self.default_ground_truth_system_prompt = default_ground_truth_system_prompt
        self.scale_factor = scale_factor
        self.categories = categories
        self.console = console

        self.result_filepath = Path("results") / "raw" / "result.jsonl"
        self.query_filepath = Path("queries") / "generated" / "queries.jsonl"

        self.queries = self._load_queries(self.query_filepath)


    def _load_queries(self, file_path: str) -> list[dict[str, Any]]:
        """Load queries from a JSONL file."""

        queries = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                queries.append(json.loads(line))

        return queries

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

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Running benchmark...",
                total=total_runs,
            )

        # Clear file
        with open(self.result_filepath, "w"):
            pass

        dataloader = DataLoader()

        for algorithm_config in self.algorithms:
            algorithm = self._load_algorithm_from_file(algorithm_config)

            # Check if algorithm configuration demands non default ground truth procedure
            ground_truth_model_name = algorithm_config.get("ground_truth", {}).get("model_name", None)
            if not ground_truth_model_name:
                ground_truth_model_name = self.default_ground_truth_model_name
            ground_truth_system_prompt = algorithm_config.get("ground_truth", {}).get("system_prompt", None)
            if not ground_truth_system_prompt:
                ground_truth_system_prompt = self.default_ground_truth_system_prompt

            for query_dict in self.queries:

                progress.update(
                    task,
                    description=(
                        f"Algorithm: [cyan]{algorithm_config['name']}[/cyan] "
                        f"| Query ID: [yellow]{query_dict['id']}[/yellow]"
                    ),
                )

                data = dataloader.load(
                    dataset=query_dict["dataset"], scale_factor=self.scale_factor
                )
                # TODO - DEBUG - Manually shortend
                data = data.head(20)

                selectivity_ground_truth = self._get_selectivity_ground_truth(ground_truth_model_name, ground_truth_system_prompt, query_dict, data)

                algorithm_kwargs = algorithm_config.get("algorithm_kwargs", {})
                algorithm.preparation(data, algorithm_kwargs)

                algorithm.reset_cost_stats()

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

        self.console.print(
            f"[green]✓[/green] Results written to [bold]{self.result_filepath}[/bold]"
        )

    def _get_selectivity_ground_truth(self, model_name: str, system_prompt: str, query_dict: dict, data: pd.DataFrame) -> int:
        """Obtain model-based selectivity ground truth."""
        backend = LotusBackend(model_name=model_name, system_prompt=system_prompt)
        selectivity_ground_truth = backend.filtering_query(query_dict, data)
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
            file.write(json.dumps(result) + "\n")
