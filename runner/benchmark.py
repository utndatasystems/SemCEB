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
from runner.models.lotus_backend import LotusBackend


class BenchmarkRunner:
    """Runs benchmark queries."""

    def __init__(
        self,
        algorithms: list[dict[str, Any]],
        default_model_name: str,
        default_system_prompt: str,
        default_using_cache_for_LLM: bool,
        scale_factor: int,
        console: Console,
    ):
        self.algorithms = algorithms
        self.default_model_name = default_model_name
        self.default_system_prompt = default_system_prompt
        self.default_using_cache_for_LLM = default_using_cache_for_LLM
        self.scale_factor = scale_factor
        self.console = console

        self.result_filepath = Path("results") / "raw" / "result.jsonl"
        self.query_filepath = Path("queries") / "generated" / "queries.jsonl"

        self.queries = self._load_queries(self.query_filepath)

        self.lotus_backend = {}

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
    ) -> Any:
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

            model_name = algorithm_config.get("model", None)
            if not model_name or model_name == "general":
                model_name = self.default_model_name

            system_prompt = algorithm_config.get("system_prompt", None)
            if not system_prompt or system_prompt == "general":
                system_prompt = self.default_system_prompt
            
            using_cache_for_LLM = algorithm_config.get("using_cache_for_LLM", None)
            if not using_cache_for_LLM or using_cache_for_LLM == "general":
                using_cache_for_LLM = self.default_using_cache_for_LLM

            for query in self.queries:

                progress.update(
                    task,
                    description=(
                        f"Algorithm: [cyan]{algorithm_config['name']}[/cyan] "
                        f"| Query ID: [yellow]{query['id']}[/yellow]"
                    ),
                )

                data = dataloader.load(
                    dataset=query["dataset"], scale_factor=self.scale_factor
                )
                # TODO - DEBUG - Manually shortend
                data = data.head(20)

                backend = self._get_LLM_backend(model_name, system_prompt, using_cache_for_LLM)

                selectivity_ground_truth = self._get_selectivity_ground_truth(query, data, backend)

                algorithm_kwargs = algorithm_config.get("algorithm_kwargs", {})
                algorithm.preparation(data, algorithm_kwargs)

                algorithm.reset_cost_stats()

                start = time.perf_counter()
                selectivity_estimation = algorithm.run(query, backend)
                time_ms = (time.perf_counter() - start) * 1000

                q_error = self._calculate_q_error(
                    selectivity_estimation, selectivity_ground_truth
                )

                self._save_result(
                    query=query,
                    name=algorithm_config["name"],
                    version=algorithm_config["version"],
                    memory_consumption=algorithm.memory_consumption,
                    cost_stats=algorithm.cost_stats,
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

    def _get_LLM_backend(self, model_name: str, system_prompt: str, using_cache_for_LLM) -> LotusBackend:
        """Get lotus backend matching the demanded configuration."""
        if model_name not in self.lotus_backend:
            self.lotus_backend[model_name] = {}
        if system_prompt not in self.lotus_backend[model_name]:
            self.lotus_backend[model_name][system_prompt] = {}
        if using_cache_for_LLM not in self.lotus_backend[model_name][system_prompt]:
            self.lotus_backend[model_name][system_prompt][using_cache_for_LLM] = LotusBackend(model_name=model_name, system_prompt=system_prompt, using_cache=using_cache_for_LLM)
        return self.lotus_backend[model_name][system_prompt][using_cache_for_LLM]

    def _get_selectivity_ground_truth(self, query: dict, data: pd.DataFrame, backend: LotusBackend) -> int:
        """Obtain model-based selectivity ground truth."""
        selectivity_ground_truth = backend.filtering_query(query, data)
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
        query: str,
        name: str,
        version: str,
        memory_consumption: int,
        cost_stats: dict,
        selectivity_ground_truth: int,
        selectivity_estimation: int,
        q_error: float,
        time_ms: float,
    ) -> None:
        """Save query result as JSONL."""

        algorithm_data = {
            "name": name,
            "version": version,
            "memory_consumption": memory_consumption,
            "cost_stats": cost_stats,
            "selectivity_ground_truth": selectivity_ground_truth,
            "selectivity_estimation": selectivity_estimation,
            "q_error": q_error,
            "time_ms": time_ms,
        }
        result = {"query": query, "algorithm": algorithm_data}

        with open(self.result_filepath, "a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")
