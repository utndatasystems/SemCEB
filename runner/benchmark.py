import sys
import importlib.util
from pathlib import Path
import json
import time
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
from data.downloader import DataDownloader
from data.loader import DataLoader


class BenchmarkRunner:
    """Runs benchmark queries."""

    def __init__(
        self, algorithms: list[dict[str, Any]], default_system_prompt: str
    ):
        self.algorithms = algorithms
        self.default_system_prompt = default_system_prompt
        self.result_filepath = Path("results") / "raw" / "result.jsonl"
        self.query_filepath = Path("queries") / "generated" / "queries.jsonl"

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

                queries.append(json.loads(line))

        return queries

    def _handle_cloud_data(self) -> bool:
        """Download data if it is not available locally."""

        downloader = DataDownloader()
        missing_files = downloader.get_missing_files()

        if missing_files:
            data_ready = downloader.download_missing_files(missing_files)

            if not data_ready:
                console = Console()
                console.print(
                    "[bold red]Benchmark aborted.[/bold red]\n"
                    "[yellow]Required benchmark data is missing, and the download was skipped.[/yellow]"
                )
                raise SystemExit(1)

    def _load_algorithm_from_file(
        self, algorithm_config: dict[str, Any]
    ) -> Any:
        """Load algorithm class from a Python file."""

        project_root = Path(__file__).resolve().parent.parent
        algorithm_filepath = (
            project_root
            / "runner"
            / "algorithms"
            / algorithm_config["filename"]
        )

        if not algorithm_filepath.exists():
            raise FileNotFoundError(
                f"Algorithm file not found: {algorithm_filepath}"
            )

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        module_name = f"runner.algorithms.{algorithm_filepath.stem}"

        spec = importlib.util.spec_from_file_location(
            module_name, algorithm_filepath
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load algorithm module: {algorithm_filepath}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        class_name = algorithm_config["class"]

        if not hasattr(module, class_name):
            raise AttributeError(
                f"Algorithm class '{class_name}' not found in {algorithm_filepath}"
            )

        algorithm_class = getattr(module, class_name)

        return algorithm_class(
            algorithm_config["name"], algorithm_config["version"]
        )

    def run(
        self, default_system_prompt: str, scale_factor: int, console: Console
    ) -> None:
        """Measure, run and store result of benchmark queries."""

        total_runs = len(self.algorithms) * len(self.queries)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
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

                for query in self.queries:
                    query_text = query["query"]

                    progress.update(
                        task,
                        description=(
                            f"Algorithm: [cyan]{algorithm_config['name']}[/cyan] "
                            f"| Query ID: [yellow]{query['id']}[/yellow]"
                        ),
                    )

                    data = dataloader.load(
                        dataset=query["dataset"], scale_factor=scale_factor
                    )
                    system_prompt = algorithm_config.get("system_prompt", None)
                    if not system_prompt or system_prompt == "general":
                        system_prompt = default_system_prompt

                    algorithm.preparation(data, system_prompt)
                    algorithm.reset_cost()

                    start = time.perf_counter()
                    selectivity_estimation = algorithm.run(query_text)
                    time_ms = (time.perf_counter() - start) * 1000

                    selectivity_ground_truth = query["selectivity_ground_truth"]
                    q_error = max(
                        selectivity_estimation / selectivity_ground_truth,
                        selectivity_ground_truth / selectivity_estimation,
                    )

                    self._save_result(
                        query=query,
                        name=algorithm_config["name"],
                        version=algorithm_config["version"],
                        cost_usd=algorithm.cost_usd,
                        selectivity_estimation=selectivity_estimation,
                        q_error=q_error,
                        time_ms=time_ms,
                    )

                    # TODO - DEBUG
                    time.sleep(0.05)

                    progress.advance(task)

                progress.console.print(
                    f"[green]✓[/green] Finished algorithm "
                    f"[bold cyan]{algorithm.name}[/bold cyan] "
                    f"on [bold]{len(self.queries)}[/bold] queries."
                )

        console.print(
            f"[green]✓[/green] Results written to [bold]{self.result_filepath}[/bold]"
        )

    def _save_result(
        self,
        query: str,
        name: str,
        version: str,
        cost_usd: float,
        selectivity_estimation: int,
        q_error: float,
        time_ms: float,
    ) -> None:
        """Save query result as JSONL."""

        algorithm_data = {
            "name": name,
            "version": version,
            "cost_usd": cost_usd,
            "selectivity_estimation": selectivity_estimation,
            "q_error": q_error,
            "time_ms": time_ms,
        }
        result = {"query": query, "algorithm": algorithm_data}

        with open(self.result_filepath, "a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")
