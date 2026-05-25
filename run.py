import sys
import tomllib
from pathlib import Path

from collections.abc import Callable
from typing import Any

from runner.benchmark import BenchmarkRunner
from results.plotter import ResultsPlotter

from utils.console import console


def generate_queries(config: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Generate queries for a benchmark run."""

    query_templates = config["general"]["query_templates"]
    max_queries_per_template = config["general"]["max_queries_per_template"]
    output_file = Path("queries") / "generated" / "queries.jsonl"
    

    # Clear file
    with open(output_file, "w"):
        pass

    generator = QueryGenerator(file_path=output_file)
    with console.status(
        "[bold green]Generating queries...[/bold green]", spinner="dots"
    ):

        for query_template_name in query_templates:
            try:
                template = generator.templates[query_template_name]
            except KeyError as error:
                available_templates = ", ".join(generator.templates.keys())
                raise ValueError(
                    f"Query template not found: '{query_template_name}'. "
                    f"Available templates: {available_templates}"
                ) from error

            generator.generate(
                template=template,
                amount=max_queries_per_template,
            )

    console.print(
        f"[green]✓[/green] Generated queries written to [bold]{output_file}[/bold]"
    )


def run_benchmark(config: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Run the benchmark."""

    runner = BenchmarkRunner(
        algorithms=config["algorithms"],
        default_ground_truth_model_name=config["general"]["ground_truth"]["model_name"],
        default_ground_truth_system_prompt=config["general"]["ground_truth"]["system_prompt"],
        scale_factor=config["general"]["data"]["scale_factor"],
        categories=config["general"]["data"]["categories"],
    )
    runner.run()


def plot_benchmark(config: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Plot the latest benchmark results."""
    plotter = ResultsPlotter()
    plotter.plot()


def parse_args(argv: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    """
    Parse CLI arguments.

    Returns:
        mode: one of "generate", "run", "plot"
        kwargs: everything else as a dictionary
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        mode = "run" # default
    else: 
        mode = argv[0]

    valid_modes = {"run", "plot"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected one of: {', '.join(valid_modes)}"
        )

    kwargs: dict[str, Any] = {}
    positional_args: list[str] = []

    i = 1
    while i < len(argv):
        arg = argv[i]

        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")

            if not key:
                raise ValueError("Empty keyword argument")

            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                kwargs[key] = argv[i + 1]
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            positional_args.append(arg)
            i += 1

    if positional_args:
        kwargs["_args"] = positional_args

    return mode, kwargs


def load_config(self, file_path: str = "config.toml") -> dict[str, Any]:
    """Loads config from a TOML file."""

    config_path = Path(file_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as file:
        config = tomllib.load(file)

    return config


def main() -> None:
    mode, kwargs = parse_args()
    console.print()
    console.rule(f"[bold cyan]Active mode: {mode}[/bold cyan]")
    console.print()

    config = load_config("config.toml")

    commands: dict[str, Callable[[dict[str, Any]], None]] = {
        "run": run_benchmark,
        "plot": plot_benchmark,
    }

    if mode not in commands:
        raise ValueError(f"Unsupported mode: {mode!r}")

    commands[mode](config, kwargs)

    console.print()
    console.rule("[bold green]Done[/bold green]")
    console.print()


if __name__ == "__main__":
    main()
