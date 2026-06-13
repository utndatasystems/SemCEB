import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
from collections.abc import Callable
from typing import Any
import logging

from semceb.utils.console import console


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    for logger_name in [
        "fontTools",
        "fontTools.subset",
        "fontTools.ttLib",
        "fontTools.ttLib.ttFont",
        "PIL",
        "matplotlib",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def run_benchmark(config: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Run the benchmark."""
    from semceb.benchmark.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(
        algorithms=config["algorithms"],
        default_ground_truth_model_name=config["general"]["ground_truth"]["model_name"],
        default_ground_truth_system_prompt=config["general"]["ground_truth"]["system_prompt"],
        scale_factor=config["general"]["data"].get("scale_factor"),
        join_scale_factor=config["general"]["data"].get("join_scale_factor"),
        categories=config["general"]["data"]["categories"],
        types=config["general"]["data"]["types"]
    )
    runner.run()


def plot_benchmark(config: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """Plot the latest benchmark results."""
    from semceb.reporting.plotter import ResultsPlotter

    plotter = ResultsPlotter()
    plotter.plot()

def print_section(title: str, style: str = "bold cyan") -> None:
    """Print a formatted CLI section header."""
    console.print()
    console.rule(f"[{style}]{title}[/{style}]")
    console.print()


def print_done() -> None:
    """Print a formatted CLI completion message."""
    console.print()
    console.rule("[bold green]Done[/bold green]")
    console.print()

def print_help() -> None:
    """Print a short overview of the available SemCEB CLI commands."""
    print_section("SemCEB Help")

    console.print(
        """Usage:
  semceb <command>

Commands:
  run       Run the benchmark
  plot      Generate plots and summary tables from benchmark results

Examples:
  semceb run
  semceb plot"""
    )

    print_done()

def parse_args(argv: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    """
    Parse CLI arguments.

    Returns:
        mode: one of "run", "plot"
        kwargs: everything else as a dictionary
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help", "help"}:
        print_help()
        sys.exit(0)

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


def load_config(file_path: str = "config.toml") -> dict[str, Any]:
    """Loads config from a TOML file."""

    config_path = Path(file_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as file:
        config = tomllib.load(file)

    return config


def main() -> None:
    configure_logging()

    mode, kwargs = parse_args()

    print_section(f"Active mode: {mode}")
    
    from dotenv import load_dotenv
    load_dotenv()

    config = load_config("config.toml")

    commands: dict[str, Callable[[dict[str, Any]], None]] = {
        "run": run_benchmark,
        "plot": plot_benchmark,
    }

    if mode not in commands:
        raise ValueError(f"Unsupported mode: {mode!r}")

    commands[mode](config, kwargs)

    print_done()


if __name__ == "__main__":
    main()
