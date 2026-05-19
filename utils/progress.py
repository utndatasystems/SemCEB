from contextlib import contextmanager
from typing import Iterator

from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from utils.console import console


def create_download_progress(*, disable: bool = False) -> Progress:
    """Create a Rich progress bar for downloads."""

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=disable,
    )


def create_benchmark_progress(*, disable: bool = False) -> Progress:
    """Create a Rich progress bar for benchmark runs."""

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        disable=disable,
    )


@contextmanager
def suspend_progress(progress: Progress) -> Iterator[None]:
    """Temporarily stop a Rich progress bar while another library prints."""

    progress.stop()

    try:
        yield
    finally:
        progress.start()