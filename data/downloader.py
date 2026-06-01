import zipfile
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

from utils.console import console
from utils.progress import create_download_progress
from rich.progress import Progress


class DataDownloader:
    """Downloads benchmark CSV data from a public cloud bucket."""

    def __init__(self):
        self.bucket_url = "https://azimmerer-semantic-selectivity-datasets.s3.eu-central-1.amazonaws.com/"

        self.filenames = [
            "dataset1.csv",
            "dataset2.csv",
            "products_filtered.parquet",
            "reviews_5core_filtered.parquet",
        ]

        self.local_data_folderpath = Path("data") / ".raw"

    def _get_missing_files(self) -> list[dict]:
        """Return configured CSV files that do not exist locally."""

        self.local_data_folderpath.mkdir(parents=True, exist_ok=True)

        missing_files = []

        for filename in self.filenames:
            local_filepath = self.local_data_folderpath / filename

            if local_filepath.exists():
                continue

            remote_url = urljoin(self.bucket_url, filename)
            size = self._get_remote_file_size(remote_url)

            missing_files.append(
                {
                    "filename": filename,
                    "url": remote_url,
                    "size": size,
                }
            )

        return missing_files

    def _download_missing_files(self, missing_files: list[dict]) -> bool:
        """Prompt user and download missing CSV files."""

        if not missing_files:
            console.print(
                "[green]✓[/green] All data files already exist locally."
            )
            return True

        total_bytes = sum(file["size"] for file in missing_files)
        download_size = self._format_download_size(total_bytes)

        answer = console.input(
            f"[bold yellow]{len(missing_files)} missing files[/bold yellow] "
            f"will be downloaded "
            f"([bold]{download_size}[/bold]).\n"
            "[bold cyan]Do you want to start downloading now?[/bold cyan] "
            "[y/N]: "
        )

        if answer.lower() not in {"y", "yes"}:
            console.print("[yellow]Download skipped.[/yellow]")
            return False

        with create_download_progress() as progress:
            task = progress.add_task(
                "Downloading data...",
                total=total_bytes,
            )

            for remote_file in missing_files:
                local_filepath = (
                    self.local_data_folderpath / remote_file["filename"]
                )

                self._download_file(
                    remote_url=remote_file["url"],
                    local_filepath=local_filepath,
                    progress=progress,
                    task=task,
                )

        console.print(
            f"[green]✓[/green] Downloaded missing data to "
            f"[bold]{self.local_data_folderpath}[/bold]"
        )

        return True

    def _get_remote_file_size(self, remote_url: str) -> int:
        """Return the remote file size in bytes."""

        try:
            with urlopen(remote_url) as response:
                size = response.headers.get("Content-Length")

                if size is None:
                    return 0

                return int(size)

        except HTTPError as error:
            console.print(
                f"[red]HTTP error while checking file:[/red] "
                f"{error.code} {error.reason}"
            )
            console.print(f"[dim]{remote_url}[/dim]")
            raise

        except URLError as error:
            console.print(
                f"[red]URL error while checking file:[/red] {error.reason}"
            )
            console.print(f"[dim]{remote_url}[/dim]")
            raise

    def _download_file(
        self,
        remote_url: str,
        local_filepath: Path,
        progress: Progress,
        task,
    ) -> None:
        """Download one file and update the total progress bar."""

        local_filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with urlopen(remote_url) as response:
                with open(local_filepath, "wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)

                        if not chunk:
                            break

                        file.write(chunk)
                        progress.update(task, advance=len(chunk))

        except HTTPError as error:
            console.print(
                f"[red]HTTP error while downloading:[/red] "
                f"{error.code} {error.reason}"
            )
            console.print(f"[dim]{remote_url}[/dim]")
            raise

        except URLError as error:
            console.print(
                f"[red]URL error while downloading:[/red] {error.reason}"
            )
            console.print(f"[dim]{remote_url}[/dim]")
            raise

    def _format_download_size(self, size_bytes: int) -> str:
        """Format download size as < 1 MB, MB, or GB."""

        one_mb = 1024**2
        one_gb = 1024**3

        if size_bytes < one_mb:
            return "< 1 MB"

        if size_bytes < one_gb:
            size_mb = size_bytes / one_mb
            return f"{size_mb:.2f} MB"

        size_gb = size_bytes / one_gb
        return f"{size_gb:.2f} GB"

    def ensure_files_available(self) -> bool:
        """Ensure data files exist locally, downloading if needed."""
        
        missing_files = self._get_missing_files()
        if missing_files:
            data_ready = self._download_missing_files(missing_files)
            return data_ready
        return True

    def _images_folder_has_files(self) -> bool:
        images_folderpath = self.local_data_folderpath / "images"

        return (
            images_folderpath.exists()
            and images_folderpath.is_dir()
            and any(images_folderpath.iterdir())
        )

    def _get_zip_uncompressed_size(self, zip_path: Path) -> int:
        """Return total uncompressed size of all files in a ZIP."""
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            return sum(
                file_info.file_size
                for file_info in zip_file.infolist()
                if not file_info.is_dir()
            )
        
    def _unzip_file(self, zip_path: Path, output_folderpath: Path) -> None:
        """Extract ZIP file into the given output folder."""
        output_folderpath.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(output_folderpath)

    def ensure_images_available(self, delete_zip_after_extract: bool = True) -> bool:
        """Ensure image files exist locally, downloading and unzipping images.zip if needed."""
        images_folderpath = self.local_data_folderpath / "images"
        zip_filepath = self.local_data_folderpath / "images.zip"
        remote_url = urljoin(self.bucket_url, "images.zip")

        if self._images_folder_has_files():
            
            if delete_zip_after_extract and zip_filepath.exists():
                zip_size = self._format_download_size(zip_filepath.stat().st_size)
                zip_filepath.unlink()

                console.print(
                    f"[green]✓[/green] Deleted leftover ZIP file "
                    f"([bold]{zip_size}[/bold] freed): "
                    f"[bold]{zip_filepath}[/bold]"
                )

            return True

        if not zip_filepath.exists():
            download_size_bytes = self._get_remote_file_size(remote_url)
            download_size = self._format_download_size(download_size_bytes)

            answer = console.input(
                f"[bold yellow]Images are missing.[/bold yellow]\n"
                f"The ZIP file will be downloaded "
                f"([bold]{download_size}[/bold]).\n"
                "[bold cyan]Do you want to start downloading now?[/bold cyan] "
                "[y/N]: "
            )

            if answer.lower() not in {"y", "yes"}:
                console.print("[yellow]Image download skipped.[/yellow]")
                return False

            with create_download_progress() as progress:
                task = progress.add_task(
                    "Downloading images.zip...",
                    total=download_size_bytes,
                )

                self._download_file(
                    remote_url=remote_url,
                    local_filepath=zip_filepath,
                    progress=progress,
                    task=task,
                )

        uncompressed_size_bytes = self._get_zip_uncompressed_size(zip_filepath)
        uncompressed_size = self._format_download_size(uncompressed_size_bytes)

        answer = console.input(
            f"[bold yellow]The ZIP file will now be extracted.[/bold yellow]\n"
            f"Uncompressed size: [bold]{uncompressed_size}[/bold]\n"
            f"Output folder: [bold]{images_folderpath}[/bold]\n"
            "[bold cyan]Do you want to unzip it now?[/bold cyan] "
            "[y/N]: "
        )

        if answer.lower() not in {"y", "yes"}:
            console.print("[yellow]Unzipping skipped.[/yellow]")
            return False

        self._unzip_file(
            zip_path=zip_filepath,
            output_folderpath=images_folderpath,
        )

        console.print(
            f"[green]✓[/green] Extracted images to "
            f"[bold]{images_folderpath}[/bold]"
        )

        if delete_zip_after_extract and zip_filepath.exists():
            zip_size = self._format_download_size(zip_filepath.stat().st_size)
            zip_filepath.unlink()

            console.print(
                f"[green]✓[/green] Deleted ZIP file "
                f"([bold]{zip_size}[/bold] freed): "
                f"[bold]{zip_filepath}[/bold]"
            )

        return True