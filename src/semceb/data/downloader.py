import zipfile
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

from rich.progress import Progress

from src.semceb.utils.console import console
from src.semceb.utils.progress import create_download_progress


class DataDownloader:
    """Downloads benchmark data from a public cloud bucket."""

    def __init__(self):
        self.bucket_url = (
            "https://azimmerer-semantic-selectivity-datasets."
            "s3.eu-central-1.amazonaws.com/"
        )

        self.filenames = [
            "amazon-reviews/images.zip",
            "amazon-reviews/products_filtered.parquet",
            "amazon-reviews/reviews_filtered.parquet",
        ]

        self.local_data_folderpath = Path("data") / "datasets"

    def ensure_files_available(self) -> bool:
        """Ensure configured files exist locally, downloading if needed."""

        missing_files = self._get_missing_files()

        if missing_files:
            return self._download_missing_files(missing_files)

        console.print("[green]✓[/green] All data files already exist locally.")
        return True

    def _get_missing_files(self) -> list[dict]:
        """Return configured files that do not exist locally.

        For ZIP files, the extracted folder also counts as available.
        Example:
            amazon-reviews/images.zip
        is considered available if:
            data/.raw/amazon-reviews/images/
        already exists and contains files.
        """

        self.local_data_folderpath.mkdir(parents=True, exist_ok=True)

        missing_files = []

        for filename in self.filenames:
            local_filepath = self.local_data_folderpath / filename

            if self._is_zip_file(filename):
                extracted_folderpath = local_filepath.with_suffix("")

                if self._folder_has_files(extracted_folderpath):
                    continue

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
        """Prompt user, download all missing files, then extract ZIP files afterwards."""

        if not missing_files:
            console.print("[green]✓[/green] All data files already exist locally.")
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

        downloaded_zip_paths: list[Path] = []

        # Phase 1: download everything first.
        with create_download_progress() as progress:
            task = progress.add_task(
                "Downloading data...",
                total=total_bytes,
            )

            for remote_file in missing_files:
                local_filepath = self.local_data_folderpath / remote_file["filename"]

                self._download_file(
                    remote_url=remote_file["url"],
                    local_filepath=local_filepath,
                    progress=progress,
                    task=task,
                )

                if self._is_zip_file(remote_file["filename"]):
                    downloaded_zip_paths.append(local_filepath)

        console.print(
            f"[green]✓[/green] Downloaded missing data to "
            f"[bold]{self.local_data_folderpath}[/bold]"
        )

        # Phase 2: after all downloads are done, ask about ZIP extraction.
        for zip_path in downloaded_zip_paths:
            self._extract_zip_next_to_file(
                zip_path=zip_path,
                delete_zip_after_extract=True,
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

    def _extract_zip_next_to_file(
        self,
        zip_path: Path,
        delete_zip_after_extract: bool = True,
    ) -> None:
        """Extract a ZIP archive next to its location.

        Example:
            data/.raw/amazon-reviews/images.zip

        becomes:
            data/.raw/amazon-reviews/images/
        """

        output_folderpath = zip_path.with_suffix("")

        uncompressed_size_bytes = self._get_zip_uncompressed_size(zip_path)
        uncompressed_size = self._format_download_size(uncompressed_size_bytes)

        answer = console.input(
            f"[bold yellow]ZIP file found:[/bold yellow] "
            f"[bold]{zip_path}[/bold]\n"
            f"Uncompressed size: [bold]{uncompressed_size}[/bold]\n"
            f"Output folder: [bold]{output_folderpath}[/bold]\n"
            "[bold cyan]Do you want to unzip it now?[/bold cyan] "
            "[y/N]: "
        )

        if answer.lower() not in {"y", "yes"}:
            console.print("[yellow]Unzipping skipped.[/yellow]")
            return

        self._unzip_file(
            zip_path=zip_path,
            output_folderpath=output_folderpath,
        )

        console.print(
            f"[green]✓[/green] Extracted ZIP file to "
            f"[bold]{output_folderpath}[/bold]"
        )

        if delete_zip_after_extract and zip_path.exists():
            zip_path.unlink()

            console.print(
                f"[green]✓[/green] Deleted ZIP file "
                f"[bold]{zip_path}[/bold]"
            )

    def _unzip_file(self, zip_path: Path, output_folderpath: Path) -> None:
        """Extract ZIP file into the given output folder."""

        output_folderpath.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(output_folderpath)

    def _get_zip_uncompressed_size(self, zip_path: Path) -> int:
        """Return total uncompressed size of all files in a ZIP."""

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            return sum(
                file_info.file_size
                for file_info in zip_file.infolist()
                if not file_info.is_dir()
            )

    def _is_zip_file(self, filename: str) -> bool:
        """Return whether a configured file is a ZIP archive."""

        return Path(filename).suffix.lower() == ".zip"

    def _folder_has_files(self, folderpath: Path) -> bool:
        """Return whether a folder exists and contains at least one file."""

        return (
            folderpath.exists()
            and folderpath.is_dir()
            and any(folderpath.iterdir())
        )

    def _format_download_size(self, size_bytes: int) -> str:
        """Format download size as B, KB, MB, or GB using decimal units."""

        one_kb = 1000
        one_mb = 1000**2
        one_gb = 1000**3

        if size_bytes < one_kb:
            return f"{size_bytes} B"

        if size_bytes < one_mb:
            return f"{size_bytes / one_kb:.2f} KB"

        if size_bytes < one_gb:
            return f"{size_bytes / one_mb:.2f} MB"

        return f"{size_bytes / one_gb:.2f} GB"