import zipfile
import ssl
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import certifi
from rich.progress import Progress
from rich.prompt import Confirm

from src.semceb.utils.console import console
from src.semceb.utils.progress import create_download_progress


class DataDownloader:
    """Downloads benchmark data from a public cloud bucket."""

    def __init__(self):
        """Initialize downloader settings, bucket URL, and local data paths."""
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
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def ensure_files_available(self) -> bool:
        """Ensure configured files exist locally, downloading if needed."""

        missing_files = self._get_missing_files()

        if missing_files:
            return self._download_missing_files(missing_files)

        console.print("[green]✓[/green] All data files exist locally.")
        return True

    def _get_missing_files(self) -> list[dict]:
        """Return configured files that do not exist locally.

        For ZIP files, the extracted folder also counts as available.
        Example:
            amazon-reviews/images.zip
        is considered available if:
            data/datasets/amazon-reviews/images/
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

        total_bytes = sum(file["size"] for file in missing_files)
        download_size = self._format_download_size(total_bytes)

        console.print(
            f"[bold yellow]{len(missing_files)} missing files[/bold yellow] "
            f"will be downloaded ([bold]{download_size}[/bold])."
        )

        if not self._confirm_download():
            console.print("[yellow]Download skipped.[/yellow]")
            return False

        downloaded_zip_paths = self._download_files_with_progress(
            missing_files=missing_files,
            total_bytes=total_bytes,
        )

        console.print(
            f"[green]✓[/green] Downloaded missing data to "
            f"[bold]{self.local_data_folderpath}[/bold]"
        )

        self._extract_downloaded_zips(downloaded_zip_paths)

        return True

    def _confirm_download(self) -> bool:
        """Ask the user whether to start the download."""

        return Confirm.ask(
            "[bold cyan]Do you want to start downloading now?[/bold cyan]",
            default=False,
        )

    def _download_files_with_progress(
        self,
        missing_files: list[dict],
        total_bytes: int,
    ) -> list[Path]:
        """Download each missing file while updating progress."""

        downloaded_zip_paths: list[Path] = []

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

        return downloaded_zip_paths

    def _extract_downloaded_zips(self, downloaded_zip_paths: list[Path]) -> None:
        """Extract any ZIP files that were downloaded."""

        for zip_path in downloaded_zip_paths:
            self._extract_zip_next_to_file(
                zip_path=zip_path,
                delete_zip_after_extract=True,
            )

    def _get_remote_file_size(self, remote_url: str) -> int:
        """Return the remote file size in bytes."""

        try:
            with self._open_url(remote_url, method="HEAD") as response:
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
            with self._open_url(remote_url) as response:
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

    def _open_url(self, remote_url: str, method: str = "GET"):
        """Open a URL using certifi's CA bundle instead of local Python certs."""

        return urlopen(
            Request(remote_url, method=method),
            context=self.ssl_context,
        )

    def _extract_zip_next_to_file(
        self,
        zip_path: Path,
        delete_zip_after_extract: bool = True,
    ) -> None:
        """Extract a ZIP archive next to its location.

        Example:
            data/datasets/amazon-reviews/images.zip

        becomes:
            data/datasets/amazon-reviews/images/
        """

        output_folderpath = zip_path.with_suffix("")

        uncompressed_size_bytes = self._get_zip_uncompressed_size(zip_path)
        uncompressed_size = self._format_download_size(uncompressed_size_bytes)

        console.print(
            f"[bold yellow]ZIP file found:[/bold yellow] "
            f"[bold]{zip_path}[/bold]\n"
            f"Uncompressed size: [bold]{uncompressed_size}[/bold]\n"
            f"Output folder: [bold]{output_folderpath}[/bold]"
        )

        should_unzip = Confirm.ask(
            "[bold cyan]Do you want to unzip it now?[/bold cyan]",
            default=False,
        )

        if not should_unzip:
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
