from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve
import xml.etree.ElementTree as ET

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


class DataDownloader:
    """Downloads benchmark data from a public cloud bucket."""

    def __init__(self):
        # TODO - DEBUG - exchange with actual aws link
        self.bucket_link = "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_year/"
        self.local_data_folderpath = Path(r"data\.raw")
        self.console = Console()

    def get_missing_files(self) -> list[dict]:
        """Return remote files that do not exist locally."""

        self.local_data_folderpath.mkdir(parents=True, exist_ok=True)

        remote_files = self._list_remote_files()
        # TODO - DEBUG - reduced to one
        remote_files = remote_files[:1]
        missing_files = []

        for remote_file in remote_files:
            local_filepath = self.local_data_folderpath / remote_file["key"]

            if not local_filepath.exists():
                missing_files.append(remote_file)

        # TODO - DEBUG
        return missing_files

    def download_missing_files(self, missing_files: list[dict]) -> bool:
        """Prompt user and download missing files."""

        if not missing_files:
            self.console.print(
                "[green]✓[/green] All data files already exist locally."
            )
            return True

        total_bytes = sum(file["size"] for file in missing_files)
        download_size = self._format_download_size(total_bytes)

        answer = self.console.input(
            f"[bold yellow]{len(missing_files)} missing files[/bold yellow] "
            f"will be downloaded "
            f"([bold]{download_size}[/bold]).\n"
            "[bold cyan]Do you want to start downloading now?[/bold cyan] "
            "[y/N]: "
        )

        if answer.lower() not in {"y", "yes"}:
            self.console.print("[yellow]Download skipped.[/yellow]")
            return False

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            total_size = sum(file["size"] for file in missing_files)

            task = progress.add_task(
                "Downloading data...",
                total=total_size,
            )

            for remote_file in missing_files:
                remote_url = remote_file["url"]
                local_filepath = self.local_data_folderpath / remote_file["key"]

                local_filepath.parent.mkdir(parents=True, exist_ok=True)

                self._download_file(
                    remote_url=remote_url,
                    local_filepath=local_filepath,
                    progress=progress,
                    task=task,
                )

        self.console.print(
            f"[green]✓[/green] Downloaded missing data to "
            f"[bold]{self.local_data_folderpath}[/bold]"
        )

        return True

    def _format_download_size(self, size_bytes: int) -> str:
        """Format download size as MB below 1 GB, otherwise as GB."""

        one_gb = 1024**3

        if size_bytes < one_gb:
            size_mb = size_bytes / (1024**2)
            return f"{size_mb:.2f} MB"

        size_gb = size_bytes / one_gb
        return f"{size_gb:.2f} GB"

    def _list_remote_files(self) -> list[dict]:
        """List all files in a public S3-style bucket path."""

        parsed_url = urlparse(self.bucket_link)

        bucket_base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        prefix = parsed_url.path.lstrip("/")

        if prefix and not prefix.endswith("/"):
            prefix += "/"

        list_url = f"{bucket_base_url}/?list-type=2&prefix={prefix}"

        remote_files = []

        while list_url:
            with urlopen(list_url) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)

            namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

            for content in root.findall("s3:Contents", namespace):
                key = content.find("s3:Key", namespace).text
                size = int(content.find("s3:Size", namespace).text)

                if key.endswith("/"):
                    continue

                relative_key = (
                    key[len(prefix) :] if key.startswith(prefix) else key
                )

                remote_files.append(
                    {
                        "key": relative_key,
                        "size": size,
                        "url": f"{bucket_base_url}/{key}",
                    }
                )

            is_truncated = root.find("s3:IsTruncated", namespace)

            if is_truncated is not None and is_truncated.text == "true":
                next_token = root.find(
                    "s3:NextContinuationToken", namespace
                ).text
                list_url = (
                    f"{bucket_base_url}/?list-type=2"
                    f"&prefix={prefix}"
                    f"&continuation-token={next_token}"
                )
            else:
                list_url = None

        return remote_files

    def _download_file(
        self,
        remote_url: str,
        local_filepath: Path,
        progress: Progress,
        task,
    ) -> None:
        """Download one file and update the total progress bar."""

        with urlopen(remote_url) as response:
            with open(local_filepath, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    file.write(chunk)
                    progress.update(task, advance=len(chunk))
