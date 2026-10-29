"""Download and safely install the Amazon runtime dataset from a private HTTPS URL."""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

REQUIRED_FILES = {"users.csv", "products.csv", "interactions.csv", "reviews.csv"}
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024


def extract_runtime_archive(archive_path: Path, destination: Path) -> None:
    """Extract only the four expected CSVs, rejecting unsafe or oversized archives."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        selected: dict[str, zipfile.ZipInfo] = {}
        parents: set[tuple[str, ...]] = set()
        extracted_size = 0
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Dataset archive contains an unsafe path.")
            if info.is_dir() or path.name not in REQUIRED_FILES:
                continue
            if path.name in selected:
                raise ValueError(f"Dataset archive contains duplicate {path.name}.")
            selected[path.name] = info
            parents.add(path.parts[:-1])
            extracted_size += info.file_size

        if set(selected) != REQUIRED_FILES:
            missing = sorted(REQUIRED_FILES - set(selected))
            raise ValueError(f"Dataset archive is missing required CSV files: {', '.join(missing)}.")
        if len(parents) != 1:
            raise ValueError("Required CSV files must be in the same archive directory.")
        if extracted_size > MAX_EXTRACTED_BYTES:
            raise ValueError("Dataset archive exceeds the allowed extracted size.")

        with tempfile.TemporaryDirectory(dir=destination) as staging_name:
            staging = Path(staging_name)
            for name, info in selected.items():
                with archive.open(info) as source, (staging / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
            for name in REQUIRED_FILES:
                os.replace(staging / name, destination / name)


def download_runtime_dataset(url: str, destination: Path) -> None:
    """Download a private HTTPS archive without logging its signed URL."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("AMAZON_RUNTIME_DATA_URL must be a valid HTTPS URL.")

    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=destination) as staging_name:
            archive_path = Path(staging_name) / "amazon-runtime.zip"
            request = urllib.request.Request(
                url, headers={"User-Agent": "Rankwise-deployment"}
            )
            downloaded_size = 0
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                archive_path.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    downloaded_size += len(chunk)
                    if downloaded_size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Dataset archive exceeds the allowed download size.")
                    output.write(chunk)
            extract_runtime_archive(archive_path, destination)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Dataset download failed with HTTP status {exc.code}.") from None
    except urllib.error.URLError:
        raise RuntimeError(
            "Dataset download failed; check the private URL and network access."
        ) from None


def ensure_runtime_dataset(destination: Path, url: str) -> None:
    """Reuse a complete persistent copy or fetch and validate it once."""
    if all(
        (destination / name).is_file() and (destination / name).stat().st_size > 0
        for name in REQUIRED_FILES
    ):
        return
    download_runtime_dataset(url, destination)


if __name__ == "__main__":
    data_url = os.environ.get("AMAZON_RUNTIME_DATA_URL", "")
    if not data_url:
        raise RuntimeError(
            "Set AMAZON_RUNTIME_DATA_URL to a private HTTPS URL for the Amazon runtime ZIP."
        )
    ensure_runtime_dataset(Path("/var/data/raw"), data_url)
