from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from deployment.prepare_render_data import (
    REQUIRED_FILES,
    download_runtime_dataset,
    extract_runtime_archive,
)


def _write_archive(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_extract_runtime_archive_installs_expected_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "amazon-runtime.zip"
    destination = tmp_path / "raw"
    _write_archive(
        archive_path,
        {f"runtime/{name}": f"{name}\n" for name in REQUIRED_FILES},
    )

    extract_runtime_archive(archive_path, destination)

    assert {path.name for path in destination.iterdir()} == REQUIRED_FILES
    assert all(path.stat().st_size > 0 for path in destination.iterdir())


def test_extract_runtime_archive_rejects_missing_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "incomplete.zip"
    _write_archive(archive_path, {"users.csv": "user_id\n"})

    with pytest.raises(ValueError, match="missing required CSV files"):
        extract_runtime_archive(archive_path, tmp_path / "raw")


def test_extract_runtime_archive_rejects_traversal_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    members = {f"runtime/{name}": "data\n" for name in REQUIRED_FILES}
    members["../users.csv"] = "attacker-controlled\n"
    _write_archive(archive_path, members)

    with pytest.raises(ValueError, match="unsafe path"):
        extract_runtime_archive(archive_path, tmp_path / "raw")


def test_download_runtime_dataset_requires_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="valid HTTPS URL"):
        download_runtime_dataset("http://example.test/data.zip", tmp_path / "raw")


def test_download_runtime_dataset_does_not_expose_signed_url_on_failure(tmp_path: Path) -> None:
    signed_url = "https://example.test/data.zip?token=private-value"
    failure = HTTPError(signed_url, 403, "Forbidden", None, None)
    with patch("deployment.prepare_render_data.urllib.request.urlopen", side_effect=failure):
        with pytest.raises(RuntimeError, match="HTTP status 403") as exc_info:
            download_runtime_dataset(signed_url, tmp_path / "raw")

    assert "private-value" not in str(exc_info.value)
