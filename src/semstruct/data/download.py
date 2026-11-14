"""Download the Coffman and Weaver evaluation subsets from the UVA Dataverse."""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from pathlib import Path

from semstruct.paths import raw_dir

logger = logging.getLogger(__name__)

DATAVERSE_API = (
    "https://dataverse.lib.virginia.edu/api/datasets/:persistentId/"
    "?persistentId=doi:10.18130/V3/KEVCF8"
)
DATAVERSE_FILE = "https://dataverse.lib.virginia.edu/api/access/datafile/{file_id}"
DATAVERSE_PAGE = "https://dataverse.lib.virginia.edu/dataset.xhtml?persistentId=doi:10.18130/V3/KEVCF8"
USER_AGENT = "semstruct-research/0.1 (academic; relational keyword search)"

# Public file ids and checksums for the evaluation subsets. The downloader
# prefers the Dataverse API and uses this table when the API is unavailable.
BENCHMARK_FILES: dict[str, list[dict[str, str | int]]] = {
    "mondial": [
        {"role": "schema", "name": "mondial-schema.pgsql.bz2", "id": 1163, "md5": "77d5f1f7add32223c66d01ebe4b4f51c"},
        {"role": "data", "name": "mondial-data.pgsql.bz2", "id": 1165, "md5": "665a11661600da959c4ff659f0a50227"},
        {"role": "topics", "name": "Mondial-topics-qrels.tar.bz2", "id": 1166, "md5": "e46ab3a0ccb94b41f54070f5f59b4289"},
    ],
    "imdb": [
        {"role": "schema", "name": "imdb2009_subset-schema.pgsql.bz2", "id": 1160, "md5": "9283c9b04a5edf8b71d4f580a927b25c"},
        {"role": "data", "name": "imdb2009_subset-data.pgsql.bz2", "id": 1169, "md5": "e78051027f93efc03b37e9baa8dbb52c"},
        {"role": "topics", "name": "IMDb-topics-qrels-V2.tar.bz2", "id": 1177, "md5": "8821e6659d5b6bce701155c1407eea64"},
    ],
    "wikipedia": [
        {"role": "schema", "name": "wikipedia2008_subset-schema.pgsql.bz2", "id": 1161, "md5": "ad1d9839be65bb44b97c98f275d6eaa0"},
        {"role": "data", "name": "wikipedia2008_subset-data.pgsql.bz2", "id": 1170, "md5": "579e4a60b4c5f11fe0550c355a348d58"},
        {"role": "topics", "name": "Wikipedia-topics-qrels.tar.bz2", "id": 1164, "md5": "bfe8e3e8ec26668412229d30279ec102"},
    ],
}


def files_for(dataset: str) -> list[dict[str, str | int]]:
    try:
        remote = _api_catalog()
    except Exception as exc:
        logger.warning("Dataverse API lookup failed (%s). Using the built-in file list.", exc)
        remote = {}
    resolved = []
    for item in BENCHMARK_FILES[dataset]:
        name = str(item["name"])
        merged = dict(item)
        if name in remote:
            merged["id"] = remote[name]["id"]
            merged["md5"] = remote[name].get("md5") or item["md5"]
        resolved.append(merged)
    return resolved


def download_dataset(dataset: str, root: Path | None = None) -> list[Path]:
    destination = raw_dir(root)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in files_for(dataset):
        path = destination / str(item["name"])
        expected = str(item["md5"])
        if path.is_file() and _md5(path) == expected:
            logger.info("already present %s", path.name)
            paths.append(path)
            continue
        url = DATAVERSE_FILE.format(file_id=item["id"])
        temporary = path.with_suffix(path.suffix + ".partial")
        logger.info("downloading %s", path.name)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            raise RuntimeError(_manual_message(dataset)) from exc
        digest = _md5(temporary)
        if digest != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {path.name}: expected {expected}, got {digest}")
        temporary.replace(path)
        paths.append(path)
    return paths


def locate_files(dataset: str, root: Path | None = None) -> dict[str, Path]:
    directory = raw_dir(root)
    located = {}
    missing = []
    for item in BENCHMARK_FILES[dataset]:
        path = directory / str(item["name"])
        if not path.is_file():
            missing.append(str(item["name"]))
            continue
        located[str(item["role"])] = path
    if missing:
        raise FileNotFoundError(_manual_message(dataset) + "\nMissing: " + ", ".join(missing))
    return located


def _api_catalog() -> dict[str, dict[str, object]]:
    request = urllib.request.Request(DATAVERSE_API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    version = payload["data"]["latestVersion"]
    catalog = {}
    for item in version.get("files", []):
        data_file = item.get("dataFile") or item
        name = item.get("label") or data_file.get("filename")
        if not name:
            continue
        catalog[str(name)] = {"id": data_file.get("id"), "md5": data_file.get("md5")}
    return catalog


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manual_message(dataset: str) -> str:
    names = ", ".join(str(item["name"]) for item in BENCHMARK_FILES[dataset])
    return (
        f"Could not download the {dataset} benchmark files. "
        f"Download them from {DATAVERSE_PAGE} and place them in data/raw/. "
        f"Required files: {names}. "
        "The benchmark is for academic, non-commercial use. Do not redistribute the dumps."
    )
