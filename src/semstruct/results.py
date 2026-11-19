"""Drop the previous run's artifacts and export the current pair files."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from semstruct.data.prepare import read_jsonl
from semstruct.paths import DATASETS, model_dir, output_dir, processed_dir, project_root

EXPORT_COLUMNS = (
    "dataset",
    "source",
    "query_id",
    "topic_id",
    "fold",
    "query",
    "keyword",
    "element_id",
    "element_type",
    "relation",
    "attribute",
    "label",
    "element_text",
)


def retire_previous_results(root: Path | None = None) -> list[str]:
    """Remove the November checkpoints, exports, log, and figures.

    Fold directories from the current run are left in place. The raw dumps
    and the manuscripts are left in place.
    """
    root = root or project_root()
    removed: list[str] = []
    for dataset in DATASETS:
        model_root = model_dir(dataset, root)
        if model_root.exists():
            for child in list(model_root.iterdir()):
                if child.is_dir() and child.name.startswith("fold-"):
                    continue
                _remove(child, removed)
        report = output_dir(dataset, root)
        for name in ("metrics.json", "run_config.json", "train_history.json"):
            path = report / name
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    exports = root / "exports"
    for obsolete in (
        exports / "training-data-2026-11-15",
        exports / "semstruct-training-data-2026-11-15.zip",
    ):
        if obsolete.exists():
            _remove(obsolete, removed)
    log = root / "docs" / "logs" / "2026-11-14-reference-run.md"
    if log.is_file():
        log.unlink()
        removed.append(str(log))
    figures = root / "docs" / "figures"
    if figures.is_dir():
        for image in figures.glob("*.png"):
            image.unlink()
            removed.append(str(image))
    return removed


def export_training_data(root: Path | None = None) -> Path:
    """Rewrite the single training export from the current pair files."""
    root = root or project_root()
    destination = root / "exports" / "training-data"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    counts: dict[str, dict] = {}
    for dataset in DATASETS:
        pairs_path = processed_dir(dataset, root) / "pairs.jsonl"
        if not pairs_path.is_file():
            continue
        rows = read_jsonl(pairs_path)
        _write_jsonl(destination / f"{dataset}.pairs.jsonl", rows)
        _write_csv(destination / f"{dataset}.pairs.csv", rows, dataset)
        counts[dataset] = _pair_counts(rows)
    (destination / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    (destination / "README.md").write_text(_export_readme(), encoding="utf-8")
    return destination


def _pair_counts(rows: list[dict]) -> dict:
    by_source: dict[str, int] = {}
    positive = 0
    for row in rows:
        by_source[row.get("source", "")] = by_source.get(row.get("source", ""), 0) + 1
        positive += int(row.get("label", 0) == 1)
    return {"pairs": len(rows), "positive": positive, "negative": len(rows) - positive, "by_source": by_source}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict], dataset: str) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {column: row.get(column, "") for column in EXPORT_COLUMNS}
            payload["dataset"] = dataset
            writer.writerow(payload)


def _export_readme() -> str:
    return """# Training export

These files are the current keyword instances. Each row is one keyword, the full query, one element, and a label. Cell text is benchmark data, so this directory stays out of git.

`fold` is the Coffman topic's test fold. Synthetic rows train every fold and have an empty fold. Training uses at most 10 other value hits per keyword. The files keep every Coffman value hit, because validation and test score the full index list.
"""


def _remove(path: Path, removed: list[str]) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(str(path))
