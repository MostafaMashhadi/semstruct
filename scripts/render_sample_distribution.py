"""Render sample-distribution figures from the current training export.

Run from the repository root, with matplotlib installed:

    python scripts/render_sample_distribution.py

matplotlib is not a project dependency. The figures read exports/training-data,
which is gitignored because it contains benchmark cell text.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "exports" / "training-data"
OUT = ROOT / "docs" / "figures"
DATASETS = ("mondial", "imdb", "wikipedia")
NAMES = {"mondial": "Mondial", "imdb": "IMDb", "wikipedia": "Wikipedia"}


def render_figures(root: Path | None = None) -> None:
    root = root or ROOT
    export = root / "exports" / "training-data"
    out = root / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = _load(export)
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _source_bars(plt, export, out / "pairs-by-source.png")
    _element_bars(plt, export, out / "element-share.png")
    _fold_bars(plt, export, out / "coffman-folds.png")
    _keyword_hist(plt, export, out / "pairs-per-keyword.png")


def _load(export: Path) -> dict[str, list[dict]]:
    loaded = {}
    for dataset in DATASETS:
        path = export / f"{dataset}.pairs.jsonl"
        if not path.is_file():
            continue
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        loaded[dataset] = rows
    return loaded


def _label_bars(ax, bars) -> None:
    """Print the exact pair count above each bar."""
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.18,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def _source_bars(plt, export: Path, path: Path) -> None:
    """Grouped bars on a log scale, so the smaller synthetic counts stay visible."""
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    labels = [NAMES[name] for name in DATASETS]
    synthetic = []
    coffman = []
    for name in DATASETS:
        counts = Counter()
        source_file = export / f"{name}.pairs.jsonl"
        if source_file.is_file():
            with source_file.open(encoding="utf-8") as handle:
                for line in handle:
                    counts[json.loads(line).get("source")] += 1
        synthetic.append(counts.get("synthetic", 0))
        coffman.append(counts.get("coffman", 0))
    positions = list(range(len(labels)))
    width = 0.36
    green = ax.bar(
        [item - width / 2 for item in positions],
        synthetic,
        width=width,
        label="Synthetic",
        color="#0f6e56",
    )
    orange = ax.bar(
        [item + width / 2 for item in positions],
        coffman,
        width=width,
        label="Coffman",
        color="#c2410c",
    )
    ax.set_yscale("log")
    _label_bars(ax, list(green) + list(orange))
    ax.set_ylim(ax.get_ylim()[0], max(synthetic + coffman) * 8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Pairs")
    ax.set_title("Keyword instances by source")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _element_bars(plt, export: Path, path: Path) -> None:
    """Grouped bars on a log scale, so the relation counts are not hidden by the values."""
    kinds = ("relation", "attribute", "value")
    colors = {"relation": "#1f77b4", "attribute": "#0f6e56", "value": "#c2410c"}
    labels = {"relation": "Relation", "attribute": "Attribute", "value": "Value"}
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    totals = {name: Counter() for name in DATASETS}
    for name in DATASETS:
        source_file = export / f"{name}.pairs.jsonl"
        if not source_file.is_file():
            continue
        with source_file.open(encoding="utf-8") as handle:
            for line in handle:
                totals[name][json.loads(line).get("element_type")] += 1
    positions = list(range(len(DATASETS)))
    width = 0.24
    for offset, kind in enumerate(kinds):
        heights = [totals[name].get(kind, 0) for name in DATASETS]
        shift = (offset - 1) * width
        bars = ax.bar(
            [item + shift for item in positions],
            heights,
            width=width,
            label=labels[kind],
            color=colors[kind],
        )
        _label_bars(ax, bars)
    ax.set_yscale("log")
    tallest = max(max(totals[name].values(), default=1) for name in DATASETS)
    ax.set_ylim(ax.get_ylim()[0], tallest * 8)
    ax.set_xticks(positions, [NAMES[name] for name in DATASETS])
    ax.set_ylabel("Pairs")
    ax.set_title("Element types")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fold_bars(plt, export: Path, path: Path) -> None:
    """Log-scaled bars. Mondial has hundreds of Coffman pairs; IMDb has hundreds of thousands."""
    colors = {"mondial": "#1f77b4", "imdb": "#ff7f0e", "wikipedia": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    width = 0.25
    folds = list(range(5))
    drawn: list[int] = []
    for offset, name in enumerate(DATASETS):
        counts: Counter[int] = Counter()
        source_file = export / f"{name}.pairs.jsonl"
        if source_file.is_file():
            with source_file.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("source") != "coffman":
                        continue
                    fold = row.get("fold")
                    if fold is None:
                        continue
                    counts[int(fold)] += 1
        heights = [counts.get(fold, 0) for fold in folds]
        positions = [fold + (offset - 1) * width for fold in folds]
        bars = ax.bar(positions, heights, width=width, label=NAMES[name], color=colors[name])
        drawn.append(max(heights, default=1))
        _label_bars(ax, bars)
    ax.set_yscale("log")
    ax.set_ylim(ax.get_ylim()[0], max(drawn, default=1) * 8)
    ax.set_xticks(folds, [str(fold) for fold in folds])
    ax.set_xlabel("Test fold")
    ax.set_ylabel("Coffman pairs")
    ax.set_title("Coffman pairs by test fold")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _keyword_hist(plt, export: Path, path: Path) -> None:
    """Log-scaled histogram. A few Coffman keywords have tens of thousands of hits."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.6, 5.2), squeeze=False)
    for axis, name in zip(axes[0], DATASETS):
        source_file = export / f"{name}.pairs.jsonl"
        grouped: dict[tuple[str, str], list] = {}
        if source_file.is_file():
            with source_file.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (str(row.get("query_id")), str(row.get("keyword")))
                    item = grouped.get(key)
                    if item is None:
                        grouped[key] = [1, row.get("source")]
                    else:
                        item[0] += 1
        synthetic = [item[0] for item in grouped.values() if item[1] == "synthetic"]
        coffman = [item[0] for item in grouped.values() if item[1] == "coffman"]
        highest = max([1, *synthetic, *coffman])
        edges = [1.0]
        while edges[-1] <= highest:
            edges.append(edges[-1] * 2)
        axis.hist(
            [synthetic, coffman],
            bins=edges,
            stacked=True,
            label=["Synthetic", "Coffman"],
            color=["#0f6e56", "#c2410c"],
        )
        axis.set_xscale("log")
        axis.set_xlim(1, edges[-1])
        pairs = sum(synthetic) + sum(coffman)
        keywords = len(synthetic) + len(coffman)
        axis.set_title(f"{NAMES[name]}\n{keywords:,} keywords, {pairs:,} pairs")
        axis.set_xlabel("Pairs per keyword")
        axis.set_ylabel("Keywords")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    try:
        render_figures()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"matplotlib is required to draw figures: {exc}") from exc
