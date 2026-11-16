"""Render the 15 November 2026 sample-distribution figures.

Run from the repository root, with matplotlib installed:

    python scripts/render_sample_distribution.py

matplotlib is not a project dependency. The figures read
exports/training-data-2026-11-15, which is gitignored because it contains
benchmark cell text. Only the counts are plotted, and the images are written
to docs/figures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "exports" / "training-data-2026-11-15"
OUT = ROOT / "docs" / "figures"

DATASETS = ("mondial", "imdb", "wikipedia")
NAMES = {"mondial": "Mondial", "imdb": "IMDb", "wikipedia": "Wikipedia"}
SPLITS = ("train", "validation", "test")
SPLIT_LABELS = {"train": "Train", "validation": "Validation", "test": "Test"}
SOURCES = ("synthetic", "coffman")
ELEMENTS = ("relation", "attribute", "value")
ELEMENT_LABELS = {"relation": "Relation", "attribute": "Attribute", "value": "Value"}

INK = "#1c1917"
MUTED = "#57534e"
LINE = "#e7e5e4"
SYN = "#0f6e56"
COF = "#c2410c"
QUERY = "#0f6e56"
PAIR = "#1d4e89"
TARGET = "#a8a29e"
REL_C = "#0f6e56"
ATTR_C = "#1d4e89"
VAL_C = "#c2410c"
POS = "#0f6e56"
NEG = "#d6d3d1"
CMAP = LinearSegmentedColormap.from_list(
    "semstruct", ["#f4f1ea", "#9dcec2", "#0f6e56", "#134e4a"]
)


def choose_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Avenir Next", "Avenir", "Helvetica Neue", "Helvetica", "DejaVu Sans"):
        if name in available:
            return name
    return "DejaVu Sans"


def style() -> None:
    font = choose_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 200,
        }
    )


def load():
    pair_count = defaultdict(int)
    query_ids = defaultdict(set)
    per_query = defaultdict(lambda: {"pairs": 0, "relevant": 0})
    for name in DATASETS:
        path = EXPORT / f"{name}.pairs.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["dataset"], row["split"], row["source"], row["element_type"], int(row["label"]))
                pair_count[key] += 1
                query_ids[(row["dataset"], row["split"], row["source"])].add(row["query_id"])
                bucket = per_query[(row["dataset"], row["source"], row["query_id"])]
                bucket["pairs"] += 1
                bucket["relevant"] += int(row["label"])
                bucket["split"] = row["split"]
    return pair_count, query_ids, per_query


def pairs(pair_count, dataset, split=None, source=None, element=None, label=None) -> int:
    total = 0
    for (ds, sp, src, el, lab), count in pair_count.items():
        if ds != dataset:
            continue
        if split is not None and sp != split:
            continue
        if source is not None and src != source:
            continue
        if element is not None and el != element:
            continue
        if label is not None and lab != label:
            continue
        total += count
    return total


def queries(query_ids, dataset, split=None, source=None) -> int:
    found = set()
    for (ds, sp, src), ids in query_ids.items():
        if ds != dataset:
            continue
        if split is not None and sp != split:
            continue
        if source is not None and src != source:
            continue
        found |= ids
    return len(found)


def finish(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(path.relative_to(ROOT))


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=8, fontweight="medium")


def bare(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.yaxis.grid(True, color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def label_vertical(ax, bars, gap: float = 0.04) -> None:
    tops = [bar.get_height() for bar in bars]
    ceiling = max(tops) if tops else 1
    ax.set_ylim(0, ceiling * (1 + gap + 0.12))
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )


def grouped_counts(path, query_ids, pair_count, kind: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    x = np.arange(len(SPLITS))
    width = 0.36
    for ax, dataset in zip(axes, DATASETS):
        if kind == "queries":
            synthetic = [queries(query_ids, dataset, split, "synthetic") for split in SPLITS]
            coffman = [queries(query_ids, dataset, split, "coffman") for split in SPLITS]
            ylabel = "Queries"
        else:
            synthetic = [pairs(pair_count, dataset, split, "synthetic") for split in SPLITS]
            coffman = [pairs(pair_count, dataset, split, "coffman") for split in SPLITS]
            ylabel = "Pairs"
        left = ax.bar(x - width / 2, synthetic, width, color=SYN, label="Synthetic", zorder=3)
        right = ax.bar(x + width / 2, coffman, width, color=COF, label="Coffman", zorder=3)
        label_vertical(ax, list(left) + list(right))
        ax.set_xticks(x, [SPLIT_LABELS[split] for split in SPLITS])
        ax.set_ylabel(ylabel)
        panel_title(ax, NAMES[dataset])
        bare(ax)
    axes[0].legend(frameon=False, loc="upper right")
    noun = "Queries" if kind == "queries" else "Pairs"
    fig.suptitle(f"{noun} by split and source", fontsize=16, fontweight="medium", x=0.02, ha="left")
    fig.text(
        0.02,
        0.90,
        "15 November 2026. A query stays in one split, and every pair of that query stays with it.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    finish(fig, path)


def split_gap_synthetic(path, query_ids, pair_count) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    x = np.arange(len(SPLITS))
    width = 0.36
    target = {"train": 80.0, "validation": 10.0, "test": 10.0}
    for ax, dataset in zip(axes, DATASETS):
        q_total = queries(query_ids, dataset, source="synthetic")
        p_total = pairs(pair_count, dataset, source="synthetic")
        q_share = [
            100 * queries(query_ids, dataset, split, "synthetic") / q_total for split in SPLITS
        ]
        p_share = [
            100 * pairs(pair_count, dataset, split, "synthetic") / p_total for split in SPLITS
        ]
        q_gap = [share - target[split] for share, split in zip(q_share, SPLITS)]
        p_gap = [share - target[split] for share, split in zip(p_share, SPLITS)]
        left = ax.bar(x - width / 2, q_gap, width, color=QUERY, label="Queries", zorder=3)
        right = ax.bar(x + width / 2, p_gap, width, color=PAIR, label="Pairs", zorder=3)
        ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
        span = max(abs(value) for value in q_gap + p_gap)
        ax.set_ylim(-span * 1.85, span * 1.85)
        for bar, share in zip(list(left) + list(right), q_share + p_share):
            height = bar.get_height()
            ax.annotate(
                f"{share:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4 if height >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
                color=INK,
            )
        ax.set_xticks(x, [SPLIT_LABELS[split] for split in SPLITS])
        ax.set_ylabel("Percentage points from 80 / 10 / 10")
        panel_title(ax, NAMES[dataset])
        bare(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Synthetic split against the 80 / 10 / 10 rule", fontsize=16, fontweight="medium", x=0.02, ha="left")
    fig.text(
        0.02,
        0.90,
        "Bar height is the gap from the rule. The label is the actual share of synthetic queries or pairs.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    finish(fig, path)


def split_gap_coffman(path, pair_count) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    x = np.arange(len(DATASETS))
    width = 0.24
    topic_share = {"train": 100 * 29 / 50, "validation": 100 * 4 / 50, "test": 100 * 17 / 50}
    colors = {"train": "#0f6e56", "validation": "#1d4e89", "test": "#c2410c"}
    gaps = {}
    shares = {}
    counts = {}
    for split in SPLITS:
        shares[split] = []
        counts[split] = []
        gaps[split] = []
        for dataset in DATASETS:
            count = pairs(pair_count, dataset, split, "coffman")
            total = pairs(pair_count, dataset, source="coffman")
            share = 100 * count / total
            shares[split].append(share)
            counts[split].append(count)
            gaps[split].append(share - topic_share[split])
    span = max(abs(value) for split in SPLITS for value in gaps[split])
    ax.set_ylim(-span * 1.7, span * 1.7)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
    for offset, split in zip((-width, 0, width), SPLITS):
        bars = ax.bar(
            x + offset,
            gaps[split],
            width,
            color=colors[split],
            label=f"{SPLIT_LABELS[split]} topics are {topic_share[split]:.0f}%",
            zorder=3,
        )
        for bar, share, count in zip(bars, shares[split], counts[split]):
            height = bar.get_height()
            ax.annotate(
                f"{share:.2f}%\n{count:,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4 if height >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(x, [NAMES[dataset] for dataset in DATASETS])
    ax.set_ylabel("Pair share minus topic share, percentage points")
    bare(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.suptitle("Coffman pairs against the topic split", fontsize=16, fontweight="medium", x=0.02, ha="left")
    fig.text(
        0.02,
        0.02,
        "Topics are 29 / 4 / 17 on every database. The label is the share of Coffman pairs, then the pair count.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    finish(fig, path)


def source_share(path, pair_count) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    x = np.arange(len(SPLITS))
    for ax, dataset in zip(axes, DATASETS):
        synthetic = [pairs(pair_count, dataset, split, "synthetic") for split in SPLITS]
        coffman = [pairs(pair_count, dataset, split, "coffman") for split in SPLITS]
        totals = [left + right for left, right in zip(synthetic, coffman)]
        syn_pct = [100 * left / total for left, total in zip(synthetic, totals)]
        cof_pct = [100 * right / total for right, total in zip(coffman, totals)]
        ax.bar(x, syn_pct, color=SYN, label="Synthetic", zorder=3)
        ax.bar(x, cof_pct, bottom=syn_pct, color=COF, label="Coffman", zorder=3)
        for index, (syn, cof, syn_n, cof_n) in enumerate(zip(syn_pct, cof_pct, synthetic, coffman)):
            ax.text(index, syn / 2, f"{syn:.2f}%\n{syn_n:,}", ha="center", va="center", fontsize=8, color="white")
            if cof < 20:
                continue
            ax.text(
                index,
                syn + cof / 2,
                f"{cof:.2f}%\n{cof_n:,}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
            )
        ax.set_ylim(0, 116)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticks(x, [SPLIT_LABELS[split] for split in SPLITS])
        ax.set_ylabel("Share of pairs")
        panel_title(ax, NAMES[dataset])
        bare(ax)
        ax.yaxis.grid(False)
        for index, (syn, cof, cof_n) in enumerate(zip(syn_pct, cof_pct, coffman)):
            if cof >= 20:
                continue
            ax.annotate(
                f"{cof:.2f}%\n{cof_n:,}",
                xy=(index, 100),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=COF,
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.90), ncol=2)
    fig.suptitle("Synthetic and Coffman share of each split", fontsize=16, fontweight="medium", x=0.02, ha="left")
    fig.text(
        0.02,
        0.90,
        "Each column is 100 percent of the pairs in that split. A Coffman share under 20 percent is printed above the bar.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    finish(fig, path)


def element_share(path, pair_count) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.8), sharey=True)
    x = np.arange(len(SPLITS))
    colors = {"relation": REL_C, "attribute": ATTR_C, "value": VAL_C}
    for ax, dataset in zip(axes, DATASETS):
        counts = {
            element: [pairs(pair_count, dataset, split, element=element) for split in SPLITS]
            for element in ELEMENTS
        }
        totals = [sum(counts[element][index] for element in ELEMENTS) for index in range(3)]
        bottom = np.zeros(3)
        for element in ELEMENTS:
            heights = [100 * counts[element][index] / totals[index] for index in range(3)]
            ax.bar(x, heights, bottom=bottom, color=colors[element], label=ELEMENT_LABELS[element], zorder=3)
            for index, height in enumerate(heights):
                if height < 6:
                    continue
                label = f"{height:.2f}%  {counts[element][index]:,}"
                ax.text(
                    index,
                    bottom[index] + height / 2,
                    label if height < 22 else f"{height:.2f}%\n{counts[element][index]:,}",
                    ha="center",
                    va="center",
                    fontsize=7 if height < 22 else 9,
                    color="white",
                )
            bottom += heights
        ax.set_ylim(0, 100)
        ax.set_xticks(x, [SPLIT_LABELS[split] for split in SPLITS])
        ax.set_ylabel("Share of pairs")
        panel_title(ax, NAMES[dataset])
        bare(ax)
        ax.yaxis.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.90), ncol=3)
    fig.suptitle("Relation, attribute, and value share of each split", fontsize=16, fontweight="medium", x=0.02, ha="left")
    fig.text(
        0.02,
        0.90,
        "Both sources are included. Relation is the bottom band, attribute is the middle band, and value is the large upper band.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    finish(fig, path)


def category_counts(path, pair_count) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    x = np.arange(len(DATASETS))
    width = 0.24
    colors = {"relation": REL_C, "attribute": ATTR_C, "value": VAL_C}
    for ax, label, title in (
        (axes[0], 1, "Relevant"),
        (axes[1], 0, "Irrelevant"),
    ):
        drawn = []
        for offset, element in zip((-width, 0, width), ELEMENTS):
            values = [pairs(pair_count, dataset, element=element, label=label) for dataset in DATASETS]
            bars = ax.bar(x + offset, values, width, color=colors[element], label=ELEMENT_LABELS[element], zorder=3)
            drawn.extend(bars)
        label_vertical(ax, drawn, gap=0.08)
        ax.set_xticks(x, [NAMES[dataset] for dataset in DATASETS])
        ax.set_ylabel("Pairs, all splits")
        panel_title(ax, title)
        bare(ax)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Relevant and irrelevant pairs by element", fontsize=16, fontweight="medium", x=0.02, ha="left")
    fig.text(
        0.02,
        0.90,
        "Train, validation, and test together. The two panels have separate scales.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    finish(fig, path)


def train_elements(path, pair_count) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    x = np.arange(len(ELEMENTS))
    width = 0.36
    for column, dataset in enumerate(DATASETS):
        for row, source in enumerate(SOURCES):
            ax = axes[row][column]
            relevant = [pairs(pair_count, dataset, "train", source, element, 1) for element in ELEMENTS]
            irrelevant = [pairs(pair_count, dataset, "train", source, element, 0) for element in ELEMENTS]
            left = ax.bar(x - width / 2, relevant, width, color=POS, label="Relevant", zorder=3)
            right = ax.bar(x + width / 2, irrelevant, width, color="#a8a29e", label="Irrelevant", zorder=3)
            label_vertical(ax, list(left) + list(right))
            ax.set_xticks(x, [ELEMENT_LABELS[element] for element in ELEMENTS])
            source_name = "Synthetic" if source == "synthetic" else "Coffman"
            panel_title(ax, f"{NAMES[dataset]} · {source_name}")
            bare(ax)
            if row == 0 and column == 0:
                ax.legend(frameon=False, loc="upper left")
            if column == 0:
                ax.set_ylabel("Train pairs")
    fig.suptitle("Training pairs by element and label", fontsize=16, fontweight="medium", x=0.02, ha="left")
    fig.text(
        0.02,
        0.935,
        "Each panel has its own scale, so the smaller Coffman row can be read. These are the pairs that updated the weights.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    finish(fig, path)


def heatmap(path, pair_count) -> None:
    rows = []
    row_labels = []
    for dataset in DATASETS:
        for split in SPLITS:
            for source in SOURCES:
                row = []
                for element in ELEMENTS:
                    for label in (1, 0):
                        row.append(pairs(pair_count, dataset, split, source, element, label))
                rows.append(row)
                source_name = "synthetic" if source == "synthetic" else "Coffman"
                row_labels.append(f"{NAMES[dataset]}  {SPLIT_LABELS[split]}  {source_name}")
    data = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(12.6, 8.6))
    image = ax.imshow(np.log1p(data), cmap=CMAP, aspect="auto")
    ceiling = np.log1p(data.max())
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if np.log1p(value) > 0.58 * ceiling else INK
            ax.text(j, i, f"{int(value):,}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(
        np.arange(6),
        ["Relation\nrelevant", "Relation\nirrelevant", "Attribute\nrelevant", "Attribute\nirrelevant", "Value\nrelevant", "Value\nirrelevant"],
    )
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("log(1 + count)", color=MUTED)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelcolor=MUTED)
    fig.suptitle("Every pair count in the 15 November corpus", fontsize=16, fontweight="medium", x=0.06, ha="left")
    fig.text(
        0.06,
        0.02,
        "Color is logarithmic so a count of 3 stays visible beside a count of 17,756. The numeral is the exact pair count.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    finish(fig, path)


def pairs_per_query(path, per_query) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0))
    for column, dataset in enumerate(DATASETS):
        for row, source in enumerate(SOURCES):
            ax = axes[row][column]
            values = np.array(
                [
                    bucket["pairs"]
                    for (ds, src, _query_id), bucket in per_query.items()
                    if ds == dataset and src == source
                ]
            )
            levels, counts = np.unique(values, return_counts=True)
            print(NAMES[dataset], source, list(zip(levels.astype(int).tolist(), counts.astype(int).tolist())))
            ax.bar(levels, counts, width=2.2, color=SYN if source == "synthetic" else COF, zorder=3)
            median = float(np.median(values))
            ax.axvline(median, color=INK, linewidth=1.0, linestyle="--", zorder=4)
            ax.set_ylim(0, max(counts) * 1.7)
            for index, (level, count) in enumerate(zip(levels, counts)):
                if source == "coffman" and count < 3:
                    continue
                on_median = abs(float(level) - median) < 0.5
                lift = 12 if source == "coffman" and index % 2 else 4
                ax.annotate(
                    f"{int(count)}",
                    xy=(level, count),
                    xytext=(-12 if on_median else 0, lift),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK,
                    bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"}
                    if on_median
                    else None,
                    clip_on=False,
                )
            source_name = "Synthetic" if source == "synthetic" else "Coffman"
            ax.set_title(
                f"{NAMES[dataset]} · {source_name}\n"
                f"n={len(values):,}    median {median:.0f}    min {int(values.min())}    max {int(values.max())}",
                loc="left",
                pad=8,
                fontweight="medium",
                fontsize=11,
            )
            ax.set_xlabel("Pairs from one query")
            if column == 0:
                ax.set_ylabel("Queries")
            bare(ax)
    fig.suptitle("How many pairs one query produces", fontsize=16, fontweight="medium", x=0.02, ha="left")
    fig.text(
        0.02,
        0.915,
        "Exact pair counts. The numeral is how many queries produced that many pairs. An unlabeled Coffman bar is one or two queries. The dashed line is the median.",
        fontsize=10,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    finish(fig, path)


def main() -> None:
    style()
    pair_count, query_ids, per_query = load()
    expected = {
        "mondial": (31840, 1229),
        "imdb": (7456, 316),
        "wikipedia": (5076, 228),
    }
    for dataset, (pair_total, query_total) in expected.items():
        got_pairs = pairs(pair_count, dataset)
        got_queries = queries(query_ids, dataset)
        if (got_pairs, got_queries) != (pair_total, query_total):
            raise SystemExit(f"{dataset}: got {got_pairs} pairs and {got_queries} queries")
    for bucket in per_query.values():
        if bucket["relevant"] * 4 != bucket["pairs"]:
            raise SystemExit("a query is not exactly one relevant pair to three irrelevant pairs")
    grouped_counts(OUT / "queries-by-split.png", query_ids, pair_count, "queries")
    grouped_counts(OUT / "pairs-by-split.png", query_ids, pair_count, "pairs")
    split_gap_synthetic(OUT / "synthetic-split-share.png", query_ids, pair_count)
    split_gap_coffman(OUT / "coffman-split-share.png", pair_count)
    source_share(OUT / "pair-source-share.png", pair_count)
    element_share(OUT / "element-share.png", pair_count)
    category_counts(OUT / "category-counts.png", pair_count)
    train_elements(OUT / "train-elements.png", pair_count)
    heatmap(OUT / "relevance-heatmap.png", pair_count)
    pairs_per_query(OUT / "pairs-per-query.png", per_query)


if __name__ == "__main__":
    main()
