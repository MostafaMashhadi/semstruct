"""Write the primary-CN report and its charts from the saved networks."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mondial", "imdb", "wikipedia")
NAMES = {"mondial": "Mondial", "imdb": "IMDb", "wikipedia": "Wikipedia"}


def render(root: Path | None = None) -> None:
    root = root or ROOT
    records = []
    for dataset in DATASETS:
        path = root / "outputs" / dataset / "networks" / "primary-cns.json"
        if path.is_file():
            records.append((dataset, json.loads(path.read_text(encoding="utf-8"))))
    if not records:
        return
    figures = root / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _charts(records, figures)
    (root / "docs" / "primary-cns.md").write_text(_markdown(records), encoding="utf-8")


def _summarize(dataset: str, document: dict) -> dict:
    queries = document["queries"]
    per_topic = [len(item["networks"]) for item in queries]
    sizes: Counter[int] = Counter()
    roles: Counter[str] = Counter()
    free_nodes = 0
    match_counts: list[int] = []
    for query in queries:
        for network in query["networks"]:
            sizes[len(network["nodes"])] += 1
            for node in network["nodes"]:
                if not node["keywords"]:
                    free_nodes += 1
                for keyword in node["keywords"]:
                    roles[keyword["role"]] += 1
                    if keyword["role"] == "value":
                        match_counts.append(int(keyword["match_count"]))
    return {
        "dataset": dataset,
        "topics": len(queries),
        "with_networks": sum(count > 0 for count in per_topic),
        "without_networks": sum(count == 0 for count in per_topic),
        "networks": sum(per_topic),
        "minimum": min(per_topic) if per_topic else 0,
        "median": statistics.median(per_topic) if per_topic else 0,
        "maximum": max(per_topic) if per_topic else 0,
        "empty": [
            {"topic_id": item["topic_id"], "query": item["query"]}
            for item in queries
            if not item["networks"]
        ],
        "sizes": sizes,
        "roles": roles,
        "free_nodes": free_nodes,
        "match_median": int(statistics.median(match_counts)) if match_counts else 0,
        "match_maximum": max(match_counts) if match_counts else 0,
        "per_topic": per_topic,
        "match_counts": match_counts,
    }


def _markdown(records: list[tuple[str, dict]]) -> str:
    summaries = [_summarize(dataset, document) for dataset, document in records]
    lines = [
        "# Primary candidate networks",
        "",
        "A primary candidate network is a join tree of tuple sets on the foreign-key graph. "
        "Tuple sets come from the database: a relation or attribute whose name matches a keyword, "
        "or the cells of one attribute that contain it. "
        "The tree has at most five relations. Every keyword appears once. "
        "Value keywords that share a relation must occur together in one row, because that row is their tuple set. "
        "A relation with no keyword is kept only when it connects keyword relations, and it is never a leaf. "
        "The same tree and the same keyword assignment are stored once. "
        "The files are `outputs/<dataset>/networks/primary-cns.json`. "
        "The IMDb file is about 162 MB, above GitHub's limit, so it stays on disk and is not in the repository. "
        "Regenerate it with `semstruct cns --dataset imdb`.",
        "",
        "| Database | Topics | With networks | Without networks | Networks | Minimum | Median | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {NAMES[item['dataset']]} | {item['topics']} | {item['with_networks']} | "
            f"{item['without_networks']} | {item['networks']:,} | {item['minimum']} | "
            f"{_format_stat(item['median'])} | {item['maximum']:,} |"
        )
    lines.extend(
        [
            "",
            "| Database | Size 1 | Size 2 | Size 3 | Size 4 | Size 5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summaries:
        sizes = item["sizes"]
        lines.append(
            "| {name} | {a} | {b} | {c} | {d} | {e} |".format(
                name=NAMES[item["dataset"]],
                a=f"{sizes[1]:,}",
                b=f"{sizes[2]:,}",
                c=f"{sizes[3]:,}",
                d=f"{sizes[4]:,}",
                e=f"{sizes[5]:,}",
            )
        )
    lines.extend(
        [
            "",
            "| Database | Relation | Attribute | Value | Free relations | Value match median | Value match maximum |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summaries:
        roles = item["roles"]
        lines.append(
            f"| {NAMES[item['dataset']]} | {roles['relation']:,} | {roles['attribute']:,} | "
            f"{roles['value']:,} | {item['free_nodes']:,} | {item['match_median']:,} | "
            f"{item['match_maximum']:,} |"
        )
    lines.extend(
        [
            "",
            "Minimum, median, and maximum count every topic, including topics with no network. "
            "A value match count is the number of cells in that attribute which contain the keyword. "
            "The network stores the count, not the cells.",
            "",
        ]
    )
    for item in summaries:
        if not item["empty"]:
            lines.append(f"{NAMES[item['dataset']]}: every topic has at least one network.")
            continue
        listed = ", ".join(f"{topic['topic_id']} ({topic['query']})" for topic in item["empty"])
        lines.append(f"{NAMES[item['dataset']]} topics with no network: {listed}.")
    lines.extend(
        [
            "",
            "The comparison charts use a logarithmic vertical axis so the smaller databases stay visible. "
            "The number printed on each bar is the exact count.",
            "",
            "## Figures",
            "",
            "![Networks per database](figures/cn-networks.png)",
            "",
            "![Network size](figures/cn-size.png)",
            "",
            "![Keyword roles](figures/cn-roles.png)",
            "",
            "![Networks per topic](figures/cn-per-topic.png)",
            "",
            "![Value tuple-set size](figures/cn-match-count.png)",
            "",
        ]
    )
    return "\n".join(lines)


def _charts(records: list[tuple[str, dict]], figures: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = [_summarize(dataset, document) for dataset, document in records]
    _bar(
        plt,
        figures / "cn-networks.png",
        [NAMES[item["dataset"]] for item in summaries],
        [item["networks"] for item in summaries],
        "Primary candidate networks",
        "Networks",
        "#1f77b4",
    )
    _grouped(
        plt,
        figures / "cn-size.png",
        summaries,
        [1, 2, 3, 4, 5],
        lambda item, size: item["sizes"][size],
        [str(size) for size in range(1, 6)],
        "Network size",
        "Networks",
        "Relations",
    )
    _grouped(
        plt,
        figures / "cn-roles.png",
        summaries,
        ["relation", "attribute", "value"],
        lambda item, role: item["roles"][role],
        ["Relation", "Attribute", "Value"],
        "Keyword roles",
        "Keyword assignments",
        "Role",
    )
    _per_topic(plt, figures / "cn-per-topic.png", summaries)
    _match_counts(plt, figures / "cn-match-count.png", summaries)


def _format_stat(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.1f}"


def _label_bars(ax, bars, *, zero_at: float, fontsize: int = 7) -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                zero_at,
                "0",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=90,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.18,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
        )


def _bar(plt, path: Path, labels: list[str], heights: list[int], title: str, ylabel: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.6))
    positions = list(range(len(labels)))
    bars = ax.bar(positions, heights, color=color)
    ax.set_yscale("log")
    _label_bars(ax, bars, zero_at=0.6)
    ax.set_xticks(positions, labels)
    ax.set_ylabel(f"{ylabel} (log scale)")
    ax.set_title(title)
    ax.set_ylim(0.5, max(heights, default=1) * 8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _grouped(plt, path, summaries, keys, getter, key_labels, title, ylabel, xlabel) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    width = 0.25
    positions = list(range(len(keys)))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    tallest = 1
    for offset, item in enumerate(summaries):
        heights = [getter(item, key) for key in keys]
        tallest = max(tallest, max(heights, default=0))
        bars = ax.bar(
            [position + (offset - 1) * width for position in positions],
            heights,
            width=width,
            label=NAMES[item["dataset"]],
            color=colors[offset % len(colors)],
        )
        _label_bars(ax, bars, zero_at=0.6)
    ax.set_yscale("log")
    ax.set_xticks(positions, key_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ylabel} (log scale)")
    ax.set_title(title)
    ax.set_ylim(0.5, tallest * 8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _per_topic(plt, path: Path, summaries: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(summaries), figsize=(7.4 * len(summaries), 5.6), squeeze=False)
    for axis, item in zip(axes[0], summaries):
        heights = item["per_topic"]
        positions = list(range(len(heights)))
        bars = axis.bar(positions, heights, color="#1f77b4", width=0.72)
        axis.set_yscale("log")
        positive = [height for height in heights if height]
        floor = min(positive) if positive else 1
        bottom = 0.3 if any(height == 0 for height in heights) else floor * 0.7
        _label_bars(axis, bars, zero_at=bottom * 1.15, fontsize=6)
        axis.set_ylim(bottom, max(heights, default=1) * 8)
        title = (
            f"{NAMES[item['dataset']]}\n"
            f"min {_format_stat(item['minimum'])}, median {_format_stat(item['median'])}, "
            f"max {item['maximum']:,}"
        )
        if item["without_networks"]:
            title += f"\n{item['without_networks']} topics have no network"
        axis.set_title(title)
        axis.set_xlabel("Coffman topic")
        axis.set_ylabel("Networks (log scale)")
        ticks = [index for index in positions if (index + 1) % 5 == 0]
        axis.set_xticks(ticks, [str(index + 1) for index in ticks])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _match_counts(plt, path: Path, summaries: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(summaries), figsize=(5.2 * len(summaries), 5.2), squeeze=False)
    for axis, item in zip(axes[0], summaries):
        values = item["match_counts"] or [1]
        highest = max(values)
        edges = [1]
        while edges[-1] <= highest:
            edges.append(edges[-1] * 2)
        counts = [0] * (len(edges) - 1)
        for value in values:
            power = 0
            while edges[power + 1] <= value:
                power += 1
            counts[power] += 1
        left = edges[:-1]
        widths = [right - start for start, right in zip(edges, edges[1:])]
        bars = axis.bar(left, counts, width=widths, align="edge", color="#c2410c")
        axis.set_xscale("log")
        for bar, height, start, right in zip(bars, counts, edges, edges[1:]):
            if not height:
                continue
            axis.text(
                (start * right) ** 0.5,
                height * 1.08,
                f"{height:,}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        axis.set_xlim(1, edges[-1])
        axis.set_ylim(0, max(counts, default=1) * 1.45)
        axis.set_title(
            f"{NAMES[item['dataset']]}\nmedian {item['match_median']:,}, max {item['match_maximum']:,}"
        )
        axis.set_xlabel("Cells in a value tuple set")
        axis.set_ylabel("Keyword assignments")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    render()
