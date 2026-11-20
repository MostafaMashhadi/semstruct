"""Write the ICN retrieval note and its charts from the saved metrics."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mondial", "imdb", "wikipedia")
NAMES = {"mondial": "Mondial", "imdb": "IMDb", "wikipedia": "Wikipedia"}
PUBLISHED = {
    "mondial": [("Efficient", 21, 0.514), ("Effective", 22, 0.495), ("SPARK", 27, 0.607), ("Cover density", 36, 0.804)],
    "imdb": [("Efficient", 4, 0.139), ("Effective", 3, 0.072), ("SPARK", 3, 0.115), ("Cover density", 13, 0.313)],
    "wikipedia": [("Efficient", 16, 0.485), ("Effective", 23, 0.594), ("SPARK", 19, 0.565), ("Cover density", 32, 0.674)],
}


def render(root: Path | None = None) -> None:
    root = root or ROOT
    records = []
    for dataset in DATASETS:
        path = root / "outputs" / dataset / "icn" / "metrics.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if len(records) != len(DATASETS):
        return
    figures = root / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _charts(records, figures)
    (root / "docs" / "icn.md").write_text(_markdown(records), encoding="utf-8")


def _markdown(records: list[dict]) -> str:
    lines = [
        "# Integrated candidate networks",
        "",
        "Each Coffman topic is scored by the DistilBERT fold that held it out. "
        "A keyword on a primary network receives one relevance probability: a relation or attribute uses its schema text, "
        "and a value keyword uses the highest probability among the cells of its stored attribute. "
        "The network score is the mean of those probabilities. "
        "Networks are ordered by that score. A lower score does not remove a network. "
        "The search stops after 1,000 distinct answers. "
        "A tuple set or a partial join that was already fetched is reused, and only the next relation is joined. "
        "A joined row is relevant when its set of search ids equals one Coffman answer. "
        "One extra id or one missing id makes the row wrong. "
        "The files are `outputs/<dataset>/icn/results.json` and `outputs/<dataset>/icn/metrics.json`.",
        "",
        "| Database | Topics | Precision | Recall | F1 | Networks evaluated | Query time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in records:
        lines.append(
            f"| {NAMES[item['dataset']]} | {item['topics']} | {item['precision']:.3f} | "
            f"{item['recall']:.3f} | {item['f1']:.3f} | {item['networks_evaluated']:,} | "
            f"{item['query_ms']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "| Database | Top-1 | MRR | MAP | Scoring | SQL |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in records:
        lines.append(
            f"| {NAMES[item['dataset']]} | {item['top1']} | {item['mrr']:.3f} | "
            f"{item['map']:.3f} | {item['scoring_ms']:.1f} ms | {item['sql_ms']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "Precision, recall, and F1 are unweighted means over the topics. "
            "A topic with no returned row has precision 0. A topic with no judged answer is left out of recall and F1. "
            "Networks evaluated counts primary networks whose join started before 1,000 answers existed. "
            "Query time is the mean of scoring time plus the incremental SQL time. "
            "Top-1, MRR, and MAP stay beside those figures. They are not the paper's precision, recall, and F1.",
            "",
            "The paper's baseline table has five rows: ICN with no DistilBERT scores, Key2Vec plus ICN, "
            "pre-trained DistilBERT plus ICN, fine-tuned DistilBERT plus ICN with the scores ignored when ordering, and SemStruct. "
            "This repository has measured only SemStruct. The other four rows are not filled from published papers.",
            "",
            "The next table is not that baseline. "
            "Its four older systems were not run here. "
            "The numbers are copied from Joel Coffman and Alfred C. Weaver, "
            "*Structured Data Retrieval Using Cover Density Ranking*, KEYS 2010, pages 1–6, Table 2. "
            "That table counts top-1 results and reciprocal rank. "
            "It is not this program's exact search-id rule, so the numbers are not placed in the paper's precision, recall, or F1 columns.",
            "",
            "| Database | System | Top-1 | Reciprocal rank |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for dataset in DATASETS:
        ours = next(item for item in records if item["dataset"] == dataset)
        lines.append(
            f"| {NAMES[dataset]} | SemStruct | {ours['top1']} | {ours['mrr']:.3f} |"
        )
        for name, top1, rank in PUBLISHED[dataset]:
            lines.append(f"| {NAMES[dataset]} | {name} | {top1} | {rank:.3f} |")
    lines.extend(
        [
            "",
            "The comparison charts use the SemStruct numbers. The count or the rounded metric is printed on each bar.",
            "",
            "## Figures",
            "",
            "![Top-1 relevant queries](figures/icn-top1.png)",
            "",
            "![Mean reciprocal rank](figures/icn-mrr.png)",
            "",
            "![Mean average precision](figures/icn-map.png)",
            "",
        ]
    )
    return "\n".join(lines)


def _charts(records: list[dict], figures: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [NAMES[item["dataset"]] for item in records]
    _bar(plt, figures / "icn-top1.png", labels, [item["top1"] for item in records], "Top-1 relevant queries", "Queries", "#1f77b4", integer=True)
    _bar(plt, figures / "icn-mrr.png", labels, [item["mrr"] for item in records], "Mean reciprocal rank", "MRR", "#ff7f0e", integer=False)
    _bar(plt, figures / "icn-map.png", labels, [item["map"] for item in records], "Mean average precision", "MAP", "#2ca02c", integer=False)


def _bar(plt, path: Path, labels: list[str], heights: list[float], title: str, ylabel: str, color: str, integer: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    positions = list(range(len(labels)))
    bars = ax.bar(positions, heights, color=color)
    for bar, height in zip(bars, heights):
        text = f"{int(height)}" if integer else f"{height:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, height, text, ha="center", va="bottom")
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(heights, default=1) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    render()
