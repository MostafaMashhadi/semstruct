"""Rewrite the project reports from the latest prepare stats and fold metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from semstruct.paths import DATASETS, output_dir, project_root
from semstruct.results import export_training_data

NAMES = {"mondial": "Mondial", "imdb": "IMDb", "wikipedia": "Wikipedia"}
WRITER_PARAGRAPH = (
    "The query texts are the synthetic queries plus the 50 Coffman topics. "
    "A label is positive only for the index hit the topic or the generator assigns to that keyword. "
    "Training keeps every schema hit and at most 10 other value hits per keyword, "
    "and the reported precision and recall score every index hit. "
    "The pre-trained numbers are the frozen [CLS] cosine."
)


def refresh_reports(root: Path | None = None) -> None:
    root = root or project_root()
    export_training_data(root)
    write_reports(root)
    script = root / "scripts" / "render_sample_distribution.py"
    if script.is_file():
        subprocess.run([sys.executable, str(script)], cwd=root, check=False)


def write_reports(root: Path | None = None) -> None:
    """Rewrite the markdown reports from the saved metrics. Does not rebuild the export."""
    root = root or project_root()
    records = {dataset: _dataset_record(dataset, root) for dataset in DATASETS}
    _write(root / "README.md", _readme(records))
    _write(root / "docs" / "training.md", _training(records))
    _write(root / "docs" / "datasets.md", _datasets(records))
    _write(root / "docs" / "sample-distribution.md", _distribution(records))
    _write(root / "docs" / "reproduction.md", _reproduction())
    _write(root / "docs" / "slides" / "semstruct-progress.md", _slides(records))
    _write(root / "docs" / "logs" / "latest-run.md", _run_log(records, root))


def _dataset_record(dataset: str, root: Path) -> dict:
    stats_path = output_dir(dataset, root) / "prepare_stats.json"
    metrics_path = output_dir(dataset, root) / "cv-metrics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    return {"stats": stats, "metrics": metrics}


def _readme(records: dict[str, dict]) -> str:
    return f"""# SemStruct

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.44%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![DistilBERT](https://img.shields.io/badge/Model-distilbert--base--uncased-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/distilbert/distilbert-base-uncased)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

SemStruct is a semantic-structural framework for relational keyword search. A user types keywords. The system has to decide which relations, attributes, and cell values those keywords refer to, and later which joined tuples answer the query.

This repository implements the offline half: one English DistilBERT ranker per database, measured with 5-fold Coffman cross-validation. The model reads one keyword, the full query, and a textual description of one element, and estimates whether that element is the intended hit. Integrated Candidate Network generation and SQL execution are specified in [docs/architecture.md](docs/architecture.md) and are not implemented here.

## This version

For Mondial, the IMDb 2009 subset, and the Wikipedia 2008 subset:

1. Load the PostgreSQL benchmark dump into SQLite.
2. Build query classes from the foreign-key graph and sample synthetic keyword queries.
3. Add the 50 Coffman topics. Each fold tests 10 topics, uses 8 to choose the checkpoint and the threshold, and trains on the other 32.
4. Fine-tune `distilbert/distilbert-base-uncased` with one linear unit and binary cross-entropy. Every synthetic instance trains every fold.
5. Report precision, recall, and F1 of the retrieved index hits, plus the mean scoring time.

{WRITER_PARAGRAPH}

## Ranker results

Pooled over all 50 Coffman topics. These are keyword-element decisions. ICN, Key2Vec, network counts, and end-to-end time are not in this table. Wikipedia is measured beside Mondial and IMDb.

{_result_section(records)}

## Sample counts

{_count_table(records)}

## Figures

![Pairs by source](docs/figures/pairs-by-source.png)

![Element types](docs/figures/element-share.png)

![Coffman pairs by test fold](docs/figures/coffman-folds.png)

![Pairs per keyword](docs/figures/pairs-per-keyword.png)

## Data

The three databases are the evaluation subsets of the Coffman and Weaver relational keyword-search benchmark, not dumps collected for this project and not the full IMDb or Wikipedia archives.

- Paper: Joel Coffman and Alfred C. Weaver, *A Framework for Evaluating Database Keyword Search Strategies*, CIKM 2010. [doi:10.1145/1871437.1871531](https://doi.org/10.1145/1871437.1871531)
- Files: Coffman and Weaver, *Benchmark for Relational Keyword Search*, University of Virginia Dataverse, 2017. [doi:10.18130/V3/KEVCF8](https://doi.org/10.18130/V3/KEVCF8)

`semstruct download` fetches the Mondial, IMDb 2009 subset, and Wikipedia 2008 subset schema, data, and topic files from that Dataverse record and checks their MD5 sums. IMDb topics are the corrected V2 assessments. The benchmark is academic and non-commercial. IMDb rows remain courtesy of IMDb. This repository's MIT license covers the code only. Filenames, checksums, and the manual download list are in [docs/datasets.md](docs/datasets.md).

## Quick start

Python 3.10 or newer. The same commands run on macOS, Windows, and Linux. Per-platform install notes are in [docs/reproduction.md](docs/reproduction.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate with `.venv\\Scripts\\activate`.

```bash
semstruct download --dataset all
semstruct prepare --dataset all
semstruct train --dataset mondial --fold 0
semstruct evaluate --dataset mondial --fold 0
semstruct train --dataset all --fold all
semstruct evaluate --dataset all --fold all
```

Checkpoints are written to `outputs/models/<dataset>/fold-<k>/` and are not uploaded. Pooled ranker numbers go to `outputs/<dataset>/cv-metrics.json`.

## Score a query

Load the fold checkpoint for the database you are querying. Pass the keyword, the full query, and one element card. The score is the sigmoid probability that the element is the intended hit.

```python
from semstruct.device import select_device
from semstruct.model.classifier import load_ranker
from semstruct.model.scoring import relevance_probabilities

tokenizer, model = load_ranker("outputs/models/mondial/fold-0")
scores = relevance_probabilities(
    model,
    tokenizer,
    ["france"],
    ["france"],
    ["value: France | attribute: country.name | relation: country"],
    batch_size=16,
    device=select_device(),
    max_length=128,
)
print(scores[0])
```

## Documentation

| Document | What it covers |
| --- | --- |
| [Architecture](docs/architecture.md) | Offline adaptation, and the online stage that is not built yet |
| [Datasets](docs/datasets.md) | Coffman and Weaver subsets, citation, and what may not be redistributed |
| [Training](docs/training.md) | Keyword instances, folds, loss, and the pooled numbers |
| [Sample distribution](docs/sample-distribution.md) | Pair counts and the charts in `docs/figures/` |
| [Reproduction](docs/reproduction.md) | macOS, Windows, Linux, and CUDA install paths |
| [Latest run](docs/logs/latest-run.md) | The plain log of the current run |
| [Slides](docs/slides/semstruct-progress.md) | Short Marp deck of what this version implements |

## Layout

```text
configs/                         experiment settings
docs/                            architecture, data, training, reproduction, run log
src/semstruct/                   download, preparation, training, evaluation
tests/                           parser, labeling, and metric tests
data/raw/                        benchmark dumps, not committed
data/processed/                  SQLite databases and training pairs, not committed
outputs/models/<dataset>/fold-k  fine-tuned checkpoints, not committed
outputs/<dataset>                metrics and run configuration
exports/training-data/           current pair export, not committed
```

## Citation

Code in this repository is MIT, copyright Mostafa Mashhadizadeh. The benchmark data is not. Cite Coffman and Weaver when you use the databases, topics, or relevance assessments. See [CITATION.cff](CITATION.cff) and [docs/references.bib](docs/references.bib).

## License

The software is released under the MIT license in [LICENSE](LICENSE). Mondial, IMDb, and Wikipedia dumps stay under the terms of the Coffman and Weaver benchmark and the upstream data owners. Do not redistribute those dumps.
"""


def _training(records: dict[str, dict]) -> str:
    return f"""# Training

The model is `distilbert/distilbert-base-uncased`. After fine-tuning, each database has five checkpoints in `outputs/models/<dataset>/fold-<k>/`. The run does not upload those checkpoints.

## Keyword instance

A row is one keyword `k`, the full query `q`, one element `e`, and a label. The model reads `[CLS] k [SEP] q [SEP] Te [SEP]`, truncated at 128 tokens. The pooled vector is the `[CLS]` hidden state. One linear unit and a sigmoid produce the relevance probability. The loss is the unweighted mean binary cross-entropy.

Element text follows the draft. A relation contributes its name, its comment when the dump has one, its attribute names, and the relations joined to it. An attribute contributes its name, type, key role, comment when present, and its relation. A value contributes the cell, its attribute, and its relation. These dumps have no relation comments, so that line is absent.

## Labels

The label is 1 when `e` is an index hit that the intended interpretation assigns to `k`.

For a synthetic query, the intended hits are the cell, the attribute, and the relation that supplied `k`. For a Coffman topic, a value is intended when it lies in a judged tuple and contains `k`. An attribute is intended when a judged tuple's cell in that attribute contains `k`, or when `k` matches the attribute name and the attribute belongs to a judged relation. A relation is intended when `k` matches the relation name and that relation is judged.

The other index hits of the same keyword are negatives. Every schema hit is kept. In the training loss only, other value hits are capped at 10 per keyword, drawn with seed 42. Validation and test score every index hit. The threshold is chosen on the full validation list.

A synthetic string that equals any Coffman topic is dropped. Every remaining synthetic instance trains every fold.

## Cross-validation

Seed 42 shuffles the 50 topic ids into 5 folds of 10. Fold `k` tests those 10 topics. Of the other 40, 8 choose the checkpoint and the decision threshold, and 32 train. A test topic never updates that fold's weights.

## Article numbers

{WRITER_PARAGRAPH}

The cells filled here are ranker results for pre-trained DistilBERT and fine-tuned DistilBERT, plus the mean time to score one test query. ICN, Key2Vec, network counts, and end-to-end time stay with the online stage. Wikipedia is measured and kept beside the Mondial and IMDb table.

{_result_section(records)}

{_count_table(records)}
"""


def _datasets(records: dict[str, dict]) -> str:
    return f"""# Datasets

Training and evaluation use the evaluation subsets of the Coffman and Weaver relational keyword-search benchmark.

> Joel Coffman and Alfred C. Weaver. A Framework for Evaluating Database Keyword Search Strategies. CIKM 2010. <https://doi.org/10.1145/1871437.1871531>

The files are distributed by the University of Virginia Dataverse:

> Coffman, Joel; Weaver, Alfred C., 2017. Benchmark for Relational Keyword Search. <https://doi.org/10.18130/V3/KEVCF8>

`semstruct download` asks the Dataverse API for these evaluation files and checks their MD5 sums.

| Dataset | Schema | Data | Topics |
| --- | --- | --- | --- |
| Mondial | `mondial-schema.pgsql.bz2` | `mondial-data.pgsql.bz2` | `Mondial-topics-qrels.tar.bz2` |
| IMDb | `imdb2009_subset-schema.pgsql.bz2` | `imdb2009_subset-data.pgsql.bz2` | `IMDb-topics-qrels-V2.tar.bz2` |
| Wikipedia | `wikipedia2008_subset-schema.pgsql.bz2` | `wikipedia2008_subset-data.pgsql.bz2` | `Wikipedia-topics-qrels.tar.bz2` |

The full IMDb dump and the full Wikipedia dump are not used. IMDb topics are the corrected V2 assessments.

## Terms

The benchmark may be used for academic, personal, and non-commercial work. It must not be resold or repurposed. IMDb rows were obtained with IMDbPY and remain courtesy of IMDb. This repository's MIT license covers the code only. Do not commit or republish the dumps. `data/raw/` and `data/processed/` are gitignored.

If the API download fails, save the files above into `data/raw/` and run `semstruct prepare`. The loader accepts the filenames exactly as published.

## Databases

Mondial is a geographical schema: countries, cities, provinces, rivers, organizations, and the foreign keys among them. The data file is a sequence of `INSERT` statements.

The IMDb subset is a small slice of the 2009 IMDbPY schema: `title`, `name`, `cast_info`, `char_name`, `movie_info`, and `role_type`. Rows arrive as PostgreSQL `COPY` data. Topics refer to a person relation; in this schema that relation is `name`.

The Wikipedia subset is a MediaWiki schema: `page`, `revision`, `pagecontent`, `pagelinks`, `mwuser`, and `user_groups`. Columns whose names contain `password`, `token`, or `email` are not used as keyword text.

Every table carries `__search_id`. The topic files identify relevant tuples by that integer. The loader stores `semstruct_search_id(search_id, relation)` so a judged tuple can be tied to its relation.

## Topics

Each numbered file in a topic archive has the form:

```text
# keyword query
([search_id, ...], [(source_id, target_id), ...])
```

Lines that start with `#` are comments. A commented-out tuple line is not a relevance judgment. The query is the first comment that is not a dataset banner such as `imdb2009` or `[reference standard]`.

Each database has 50 of these queries. They are dealt into 5 folds of 10. A fold tests its 10 topics, holds out 8 for the checkpoint and the threshold, and trains on the other 32. A synthetic query whose normalized text equals a topic is dropped.

{WRITER_PARAGRAPH}

{_result_section(records)}

{_count_table(records)}
"""


def _distribution(records: dict[str, dict]) -> str:
    prepared = [dataset for dataset, record in records.items() if record["stats"]]
    if not prepared:
        body = "Prepare has not written a pair file yet."
    else:
        body = _count_table(records) + "\n\n" + _source_table(records)
    return f"""# Sample distribution

This note describes the current keyword instances. A row is one keyword, the full query, and one element. Synthetic rows train every fold. Coffman rows carry the test fold of their topic. The row files are in `exports/training-data/`. Those files contain benchmark cell text and are not part of the git history.

{body}

## Figures

![Pairs by source](figures/pairs-by-source.png)

![Element types](figures/element-share.png)

![Coffman pairs by test fold](figures/coffman-folds.png)

![Pairs per keyword](figures/pairs-per-keyword.png)

The figures are rendered from that export by `scripts/render_sample_distribution.py`.
"""


def _reproduction() -> str:
    return """# Reproduction

The commands are the same on macOS, Windows, and Linux. Paths in the code use `pathlib`. Dataset names are lowercase. Text files are UTF-8. PostgreSQL does not need to be installed. The loader reads the plain dumps itself.

Tested on macOS with Python 3.12. The Trainer imports Accelerate, which is listed in `requirements.txt`. Windows and Linux use the same code. Install a CUDA build of PyTorch on NVIDIA machines. An Intel Mac, or any machine without CUDA or MPS, trains on CPU.

## macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Apple Silicon the default PyTorch wheel uses MPS. On Linux without an NVIDIA GPU it uses CPU. After each database the CLI releases the CUDA or MPS cache. That matters on an 8 GB machine.

## Windows

```bat
py -3 -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## NVIDIA CUDA

Install PyTorch from the CUDA wheel index that matches the driver, then install the rest of the requirements. The training arguments enable pinned memory only when CUDA is visible.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Experiment

```bash
semstruct download --dataset all
semstruct prepare --dataset mondial
semstruct train --dataset mondial --fold 0
semstruct evaluate --dataset mondial --fold 0
semstruct train --dataset mondial --fold all
semstruct evaluate --dataset mondial --fold all
```

Repeat for `imdb` and `wikipedia`, one database at a time. `--fold all` runs folds 0 through 4. Each fold writes `outputs/<dataset>/fold-<k>/run.log`, `run_config.json`, and `train_history.json`. Evaluation writes `metrics.json` in that directory and pools finished folds into `outputs/<dataset>/cv-metrics.json`.

`prepare` reuses SQLite unless `--rebuild` is set. It rewrites the pair files, stamps each Coffman topic with its test fold, and copies `prepare_stats.json` to `outputs/<dataset>/`.

## Outputs

| Path | Contents |
| --- | --- |
| `data/processed/<dataset>/pairs.jsonl` | Keyword instances |
| `data/processed/<dataset>/prepare_stats.json` | Counts, categories, and fold roles |
| `outputs/<dataset>/prepare_stats.json` | The same prepare record |
| `outputs/models/<dataset>/fold-<k>/` | Fine-tuned encoder and `scorer.pt` |
| `outputs/<dataset>/fold-<k>/run.log` | Plain lines from that fold |
| `outputs/<dataset>/fold-<k>/metrics.json` | Threshold, precision, recall, F1, scoring time |
| `outputs/<dataset>/cv-metrics.json` | Pooled ranker numbers for the folds that have finished |
| `exports/training-data/` | Current pair export |

## Tests

The unit tests do not download a model. They cover the keyword instance, the 10/8/32 fold roles, retrieval precision and recall, and the plain log lines.

```bash
python -m pytest
```
"""


def _slides(records: dict[str, dict]) -> str:
    table = _result_section(records)
    return f"""---
marp: true
theme: semstruct
size: 16:9
paginate: true
footer: SemStruct
---

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "" -->

# SemStruct

One DistilBERT ranker per database, scored on Coffman topics with 5-fold cross-validation.

---

<p class="eyebrow">The task</p>

# A keyword still has to land in the schema

A person types a few words. The model scores one keyword, in the context of the whole query, against one relation, attribute, or cell.

---

<p class="eyebrow">This version</p>

# The offline ranker

<div class="split">
  <div class="card">
    <p class="kicker">Built</p>
    <p>Keyword instances, a sigmoid head, and five folds on Mondial, IMDb, and Wikipedia.</p>
  </div>
  <div class="card">
    <p class="kicker">Later</p>
    <p>Candidate networks, SQL execution, and ranking of the joined answers.</p>
  </div>
</div>

---

<p class="eyebrow">Folds</p>

# Ten topics out, eight to choose, thirty-two to train

Each fold tests 10 Coffman topics. Eight of the rest choose the checkpoint and the threshold. The other 32 train, with every synthetic instance. A test topic never updates that fold.

---

<p class="eyebrow">Ranker results</p>

# Precision, recall, and F1

{table}

<p class="note">These are keyword-element decisions, not end-to-end ICN retrieval. Wikipedia sits beside the Mondial and IMDb article rows.</p>
"""


def _run_log(records: dict[str, dict], root: Path) -> str:
    sections = ["# Latest run", "", "Plain lines from the current prepare and fold logs.", ""]
    for dataset in DATASETS:
        prepare_log = output_dir(dataset, root) / "prepare.log"
        if prepare_log.is_file():
            sections.append(f"## {NAMES[dataset]} prepare")
            sections.append("")
            sections.append("```text")
            sections.append(prepare_log.read_text(encoding="utf-8").rstrip())
            sections.append("```")
            sections.append("")
        for fold in range(5):
            path = output_dir(dataset, root) / f"fold-{fold}" / "run.log"
            if not path.is_file():
                continue
            sections.append(f"## {NAMES[dataset]} fold {fold}")
            sections.append("")
            sections.append("```text")
            sections.append(path.read_text(encoding="utf-8").rstrip())
            sections.append("```")
            sections.append("")
        metrics = records[dataset]["metrics"]
        if metrics:
            sections.append(f"## {NAMES[dataset]} pooled")
            sections.append("")
            sections.append(_result_section({dataset: records[dataset]}))
            sections.append("")
    if len(sections) <= 4:
        sections.append("No run log has been written yet.")
    return "\n".join(sections).rstrip() + "\n"


def _count_table(records: dict[str, dict]) -> str:
    lines = [
        "| Database | Synthetic queries | Synthetic pairs | Coffman topics | Coffman pairs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    any_row = False
    for dataset in DATASETS:
        stats = records[dataset]["stats"]
        if not stats:
            continue
        any_row = True
        coffman = stats.get("coffman") or {}
        lines.append(
            f"| {NAMES[dataset]} | {_fmt(stats.get('synthetic_queries', 0))} | "
            f"{_fmt(stats.get('synthetic_pairs', 0))} | {_fmt(stats.get('topics', 0))} | "
            f"{_fmt(coffman.get('pairs', 0))} |"
        )
    if not any_row:
        return "No database has been prepared yet."
    lines.append("")
    lines.append(_category_table(records))
    return "\n".join(lines)


def _category_table(records: dict[str, dict]) -> str:
    lines = [
        "| Database | Category | Positive | Negative | Total |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        stats = records[dataset]["stats"]
        if not stats or "categories" not in stats:
            continue
        for category in ("relation", "attribute", "value", "all"):
            counts = stats["categories"][category]
            lines.append(
                f"| {NAMES[dataset]} | {category.title()} | {_fmt(counts['positive'])} | "
                f"{_fmt(counts['negative'])} | {_fmt(counts['total'])} |"
            )
    return "\n".join(lines)


def _source_table(records: dict[str, dict]) -> str:
    lines = [
        "| Database | Source | Positive | Negative | Total |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        stats = records[dataset]["stats"]
        if not stats:
            continue
        for source, counts in (stats.get("categories_by_source") or {}).items():
            for category in ("relation", "attribute", "value"):
                block = counts[category]
                lines.append(
                    f"| {NAMES[dataset]} | {source} {category} | {_fmt(block['positive'])} | "
                    f"{_fmt(block['negative'])} | {_fmt(block['total'])} |"
                )
    return "\n".join(lines)


def _result_section(records: dict[str, dict]) -> str:
    lines = [
        "| Database | System | Precision | Recall | F1 | Scoring |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    labels = (
        ("finetuned_distilbert", "Fine-tuned"),
        ("pretrained_distilbert", "Pre-trained"),
    )
    any_row = False
    for dataset in DATASETS:
        if dataset not in records:
            continue
        metrics = records[dataset]["metrics"]
        if not metrics:
            continue
        done = len(metrics.get("folds_completed", []))
        expected = int(metrics.get("folds_expected", 5))
        if done != expected:
            continue
        any_row = True
        for key, label in labels:
            block = metrics["ranker_results"][key]
            lines.append(
                f"| {NAMES[dataset]} | {label} | {_num(block['precision'])} | {_num(block['recall'])} | "
                f"{_num(block['f1'])} | {_scoring(block['scoring_ms'])} |"
            )
    if not any_row:
        return "No database has all five folds yet, so the pooled table is not written."
    return "\n".join(lines)


def _scoring(milliseconds: float) -> str:
    value = float(milliseconds)
    if value >= 1000:
        return f"{value / 1000:.1f} s"
    return f"{value:.0f} ms"


def _fmt(value) -> str:
    return f"{int(value):,}"


def _num(value) -> str:
    return f"{float(value):.3f}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
