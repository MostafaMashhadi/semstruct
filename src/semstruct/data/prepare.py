"""Turn one downloaded benchmark into schema, query classes, and training pairs."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from semstruct.data.download import locate_files
from semstruct.data.dump import load_sqlite
from semstruct.data.elements import schema_element_records
from semstruct.data.instances import build_instances, build_topic_instances, category_counts, sample_example_values
from semstruct.data.query_classes import select_query_classes
from semstruct.data.topics import extract_topics, load_topics, topic_query_norms
from semstruct.paths import processed_dir, ranker_dir
from semstruct.runlog import emit, format_prepare

logger = logging.getLogger(__name__)


def prepare_dataset(dataset: str, config: dict, root: Path | None = None, rebuild: bool = False) -> dict:
    import sqlite3

    files = locate_files(dataset, root)
    destination = processed_dir(dataset, root)
    destination.mkdir(parents=True, exist_ok=True)
    sqlite_path = destination / "database.sqlite"
    schema_path = destination / "schema.json"
    reused = not rebuild and sqlite_path.exists() and schema_path.exists()
    if not reused:
        logger.info("loading %s into %s", dataset, sqlite_path)
        schema = load_sqlite(files["schema"], files["data"], sqlite_path)
        payload = schema.as_dict()
        schema_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        from semstruct.data.schema import Schema

        schema = Schema.from_dict(json.loads(schema_path.read_text(encoding="utf-8")))
        logger.info("reusing %s", sqlite_path)

    topics_dir = destination / "topics"
    if rebuild or not any(topics_dir.rglob("*.txt")):
        extract_topics(files["topics"], topics_dir)
    topics = load_topics(topics_dir)

    classes = select_query_classes(
        schema,
        max_relations=int(config["max_relations"]),
        max_classes=int(config["max_classes"]),
        seed=int(config["seed"]),
    )
    write_jsonl(
        destination / "classes.jsonl",
        [{"class_id": class_id, "relations": relations} for class_id, relations in classes],
    )

    connection = sqlite3.connect(sqlite_path)
    try:
        sampled = sample_example_values(
            connection,
            schema,
            per_attribute=max(int(config["max_values_per_attribute"]), int(config["example_values_per_attribute"])),
            max_chars=int(config["max_value_chars"]),
            rng=random.Random(int(config["seed"])),
        )
        example_limit = int(config["example_values_per_attribute"])
        examples = {key: values[:example_limit] for key, values in sampled.items()}
        (destination / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
        queries, pairs, stats = build_instances(
            connection,
            schema,
            classes,
            examples,
            sampled,
            topic_query_norms(topics),
            config,
        )
        topic_queries, topic_pairs, topic_stats, topic_splits = build_topic_instances(
            connection,
            schema,
            topics,
            examples,
            sampled,
            config,
        )
    finally:
        connection.close()
    queries.extend(topic_queries)
    pairs.extend(topic_pairs)
    stats["synthetic_queries"] = stats["queries"]
    stats["synthetic_pairs"] = stats["pairs"]
    stats["queries"] = len(queries)
    stats["pairs"] = len(pairs)
    stats["coffman"] = topic_stats
    for name in ("train", "validation", "test"):
        stats["splits"][name] = sum(1 for query in queries if query["split"] == name)
    stats["categories"] = category_counts(pairs)
    stats["categories_by_source"] = {
        source: category_counts([pair for pair in pairs if pair.get("source") == source])
        for source in ("synthetic", "coffman")
    }
    topic_rows = []
    for topic in topics:
        row = topic.as_dict()
        row["fold"] = topic_splits.get(str(topic.topic_id))
        topic_rows.append(row)
    write_jsonl(destination / "topics.jsonl", topic_rows)

    write_jsonl(destination / "elements.jsonl", schema_element_records(schema, examples))
    write_jsonl(destination / "queries.jsonl", queries)
    write_jsonl(destination / "pairs.jsonl", pairs)
    stats["dataset"] = dataset
    stats["topics"] = len(topics)
    stats["relations"] = len(schema.relations)
    stats["reused_sqlite"] = reused
    (destination / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    report_dir = ranker_dir(dataset, root)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    roles = (stats.get("coffman") or {}).get("fold_roles", {}).get("0", {})
    emit(
        format_prepare(
            dataset,
            reused,
            int(stats.get("synthetic_queries", stats.get("queries", 0))),
            int(stats.get("synthetic_pairs", 0)),
            int(stats.get("topics", 0)),
            int((stats.get("coffman") or {}).get("pairs", 0)),
            int(roles.get("train", 0)),
            int(roles.get("validation", 0)),
            int(roles.get("test", 0)),
            destination / "pairs.jsonl",
        ),
        report_dir / "prepare.log",
    )
    from semstruct.reports import refresh_reports

    refresh_reports(root)
    return stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
