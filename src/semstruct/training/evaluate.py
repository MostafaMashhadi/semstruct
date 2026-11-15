"""Evaluate a fine-tuned classifier against lexical and frozen-embedding baselines."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from semstruct.config import load_config
from semstruct.data.elements import (
    attribute_element_id,
    attribute_text,
    relation_element_id,
    relation_text,
)
from semstruct.data.prepare import read_jsonl
from semstruct.data.schema import Schema
from semstruct.device import release_device_cache, select_device
from semstruct.model.classifier import load_classifier
from semstruct.model.scoring import keyword_relevance, query_class_score, relevance_probabilities
from semstruct.paths import model_dir, output_dir, processed_dir, project_root
from semstruct.training.baselines import cosine_from_texts, overlap_scores
from semstruct.training.metrics import best_f1_threshold, classification_report, ranking_report

logger = logging.getLogger(__name__)


def evaluate_dataset(dataset: str, config: dict | None = None, root: Path | None = None, batch_size: int | None = None) -> dict:
    root = root or project_root()
    config = dict(config or load_config(dataset, root=root))
    if batch_size is not None:
        config["batch_size"] = batch_size
    directory = processed_dir(dataset, root)
    pairs = read_jsonl(directory / "pairs.jsonl")
    queries = read_jsonl(directory / "queries.jsonl")
    classes = read_jsonl(directory / "classes.jsonl")
    topics = read_jsonl(directory / "topics.jsonl")
    examples = json.loads((directory / "examples.json").read_text(encoding="utf-8"))
    schema = Schema.from_dict(json.loads((directory / "schema.json").read_text(encoding="utf-8")))
    checkpoint = model_dir(dataset, root)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"No fine-tuned model at {checkpoint}. Run train first.")

    device = select_device()
    tokenizer, model = load_classifier(str(checkpoint))
    batch = int(config["batch_size"])
    max_length = int(config["max_length"])
    validation = [row for row in pairs if row["split"] == "validation"]
    test = [row for row in pairs if row["split"] == "test"]
    test_queries = [query for query in queries if query["split"] == "test"]

    systems = {
        "distilbert": _cross_encoder_scorer(model, tokenizer, batch, device, max_length),
        "token_overlap": lambda queries_batch, texts: overlap_scores(queries_batch, texts),
        "frozen_distilbert": _frozen_scorer(config["model_id"], batch, device, max_length),
    }
    report: dict = {"dataset": dataset, "device": str(device), "systems": {}}
    schema_scores: dict[str, list[tuple[dict, dict, float]]] = {}
    for name, scorer in systems.items():
        logger.info("scoring %s with %s", dataset, name)
        validation_scores = scorer([row["query"] for row in validation], [row["element_text"] for row in validation])
        test_scores = scorer([row["query"] for row in test], [row["element_text"] for row in test])
        threshold = best_f1_threshold([int(row["label"]) for row in validation], validation_scores)
        rendered = _score_schema(scorer, test_queries, schema, examples)
        schema_scores[name] = rendered
        report["systems"][name] = {
            "validation": _pair_block(validation, validation_scores, threshold),
            "test": _pair_block(test, test_scores, threshold),
            "schema_ranking": _schema_ranking(rendered),
            "query_class_ranking": _class_ranking(rendered, test_queries, classes, schema),
        }

    search_map = _search_map(directory / "database.sqlite")
    report["coffman"] = _coffman_report(
        topics,
        schema,
        examples,
        search_map,
        systems,
        model,
        tokenizer,
        batch,
        device,
        max_length,
    )
    destination = output_dir(dataset, root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", destination / "metrics.json")
    systems.clear()
    del model, tokenizer
    release_device_cache()
    return report


def _cross_encoder_scorer(model, tokenizer, batch_size: int, device, max_length: int):
    def score(queries: list[str], texts: list[str]) -> list[float]:
        return relevance_probabilities(
            model,
            tokenizer,
            queries,
            texts,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )

    return score


def _frozen_scorer(model_id: str, batch_size: int, device, max_length: int):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    def score(queries: list[str], texts: list[str]) -> list[float]:
        if not queries:
            return []
        return cosine_from_texts(
            model,
            tokenizer,
            queries,
            texts,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )

    return score


def _pair_block(pairs: list[dict], scores: list[float], threshold: float) -> dict:
    labels = [int(row["label"]) for row in pairs]
    block = {
        "classification": classification_report(labels, scores, threshold),
        "classification_at_0.5": classification_report(labels, scores, 0.5),
        "ranking_paired": ranking_report(_grouped(pairs, scores, lambda row: int(row["label"]))),
        "by_element_type": {},
    }
    for element_type in ("relation", "attribute", "value"):
        chosen = [(row, score) for row, score in zip(pairs, scores) if row["element_type"] == element_type]
        if not chosen:
            continue
        chosen_rows = [row for row, _score in chosen]
        chosen_scores = [score for _row, score in chosen]
        block["by_element_type"][element_type] = {
            "classification": classification_report(
                [int(row["label"]) for row in chosen_rows],
                chosen_scores,
                threshold,
            ),
            "ranking_paired": ranking_report(
                _grouped(chosen_rows, chosen_scores, lambda row: int(row["label"]))
            ),
        }
    return block


def _grouped(pairs: list[dict], scores: list[float], label_of) -> dict[str, list[tuple[float, int]]]:
    grouped: dict[str, list[tuple[float, int]]] = {}
    for row, score in zip(pairs, scores):
        grouped.setdefault(row["query_id"], []).append((float(score), int(label_of(row))))
    return grouped


def _score_schema(scorer, queries: list[dict], schema: Schema, examples: dict[str, list[str]]):
    flat_queries: list[str] = []
    flat_texts: list[str] = []
    flat_rows: list[tuple[dict, dict]] = []
    for query in queries:
        for row in _schema_rows(schema, examples, query["keywords"]):
            flat_queries.append(query["query"])
            flat_texts.append(row["text"])
            flat_rows.append((query, row))
    scores = scorer(flat_queries, flat_texts) if flat_queries else []
    return [(query, row, score) for (query, row), score in zip(flat_rows, scores)]


def _schema_rows(schema: Schema, examples: dict[str, list[str]], keywords: list[str]) -> list[dict]:
    rows = []
    for relation_name in sorted(schema.relations):
        relation = schema.relations[relation_name]
        rows.append(
            {
                "element_id": relation_element_id(relation.name),
                "element_type": "relation",
                "relation": relation.name,
                "attribute": "",
                "text": relation_text(relation),
            }
        )
        for attribute in relation.keyword_attributes():
            key = f"{relation.name}.{attribute.name}"
            rows.append(
                {
                    "element_id": attribute_element_id(relation.name, attribute.name),
                    "element_type": "attribute",
                    "relation": relation.name,
                    "attribute": attribute.name,
                    "text": attribute_text(relation, attribute, examples.get(key, []), keywords),
                }
            )
    return rows


def _schema_ranking(scored_rows: list[tuple[dict, dict, float]]) -> dict:
    grouped: dict[str, list[tuple[float, int]]] = {}
    for query, row, score in scored_rows:
        if row["element_type"] == "relation":
            label = int(row["relation"] in query["relations"])
        else:
            label = int(f"{row['relation']}.{row['attribute']}" in query["keyword_attributes"])
        grouped.setdefault(query["query_id"], []).append((float(score), label))
    return ranking_report(grouped)


def _class_ranking(scored_rows, queries: list[dict], classes: list[dict], schema: Schema) -> dict:
    by_query: dict[str, dict[str, float]] = {}
    for query, row, score in scored_rows:
        by_query.setdefault(query["query_id"], {})[row["element_id"]] = float(score)
    reciprocal = []
    hits = []
    query_by_id = {query["query_id"]: query for query in queries}
    class_elements = {item["class_id"]: _class_element_ids(schema, item["relations"]) for item in classes}
    for query_id, scores in by_query.items():
        gold = query_by_id[query_id]["class_id"]
        if gold not in class_elements:
            continue
        ranked = []
        for class_id, element_ids in class_elements.items():
            values = [scores[element_id] for element_id in element_ids if element_id in scores]
            ranked.append((query_class_score(values), class_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        labels = [1 if class_id == gold else 0 for _score, class_id in ranked]
        rank = labels.index(1) + 1
        reciprocal.append(1.0 / rank)
        hits.append(1.0 if rank == 1 else 0.0)
    if not reciprocal:
        return {"mrr": None, "recall@1": None, "queries": 0}
    return {
        "mrr": sum(reciprocal) / len(reciprocal),
        "recall@1": sum(hits) / len(hits),
        "queries": len(reciprocal),
    }


def _class_element_ids(schema: Schema, relations: list[str]) -> list[str]:
    element_ids = []
    for relation_name in relations:
        element_ids.append(relation_element_id(relation_name))
        relation = schema.relations[relation_name]
        for attribute in relation.keyword_attributes():
            element_ids.append(attribute_element_id(relation_name, attribute.name))
    return element_ids


def _search_map(sqlite_path: Path) -> dict[int, str]:
    if not sqlite_path.exists():
        return {}
    connection = sqlite3.connect(sqlite_path)
    try:
        rows = connection.execute("SELECT search_id, relation FROM semstruct_search_id").fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        connection.close()
    return {int(search_id): relation for search_id, relation in rows}


def _coffman_report(topics, schema, examples, search_map, systems, model, tokenizer, batch_size, device, max_length) -> dict:
    relation_rows = [
        {"element_id": relation_element_id(name), "relation": name, "text": relation_text(schema.relations[name])}
        for name in sorted(schema.relations)
    ]
    qualitative = []
    system_groups = {name: {} for name in systems}
    skipped = 0
    unresolved = 0
    scored_topics = [topic for topic in topics if topic.get("split", "test") == "test"]
    for topic in scored_topics:
        gold = set()
        for search_id in topic["search_ids"]:
            relation = search_map.get(int(search_id))
            if relation is None:
                unresolved += 1
                continue
            gold.add(relation)
        attribute_rows = [
            row
            for row in _schema_rows(schema, examples, topic["query"].split())
            if row["element_type"] == "attribute"
        ]
        per_system = {}
        for name, scorer in systems.items():
            relation_scores = scorer([topic["query"]] * len(relation_rows), [row["text"] for row in relation_rows])
            attribute_scores = scorer(
                [topic["query"]] * len(attribute_rows),
                [row["text"] for row in attribute_rows],
            )
            ranked_relations = sorted(
                (
                    {"relation": row["relation"], "score": score}
                    for row, score in zip(relation_rows, relation_scores)
                ),
                key=lambda item: (-item["score"], item["relation"]),
            )
            ranked_attributes = sorted(
                (
                    {
                        "attribute": f"{row['relation']}.{row['attribute']}",
                        "score": score,
                    }
                    for row, score in zip(attribute_rows, attribute_scores)
                ),
                key=lambda item: (-item["score"], item["attribute"]),
            )
            per_system[name] = {
                "top_relations": ranked_relations[:5],
                "top_attributes": ranked_attributes[:5],
            }
            if gold:
                system_groups[name].setdefault(topic["topic_id"], [])
                system_groups[name][topic["topic_id"]] = [
                    (item["score"], int(item["relation"] in gold)) for item in ranked_relations
                ]
        if not gold:
            skipped += 1
        top_relations = per_system["distilbert"]["top_relations"]
        keyword_rows = []
        for item in top_relations:
            relation = item["relation"]
            text = next(row["text"] for row in relation_rows if row["relation"] == relation)
            keyword_rows.append(
                keyword_relevance(
                    model,
                    tokenizer,
                    topic["query"],
                    text,
                    batch_size=batch_size,
                    device=device,
                    max_length=max_length,
                )
            )
        qualitative.append(
            {
                "topic_id": topic["topic_id"],
                "query": topic["query"],
                "gold_relations": sorted(gold),
                "systems": per_system,
                "distilbert_keyword_scores": keyword_rows,
            }
        )
    ranking = {}
    for name, grouped in system_groups.items():
        ranking[name] = ranking_report(grouped) if grouped else {"map": None, "mrr": None, "recall@5": None, "recall@10": None, "queries": 0}
    return {
        "relation_ranking": ranking,
        "topics": len(topics),
        "topics_in_report": len(scored_topics),
        "topics_used_in_training_or_validation": len(topics) - len(scored_topics),
        "topics_without_resolved_tuples": skipped,
        "unresolved_search_ids": unresolved,
        "qualitative": qualitative,
        "note": (
            "One third of the Coffman topics are the test slice scored here. "
            "The other two thirds are training and validation pairs, so they are not in this report. "
            "Gold relations are the relations of tuples cited by __search_id. "
            "A cell in a judged tuple is a positive attribute or value when it contains a query keyword."
        ),
    }
