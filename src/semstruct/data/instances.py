"""Build leakage-safe query–element pairs from query classes."""

from __future__ import annotations

import logging
import random
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

from semstruct.data.elements import (
    attribute_element_id,
    attribute_text,
    content_tokens,
    keyword_from_value,
    normalize_text,
    relation_element_id,
    relation_text,
    value_element_id,
    value_text,
)
from semstruct.data.schema import Attribute, Relation, Schema, quote_ident

logger = logging.getLogger(__name__)


def sample_example_values(
    connection: sqlite3.Connection,
    schema: Schema,
    per_attribute: int,
    max_chars: int,
    rng: random.Random,
) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {}
    for relation_name in sorted(schema.relations):
        relation = schema.relations[relation_name]
        for attribute in relation.keyword_attributes():
            key = f"{relation.name}.{attribute.name}"
            examples[key] = _sample_column(
                connection,
                relation.name,
                attribute.name,
                per_attribute,
                max_chars,
                rng,
            )
    return examples


def build_instances(
    connection: sqlite3.Connection,
    schema: Schema,
    classes: list[tuple[str, list[str]]],
    examples: dict[str, list[str]],
    content_values: dict[str, list[str]],
    topic_norms: set[str],
    config: dict,
) -> tuple[list[dict], list[dict], dict]:
    """Return queries, pairs, and a small stats dict.

    Splits are assigned by query, after an exact-string filter against the
    Coffman topics. Pairs are then capped by dropping whole queries.
    """
    rng = random.Random(int(config["seed"]))
    value_pool = _content_pool(content_values, schema, int(config["max_value_chars"]))
    relation_pool = [relation_text_record(schema.relations[name]) for name in sorted(schema.relations)]
    attribute_pool = _attribute_pool(schema, examples)
    queries: list[dict] = []
    pairs_by_query: dict[str, list[dict]] = {}
    seen_queries: set[str] = set()
    skipped_topics = 0
    skipped_empty = 0
    max_chars = int(config["max_value_chars"])
    timeout = float(config.get("join_timeout_seconds", 15))

    for class_index, (class_id, relations) in enumerate(classes):
        rows = _sample_class_rows(
            connection,
            schema,
            relations,
            int(config["queries_per_class"]),
            timeout,
        )
        if not rows:
            skipped_empty += 1
            continue
        for row in rows:
            built = _query_from_row(schema, relations, row, rng, config)
            if built is None:
                skipped_empty += 1
                continue
            keywords, keyword_attributes, keyword_values = built
            query = " ".join(keywords)
            normalized = normalize_text(query)
            if not normalized or normalized in seen_queries:
                continue
            if normalized in topic_norms:
                skipped_topics += 1
                continue
            seen_queries.add(normalized)
            query_id = f"q{len(queries):05d}"
            query_record = {
                "query_id": query_id,
                "query": query,
                "class_id": class_id,
                "relations": list(relations),
                "keyword_attributes": list(keyword_attributes),
                "keywords": list(keywords),
            }
            pair_rows = _pairs_for_query(
                schema,
                examples,
                query_record,
                keyword_values,
                relation_pool,
                attribute_pool,
                value_pool,
                rng,
                int(config["negatives_per_positive"]),
                float(config["hard_negative_ratio"]),
                max_chars,
            )
            if not pair_rows:
                seen_queries.remove(normalized)
                continue
            queries.append(query_record)
            pairs_by_query[query_id] = pair_rows
        if (class_index + 1) % 50 == 0:
            logger.info("generated queries from %s/%s classes", class_index + 1, len(classes))

    kept_ids = _cap_query_ids(list(pairs_by_query), pairs_by_query, int(config["max_pairs"]), int(config["seed"]))
    kept_queries = [query for query in queries if query["query_id"] in set(kept_ids)]
    split_of = assign_splits(
        [query["query_id"] for query in kept_queries],
        int(config["seed"]),
        float(config["train_ratio"]),
        float(config["val_ratio"]),
    )
    final_queries = []
    final_pairs = []
    for query in kept_queries:
        query["split"] = split_of[query["query_id"]]
        final_queries.append(query)
        for pair in pairs_by_query[query["query_id"]]:
            pair["split"] = query["split"]
            final_pairs.append(pair)
    stats = {
        "classes": len(classes),
        "queries": len(final_queries),
        "pairs": len(final_pairs),
        "skipped_topic_collisions": skipped_topics,
        "skipped_empty_rows": skipped_empty,
        "splits": {
            name: sum(1 for query in final_queries if query["split"] == name)
            for name in ("train", "validation", "test")
        },
    }
    logger.info("prepared %s", stats)
    return final_queries, final_pairs, stats


def assign_splits(query_ids: list[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, str]:
    if len(query_ids) < 3:
        raise ValueError(f"Need at least 3 queries for train/validation/test splits, found {len(query_ids)}")
    ordered = sorted(query_ids)
    generator = random.Random(seed)
    generator.shuffle(ordered)
    n_train, n_val, _n_test = _split_counts(len(ordered), train_ratio, val_ratio)
    splits = {}
    for index, query_id in enumerate(ordered):
        if index < n_train:
            splits[query_id] = "train"
        elif index < n_train + n_val:
            splits[query_id] = "validation"
        else:
            splits[query_id] = "test"
    return splits


def relation_text_record(relation: Relation) -> dict[str, str]:
    return {
        "element_id": relation_element_id(relation.name),
        "element_type": "relation",
        "relation": relation.name,
        "attribute": "",
        "text": relation_text(relation),
    }


def _split_counts(count: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    n_train = int(count * train_ratio)
    n_val = int(count * val_ratio)
    n_test = count - n_train - n_val
    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1
    n_train = count - n_val - n_test
    if n_train < 1:
        raise ValueError("Split ratios leave no training queries")
    return n_train, n_val, n_test


def _cap_query_ids(
    query_ids: list[str],
    pairs_by_query: dict[str, list[dict]],
    max_pairs: int,
    seed: int,
) -> list[str]:
    ordered = list(query_ids)
    random.Random(seed).shuffle(ordered)
    kept: list[str] = []
    total = 0
    for query_id in ordered:
        size = len(pairs_by_query[query_id])
        if size > max_pairs:
            continue
        if total + size > max_pairs:
            continue
        kept.append(query_id)
        total += size
    if len(kept) < 3:
        raise ValueError(
            f"Pair cap {max_pairs} left {len(kept)} queries. Raise max_pairs or lower negatives_per_positive."
        )
    return kept


def _query_from_row(
    schema: Schema,
    relations: list[str],
    row: dict[str, str | None],
    rng: random.Random,
    config: dict,
) -> tuple[list[str], list[str], list[tuple[str, str, str]]] | None:
    candidates: list[tuple[str, str, str, str]] = []
    max_chars = int(config["max_value_chars"])
    for relation_name in relations:
        relation = schema.relations[relation_name]
        for attribute in relation.keyword_attributes():
            key = f"{relation_name}.{attribute.name}"
            raw = row.get(key)
            if raw is None:
                continue
            keyword = keyword_from_value(str(raw), max_chars)
            if keyword is None:
                continue
            candidates.append((key, keyword, str(raw), relation_name))
    if not candidates:
        return None
    rng.shuffle(candidates)
    target = 2 if len(candidates) >= 2 and int(config["max_keywords"]) >= 2 else 1
    target = min(target, int(config["max_keywords"]), len(candidates))
    target = max(target, min(int(config["min_keywords"]), len(candidates)))
    chosen = []
    used_keywords: set[str] = set()
    for key, keyword, raw, relation_name in candidates:
        if keyword in used_keywords:
            continue
        chosen.append((key, keyword, raw, relation_name))
        used_keywords.add(keyword)
        if len(chosen) >= target:
            break
    if len(chosen) < int(config["min_keywords"]):
        return None
    keywords = [item[1] for item in chosen]
    attributes = [item[0] for item in chosen]
    values = [(item[3], item[0].split(".", 1)[1], item[2]) for item in chosen]
    return keywords, attributes, values


def _pairs_for_query(
    schema: Schema,
    examples: dict[str, list[str]],
    query: dict,
    keyword_values: list[tuple[str, str, str]],
    relation_pool: list[dict[str, str]],
    attribute_pool: list[dict[str, str]],
    value_pool: list[dict[str, str]],
    rng: random.Random,
    negatives_per_positive: int,
    hard_negative_ratio: float,
    max_chars: int,
) -> list[dict]:
    keywords: list[str] = query["keywords"]
    class_relations = set(query["relations"])
    keyword_attributes = set(query["keyword_attributes"])
    positives: list[dict] = []
    for relation_name in query["relations"]:
        relation = schema.relations[relation_name]
        positives.append(
            _pair(query, relation_element_id(relation_name), "relation", relation_name, "", relation_text(relation), 1)
        )
    for attribute_key in query["keyword_attributes"]:
        relation_name, attribute_name = attribute_key.split(".", 1)
        relation = schema.relations[relation_name]
        attribute = relation.attribute(attribute_name)
        text = attribute_text(relation, attribute, examples.get(attribute_key, []), keywords)
        positives.append(
            _pair(query, attribute_element_id(relation_name, attribute_name), "attribute", relation_name, attribute_name, text, 1)
        )
    for relation_name, attribute_name, raw in keyword_values:
        positives.append(
            _pair(
                query,
                value_element_id(relation_name, attribute_name, raw),
                "value",
                relation_name,
                attribute_name,
                value_text(relation_name, attribute_name, raw, max_chars),
                1,
            )
        )
    positive_ids = {item["element_id"] for item in positives}
    query_tokens = content_tokens(query["query"])
    n_negative = negatives_per_positive * len(positives)
    n_hard = int(round(n_negative * hard_negative_ratio))
    negatives = _sample_negatives(
        query,
        positive_ids,
        class_relations,
        keyword_attributes,
        relation_pool,
        attribute_pool,
        value_pool,
        query_tokens,
        keywords,
        n_hard,
        n_negative - n_hard,
        rng,
        examples,
        schema,
        max_chars,
    )
    return positives + negatives


def _sample_negatives(
    query: dict,
    positive_ids: set[str],
    class_relations: set[str],
    keyword_attributes: set[str],
    relation_pool: list[dict[str, str]],
    attribute_pool: list[dict[str, str]],
    value_pool: list[dict[str, str]],
    query_tokens: set[str],
    keywords: list[str],
    n_hard: int,
    n_random: int,
    rng: random.Random,
    examples: dict[str, list[str]],
    schema: Schema,
    max_chars: int,
) -> list[dict]:
    hard: list[dict] = []
    random_pool: list[dict] = []
    for record in relation_pool:
        if record["element_id"] in positive_ids or record["relation"] in class_relations:
            continue
        rendered = dict(record)
        if query_tokens & content_tokens(rendered["text"]):
            hard.append(rendered)
        else:
            random_pool.append(rendered)
    for record in attribute_pool:
        attribute_key = f"{record['relation']}.{record['attribute']}"
        if record["element_id"] in positive_ids or attribute_key in keyword_attributes:
            continue
        relation = schema.relations[record["relation"]]
        attribute = relation.attribute(record["attribute"])
        text = attribute_text(relation, attribute, examples.get(attribute_key, []), keywords)
        rendered = dict(record)
        rendered["text"] = text
        if query_tokens & content_tokens(text):
            hard.append(rendered)
        else:
            random_pool.append(rendered)
    keyword_norms = {normalize_text(keyword) for keyword in keywords}
    for record in value_pool:
        if record["element_id"] in positive_ids:
            continue
        normalized_value = normalize_text(record.get("value", ""))
        contains_keyword = any(keyword and keyword in normalized_value for keyword in keyword_norms)
        shares_token = bool(query_tokens & content_tokens(record.get("value", "")))
        if shares_token or contains_keyword:
            hard.append(record)
        elif not contains_keyword:
            random_pool.append(record)
    rng.shuffle(hard)
    rng.shuffle(random_pool)
    chosen: list[dict] = []
    used: set[str] = set(positive_ids)
    for source, count in ((hard, n_hard), (random_pool, n_random)):
        if count <= 0:
            continue
        for record in source:
            if record["element_id"] in used:
                continue
            chosen.append(
                _pair(
                    query,
                    record["element_id"],
                    record["element_type"],
                    record["relation"],
                    record.get("attribute", ""),
                    record["text"],
                    0,
                )
            )
            used.add(record["element_id"])
            count -= 1
            if count == 0:
                break
    if len(chosen) < n_hard + n_random:
        for record in hard + random_pool:
            if record["element_id"] in used:
                continue
            chosen.append(
                _pair(
                    query,
                    record["element_id"],
                    record["element_type"],
                    record["relation"],
                    record.get("attribute", ""),
                    record["text"],
                    0,
                )
            )
            used.add(record["element_id"])
            if len(chosen) >= n_hard + n_random:
                break
    return chosen


def _pair(
    query: dict,
    element_id: str,
    element_type: str,
    relation: str,
    attribute: str,
    text: str,
    label: int,
) -> dict:
    return {
        "query_id": query["query_id"],
        "query": query["query"],
        "element_id": element_id,
        "element_type": element_type,
        "relation": relation,
        "attribute": attribute,
        "element_text": text,
        "label": label,
        "class_id": query["class_id"],
    }


def _attribute_pool(schema: Schema, examples: dict[str, list[str]]) -> list[dict[str, str]]:
    pool = []
    for relation_name in sorted(schema.relations):
        relation = schema.relations[relation_name]
        for attribute in relation.keyword_attributes():
            key = f"{relation.name}.{attribute.name}"
            pool.append(
                {
                    "element_id": attribute_element_id(relation.name, attribute.name),
                    "element_type": "attribute",
                    "relation": relation.name,
                    "attribute": attribute.name,
                    "text": attribute_text(relation, attribute, examples.get(key, [])),
                }
            )
    return pool


def _content_pool(examples: dict[str, list[str]], schema: Schema, max_chars: int) -> list[dict[str, str]]:
    pool = []
    seen: set[str] = set()
    for key, values in examples.items():
        relation_name, attribute_name = key.split(".", 1)
        if relation_name not in schema.relations:
            continue
        for value in values:
            element_id = value_element_id(relation_name, attribute_name, value)
            if element_id in seen:
                continue
            seen.add(element_id)
            pool.append(
                {
                    "element_id": element_id,
                    "element_type": "value",
                    "relation": relation_name,
                    "attribute": attribute_name,
                    "value": value,
                    "text": value_text(relation_name, attribute_name, value, max_chars),
                }
            )
    return pool


def _sample_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    limit: int,
    max_chars: int,
    rng: random.Random,
) -> list[str]:
    maximum = connection.execute(f"SELECT MAX(rowid) FROM {quote_ident(table)}").fetchone()[0]
    if not maximum:
        return []
    found: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(found) < limit and attempts < limit * 30:
        attempts += 1
        row_id = rng.randint(1, int(maximum))
        row = connection.execute(
            f"SELECT {quote_ident(column)} FROM {quote_ident(table)} WHERE rowid = ?",
            (row_id,),
        ).fetchone()
        if row is None or row[0] is None:
            continue
        text = str(row[0]).strip()
        if not text or len(text) > max_chars or not any(character.isalpha() for character in text):
            continue
        marker = normalize_text(text)
        if marker in seen:
            continue
        seen.add(marker)
        found.append(text)
    return found


def _sample_class_rows(
    connection: sqlite3.Connection,
    schema: Schema,
    relations: list[str],
    limit: int,
    timeout: float,
) -> list[dict[str, str | None]]:
    sql = _join_sql(schema, relations, limit)
    if sql is None:
        return []
    try:
        with _time_limit(connection, timeout):
            cursor = connection.execute(sql)
            names = [item[0] for item in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]
    except sqlite3.OperationalError as exc:
        logger.warning("skipping class %s: %s", "|".join(relations), exc)
        return []


def _join_sql(schema: Schema, relations: list[str], limit: int) -> str | None:
    aliases = {name: f"r{index}" for index, name in enumerate(relations)}
    selected: list[str] = []
    for relation_name in relations:
        relation = schema.relations[relation_name]
        for attribute in relation.keyword_attributes():
            selected.append(
                "substr({alias}.{column}, 1, 200) AS {label}".format(
                    alias=aliases[relation_name],
                    column=quote_ident(attribute.name),
                    label=quote_ident(f"{relation_name}.{attribute.name}"),
                )
            )
    if not selected:
        return None
    joins, connected = _join_clauses(schema, relations, aliases)
    if connected != set(relations):
        return None
    sql = (
        f"SELECT {', '.join(selected)} FROM {quote_ident(relations[0])} AS {aliases[relations[0]]} "
        + " ".join(joins)
        + f" LIMIT {int(limit)}"
    )
    return sql


def _join_clauses(
    schema: Schema,
    relations: list[str],
    aliases: dict[str, str],
) -> tuple[list[str], set[str]]:
    connected = {relations[0]}
    remaining = set(relations[1:])
    clauses: list[str] = []
    guard = 0
    while remaining and guard < len(relations) + 2:
        guard += 1
        progress = False
        for relation_name in list(remaining):
            link = _find_link(schema, relation_name, connected)
            if link is None:
                continue
            clauses.append(_link_sql(link, aliases))
            connected.add(relation_name)
            remaining.remove(relation_name)
            progress = True
        if not progress:
            break
    return clauses, connected


def _find_link(schema: Schema, relation_name: str, connected: set[str]):
    relation = schema.relations[relation_name]
    for foreign_key in relation.foreign_keys:
        if foreign_key.ref_table in connected and foreign_key.ref_table != relation_name:
            return ("out", relation_name, foreign_key)
    for other in connected:
        for foreign_key in schema.relations[other].foreign_keys:
            if foreign_key.ref_table == relation_name and other != relation_name:
                return ("in", other, foreign_key)
    return None


def _link_sql(link, aliases: dict[str, str]) -> str:
    direction, source, foreign_key = link
    if direction == "out":
        left = source
        right = foreign_key.ref_table
        left_columns = foreign_key.columns
        right_columns = foreign_key.ref_columns
        joined = source
    else:
        left = source
        right = foreign_key.ref_table
        left_columns = foreign_key.columns
        right_columns = foreign_key.ref_columns
        joined = foreign_key.ref_table
    conditions = [
        f"{aliases[left]}.{quote_ident(left_column)} = {aliases[right]}.{quote_ident(right_column)}"
        for left_column, right_column in zip(left_columns, right_columns)
    ]
    return f"JOIN {quote_ident(joined)} AS {aliases[joined]} ON {' AND '.join(conditions)}"


@contextmanager
def _time_limit(connection: sqlite3.Connection, seconds: float) -> Iterator[None]:
    start = time.monotonic()

    def handler() -> int:
        return 1 if time.monotonic() - start > seconds else 0

    connection.set_progress_handler(handler, 10000)
    try:
        yield
    finally:
        connection.set_progress_handler(None, 0)


def attribute_records_for_class(schema: Schema, relations: list[str]) -> list[Attribute]:
    records = []
    for relation_name in relations:
        records.extend(schema.relations[relation_name].keyword_attributes())
    return records


def group_pairs(pairs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["query_id"]].append(pair)
    return grouped
