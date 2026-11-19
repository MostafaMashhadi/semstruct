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
    STOPWORDS,
    attribute_element_id,
    attribute_text,
    joined_relation_names,
    keyword_from_value,
    normalize_text,
    relation_element_id,
    relation_text,
    tokenize,
    value_element_id,
    value_text,
)
from semstruct.data.schema import Attribute, Relation, Schema, quote_ident
from semstruct.runlog import emit

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
    """Return synthetic keyword instances. Every one of them is training data.

    A query string that equals a Coffman topic is dropped. Value negatives are
    capped here because these pairs are never used as validation or test.
    """
    del examples, content_values
    rng = random.Random(int(config["seed"]))
    queries: list[dict] = []
    seen_queries: set[str] = set()
    skipped_topics = 0
    skipped_empty = 0
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
            sources = [
                {
                    "keyword": keyword,
                    "relation": relation_name,
                    "attribute": attribute_name,
                    "value": raw,
                }
                for keyword, (relation_name, attribute_name, raw) in zip(keywords, keyword_values)
            ]
            queries.append(
                {
                    "query_id": f"q{len(queries):05d}",
                    "query": query,
                    "source": "synthetic",
                    "class_id": class_id,
                    "relations": list(relations),
                    "keyword_attributes": list(keyword_attributes),
                    "keywords": list(keywords),
                    "keyword_sources": sources,
                    "split": "train",
                }
            )
        if (class_index + 1) % 50 == 0:
            logger.info("generated queries from %s/%s classes", class_index + 1, len(classes))

    pairs = build_keyword_pairs(connection, schema, queries, config)
    pairs_by_query: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        pairs_by_query[pair["query_id"]].append(pair)
    queries = [query for query in queries if pairs_by_query.get(query["query_id"])]
    kept_ids = _cap_query_ids(
        [query["query_id"] for query in queries],
        pairs_by_query,
        int(config["max_pairs"]),
        int(config["seed"]),
    )
    kept = set(kept_ids)
    final_queries = [query for query in queries if query["query_id"] in kept]
    final_pairs = [pair for pair in pairs if pair["query_id"] in kept]
    for pair in final_pairs:
        pair["split"] = "train"
    stats = {
        "classes": len(classes),
        "queries": len(final_queries),
        "pairs": len(final_pairs),
        "skipped_topic_collisions": skipped_topics,
        "skipped_empty_rows": skipped_empty,
        "value_negative_cap": int(config.get("value_negative_cap", 10)),
        "splits": {"train": len(final_queries), "validation": 0, "test": 0},
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


def _topic_sort_key(topic_id: str) -> tuple:
    text = str(topic_id)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def assign_topic_folds(topic_ids: list[str], seed: int, n_folds: int = 5) -> dict[str, int]:
    """Deal topic ids into folds. Fifty topics become five folds of ten.

    Fewer than five topics use one topic per fold, so a three-topic test still
    has a defined assignment.
    """
    ordered = sorted({str(topic_id) for topic_id in topic_ids}, key=_topic_sort_key)
    if not ordered:
        return {}
    generator = random.Random(seed)
    generator.shuffle(ordered)
    if len(ordered) < n_folds:
        return {topic_id: index for index, topic_id in enumerate(ordered)}
    return {topic_id: index % n_folds for index, topic_id in enumerate(ordered)}


def fold_roles(topic_ids: list[str], fold: int, seed: int, n_folds: int = 5) -> dict[str, str]:
    """Role of every topic while training fold `fold`.

    That fold's topics are the test set. Of the rest, eight choose the checkpoint
    and the threshold when forty topics remain, and the other thirty-two train.
    The shuffle of those remaining topics uses a fold-specific seed.
    """
    assignment = assign_topic_folds(topic_ids, seed, n_folds)
    test_ids = [topic_id for topic_id, assigned in assignment.items() if assigned == fold]
    others = sorted(
        (topic_id for topic_id, assigned in assignment.items() if assigned != fold),
        key=_topic_sort_key,
    )
    generator = random.Random(seed + 1000 + int(fold))
    generator.shuffle(others)
    n_validation = _validation_count(len(others))
    roles = {topic_id: "test" for topic_id in test_ids}
    for index, topic_id in enumerate(others):
        roles[topic_id] = "validation" if index < n_validation else "train"
    return roles


def _validation_count(n_other: int) -> int:
    if n_other >= 40:
        return 8
    if n_other >= 2:
        return 1
    return 0


def folds_for_topics(topic_ids: list[str], n_folds: int = 5) -> list[int]:
    count = len({str(topic_id) for topic_id in topic_ids})
    if count < n_folds:
        return list(range(count))
    return list(range(n_folds))


def category_counts(pairs: list[dict]) -> dict[str, dict[str, int]]:
    """Count relation, attribute, and value pairs, with positives and negatives separate."""
    counts: dict[str, dict[str, int]] = {}
    for element_type in ("relation", "attribute", "value"):
        chosen = [pair for pair in pairs if pair["element_type"] == element_type]
        positive = sum(int(pair["label"]) == 1 for pair in chosen)
        negative = sum(int(pair["label"]) == 0 for pair in chosen)
        counts[element_type] = {"positive": positive, "negative": negative, "total": positive + negative}
    positive = sum(int(pair["label"]) == 1 for pair in pairs)
    counts["all"] = {"positive": positive, "negative": len(pairs) - positive, "total": len(pairs)}
    return counts


def build_topic_instances(
    connection: sqlite3.Connection,
    schema: Schema,
    topics: list,
    examples: dict[str, list[str]],
    content_values: dict[str, list[str]],
    config: dict,
) -> tuple[list[dict], list[dict], dict, dict[str, int]]:
    """Turn each Coffman keyword into index-hit instances and stamp its test fold.

    Validation and test keep every value hit. The training cap is applied later,
    when a fold selects its 32 training topics.
    """
    del examples, content_values
    topic_ids = [str(topic.topic_id) for topic in topics]
    folds = assign_topic_folds(topic_ids, int(config["seed"]), int(config.get("n_folds", 5)))
    search_map = _search_map(connection)
    queries: list[dict] = []
    skipped = 0
    for topic in topics:
        topic_id = str(topic.topic_id)
        keywords = [token for token in tokenize(topic.query) if token not in STOPWORDS and len(token) > 1]
        if not keywords:
            keywords = [token for token in tokenize(topic.query) if token]
        relations: list[str] = []
        seen_relations: set[str] = set()
        search_ids: list[int] = []
        for search_id in topic.search_ids:
            relation_name = search_map.get(int(search_id))
            if relation_name is None or relation_name not in schema.relations:
                continue
            search_ids.append(int(search_id))
            if relation_name not in seen_relations:
                seen_relations.add(relation_name)
                relations.append(relation_name)
        if not relations or not keywords:
            skipped += 1
            continue
        queries.append(
            {
                "query_id": f"t{int(topic_id):05d}",
                "query": topic.query,
                "source": "coffman",
                "class_id": f"topic:{topic_id}",
                "relations": relations,
                "keyword_attributes": [],
                "keywords": keywords,
                "keyword_sources": [
                    {"keyword": keyword, "relation": "", "attribute": "", "value": ""} for keyword in keywords
                ],
                "topic_id": topic_id,
                "fold": folds[topic_id],
                "search_ids": search_ids,
                "split": "coffman",
            }
        )
    pairs = build_keyword_pairs(connection, schema, queries, config)
    attributes_by_query: dict[str, set[str]] = defaultdict(set)
    used_ids: set[str] = set()
    for pair in pairs:
        used_ids.add(pair["query_id"])
        if pair["element_type"] == "attribute" and int(pair["label"]) == 1 and pair["attribute"]:
            attributes_by_query[pair["query_id"]].add(f"{pair['relation']}.{pair['attribute']}")
    kept_queries = []
    for query in queries:
        if query["query_id"] not in used_ids:
            skipped += 1
            continue
        query["keyword_attributes"] = sorted(attributes_by_query.get(query["query_id"], ()))
        kept_queries.append(query)
    role_summary = {}
    for fold in folds_for_topics(topic_ids, int(config.get("n_folds", 5))):
        roles = fold_roles(topic_ids, fold, int(config["seed"]), int(config.get("n_folds", 5)))
        role_summary[str(fold)] = {
            name: sum(role == name for role in roles.values()) for name in ("train", "validation", "test")
        }
    stats = {
        "topics": len(topics),
        "topics_used": len(kept_queries),
        "topics_skipped": skipped,
        "pairs": len(pairs),
        "folds": {str(fold): sum(assigned == fold for assigned in folds.values()) for fold in sorted(set(folds.values()))},
        "fold_roles": role_summary,
        "value_negative_cap": int(config.get("value_negative_cap", 10)),
    }
    logger.info("prepared coffman topics %s", stats)
    return kept_queries, pairs, stats, folds


def _search_map(connection: sqlite3.Connection) -> dict[int, str]:
    try:
        rows = connection.execute("SELECT search_id, relation FROM semstruct_search_id").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {int(search_id): str(relation) for search_id, relation in rows}


def _row_by_search_id(connection: sqlite3.Connection, relation_name: str, search_id: int) -> dict | None:
    statement = (
        f"SELECT * FROM {quote_ident(relation_name)} WHERE {quote_ident('__search_id')} = ?"
    )
    cursor = connection.execute(statement, (str(search_id),))
    row = cursor.fetchone()
    if row is None:
        return None
    names = [item[0] for item in cursor.description]
    return dict(zip(names, row))


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


def cap_value_negatives(pairs: list[dict], seed: int, cap: int = 10) -> list[dict]:
    """Keep every schema hit and every intended hit. Cap other value hits per keyword."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for pair in pairs:
        key = (str(pair["query_id"]), str(pair["keyword"]))
        if key not in grouped:
            order.append(key)
        grouped[key].append(pair)
    generator = random.Random(seed)
    kept: list[dict] = []
    for key in order:
        rows = grouped[key]
        negatives = [
            row for row in rows if row["element_type"] == "value" and int(row["label"]) == 0
        ]
        rest = [
            row for row in rows if not (row["element_type"] == "value" and int(row["label"]) == 0)
        ]
        if len(negatives) > cap:
            negatives = sorted(negatives, key=lambda row: row["element_id"])
            generator.shuffle(negatives)
            negatives = negatives[:cap]
        kept.extend(rest)
        kept.extend(negatives)
    return kept


def build_keyword_pairs(
    connection: sqlite3.Connection,
    schema: Schema,
    queries: list[dict],
    config: dict,
) -> list[dict]:
    """One row per keyword, query, and index hit."""
    if not queries:
        return []
    keywords = [
        source["keyword"]
        for query in queries
        for source in query.get("keyword_sources", [])
        if source.get("keyword")
    ]
    index = _build_value_index(connection, schema, keywords, int(config["max_value_chars"]))
    pairs: list[dict] = []
    max_chars = int(config["max_value_chars"])
    for query in queries:
        for source in query.get("keyword_sources", []):
            pairs.extend(_pairs_for_keyword(schema, query, source, index, max_chars))
    synthetic = [pair for pair in pairs if pair.get("source") == "synthetic"]
    other = [pair for pair in pairs if pair.get("source") != "synthetic"]
    if synthetic:
        synthetic = cap_value_negatives(
            synthetic,
            int(config["seed"]),
            int(config.get("value_negative_cap", 10)),
        )
    return synthetic + other


def select_fold_pairs(
    pairs: list[dict],
    topic_ids: list[str],
    fold: int,
    seed: int,
    cap: int = 10,
    n_folds: int = 5,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Training pairs are capped. Validation and test keep every index hit."""
    roles = fold_roles(topic_ids, fold, seed, n_folds)
    train: list[dict] = []
    validation: list[dict] = []
    test: list[dict] = []
    for pair in pairs:
        if pair.get("source") == "synthetic":
            train.append(pair)
            continue
        role = roles.get(str(pair.get("topic_id")))
        if role == "train":
            train.append(pair)
        elif role == "validation":
            validation.append(pair)
        elif role == "test":
            test.append(pair)
    return cap_value_negatives(train, seed, cap), validation, test


def _contains_keyword_tokens(text_tokens: list[str], key_tokens: list[str]) -> bool:
    if not key_tokens:
        return False
    if len(key_tokens) == 1:
        return key_tokens[0] in text_tokens
    width = len(key_tokens)
    for start in range(len(text_tokens) - width + 1):
        if text_tokens[start : start + width] == key_tokens:
            return True
    return False


def _name_matches(name: str, keyword: str) -> bool:
    return tokenize(name) == tokenize(keyword)


def _build_value_index(
    connection: sqlite3.Connection,
    schema: Schema,
    keywords: list[str],
    max_chars: int,
) -> dict[str, dict[str, dict]]:
    """Map a normalized keyword to the value and attribute elements that contain it."""
    token_to_keywords: dict[str, set[str]] = defaultdict(set)
    key_tokens: dict[str, list[str]] = {}
    for keyword in keywords:
        normalized = normalize_text(keyword)
        if not normalized:
            continue
        tokens = tokenize(keyword)
        if not tokens:
            continue
        key_tokens[normalized] = tokens
        for token in set(tokens):
            token_to_keywords[token].add(normalized)
    index: dict[str, dict[str, dict]] = defaultdict(dict)
    if not token_to_keywords:
        return index
    for relation_name in sorted(schema.relations):
        relation = schema.relations[relation_name]
        columns = relation.keyword_attributes()
        if not columns:
            continue
        names = [attribute.name for attribute in columns]
        has_search = any(attribute.name == "__search_id" for attribute in relation.attributes)
        selected = names + (["__search_id"] if has_search else [])
        statement = (
            "SELECT "
            + ", ".join(quote_ident(name) for name in selected)
            + f" FROM {quote_ident(relation.name)}"
        )
        emit(f"prepare | indexing {relation.name}")
        for row in connection.execute(statement):
            search_id = int(row[-1]) if has_search and row[-1] is not None else None
            values = row[:-1] if has_search else row
            for attribute_name, raw in zip(names, values):
                if raw is None:
                    continue
                text = str(raw).strip()
                if not text or not any(character.isalpha() for character in text):
                    continue
                tokens = tokenize(text)
                if not tokens:
                    continue
                matched: set[str] = set()
                for token in set(tokens):
                    matched.update(token_to_keywords.get(token, ()))
                if not matched:
                    continue
                for normalized in matched:
                    if not _contains_keyword_tokens(tokens, key_tokens[normalized]):
                        continue
                    _add_posting(
                        index,
                        normalized,
                        _value_posting(relation.name, attribute_name, text, search_id, max_chars),
                    )
                    _add_posting(
                        index,
                        normalized,
                        _attribute_posting(schema, relation, attribute_name, search_id),
                    )
    return index


def _add_posting(index: dict[str, dict[str, dict]], keyword: str, posting: dict) -> None:
    bucket = index[keyword]
    existing = bucket.get(posting["element_id"])
    if existing is None:
        bucket[posting["element_id"]] = posting
        return
    existing["search_ids"].update(posting["search_ids"])


def _value_posting(relation_name: str, attribute_name: str, value: str, search_id: int | None, max_chars: int) -> dict:
    posting = {
        "element_id": value_element_id(relation_name, attribute_name, value),
        "element_type": "value",
        "relation": relation_name,
        "attribute": attribute_name,
        "text": value_text(relation_name, attribute_name, value, max_chars),
        "value": value,
        "search_ids": set(),
    }
    if search_id is not None:
        posting["search_ids"].add(int(search_id))
    return posting


def _attribute_posting(schema: Schema, relation: Relation, attribute_name: str, search_id: int | None) -> dict:
    attribute = relation.attribute(attribute_name)
    posting = {
        "element_id": attribute_element_id(relation.name, attribute_name),
        "element_type": "attribute",
        "relation": relation.name,
        "attribute": attribute_name,
        "text": attribute_text(relation, attribute),
        "value": "",
        "search_ids": set(),
    }
    if search_id is not None:
        posting["search_ids"].add(int(search_id))
    return posting


def _relation_posting(schema: Schema, relation: Relation) -> dict:
    return {
        "element_id": relation_element_id(relation.name),
        "element_type": "relation",
        "relation": relation.name,
        "attribute": "",
        "text": relation_text(relation, joined_relation_names(schema, relation.name)),
        "value": "",
        "search_ids": set(),
    }


def _hits_for_keyword(schema: Schema, index: dict[str, dict[str, dict]], keyword: str) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for posting in index.get(normalize_text(keyword), {}).values():
        merged[posting["element_id"]] = posting
    for relation_name in sorted(schema.relations):
        relation = schema.relations[relation_name]
        if _name_matches(relation.name, keyword):
            posting = _relation_posting(schema, relation)
            merged.setdefault(posting["element_id"], posting)
        for attribute in relation.keyword_attributes():
            if _name_matches(attribute.name, keyword):
                posting = _attribute_posting(schema, relation, attribute.name, None)
                merged.setdefault(posting["element_id"], posting)
    return merged


def _ensure_synthetic_intended(schema: Schema, source: dict, hits: dict[str, dict], max_chars: int) -> None:
    relation_name = source.get("relation") or ""
    attribute_name = source.get("attribute") or ""
    value = source.get("value") or ""
    if not relation_name or relation_name not in schema.relations:
        return
    relation = schema.relations[relation_name]
    hits.setdefault(relation_element_id(relation_name), _relation_posting(schema, relation))
    if attribute_name:
        hits.setdefault(
            attribute_element_id(relation_name, attribute_name),
            _attribute_posting(schema, relation, attribute_name, None),
        )
    if attribute_name and value:
        posting = _value_posting(relation_name, attribute_name, value, None, max_chars)
        hits.setdefault(posting["element_id"], posting)


def _label_hit(query: dict, source: dict, hit: dict, keyword: str) -> int:
    if query.get("source") == "synthetic":
        if hit["element_type"] == "value" and hit["element_id"] == value_element_id(
            source.get("relation", ""), source.get("attribute", ""), source.get("value", "")
        ):
            return 1
        if (
            hit["element_type"] == "attribute"
            and hit["relation"] == source.get("relation")
            and hit["attribute"] == source.get("attribute")
        ):
            return 1
        if hit["element_type"] == "relation" and hit["relation"] == source.get("relation"):
            return 1
        return 0
    judged = {int(search_id) for search_id in query.get("search_ids", [])}
    judged_relations = set(query.get("relations", []))
    search_ids = {int(search_id) for search_id in hit.get("search_ids", ())}
    if hit["element_type"] == "value":
        return 1 if search_ids & judged else 0
    if hit["element_type"] == "attribute":
        name_match = _name_matches(hit["attribute"], keyword) and hit["relation"] in judged_relations
        cell_match = bool(search_ids & judged)
        return 1 if name_match or cell_match else 0
    if hit["element_type"] == "relation":
        return 1 if _name_matches(hit["relation"], keyword) and hit["relation"] in judged_relations else 0
    return 0


def _pairs_for_keyword(
    schema: Schema,
    query: dict,
    source: dict,
    index: dict[str, dict[str, dict]],
    max_chars: int,
) -> list[dict]:
    keyword = source.get("keyword") or ""
    if not keyword:
        return []
    hits = _hits_for_keyword(schema, index, keyword)
    if query.get("source") == "synthetic":
        _ensure_synthetic_intended(schema, source, hits, max_chars)
    pairs = []
    for hit in hits.values():
        pairs.append(
            _pair(
                query,
                keyword,
                hit["element_id"],
                hit["element_type"],
                hit["relation"],
                hit["attribute"],
                hit["text"],
                _label_hit(query, source, hit, keyword),
            )
        )
    return pairs


def _pair(
    query: dict,
    keyword: str,
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
        "keyword": keyword,
        "element_id": element_id,
        "element_type": element_type,
        "relation": relation,
        "attribute": attribute,
        "element_text": text,
        "label": label,
        "class_id": query["class_id"],
        "source": query.get("source", "synthetic"),
        "topic_id": query.get("topic_id", ""),
        "fold": query.get("fold"),
        "split": query.get("split", "train"),
    }


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
