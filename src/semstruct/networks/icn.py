"""Score primary candidate networks and evaluate the joined answers.

A topic is scored by the fold that held it out. The network score is the mean
keyword probability. The best networks become foreign-key joins on the SQLite
copy of the benchmark. A joined row is relevant when its search-id set equals
one Coffman answer.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from semstruct.config import load_config
from semstruct.data.elements import attribute_text, relation_text, tokenize, value_text
from semstruct.data.elements import joined_relation_names
from semstruct.data.instances import assign_topic_folds
from semstruct.data.schema import Schema, quote_ident, schema_graph
from semstruct.data.topics import load_topics
from semstruct.device import release_device_cache, select_device
from semstruct.networks.primary import _contains_keyword_tokens
from semstruct.paths import fold_model_dir, fold_output_dir, icn_dir, networks_dir, processed_dir, project_root
from semstruct.ranker.scoring import relevance_probabilities
from semstruct.runlog import emit

MAX_NETWORKS = 30
MAX_ANSWERS = 1000


def evaluate_dataset(dataset: str, root: Path | None = None, config: dict | None = None) -> dict:
    """Score, join, and measure one database. The ranker weights are not updated."""
    root = root or project_root()
    config = config or load_config(dataset, root=root)
    directory = processed_dir(dataset, root)
    schema = Schema.from_dict(json.loads((directory / "schema.json").read_text(encoding="utf-8")))
    document = json.loads((networks_dir(dataset, root) / "primary-cns.json").read_text(encoding="utf-8"))
    topics = {topic.topic_id: topic for topic in load_topics(directory / "topics")}
    topic_ids = [str(query["topic_id"]) for query in document["queries"]]
    folds = assign_topic_folds(topic_ids, int(config["seed"]), int(config.get("n_folds", 5)))
    thresholds = {
        fold: _fold_threshold(dataset, fold, root)
        for fold in sorted(set(folds.values()))
    }
    graph = schema_graph(schema)
    connection = sqlite3.connect(directory / "database.sqlite")
    destination = icn_dir(dataset, root)
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "run.log"
    try:
        queries = _evaluate_queries(
            dataset,
            document["queries"],
            topics,
            folds,
            thresholds,
            schema,
            graph,
            connection,
            config,
            log_path,
            root,
        )
    finally:
        connection.close()
        release_device_cache()
    metrics = _metrics(dataset, queries)
    payload = {
        "dataset": dataset,
        "max_networks": MAX_NETWORKS,
        "max_answers": MAX_ANSWERS,
        "queries": queries,
    }
    (destination / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    emit(
        (
            f"{dataset} | icn | topics {metrics['topics']} | top-1 {metrics['top1']} | "
            f"mrr {metrics['mrr']:.3f} | map {metrics['map']:.3f} | "
            f"scoring {metrics['scoring_ms']:.1f} ms | sql {metrics['sql_ms']:.1f} ms | {destination}"
        ),
        log_path,
    )
    return metrics


def select_networks(networks: list[dict], threshold: float, limit: int = MAX_NETWORKS) -> list[dict]:
    """Keep networks at or above the threshold, then the best `limit`.

    An empty selection falls back to the single best network. Fewer relations
    win a tie, then the network id.
    """
    if not networks:
        return []
    kept = [network for network in networks if float(network["score"]) >= threshold]
    if not kept:
        kept = [max(networks, key=lambda item: (float(item["score"]), -int(item["size"]), item["network_id"]))]
    kept.sort(key=lambda item: (-float(item["score"]), int(item["size"]), item["network_id"]))
    return kept[:limit]


def answer_is_relevant(search_ids: list[int], answers: list[list[int]]) -> bool:
    """A join is relevant only when its search ids equal one judged answer."""
    found = set(search_ids)
    return any(found == set(answer) for answer in answers)


def retrieval_metrics(labels_by_topic: list[list[int]]) -> dict[str, float]:
    """Top-1 count, mean reciprocal rank, and mean average precision.

    A topic with no relevant answer contributes zero to the rank and the precision.
    """
    if not labels_by_topic:
        return {"top1": 0.0, "mrr": 0.0, "map": 0.0}
    top1 = 0
    reciprocal = 0.0
    average = 0.0
    for labels in labels_by_topic:
        if labels and labels[0] == 1:
            top1 += 1
        reciprocal += _reciprocal(labels)
        average += _average_precision(labels)
    count = len(labels_by_topic)
    return {"top1": top1, "mrr": reciprocal / count, "map": average / count}


def joined_answers(
    connection: sqlite3.Connection,
    schema: Schema,
    graph: dict[str, set[str]],
    nodes: list[dict],
    row_ids: dict[tuple[str, str], set[int]],
    limit: int = MAX_ANSWERS,
) -> list[list[int]]:
    """Run one join tree and return each row's search ids, in stable order."""
    relations = [node["relation"] for node in nodes]
    if len(set(relations)) != len(relations):
        return []
    edges = _tree_edges(relations, graph)
    if edges is None:
        return []
    filters: dict[str, set[int] | None] = {}
    for node in nodes:
        keywords = [item["keyword"] for item in node["keywords"] if item.get("role") == "value"]
        if not keywords:
            filters[node["relation"]] = None
            continue
        shared: set[int] | None = None
        for keyword in keywords:
            hits = row_ids.get((node["relation"], keyword), set())
            shared = set(hits) if shared is None else shared & hits
        if not shared:
            return []
        filters[node["relation"]] = shared
    return _execute_join(connection, schema, relations, edges, filters, limit)


def _evaluate_queries(
    dataset,
    queries,
    topics,
    folds,
    thresholds,
    schema,
    graph,
    connection,
    config,
    log_path,
    root,
) -> list[dict]:
    from semstruct.ranker.classifier import load_ranker

    by_fold: dict[int, list[dict]] = {}
    for query in queries:
        by_fold.setdefault(folds[str(query["topic_id"])], []).append(query)
    records = []
    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])
    max_chars = int(config["max_value_chars"])
    for fold in sorted(by_fold):
        checkpoint = fold_model_dir(dataset, fold, root)
        tokenizer, model = load_ranker(str(checkpoint))
        device = select_device()
        model.to(device)
        model.eval()
        fold_networks = [network for query in by_fold[fold] for network in query.get("networks", [])]
        index_started = time.perf_counter()
        cells, row_ids = _value_index(connection, schema, fold_networks, max_chars)
        index_share = ((time.perf_counter() - index_started) * 1000) / max(len(by_fold[fold]), 1)
        cache = _ScoreCache()
        try:
            for query in by_fold[fold]:
                records.append(
                    _evaluate_query(
                        query,
                        topics.get(str(query["topic_id"])),
                        fold,
                        thresholds[fold],
                        schema,
                        graph,
                        connection,
                        tokenizer,
                        model,
                        device,
                        batch_size,
                        max_length,
                        cells,
                        row_ids,
                        cache,
                        log_path,
                        dataset,
                        index_share,
                    )
                )
        finally:
            del model
            release_device_cache()
    records.sort(key=lambda item: int(item["topic_id"]))
    return records


def _evaluate_query(
    query,
    topic,
    fold,
    threshold,
    schema,
    graph,
    connection,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
    cells,
    row_ids,
    cache,
    log_path,
    dataset,
    index_share,
) -> dict:
    answers = list(topic.answers) if topic is not None else []
    networks = query.get("networks", [])
    started = time.perf_counter()
    assignments: dict[tuple, float] = {}
    for network in networks:
        for node in network["nodes"]:
            for item in node["keywords"]:
                key = (str(item["keyword"]), node["relation"], str(item.get("role")), str(item.get("attribute", "")))
                if key not in assignments:
                    assignments[key] = _keyword_probability(item, node["relation"], query["query"], schema, cells, cache)
    cache.run(model, tokenizer, batch_size, device, max_length)
    for keyword, relation_name, role, attribute in list(assignments):
        item = {"keyword": keyword, "role": role, "attribute": attribute}
        assignments[(keyword, relation_name, role, attribute)] = _keyword_probability(
            item, relation_name, query["query"], schema, cells, cache
        )
    scored = []
    for network in networks:
        probabilities = [
            assignments[(str(item["keyword"]), node["relation"], str(item.get("role")), str(item.get("attribute", "")))]
            for node in network["nodes"]
            for item in node["keywords"]
        ]
        scored.append(
            {
                "network_id": network["network_id"],
                "nodes": network["nodes"],
                "score": sum(probabilities) / len(probabilities) if probabilities else 0.0,
                "size": len(network["nodes"]),
            }
        )
    scoring_ms = (time.perf_counter() - started) * 1000 + index_share
    selected = select_networks(scored, threshold)
    sql_started = time.perf_counter()
    ranked = _rank_selected(connection, schema, graph, selected, row_ids, answers)
    sql_ms = (time.perf_counter() - sql_started) * 1000
    relevant = sum(1 for answer in ranked if answer["relevant"])
    emit(
        (
            f"{dataset} | icn | topic {query['topic_id']} | fold {fold} | "
            f"networks {len(networks)} | selected {len(selected)} | "
            f"answers {len(ranked)} | relevant {relevant}"
        ),
        log_path,
    )
    return {
        "topic_id": str(query["topic_id"]),
        "query": query["query"],
        "fold": fold,
        "threshold": threshold,
        "networks_scored": len(networks),
        "networks_selected": len(selected),
        "scoring_ms": scoring_ms,
        "sql_ms": sql_ms,
        "answers": ranked,
    }


def _rank_selected(connection, schema, graph, selected, row_ids, answers) -> list[dict]:
    found: dict[frozenset[int], dict] = {}
    seen_sql: set[tuple] = set()
    for network in selected:
        if len(found) >= MAX_ANSWERS:
            break
        signature = _predicate_key(network)
        if signature in seen_sql:
            continue
        seen_sql.add(signature)
        rows = joined_answers(connection, schema, graph, network["nodes"], row_ids, MAX_ANSWERS)
        for search_ids in rows:
            key = frozenset(search_ids)
            current = found.get(key)
            if current is not None and float(current["score"]) >= float(network["score"]):
                continue
            found[key] = {
                "score": float(network["score"]),
                "network_id": network["network_id"],
                "search_ids": sorted(key),
                "relevant": answer_is_relevant(sorted(key), answers),
                "size": int(network["size"]),
            }
    ordered = sorted(
        found.values(),
        key=lambda item: (-float(item["score"]), int(item["size"]), item["search_ids"], item["network_id"]),
    )
    ranked = []
    for rank, item in enumerate(ordered[:MAX_ANSWERS], start=1):
        ranked.append(
            {
                "rank": rank,
                "score": item["score"],
                "network_id": item["network_id"],
                "search_ids": item["search_ids"],
                "relevant": item["relevant"],
            }
        )
    return ranked


def _keyword_probability(item, relation_name, query, schema, cells, cache) -> float:
    keyword = str(item["keyword"])
    role = str(item.get("role"))
    if role == "value":
        texts = cells.get((keyword, relation_name, str(item.get("attribute", ""))), ())
        if not texts:
            return 0.0
        return max(cache.lookup(keyword, query, text) for text in texts)
    relation = schema.relations[relation_name]
    if role == "attribute":
        text = attribute_text(relation, relation.attribute(str(item["attribute"])))
    else:
        text = relation_text(relation, joined_relation_names(schema, relation_name))
    return cache.lookup(keyword, query, text)


def _value_index(connection, schema, networks, max_chars):
    """Row ids per value keyword, and the cell texts of each stored attribute."""
    keywords_by_relation: dict[str, set[str]] = {}
    attributes: dict[tuple[str, str, str], set[str]] = {}
    for network in networks:
        for node in network["nodes"]:
            for item in node["keywords"]:
                if item.get("role") != "value":
                    continue
                keywords_by_relation.setdefault(node["relation"], set()).add(str(item["keyword"]))
                attributes[(str(item["keyword"]), node["relation"], str(item.get("attribute", "")))] = set()
    row_ids: dict[tuple[str, str], set[int]] = {}
    key_tokens = {keyword: tokenize(keyword) for keywords in keywords_by_relation.values() for keyword in keywords}
    token_to_keywords: dict[str, set[str]] = {}
    for keyword, tokens in key_tokens.items():
        for token in set(tokens):
            token_to_keywords.setdefault(token, set()).add(keyword)
    for relation_name, keywords in keywords_by_relation.items():
        relation = schema.relations[relation_name]
        columns = relation.keyword_attributes()
        if not columns:
            continue
        names = [attribute.name for attribute in columns]
        statement = (
            f"SELECT {quote_ident('__search_id')}, "
            + ", ".join(quote_ident(name) for name in names)
            + f" FROM {quote_ident(relation.name)}"
        )
        for row in connection.execute(statement):
            search_id = int(row[0])
            for attribute_name, raw in zip(names, row[1:]):
                if raw is None:
                    continue
                text = str(raw).strip()
                if not text:
                    continue
                tokens = tokenize(text)
                if not tokens:
                    continue
                found: set[str] = set()
                for token in set(tokens):
                    found.update(token_to_keywords.get(token, ()))
                for keyword in found:
                    if keyword not in keywords:
                        continue
                    if _contains_keyword_tokens(tokens, key_tokens[keyword]):
                        row_ids.setdefault((relation_name, keyword), set()).add(search_id)
                        key = (keyword, relation_name, attribute_name)
                        if key in attributes:
                            attributes[key].add(value_text(relation_name, attribute_name, text, max_chars))
    cells = {key: tuple(sorted(texts)) for key, texts in attributes.items()}
    return cells, row_ids


class _ScoreCache:
    def __init__(self) -> None:
        self._pending: list[tuple[str, str, str]] = []
        self._index: dict[tuple[str, str, str], int] = {}
        self._scores: list[float] = []

    def lookup(self, keyword: str, query: str, text: str) -> float:
        key = (keyword, query, text)
        if key not in self._index:
            self._index[key] = len(self._pending)
            self._pending.append(key)
        position = self._index[key]
        if position < len(self._scores):
            return self._scores[position]
        return 0.0

    def run(self, model, tokenizer, batch_size: int, device, max_length: int) -> None:
        fresh = self._pending[len(self._scores) :]
        if not fresh:
            return
        self._scores.extend(
            relevance_probabilities(
                model,
                tokenizer,
                [item[0] for item in fresh],
                [item[1] for item in fresh],
                [item[2] for item in fresh],
                batch_size=batch_size,
                device=device,
                max_length=max_length,
            )
        )


def _predicate_key(network: dict) -> tuple:
    predicates = []
    for node in network["nodes"]:
        values = tuple(
            sorted(
                (str(item["keyword"]), str(item.get("attribute", "")))
                for item in node["keywords"]
                if item.get("role") == "value"
            )
        )
        predicates.append((node["relation"], values))
    return tuple(predicates)


def _tree_edges(relations: list[str], graph: dict[str, set[str]]) -> list[tuple[str, str]] | None:
    if not relations:
        return None
    seen = {relations[0]}
    edges = []
    for name in relations[1:]:
        parent = next((earlier for earlier in relations if earlier in seen and name in graph.get(earlier, ())), None)
        if parent is None:
            return None
        edges.append((parent, name))
        seen.add(name)
    return edges


def _execute_join(connection, schema, relations, edges, filters, limit: int) -> list[list[int]]:
    aliases = {relation: f"r{index}" for index, relation in enumerate(relations)}
    connection.execute("DROP TABLE IF EXISTS semstruct_icn_keep")
    connection.execute("CREATE TEMP TABLE semstruct_icn_keep (slot INTEGER NOT NULL, search_id INTEGER NOT NULL)")
    for relation, allowed in filters.items():
        if allowed is None:
            continue
        slot = relations.index(relation)
        connection.executemany(
            "INSERT INTO semstruct_icn_keep (slot, search_id) VALUES (?, ?)",
            ((slot, int(search_id)) for search_id in allowed),
        )
    select = ", ".join(f"{aliases[relation]}.{quote_ident('__search_id')}" for relation in relations)
    statement = f"SELECT {select} FROM {quote_ident(relations[0])} AS {aliases[relations[0]]}"
    for parent, child in edges:
        condition = _join_condition(schema, parent, child, aliases)
        if condition is None:
            connection.execute("DROP TABLE IF EXISTS semstruct_icn_keep")
            return []
        statement += f" JOIN {quote_ident(child)} AS {aliases[child]} ON {condition}"
    where = []
    for relation, allowed in filters.items():
        if allowed is None:
            continue
        slot = relations.index(relation)
        alias = aliases[relation]
        where.append(
            f"{alias}.{quote_ident('__search_id')} IN "
            f"(SELECT search_id FROM semstruct_icn_keep WHERE slot = {slot})"
        )
    if where:
        statement += " WHERE " + " AND ".join(where)
    order = ", ".join(f"{aliases[relation]}.{quote_ident('__search_id')}" for relation in relations)
    statement += f" ORDER BY {order} LIMIT {int(limit)}"
    rows = [list(map(int, row)) for row in connection.execute(statement)]
    connection.execute("DROP TABLE IF EXISTS semstruct_icn_keep")
    return rows


def _join_condition(schema: Schema, parent: str, child: str, aliases: dict[str, str]) -> str | None:
    pair = _foreign_key(schema, child, parent)
    if pair is not None:
        source, target, foreign_key = child, parent, pair
    else:
        pair = _foreign_key(schema, parent, child)
        if pair is None:
            return None
        source, target, foreign_key = parent, child, pair
    parts = [
        f"{aliases[source]}.{quote_ident(column)} = {aliases[target]}.{quote_ident(referenced)}"
        for column, referenced in zip(foreign_key.columns, foreign_key.ref_columns)
    ]
    return " AND ".join(parts)


def _foreign_key(schema: Schema, source: str, target: str):
    matches = [
        foreign_key
        for foreign_key in schema.relations[source].foreign_keys
        if foreign_key.ref_table == target
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item.columns, item.ref_columns))
    return matches[0]


def _fold_threshold(dataset: str, fold: int, root: Path) -> float:
    path = fold_output_dir(dataset, fold, root) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"No validation threshold at {path}. Run evaluate --fold {fold} first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["finetuned_distilbert"]["threshold"])


def _metrics(dataset: str, queries: list[dict]) -> dict:
    labels = [[1 if answer["relevant"] else 0 for answer in query["answers"]] for query in queries]
    measured = retrieval_metrics(labels)
    scoring = [float(query["scoring_ms"]) for query in queries]
    sql = [float(query["sql_ms"]) for query in queries]
    return {
        "dataset": dataset,
        "topics": len(queries),
        "top1": int(measured["top1"]),
        "mrr": measured["mrr"],
        "map": measured["map"],
        "scoring_ms": sum(scoring) / len(scoring) if scoring else 0.0,
        "sql_ms": sum(sql) / len(sql) if sql else 0.0,
    }


def _reciprocal(labels: list[int]) -> float:
    for rank, label in enumerate(labels, start=1):
        if label:
            return 1.0 / rank
    return 0.0


def _average_precision(labels: list[int]) -> float:
    relevant = sum(labels)
    if relevant == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, label in enumerate(labels, start=1):
        if label:
            hits += 1
            total += hits / rank
    return total / relevant
