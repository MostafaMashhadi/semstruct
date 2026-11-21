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
from semstruct.paths import fold_model_dir, icn_dir, networks_dir, processed_dir, project_root
from semstruct.ranker.scoring import relevance_probabilities
from semstruct.runlog import emit

MAX_ANSWERS = 1000
ORDERS = ("semantic", "size")


def evaluate_dataset(
    dataset: str,
    root: Path | None = None,
    config: dict | None = None,
    k: int = MAX_ANSWERS,
    order: str = "semantic",
    tag: str = "",
) -> dict:
    """Score, join, and measure one database. The ranker weights are not updated.

    k: stop after this many distinct answers per topic.
    order: network ordering, either "semantic" (model score) or "size" (smallest first).
    tag: optional suffix for the output folder name, used to repeat a run.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    if k < 1:
        raise ValueError("k must be at least 1")
    root = root or project_root()
    config = config or load_config(dataset, root=root)
    directory = processed_dir(dataset, root)
    schema = Schema.from_dict(json.loads((directory / "schema.json").read_text(encoding="utf-8")))
    document = json.loads((networks_dir(dataset, root) / "primary-cns.json").read_text(encoding="utf-8"))
    topics = {topic.topic_id: topic for topic in load_topics(directory / "topics")}
    topic_ids = [str(query["topic_id"]) for query in document["queries"]]
    folds = assign_topic_folds(topic_ids, int(config["seed"]), int(config.get("n_folds", 5)))
    graph = schema_graph(schema)
    connection = sqlite3.connect(directory / "database.sqlite")
    destination = icn_dir(dataset, root) / f"{order}-k{k}{'-' + tag if tag else ''}"
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "run.log"
    try:
        queries = _evaluate_queries(
            dataset,
            document["queries"],
            topics,
            folds,
            schema,
            graph,
            connection,
            config,
            log_path,
            root,
            k,
            order,
        )
    finally:
        connection.close()
        release_device_cache()
    metrics = _metrics(dataset, queries)
    payload = {
        "dataset": dataset,
        "max_answers": k,
        "order": order,
        "queries": queries,
    }
    (destination / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    emit(
        (
            f"{dataset} | icn | topics {metrics['topics']} | top-1 {metrics['top1']} | "
            f"precision {metrics['precision']:.3f} | recall {metrics['recall']:.3f} | "
            f"f1 {metrics['f1']:.3f} | mrr {metrics['mrr']:.3f} | map {metrics['map']:.3f} | "
            f"networks {metrics['networks_evaluated']} | "
            f"query {metrics['query_ms']:.1f} ms | {destination}"
        ),
        log_path,
    )
    return metrics


def order_networks(networks: list[dict]) -> list[dict]:
    """Order every network. A low score changes the order and does not remove it.

    Higher scores come first. Fewer relations win a tie, then the network id.
    """
    return sorted(
        networks,
        key=lambda item: (-float(item["score"]), int(item["size"]), item["network_id"]),
    )


def answer_is_relevant(search_ids: list[int], answers: list[list[int]]) -> bool:
    """A join is relevant only when its search ids equal one judged answer."""
    found = set(search_ids)
    return any(found == set(answer) for answer in answers)


def answer_quality(relevant_flags: list[bool], judged: int) -> tuple[float, float | None, float | None]:
    """Precision, recall, and F1 for one topic under the exact search-id rule.

    A topic with no returned row has precision 0. A topic with no judged answer
    is not given a recall or an F1.
    """
    exact = sum(1 for flag in relevant_flags if flag)
    precision = exact / len(relevant_flags) if relevant_flags else 0.0
    if judged <= 0:
        return precision, None, None
    recall = exact / judged
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


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
    limit: int | None = None,
) -> list[list[int]]:
    """Run one join tree and return each row's search ids, in stable order."""
    cache = _JoinCache()
    rows = cache.combinations(connection, schema, graph, nodes, row_ids)
    if limit is not None:
        rows = rows[:limit]
    return [list(row) for row in rows]


def _collect_answers(connection, schema, graph, ordered, row_ids, answers, limit: int = MAX_ANSWERS):
    """Walk networks in score order and stop once `limit` distinct answers exist."""
    cache = _JoinCache()
    found: dict[frozenset[int], dict] = {}
    evaluated = 0
    for network in ordered:
        if len(found) >= limit:
            break
        evaluated += 1
        for combo in cache.combinations(connection, schema, graph, network["nodes"], row_ids):
            if len(found) >= limit:
                break
            key = frozenset(combo)
            if key in found:
                continue
            found[key] = {
                "score": float(network["score"]),
                "network_id": network["network_id"],
                "search_ids": sorted(key),
                "relevant": answer_is_relevant(sorted(key), answers),
                "size": int(network["size"]),
            }
    ranked_rows = sorted(
        found.values(),
        key=lambda item: (-float(item["score"]), int(item["size"]), item["search_ids"], item["network_id"]),
    )
    ranked = []
    for rank, item in enumerate(ranked_rows[:limit], start=1):
        ranked.append(
            {
                "rank": rank,
                "score": item["score"],
                "network_id": item["network_id"],
                "search_ids": item["search_ids"],
                "relevant": item["relevant"],
            }
        )
    return ranked, evaluated, cache.sql_ms


def _evaluate_queries(
    dataset,
    queries,
    topics,
    folds,
    schema,
    graph,
    connection,
    config,
    log_path,
    root,
    k,
    order,
) -> list[dict]:

    by_fold: dict[int, list[dict]] = {}
    for query in queries:
        by_fold.setdefault(folds[str(query["topic_id"])], []).append(query)
    records = []
    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])
    max_chars = int(config["max_value_chars"])
    for fold in sorted(by_fold):
        if order == "semantic":
            from semstruct.ranker.classifier import load_ranker

            checkpoint = fold_model_dir(dataset, fold, root)
            tokenizer, model = load_ranker(str(checkpoint))
            device = select_device()
            model.to(device)
            model.eval()
        else:
            tokenizer = None
            model = None
            device = None
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
                        k,
                        order,
                    )
                )
        finally:
            if model is not None:
                del model
                release_device_cache()
    records.sort(key=lambda item: int(item["topic_id"]))
    return records


def _evaluate_query(
    query,
    topic,
    fold,
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
    k,
    order,
) -> dict:
    answers = list(topic.answers) if topic is not None else []
    networks = query.get("networks", [])
    started = time.perf_counter()
    if order == "size":
        scored = [
            {
                "network_id": network["network_id"],
                "nodes": network["nodes"],
                "score": 0.0,
                "size": len(network["nodes"]),
            }
            for network in networks
        ]
    else:
        scored = _score_networks(
            networks,
            query,
            schema,
            cells,
            cache,
            tokenizer,
            model,
            device,
            batch_size,
            max_length,
        )
    scoring_ms = (time.perf_counter() - started) * 1000 + index_share
    ordered = order_networks(scored)
    ranked, evaluated, sql_ms = _collect_answers(connection, schema, graph, ordered, row_ids, answers, limit=k)
    relevant = sum(1 for answer in ranked if answer["relevant"])
    emit(
        (
            f"{dataset} | icn | topic {query['topic_id']} | fold {fold} | "
            f"networks {len(networks)} | evaluated {evaluated} | "
            f"answers {len(ranked)} | relevant {relevant}"
        ),
        log_path,
    )
    return {
        "topic_id": str(query["topic_id"]),
        "query": query["query"],
        "fold": fold,
        "networks_scored": len(networks),
        "networks_evaluated": evaluated,
        "judged": len(answers),
        "scoring_ms": scoring_ms,
        "sql_ms": sql_ms,
        "answers": ranked,
    }


def _score_networks(
    networks,
    query,
    schema,
    cells,
    cache,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
):
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
    return scored


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


class _JoinCache:
    """Tuple sets and partial joins for one topic. A stored prefix is not fetched again."""

    def __init__(self) -> None:
        self.rows: dict[tuple, list[tuple[int, ...]]] = {}
        self.extensions = 0
        self.sql_ms = 0.0

    def combinations(self, connection, schema, graph, nodes, row_ids) -> list[tuple[int, ...]]:
        relations = [node["relation"] for node in nodes]
        if len(set(relations)) != len(relations):
            return []
        edges = _tree_edges(relations, graph)
        if edges is None:
            return []
        predicates = {node["relation"]: _value_keywords(node) for node in nodes}
        return self._materialize(connection, schema, edges, predicates, frozenset(relations), row_ids)

    def _materialize(self, connection, schema, edges, predicates, subset, row_ids) -> list[tuple[int, ...]]:
        key = tuple(sorted((relation, predicates[relation]) for relation in subset))
        cached = self.rows.get(key)
        if cached is not None:
            return cached
        ordered = [relation for relation, _keywords in key]
        if len(subset) == 1:
            started = time.perf_counter()
            identifiers = _relation_ids(connection, ordered[0], predicates[ordered[0]], row_ids)
            if not predicates[ordered[0]]:
                self.sql_ms += (time.perf_counter() - started) * 1000
            stored = [(value,) for value in identifiers]
            self.rows[key] = stored
            return stored
        added = _subset_leaves(subset, edges)[-1]
        rest = subset - {added}
        base = self._materialize(connection, schema, edges, predicates, rest, row_ids)
        if not base:
            self.rows[key] = []
            return []
        neighbor = _neighbor(added, rest, edges)
        base_order = [relation for relation, _keywords in tuple(sorted((relation, predicates[relation]) for relation in rest))]
        started = time.perf_counter()
        extended = _extend_join(
            connection,
            schema,
            base_order,
            base,
            neighbor,
            added,
            predicates[added],
            row_ids,
        )
        self.sql_ms += (time.perf_counter() - started) * 1000
        self.extensions += 1
        canonical = {relation: index for index, relation in enumerate(ordered)}
        base_index = {relation: index for index, relation in enumerate(base_order)}
        aligned = []
        for combo in extended:
            row = [0] * len(ordered)
            for relation, index in base_index.items():
                row[canonical[relation]] = combo[index]
            row[canonical[added]] = combo[-1]
            aligned.append(tuple(row))
        self.rows[key] = aligned
        return aligned


def _value_keywords(node: dict) -> tuple[str, ...]:
    return tuple(sorted(str(item["keyword"]) for item in node["keywords"] if item.get("role") == "value"))


def _relation_ids(connection, relation: str, keywords: tuple[str, ...], row_ids) -> list[int]:
    if not keywords:
        statement = f"SELECT {quote_ident('__search_id')} FROM {quote_ident(relation)} ORDER BY 1"
        return [int(row[0]) for row in connection.execute(statement)]
    shared: set[int] | None = None
    for keyword in keywords:
        hits = row_ids.get((relation, keyword), set())
        shared = set(hits) if shared is None else shared & hits
        if not shared:
            return []
    return sorted(shared or ())


def _subset_leaves(subset: frozenset[str], edges: list[tuple[str, str]]) -> list[str]:
    degree = {relation: 0 for relation in subset}
    for left, right in edges:
        if left in subset and right in subset:
            degree[left] += 1
            degree[right] += 1
    return sorted(relation for relation, count in degree.items() if count <= 1)


def _neighbor(relation: str, rest: frozenset[str], edges: list[tuple[str, str]]) -> str:
    for left, right in edges:
        if left == relation and right in rest:
            return right
        if right == relation and left in rest:
            return left
    raise ValueError(f"{relation} is not connected to the stored prefix")


def _extend_join(connection, schema, base_order, base_rows, neighbor, added, keywords, row_ids) -> list[tuple[int, ...]]:
    allowed = _relation_ids(connection, added, keywords, row_ids) if keywords else None
    if allowed is not None and not allowed:
        return []
    connection.execute("DROP TABLE IF EXISTS semstruct_icn_partial")
    columns = ", ".join(f"c{index} INTEGER NOT NULL" for index in range(len(base_order)))
    connection.execute(f"CREATE TEMP TABLE semstruct_icn_partial ({columns})")
    placeholders = ", ".join("?" for _ in base_order)
    connection.executemany(
        f"INSERT INTO semstruct_icn_partial VALUES ({placeholders})",
        base_rows,
    )
    connection.execute("DROP TABLE IF EXISTS semstruct_icn_keep")
    if allowed is not None:
        connection.execute("CREATE TEMP TABLE semstruct_icn_keep (search_id INTEGER NOT NULL)")
        connection.executemany(
            "INSERT INTO semstruct_icn_keep (search_id) VALUES (?)",
            ((value,) for value in allowed),
        )
    aliases = {neighbor: "boundary", added: "added"}
    condition = _join_condition(schema, neighbor, added, aliases)
    if condition is None:
        connection.execute("DROP TABLE IF EXISTS semstruct_icn_partial")
        connection.execute("DROP TABLE IF EXISTS semstruct_icn_keep")
        return []
    neighbor_index = base_order.index(neighbor)
    selected = ", ".join(f"partial.c{index}" for index in range(len(base_order)))
    statement = (
        f"SELECT {selected}, added.{quote_ident('__search_id')} "
        f"FROM semstruct_icn_partial AS partial "
        f"JOIN {quote_ident(neighbor)} AS boundary "
        f"ON boundary.{quote_ident('__search_id')} = partial.c{neighbor_index} "
        f"JOIN {quote_ident(added)} AS added ON {condition}"
    )
    if allowed is not None:
        statement += (
            f" WHERE added.{quote_ident('__search_id')} IN (SELECT search_id FROM semstruct_icn_keep)"
        )
    order = ", ".join(
        [f"partial.c{index}" for index in range(len(base_order))] + [f"added.{quote_ident('__search_id')}"]
    )
    statement += f" ORDER BY {order}"
    rows = [tuple(int(value) for value in row) for row in connection.execute(statement)]
    connection.execute("DROP TABLE IF EXISTS semstruct_icn_partial")
    connection.execute("DROP TABLE IF EXISTS semstruct_icn_keep")
    return rows


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


def _metrics(dataset: str, queries: list[dict]) -> dict:
    labels = [[1 if answer["relevant"] else 0 for answer in query["answers"]] for query in queries]
    measured = retrieval_metrics(labels)
    scoring = [float(query["scoring_ms"]) for query in queries]
    sql = [float(query["sql_ms"]) for query in queries]
    precisions = []
    recalls = []
    f1_scores = []
    for query in queries:
        flags = [bool(answer["relevant"]) for answer in query["answers"]]
        precision, recall, f1 = answer_quality(flags, int(query.get("judged", 0)))
        precisions.append(precision)
        if recall is not None and f1 is not None:
            recalls.append(recall)
            f1_scores.append(f1)
    count = len(queries)
    return {
        "dataset": dataset,
        "topics": count,
        "top1": int(measured["top1"]),
        "mrr": measured["mrr"],
        "map": measured["map"],
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "networks_scored": sum(int(query["networks_scored"]) for query in queries),
        "networks_evaluated": sum(int(query["networks_evaluated"]) for query in queries),
        "scoring_ms": sum(scoring) / count if count else 0.0,
        "sql_ms": sum(sql) / count if count else 0.0,
        "query_ms": sum(left + right for left, right in zip(scoring, sql)) / count if count else 0.0,
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
