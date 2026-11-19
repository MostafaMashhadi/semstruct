"""Primary candidate networks as bounded join trees of tuple sets.

The construction follows DISCOVER and the ICN paper: a network is a tree of
relations on the foreign-key graph, not one network per matching cell.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

from semstruct.data.elements import STOPWORDS, tokenize
from semstruct.data.prepare import read_jsonl
from semstruct.data.schema import Schema, quote_ident, schema_graph
from semstruct.paths import networks_dir, processed_dir, project_root, ranker_dir
from semstruct.runlog import emit

MAX_RELATIONS = 5


@dataclass(frozen=True)
class TupleSet:
    keyword: str
    role: str
    relation: str
    attribute: str
    match_count: int

    def as_keyword(self) -> dict[str, object]:
        payload: dict[str, object] = {"keyword": self.keyword, "role": self.role}
        if self.attribute:
            payload["attribute"] = self.attribute
        payload["match_count"] = self.match_count
        return payload


def query_keywords(query: str) -> list[str]:
    tokens = [token for token in tokenize(query) if token not in STOPWORDS and len(token) > 1]
    if not tokens:
        tokens = [token for token in tokenize(query) if token]
    ordered: list[str] = []
    for token in tokens:
        if token not in ordered:
            ordered.append(token)
    return ordered


def generate_networks(
    connection: sqlite3.Connection,
    schema: Schema,
    topics: list[dict],
    *,
    max_relations: int = MAX_RELATIONS,
) -> dict:
    """Return the primary-CN document for these topics."""
    keywords = sorted({keyword for topic in topics for keyword in query_keywords(str(topic["query"]))})
    value_sets, row_bits = _value_tuple_sets(connection, schema, keywords)
    graph = schema_graph(schema)
    queries = []
    for topic in topics:
        topic_keywords = query_keywords(str(topic["query"]))
        choices = {
            keyword: _tuple_sets_for_keyword(schema, keyword, value_sets.get(keyword, []))
            for keyword in topic_keywords
        }
        networks = _networks_for_keywords(choices, graph, max_relations, row_bits)
        for index, network in enumerate(networks, start=1):
            network["network_id"] = f"t{int(topic['topic_id']):05d}-{index}"
        emit(f"primary cns | topic {int(topic['topic_id'])} | networks {len(networks)}")
        queries.append(
            {
                "query_id": f"t{int(topic['topic_id']):05d}",
                "topic_id": str(topic["topic_id"]),
                "query": topic["query"],
                "networks": networks,
            }
        )
    return {"max_relations": max_relations, "queries": queries}


def generate_dataset(dataset: str, root: Path | None = None, max_relations: int = MAX_RELATIONS) -> dict:
    root = root or project_root()
    directory = processed_dir(dataset, root)
    schema = Schema.from_dict(json.loads((directory / "schema.json").read_text(encoding="utf-8")))
    topics = read_jsonl(directory / "topics.jsonl")
    connection = sqlite3.connect(directory / "database.sqlite")
    try:
        document = generate_networks(connection, schema, topics, max_relations=max_relations)
    finally:
        connection.close()
    document["dataset"] = dataset
    destination = networks_dir(dataset, root)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "primary-cns.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    empty = [item["topic_id"] for item in document["queries"] if not item["networks"]]
    network_count = sum(len(item["networks"]) for item in document["queries"])
    summary = {
        "dataset": dataset,
        "topics": len(document["queries"]),
        "networks": network_count,
        "topics_without_networks": empty,
        "path": str(path),
    }
    log_dir = ranker_dir(dataset, root)
    log_dir.mkdir(parents=True, exist_ok=True)
    emit(
        (
            f"{dataset} | primary cns | topics {summary['topics']} | "
            f"networks {network_count} | topics without networks {len(empty)}"
            + (f" | {', '.join(empty)}" if empty else "")
            + f" | {path}"
        ),
        log_dir / "prepare.log",
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _tuple_sets_for_keyword(schema: Schema, keyword: str, value_sets: list[TupleSet]) -> list[TupleSet]:
    found = list(value_sets)
    seen = {(item.role, item.relation, item.attribute) for item in found}
    for relation_name in sorted(schema.relations):
        relation = schema.relations[relation_name]
        if _name_matches(relation.name, keyword) and ("relation", relation.name, "") not in seen:
            found.append(TupleSet(keyword, "relation", relation.name, "", 1))
        for attribute in relation.keyword_attributes():
            if _name_matches(attribute.name, keyword):
                key = ("attribute", relation.name, attribute.name)
                if key not in seen:
                    found.append(TupleSet(keyword, "attribute", relation.name, attribute.name, 1))
    found.sort(key=lambda item: (item.role, item.relation, item.attribute))
    return found


def _value_tuple_sets(
    connection: sqlite3.Connection,
    schema: Schema,
    keywords: list[str],
) -> tuple[dict[str, list[TupleSet]], dict[tuple[str, str], int]]:
    """Cell counts per keyword, plus the rows in which each keyword occurs.

    Keywords placed on the same relation are one tuple set only when a single
    row contains every one of them. The bitsets record those rows.
    """
    token_to_keywords: dict[str, set[str]] = defaultdict(set)
    key_tokens = {keyword: tokenize(keyword) for keyword in keywords}
    for keyword, tokens in key_tokens.items():
        for token in set(tokens):
            token_to_keywords[token].add(keyword)
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    row_bits: dict[tuple[str, str], int] = defaultdict(int)
    if token_to_keywords:
        for relation_name in sorted(schema.relations):
            relation = schema.relations[relation_name]
            columns = relation.keyword_attributes()
            if not columns:
                continue
            names = [attribute.name for attribute in columns]
            statement = (
                "SELECT "
                + ", ".join(quote_ident(name) for name in names)
                + f" FROM {quote_ident(relation.name)}"
            )
            emit(f"primary cns | indexing {relation.name}")
            for row_index, row in enumerate(connection.execute(statement)):
                bit = 1 << row_index
                matched_in_row: set[str] = set()
                for attribute_name, raw in zip(names, row):
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
                    for keyword in matched:
                        if _contains_keyword_tokens(tokens, key_tokens[keyword]):
                            counts[(keyword, relation.name, attribute_name)] += 1
                            matched_in_row.add(keyword)
                for keyword in matched_in_row:
                    row_bits[(keyword, relation.name)] |= bit
    grouped: dict[str, list[TupleSet]] = defaultdict(list)
    collapsed: dict[tuple[str, str], tuple[str, int]] = {}
    for (keyword, relation_name, attribute_name), count in counts.items():
        key = (keyword, relation_name)
        current = collapsed.get(key)
        if current is None or count > current[1]:
            collapsed[key] = (attribute_name, count)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for (keyword, relation_name, _attribute_name), count in counts.items():
        totals[(keyword, relation_name)] += count
    for (keyword, relation_name), (attribute_name, _best) in sorted(collapsed.items()):
        grouped[keyword].append(
            TupleSet(keyword, "value", relation_name, attribute_name, totals[(keyword, relation_name)])
        )
    return grouped, row_bits


def _networks_for_keywords(
    choices: dict[str, list[TupleSet]],
    graph: dict[str, set[str]],
    max_relations: int,
    row_bits: dict[tuple[str, str], int],
) -> list[dict]:
    keywords = list(choices)
    if not keywords or any(not choices[keyword] for keyword in keywords):
        return []
    valid_masks = _valid_keyword_masks(keywords, choices, row_bits)
    relations = sorted(valid_masks)
    if not relations:
        return []
    networks: list[dict] = []
    seen: set[tuple] = set()
    connectors: dict[frozenset[str], list[frozenset[str]]] = {}
    role_cache: dict[tuple[str, int], tuple[tuple[TupleSet, ...], ...]] = {}

    def roles_for(relation_name: str, mask: int) -> tuple[tuple[TupleSet, ...], ...]:
        key = (relation_name, mask)
        cached = role_cache.get(key)
        if cached is not None:
            return cached
        chosen_keywords = [keywords[index] for index in range(len(keywords)) if mask >> index & 1]
        found: list[tuple[TupleSet, ...]] = []

        def pick(index: int, acc: int | None, chosen: list[TupleSet]) -> None:
            if index == len(chosen_keywords):
                found.append(tuple(chosen))
                return
            keyword = chosen_keywords[index]
            for choice in choices[keyword]:
                if choice.relation != relation_name:
                    continue
                if choice.role == "value":
                    bits = row_bits.get((keyword, relation_name), 0)
                    shared = bits if acc is None else acc & bits
                    if not shared:
                        continue
                    chosen.append(choice)
                    pick(index + 1, shared, chosen)
                    chosen.pop()
                else:
                    chosen.append(choice)
                    pick(index + 1, acc, chosen)
                    chosen.pop()

        pick(0, None, [])
        stored = tuple(found)
        role_cache[key] = stored
        return stored

    def emit_groups(groups: dict[str, tuple[TupleSet, ...]]) -> None:
        terminals = sorted(groups)
        key = frozenset(terminals)
        if key not in connectors:
            connectors[key] = _connecting_node_sets(terminals, graph, max_relations)
        by_relation = {name: list(items) for name, items in groups.items()}
        for node_set in connectors[key]:
            signature = _signature(node_set, by_relation)
            if signature in seen:
                continue
            seen.add(signature)
            ordered = _bfs_order(node_set, min(terminals), graph)
            nodes = []
            for relation_name in ordered:
                keyword_sets = sorted(
                    by_relation.get(relation_name, []),
                    key=lambda item: (item.keyword, item.role, item.attribute),
                )
                nodes.append(
                    {
                        "relation": relation_name,
                        "keywords": [item.as_keyword() for item in keyword_sets],
                    }
                )
            networks.append({"network_id": "", "nodes": nodes})

    full = (1 << len(keywords)) - 1
    for size in range(1, min(max_relations, len(relations)) + 1):
        for combo in combinations(relations, size):
            if any(not valid_masks[name] for name in combo):
                continue
            terminal_key = frozenset(combo)
            if terminal_key not in connectors:
                connectors[terminal_key] = _connecting_node_sets(list(combo), graph, max_relations)
            if not connectors[terminal_key]:
                continue
            if not _keywords_fit(keywords, choices, set(combo)):
                continue
            allowed = tuple(valid_masks[name] for name in combo)

            @lru_cache(maxsize=None)
            def covers(index: int, remaining: int) -> tuple[tuple[tuple[str, int], ...], ...]:
                if index == len(combo):
                    return ((),) if remaining == 0 else ()
                found: list[tuple[tuple[str, int], ...]] = []
                subset = remaining
                relation_name = combo[index]
                relation_masks = allowed[index]
                while subset:
                    if subset in relation_masks:
                        for tail in covers(index + 1, remaining ^ subset):
                            found.append(((relation_name, subset),) + tail)
                    subset = (subset - 1) & remaining
                return tuple(found)

            for chosen in covers(0, full):
                _expand_roles(chosen, roles_for, emit_groups)
            covers.cache_clear()
    return networks


def _expand_roles(chosen, roles_for, emit_groups) -> None:
    pools = [roles_for(relation_name, mask) for relation_name, mask in chosen]
    if any(not pool for pool in pools):
        return
    for picks in product(*pools):
        emit_groups({relation_name: items for (relation_name, _mask), items in zip(chosen, picks)})


def _keywords_fit(keywords: list[str], choices: dict[str, list[TupleSet]], relations: set[str]) -> bool:
    return all(any(item.relation in relations for item in choices[keyword]) for keyword in keywords)


def _valid_keyword_masks(
    keywords: list[str],
    choices: dict[str, list[TupleSet]],
    row_bits: dict[tuple[str, str], int],
) -> dict[str, set[int]]:
    """Keyword subsets that can share a relation.

    Value keywords in the subset must occur together in one row. A keyword
    matched through a relation or attribute name does not need a row.
    """
    masks: dict[str, set[int]] = {}
    relations = sorted({item.relation for options in choices.values() for item in options})
    width = len(keywords)
    for relation_name in relations:
        value_counts = [0] * width
        schema_counts = [0] * width
        bits = [0] * width
        for index, keyword in enumerate(keywords):
            for item in choices[keyword]:
                if item.relation != relation_name:
                    continue
                if item.role == "value":
                    value_counts[index] += 1
                    bits[index] = row_bits.get((keyword, relation_name), 0)
                else:
                    schema_counts[index] += 1
        found: set[int] = set()

        def walk(index: int, mask: int, shared: int | None) -> None:
            if index == width:
                if mask:
                    found.add(mask)
                return
            walk(index + 1, mask, shared)
            if value_counts[index] and bits[index]:
                updated = bits[index] if shared is None else shared & bits[index]
                if updated:
                    walk(index + 1, mask | (1 << index), updated)
            if schema_counts[index]:
                walk(index + 1, mask | (1 << index), shared)

        walk(0, 0, None)
        if found:
            masks[relation_name] = found
    return masks


def _connecting_node_sets(
    terminals: list[str],
    adjacency: dict[str, set[str]],
    limit: int,
) -> list[frozenset[str]]:
    terminals_set = set(terminals)
    if not terminals_set or len(terminals_set) > limit:
        return []
    found: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()

    def connected(nodes: set[str]) -> bool:
        start = next(iter(terminals_set))
        if start not in nodes:
            return False
        reached: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            for neighbor in adjacency.get(node, ()):
                if neighbor in nodes and neighbor not in reached:
                    stack.append(neighbor)
        return terminals_set <= reached

    def minimal(nodes: set[str]) -> bool:
        if not connected(nodes):
            return False
        for node in nodes:
            degree = sum(1 for neighbor in adjacency.get(node, ()) if neighbor in nodes)
            if degree <= 1 and node not in terminals_set:
                return False
        for node in nodes - terminals_set:
            if connected(nodes - {node}):
                return False
        return True

    def rec(nodes: set[str]) -> None:
        key = frozenset(nodes)
        if key in seen or len(nodes) > limit:
            return
        seen.add(key)
        if terminals_set <= nodes and connected(nodes):
            if minimal(nodes):
                found.append(key)
            return
        if len(nodes) == limit:
            return
        neighbors: set[str] = set()
        for node in nodes:
            neighbors.update(adjacency.get(node, ()))
        for nxt in sorted(neighbors - nodes):
            rec(nodes | {nxt})

    rec(set(terminals_set))
    return found


def _signature(node_set: frozenset[str], by_relation: dict[str, list[TupleSet]]) -> tuple:
    assigned = []
    for relation_name in sorted(node_set):
        keywords = tuple(
            sorted(
                (item.keyword, item.role, item.attribute)
                for item in by_relation.get(relation_name, [])
            )
        )
        assigned.append((relation_name, keywords))
    return tuple(assigned)


def _bfs_order(nodes: frozenset[str], start: str, adjacency: dict[str, set[str]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node in seen or node not in nodes:
            continue
        seen.add(node)
        ordered.append(node)
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor in nodes and neighbor not in seen:
                queue.append(neighbor)
    for node in sorted(nodes - seen):
        ordered.append(node)
    return ordered


def _name_matches(name: str, keyword: str) -> bool:
    return tokenize(name) == tokenize(keyword)


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
