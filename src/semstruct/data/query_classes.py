"""Connected relation sets. A query class is one of these sets."""

from __future__ import annotations

import random

from semstruct.data.schema import Schema, schema_graph


def connected_subgraphs(adjacency: dict[str, set[str]], max_size: int) -> list[frozenset[str]]:
    nodes = sorted(adjacency)
    found: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    frontier: list[frozenset[str]] = []
    for node in nodes:
        subset = frozenset([node])
        seen.add(subset)
        found.append(subset)
        frontier.append(subset)
    for _size in range(2, max_size + 1):
        grown: list[frozenset[str]] = []
        for subset in frontier:
            neighbors: set[str] = set()
            for node in subset:
                neighbors |= adjacency[node]
            for candidate in sorted(neighbors - subset):
                nxt = subset | {candidate}
                if nxt in seen:
                    continue
                seen.add(nxt)
                found.append(nxt)
                grown.append(nxt)
        frontier = grown
    return found


def select_query_classes(schema: Schema, max_relations: int, max_classes: int, seed: int) -> list[tuple[str, list[str]]]:
    """Return `(class_id, relation names)` with a deterministic cap."""
    classes = connected_subgraphs(schema_graph(schema), max_relations)
    singles = [item for item in classes if len(item) == 1]
    larger = [item for item in classes if len(item) > 1]
    larger = sorted(larger, key=lambda item: (len(item), tuple(sorted(item))))
    if len(singles) >= max_classes:
        chosen = sorted(singles, key=lambda item: tuple(sorted(item)))[:max_classes]
    elif len(singles) + len(larger) > max_classes:
        generator = random.Random(seed)
        generator.shuffle(larger)
        chosen = singles + larger[: max_classes - len(singles)]
    else:
        chosen = singles + larger
    records = []
    for subset in chosen:
        names = sorted(subset)
        records.append(("class:" + "|".join(names), names))
    records.sort(key=lambda item: item[0])
    return records


def class_is_connected(adjacency: dict[str, set[str]], relations: list[str]) -> bool:
    if not relations:
        return False
    seen = {relations[0]}
    stack = [relations[0]]
    allowed = set(relations)
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == allowed
