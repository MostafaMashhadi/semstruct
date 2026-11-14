"""Text forms of relations, attributes, and cell values."""

from __future__ import annotations

import re

from semstruct.data.schema import Attribute, Relation, Schema

STOPWORDS = frozenset(
    """
    a an the of and or to for in on at by from with as is are was were be been
    that this these those it its into over under between among per than then
    """.split()
)

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip().lower()


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(value.replace("_", " "))]


def content_tokens(value: str) -> set[str]:
    tokens = [token for token in tokenize(value) if token not in STOPWORDS and len(token) > 1]
    if tokens:
        return set(tokens)
    return {token for token in tokenize(value) if token}


def has_letter(value: str) -> bool:
    return any(character.isalpha() for character in value)


def keyword_from_value(value: str, max_chars: int) -> str | None:
    """Turn one cell into the keyword string a user might type."""
    if value is None:
        return None
    surface = re.sub(r"\s+", " ", str(value).replace("_", " ")).strip()
    if not surface or len(surface) > max_chars or not has_letter(surface):
        return None
    tokens = tokenize(surface)
    if not tokens:
        return None
    if len(tokens) <= 4:
        keyword = " ".join(tokens)
    else:
        kept = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
        if not kept:
            return None
        keyword = " ".join(kept[:2])
    if len(keyword) < 2:
        return None
    return keyword


def conflicts_with_keywords(example: str, keywords: list[str]) -> bool:
    normalized_example = normalize_text(example)
    if not normalized_example:
        return False
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue
        if (
            normalized_keyword == normalized_example
            or normalized_keyword in normalized_example
            or normalized_example in normalized_keyword
        ):
            return True
    return False


def reference_label(relation: Relation, attribute_name: str) -> str | None:
    labels = []
    for foreign_key in relation.foreign_keys:
        if attribute_name in foreign_key.columns:
            labels.append(f"{foreign_key.ref_table}({', '.join(foreign_key.ref_columns)})")
    if not labels:
        return None
    return "; ".join(labels)


def relation_text(relation: Relation) -> str:
    attributes = ", ".join(attribute.name for attribute in relation.attributes)
    parts = [f"[RELATION] name: {relation.name}", f"attributes: {attributes}"]
    if relation.primary_key:
        parts.append("primary key: " + ", ".join(relation.primary_key))
    if relation.foreign_keys:
        edges = []
        for foreign_key in relation.foreign_keys:
            source = ", ".join(foreign_key.columns)
            target = f"{foreign_key.ref_table}({', '.join(foreign_key.ref_columns)})"
            edges.append(f"{source} -> {target}")
        parts.append("foreign keys: " + "; ".join(edges))
    if relation.comment:
        parts.append(f"description: {relation.comment}")
    return " | ".join(parts)


def attribute_text(
    relation: Relation,
    attribute: Attribute,
    examples: list[str],
    query_keywords: list[str] | None = None,
) -> str:
    keywords = query_keywords or []
    safe = [example for example in examples if not conflicts_with_keywords(example, keywords)]
    parts = [
        f"[ATTRIBUTE] name: {relation.name}.{attribute.name}",
        f"type: {attribute.data_type}",
        f"relation: {relation.name}",
    ]
    refers_to = reference_label(relation, attribute.name)
    if refers_to:
        parts.append(f"refers to: {refers_to}")
    if safe:
        parts.append("example values: " + "; ".join(safe))
    return " | ".join(parts)


def value_text(relation_name: str, attribute_name: str, value: str, max_chars: int) -> str:
    shown = re.sub(r"\s+", " ", str(value).replace("\n", " ")).strip()
    if len(shown) > max_chars:
        shown = shown[:max_chars]
    normalized = normalize_text(shown)
    parts = [f'[VALUE] "{shown}"']
    if normalized and normalized != shown.lower():
        parts.append(f"normalized: {normalized}")
    parts.append(f"attribute: {relation_name}.{attribute_name}")
    parts.append(f"relation: {relation_name}")
    return " | ".join(parts)


def relation_element_id(relation_name: str) -> str:
    return f"relation:{relation_name}"


def attribute_element_id(relation_name: str, attribute_name: str) -> str:
    return f"attribute:{relation_name}.{attribute_name}"


def value_element_id(relation_name: str, attribute_name: str, value: str) -> str:
    return f"value:{relation_name}.{attribute_name}:{normalize_text(value)}"


def schema_element_records(schema: Schema, examples: dict[str, list[str]]) -> list[dict[str, str]]:
    records = []
    for relation_name in sorted(schema.relations):
        relation = schema.relations[relation_name]
        records.append(
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
            records.append(
                {
                    "element_id": attribute_element_id(relation.name, attribute.name),
                    "element_type": "attribute",
                    "relation": relation.name,
                    "attribute": attribute.name,
                    "text": attribute_text(relation, attribute, examples.get(key, [])),
                }
            )
    return records
