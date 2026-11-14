"""Relational schema objects and identifier quoting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_EXCLUDED_NAME_PARTS = ("password", "passwd", "token", "email")
_TEXT_TYPES = {
    "text",
    "varchar",
    "character varying",
    "char",
    "character",
    "name",
    "citext",
    "string",
    "clob",
    "bpchar",
}


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def unqualified(name: str) -> str:
    """Drop a schema qualifier and surrounding quotes. `public.country` becomes `country`."""
    text = name.strip().rstrip(";")
    if text.upper().startswith("ONLY "):
        text = text[5:].strip()
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    for char in text:
        if char == '"':
            in_quote = not in_quote
            current.append(char)
        elif char == "." and not in_quote:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    last = parts[-1].strip() if parts else text
    if len(last) >= 2 and last.startswith('"') and last.endswith('"'):
        return last[1:-1].replace('""', '"')
    return last


def is_text_type(data_type: str) -> bool:
    base = re.sub(r"\(.*\)", "", data_type).strip().lower()
    return base in _TEXT_TYPES


@dataclass
class ForeignKey:
    columns: list[str]
    ref_table: str
    ref_columns: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "ref_table": self.ref_table,
            "ref_columns": list(self.ref_columns),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ForeignKey:
        return cls(
            columns=list(payload["columns"]),  # type: ignore[arg-type]
            ref_table=str(payload["ref_table"]),
            ref_columns=list(payload["ref_columns"]),  # type: ignore[arg-type]
        )


@dataclass
class Attribute:
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Attribute:
        return cls(
            name=str(payload["name"]),
            data_type=str(payload["data_type"]),
            nullable=bool(payload.get("nullable", True)),
            primary_key=bool(payload.get("primary_key", False)),
        )


def is_keyword_attribute(attribute: Attribute) -> bool:
    """Text columns that can supply a keyword. Internal ids and credentials are excluded."""
    if attribute.name == "__search_id":
        return False
    lowered = attribute.name.lower()
    if any(part in lowered for part in _EXCLUDED_NAME_PARTS):
        return False
    return is_text_type(attribute.data_type)


@dataclass
class Relation:
    name: str
    attributes: list[Attribute] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    comment: str | None = None

    def attribute(self, name: str) -> Attribute:
        for item in self.attributes:
            if item.name == name:
                return item
        raise KeyError(name)

    def keyword_attributes(self) -> list[Attribute]:
        return [item for item in self.attributes if is_keyword_attribute(item)]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "attributes": [item.as_dict() for item in self.attributes],
            "primary_key": list(self.primary_key),
            "foreign_keys": [item.as_dict() for item in self.foreign_keys],
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Relation:
        return cls(
            name=str(payload["name"]),
            attributes=[Attribute.from_dict(item) for item in payload.get("attributes", [])],  # type: ignore[union-attr]
            primary_key=list(payload.get("primary_key", [])),  # type: ignore[arg-type]
            foreign_keys=[ForeignKey.from_dict(item) for item in payload.get("foreign_keys", [])],  # type: ignore[union-attr]
            comment=payload.get("comment") if payload.get("comment") is None else str(payload.get("comment")),
        )


@dataclass
class Schema:
    relations: dict[str, Relation] = field(default_factory=dict)

    def add(self, relation: Relation) -> None:
        self.relations[relation.name] = relation

    def as_dict(self) -> dict[str, object]:
        return {"relations": [relation.as_dict() for relation in self.relations.values()]}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Schema:
        relations = [Relation.from_dict(item) for item in payload.get("relations", [])]  # type: ignore[union-attr]
        return cls(relations={relation.name: relation for relation in relations})


def schema_graph(schema: Schema) -> dict[str, set[str]]:
    """Undirected foreign-key graph. A self-reference does not connect two relations."""
    adjacency = {name: set() for name in schema.relations}
    for relation in schema.relations.values():
        for foreign_key in relation.foreign_keys:
            if foreign_key.ref_table == relation.name:
                continue
            if foreign_key.ref_table not in adjacency:
                continue
            adjacency[relation.name].add(foreign_key.ref_table)
            adjacency[foreign_key.ref_table].add(relation.name)
    return adjacency
