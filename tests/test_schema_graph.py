from semstruct.data.query_classes import class_is_connected, connected_subgraphs, select_query_classes
from semstruct.data.schema import Attribute, ForeignKey, Relation, Schema, schema_graph


def _mini_schema() -> Schema:
    schema = Schema()
    schema.add(Relation("country", [Attribute("code", "character varying"), Attribute("name", "text")]))
    schema.add(
        Relation(
            "city",
            [Attribute("name", "text"), Attribute("country", "character varying")],
            foreign_keys=[ForeignKey(["country"], "country", ["code"])],
        )
    )
    schema.add(
        Relation(
            "river",
            [Attribute("name", "text"), Attribute("country", "character varying")],
            foreign_keys=[ForeignKey(["country"], "country", ["code"])],
        )
    )
    return schema


def test_size_two_classes_follow_foreign_keys():
    adjacency = schema_graph(_mini_schema())
    classes = connected_subgraphs(adjacency, max_size=2)
    as_sets = {tuple(sorted(item)) for item in classes}
    assert ("city", "country") in as_sets
    assert ("country", "river") in as_sets
    assert ("city", "river") not in as_sets


def test_size_three_class_is_connected_and_capped():
    schema = _mini_schema()
    adjacency = schema_graph(schema)
    classes = connected_subgraphs(adjacency, max_size=3)
    triple = frozenset({"country", "city", "river"})
    assert triple in classes
    assert class_is_connected(adjacency, ["country", "city", "river"])
    assert not class_is_connected(adjacency, ["city", "river"])
    capped = select_query_classes(schema, max_relations=3, max_classes=2, seed=1)
    assert len(capped) == 2
    assert all(class_id.startswith("class:") for class_id, _names in capped)
