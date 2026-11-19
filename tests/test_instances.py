import sqlite3
from pathlib import Path

from semstruct.data.dump import load_sqlite
from semstruct.data.elements import normalize_text
from semstruct.data.instances import (
    assign_splits,
    assign_topic_folds,
    build_instances,
    build_keyword_pairs,
    build_topic_instances,
    cap_value_negatives,
    fold_roles,
    sample_example_values,
)
from semstruct.data.topics import Topic
from semstruct.data.query_classes import select_query_classes

SCHEMA = """
CREATE TABLE country (
    code character varying(4) NOT NULL,
    name character varying(32) NOT NULL,
    __search_id integer NOT NULL
);
CREATE TABLE city (
    name character varying(35) NOT NULL,
    country character varying(4) NOT NULL,
    __search_id integer NOT NULL
);
CREATE TABLE river (
    name character varying(35) NOT NULL,
    country character varying(4) NOT NULL,
    __search_id integer NOT NULL
);
ALTER TABLE ONLY country ADD CONSTRAINT country_pkey PRIMARY KEY (code);
ALTER TABLE ONLY city ADD CONSTRAINT city_country_fkey FOREIGN KEY (country) REFERENCES country(code);
ALTER TABLE ONLY river ADD CONSTRAINT river_country_fkey FOREIGN KEY (country) REFERENCES country(code);
"""

DATA = (
    "INSERT INTO country (code, name, __search_id) VALUES ('CH', 'Switzerland', 1);\n"
    "INSERT INTO country (code, name, __search_id) VALUES ('FR', 'France', 2);\n"
    "COPY city (name, country, __search_id) FROM stdin;\n"
    "Bern\tCH\t3\n"
    "Paris\tFR\t4\n"
    "Zurich\tCH\t5\n"
    "France\tFR\t8\n"
    "\\.\n"
    "COPY river (name, country, __search_id) FROM stdin;\n"
    "Rhine\tCH\t6\n"
    "Seine\tFR\t7\n"
    "\\.\n"
)

CONFIG = {
    "seed": 3,
    "max_relations": 3,
    "max_classes": 20,
    "queries_per_class": 6,
    "negatives_per_positive": 2,
    "hard_negative_ratio": 0.5,
    "max_pairs": 500,
    "max_values_per_attribute": 10,
    "example_values_per_attribute": 5,
    "max_keywords": 2,
    "min_keywords": 1,
    "max_value_chars": 80,
    "join_timeout_seconds": 5,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "value_negative_cap": 10,
    "n_folds": 5,
}


def _build(tmp_path: Path):
    schema_path = tmp_path / "schema.sql"
    data_path = tmp_path / "data.sql"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    data_path.write_text(DATA, encoding="utf-8")
    database = tmp_path / "mini.sqlite"
    schema = load_sqlite(schema_path, data_path, database)
    connection = sqlite3.connect(database)
    sampled = sample_example_values(connection, schema, 10, 80, __import__("random").Random(3))
    examples = {key: values[:5] for key, values in sampled.items()}
    classes = select_query_classes(schema, 3, 20, 3)
    queries, pairs, _stats = build_instances(
        connection,
        schema,
        classes,
        examples,
        sampled,
        {normalize_text("france")},
        CONFIG,
    )
    connection.close()
    return schema, classes, queries, pairs


def test_fifty_topics_split_ten_eight_thirty_two():
    topic_ids = [str(index) for index in range(1, 51)]
    folds = assign_topic_folds(topic_ids, seed=42)
    assert set(folds) == set(topic_ids)
    assert sorted(folds.values()).count(0) == 10
    assert {fold: list(folds.values()).count(fold) for fold in range(5)} == {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}
    tested = []
    for fold in range(5):
        roles = fold_roles(topic_ids, fold, seed=42)
        assert sum(role == "test" for role in roles.values()) == 10
        assert sum(role == "validation" for role in roles.values()) == 8
        assert sum(role == "train" for role in roles.values()) == 32
        tested.extend(topic_id for topic_id, role in roles.items() if role == "test")
        assert {topic_id for topic_id, assigned in folds.items() if assigned == fold} == {
            topic_id for topic_id, role in roles.items() if role == "test"
        }
    assert sorted(tested, key=int) == topic_ids


def test_value_negatives_are_capped_at_ten():
    pairs = [
        {"query_id": "q", "keyword": "france", "element_type": "value", "label": 0, "element_id": f"v{index}"}
        for index in range(25)
    ]
    pairs.append({"query_id": "q", "keyword": "france", "element_type": "value", "label": 1, "element_id": "gold"})
    pairs.append({"query_id": "q", "keyword": "france", "element_type": "relation", "label": 0, "element_id": "rel"})
    capped = cap_value_negatives(pairs, seed=42, cap=10)
    assert sum(pair["element_type"] == "value" and pair["label"] == 0 for pair in capped) == 10
    assert any(pair["element_id"] == "gold" for pair in capped)
    assert any(pair["element_id"] == "rel" for pair in capped)


def test_coffman_topic_becomes_relation_attribute_and_value_pairs(tmp_path: Path):
    schema_path = tmp_path / "schema.sql"
    data_path = tmp_path / "data.sql"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    data_path.write_text(DATA, encoding="utf-8")
    database = tmp_path / "mini.sqlite"
    schema = load_sqlite(schema_path, data_path, database)
    connection = sqlite3.connect(database)
    sampled = sample_example_values(connection, schema, 10, 80, __import__("random").Random(3))
    examples = {key: values[:5] for key, values in sampled.items()}
    topics = [
        Topic(topic_id="1", query="france", search_ids=[2]),
        Topic(topic_id="2", query="switzerland", search_ids=[1]),
        Topic(topic_id="3", query="rhine", search_ids=[6]),
    ]
    queries, pairs, _stats, folds = build_topic_instances(
        connection,
        schema,
        topics,
        examples,
        sampled,
        CONFIG,
    )
    connection.close()
    assert set(folds.values()) == {0, 1, 2}
    france = next(query for query in queries if query["query"] == "france")
    assert france["source"] == "coffman"
    france_pairs = [pair for pair in pairs if pair["query_id"] == france["query_id"] and pair["keyword"] == "france"]
    assert france_pairs
    assert {pair["fold"] for pair in france_pairs} == {france["fold"]}
    values = {
        (pair["relation"], pair["attribute"], pair["label"])
        for pair in france_pairs
        if pair["element_type"] == "value"
    }
    assert ("country", "name", 1) in values
    assert ("city", "name", 0) in values
    attributes = {
        (pair["relation"], pair["attribute"], pair["label"])
        for pair in france_pairs
        if pair["element_type"] == "attribute"
    }
    assert ("country", "name", 1) in attributes
    assert ("city", "name", 0) in attributes


def test_splits_are_query_disjoint():
    splits = assign_splits(["q1", "q2", "q3", "q4", "q5"], seed=1, train_ratio=0.8, val_ratio=0.1)
    assert set(splits) == {"q1", "q2", "q3", "q4", "q5"}
    assert set(splits.values()) == {"train", "validation", "test"}


def test_labels_match_query_classes_and_examples_do_not_leak(tmp_path: Path):
    _schema, _classes, queries, pairs = _build(tmp_path)
    assert len(queries) >= 3
    by_query = {query["query_id"]: query for query in queries}
    assert all(query["query"] != "france" for query in queries)
    assert all(query["split"] == "train" for query in queries)
    negatives: dict[tuple[str, str], int] = {}
    for pair in pairs:
        query = by_query[pair["query_id"]]
        assert pair["keyword"]
        assert pair["split"] == "train"
        source = next(item for item in query["keyword_sources"] if item["keyword"] == pair["keyword"])
        if pair["label"] == 1 and pair["element_type"] == "relation":
            assert pair["relation"] == source["relation"]
        if pair["label"] == 1 and pair["element_type"] == "attribute":
            assert pair["relation"] == source["relation"]
            assert pair["attribute"] == source["attribute"]
        if pair["label"] == 1 and pair["element_type"] == "value":
            assert pair["keyword"] in normalize_text(pair["element_text"])
        if pair["element_type"] == "value" and pair["label"] == 0:
            key = (pair["query_id"], pair["keyword"])
            negatives[key] = negatives.get(key, 0) + 1
    assert all(count <= 10 for count in negatives.values())


def test_synthetic_keyword_keeps_its_source_and_one_other_hit(tmp_path: Path):
    schema_path = tmp_path / "schema.sql"
    data_path = tmp_path / "data.sql"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    data_path.write_text(DATA, encoding="utf-8")
    database = tmp_path / "mini.sqlite"
    schema = load_sqlite(schema_path, data_path, database)
    connection = sqlite3.connect(database)
    query = {
        "query_id": "q00000",
        "query": "france paris",
        "source": "synthetic",
        "class_id": "country|city",
        "relations": ["country", "city"],
        "keyword_attributes": ["country.name", "city.name"],
        "keywords": ["france", "paris"],
        "keyword_sources": [
            {"keyword": "france", "relation": "country", "attribute": "name", "value": "France"},
            {"keyword": "paris", "relation": "city", "attribute": "name", "value": "Paris"},
        ],
        "split": "train",
    }
    pairs = build_keyword_pairs(connection, schema, [query], CONFIG)
    connection.close()
    france = [pair for pair in pairs if pair["keyword"] == "france"]
    assert any(pair["element_type"] == "relation" and pair["relation"] == "country" and pair["label"] == 1 for pair in france)
    assert any(pair["element_type"] == "attribute" and pair["relation"] == "country" and pair["label"] == 1 for pair in france)
    assert any(pair["element_type"] == "value" and pair["relation"] == "country" and pair["label"] == 1 for pair in france)
    assert any(pair["element_type"] == "value" and pair["relation"] == "city" and pair["label"] == 0 for pair in france)
    assert sum(pair["element_type"] == "value" and pair["label"] == 0 for pair in france) <= 10


def test_same_query_is_not_in_two_classes(tmp_path: Path):
    _schema, classes, queries, _pairs = _build(tmp_path)
    assert any(len(relations) >= 2 for _class_id, relations in classes)
    texts = [query["query"] for query in queries]
    assert len(texts) == len(set(texts))
    assert (tmp_path / "mini.sqlite").exists()
