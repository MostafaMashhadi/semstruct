import sqlite3
from pathlib import Path

from semstruct.data.dump import load_sqlite
from semstruct.data.elements import normalize_text
from semstruct.data.instances import assign_splits, build_instances, sample_example_values
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


def test_splits_are_query_disjoint():
    splits = assign_splits(["q1", "q2", "q3", "q4", "q5"], seed=1, train_ratio=0.8, val_ratio=0.1)
    assert set(splits) == {"q1", "q2", "q3", "q4", "q5"}
    assert set(splits.values()) == {"train", "validation", "test"}


def test_labels_match_query_classes_and_examples_do_not_leak(tmp_path: Path):
    _schema, _classes, queries, pairs = _build(tmp_path)
    assert len(queries) >= 3
    by_query = {query["query_id"]: query for query in queries}
    splits: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    assert all(query["query"] != "france" for query in queries)
    for pair in pairs:
        query = by_query[pair["query_id"]]
        splits[pair["split"]].add(pair["query_id"])
        if pair["label"] == 1 and pair["element_type"] == "relation":
            assert pair["relation"] in query["relations"]
        if pair["label"] == 0 and pair["element_type"] == "relation":
            assert pair["relation"] not in query["relations"]
        if pair["label"] == 1 and pair["element_type"] == "attribute":
            assert f"{pair['relation']}.{pair['attribute']}" in query["keyword_attributes"]
        if pair["element_type"] == "attribute" and "example values:" in pair["element_text"]:
            examples = pair["element_text"].split("example values: ", 1)[1]
            for keyword in query["keywords"]:
                assert normalize_text(keyword) not in normalize_text(examples)
        if pair["label"] == 1 and pair["element_type"] == "value":
            assert any(keyword in normalize_text(pair["element_text"]) for keyword in query["keywords"])
    assert splits["train"].isdisjoint(splits["validation"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["validation"].isdisjoint(splits["test"])
    assert all(splits.values())


def test_same_query_is_not_in_two_classes(tmp_path: Path):
    _schema, classes, queries, _pairs = _build(tmp_path)
    assert any(len(relations) >= 2 for _class_id, relations in classes)
    texts = [query["query"] for query in queries]
    assert len(texts) == len(set(texts))
    assert (tmp_path / "mini.sqlite").exists()
