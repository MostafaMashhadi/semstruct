import json
import sqlite3
from pathlib import Path

from semstruct.data.dump import load_sqlite
from semstruct.data.schema import schema_graph
from semstruct.networks.primary import generate_networks

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
    "\\.\n"
    "COPY river (name, country, __search_id) FROM stdin;\n"
    "Rhine\tCH\t6\n"
    "Seine\tFR\t7\n"
    "\\.\n"
)


def _database(tmp_path: Path):
    schema_path = tmp_path / "schema.sql"
    data_path = tmp_path / "data.sql"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    data_path.write_text(DATA, encoding="utf-8")
    database = tmp_path / "mini.sqlite"
    schema = load_sqlite(schema_path, data_path, database)
    return schema, sqlite3.connect(database)


def _network(document: dict, query: str) -> dict:
    topic = next(item for item in document["queries"] if item["query"] == query)
    assert topic["networks"]
    return topic


def test_keywords_join_through_the_linking_relation(tmp_path: Path):
    schema, connection = _database(tmp_path)
    document = generate_networks(
        connection,
        schema,
        [{"topic_id": "1", "query": "bern rhine"}],
    )
    connection.close()
    topic = _network(document, "bern rhine")
    relations = [node["relation"] for node in topic["networks"][0]["nodes"]]
    assert relations[0] in {"city", "river"}
    assert "country" in relations
    assert set(relations) == {"city", "country", "river"}
    free = [node for node in topic["networks"][0]["nodes"] if not node["keywords"]]
    assert [node["relation"] for node in free] == ["country"]
    keywords = {
        item["keyword"]
        for node in topic["networks"][0]["nodes"]
        for item in node["keywords"]
    }
    assert keywords == {"bern", "rhine"}


def test_a_free_relation_is_not_a_leaf(tmp_path: Path):
    schema, connection = _database(tmp_path)
    document = generate_networks(connection, schema, [{"topic_id": "1", "query": "bern rhine"}])
    connection.close()
    graph = schema_graph(schema)
    for network in document["queries"][0]["networks"]:
        nodes = {node["relation"] for node in network["nodes"]}
        for node in network["nodes"]:
            degree = sum(1 for neighbor in graph[node["relation"]] if neighbor in nodes)
            if degree <= 1:
                assert node["keywords"]


def test_networks_larger_than_the_cap_are_dropped(tmp_path: Path):
    schema, connection = _database(tmp_path)
    document = generate_networks(
        connection,
        schema,
        [{"topic_id": "1", "query": "bern rhine"}],
        max_relations=2,
    )
    connection.close()
    assert document["queries"][0]["networks"] == []


def test_value_keywords_on_one_relation_share_a_row(tmp_path: Path):
    schema, connection = _database(tmp_path)
    document = generate_networks(
        connection,
        schema,
        [{"topic_id": "1", "query": "bern paris"}],
    )
    connection.close()
    assert document["queries"][0]["networks"] == []


def test_duplicate_trees_are_stored_once(tmp_path: Path):
    schema, connection = _database(tmp_path)
    document = generate_networks(connection, schema, [{"topic_id": "1", "query": "bern rhine"}])
    connection.close()
    signatures = [
        tuple((node["relation"], tuple(item["keyword"] for item in node["keywords"])) for node in network["nodes"])
        for network in document["queries"][0]["networks"]
    ]
    assert len(signatures) == len(set(signatures))
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert path.is_file()
