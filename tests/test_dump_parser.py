from pathlib import Path

from semstruct.data.dump import (
    load_sqlite,
    parse_copy_header,
    parse_insert,
    read_schema,
    split_copy_row,
    unescape_copy_field,
)
from semstruct.paths import project_root

SCHEMA = """
CREATE TABLE country (
    code character varying(4) NOT NULL,
    name character varying(32) NOT NULL,
    capital character varying(35),
    __search_id integer NOT NULL,
    CONSTRAINT country_code_check CHECK ((code <> ''::text))
);

CREATE FUNCTION skip_me() RETURNS integer
    AS $$
 BEGIN
 SELECT 1;
 END;
 $$
    LANGUAGE plpgsql;

CREATE TABLE city (
    name character varying(35) NOT NULL,
    country character varying(4) REFERENCES country(code),
    population integer,
    __search_id integer NOT NULL
);

CREATE TABLE river (
    name character varying(35) NOT NULL,
    country character varying(4) NOT NULL,
    __search_id integer NOT NULL
);

ALTER TABLE ONLY country
    ADD CONSTRAINT country_pkey PRIMARY KEY (code);

ALTER TABLE ONLY city
    ADD CONSTRAINT city_pkey PRIMARY KEY (name, country);

ALTER TABLE ONLY river
    ADD CONSTRAINT river_country_fkey FOREIGN KEY (country) REFERENCES country(code);
"""

DATA = (
    "INSERT INTO country (code, name, capital, __search_id) VALUES ('CH', 'Cote d''Ivoire', 'Bern', 1);\n"
    "INSERT INTO country (code, name, capital, __search_id) VALUES ('FR', NULL, 'Paris', 2);\n"
    "COPY city (name, country, population, __search_id) FROM stdin;\n"
    "Bern\tCH\t133000\t3\n"
    "Paris\tFR\t\\N\t4\n"
    "\\.\n"
    "COPY river (name, country, __search_id) FROM stdin;\n"
    "Rhine\tCH\t6\n"
    "Seine\tFR\t7\n"
    "\\.\n"
)


def test_schema_survives_dollar_quotes(tmp_path: Path):
    schema_path = tmp_path / "schema.sql"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    schema = read_schema(schema_path)
    assert set(schema.relations) == {"country", "city", "river"}
    assert schema.relations["country"].primary_key == ["code"]
    assert schema.relations["city"].primary_key == ["name", "country"]
    city_refs = {(tuple(item.columns), item.ref_table) for item in schema.relations["city"].foreign_keys}
    assert (("country",), "country") in city_refs
    river_refs = {item.ref_table for item in schema.relations["river"].foreign_keys}
    assert river_refs == {"country"}
    assert schema.relations["country"].attribute("name").data_type.startswith("character varying")


def test_insert_and_copy_literals():
    parsed = parse_insert("INSERT INTO country (code, name) VALUES ('Cote d''Ivoire', 'it\\'s');")
    assert parsed is not None
    _table, columns, tuples = parsed
    assert columns == ["code", "name"]
    assert tuples == [["Cote d'Ivoire", "it's"]]
    header = parse_copy_header("COPY public.city (name, country) FROM stdin;")
    assert header == ("city", ["name", "country"])
    assert split_copy_row("Bern\tCH\t\\N") == ["Bern", "CH", None]
    assert unescape_copy_field(r"A\\B") == "A\\B"


def test_sqlite_round_trip(tmp_path: Path):
    schema_path = tmp_path / "schema.sql"
    data_path = tmp_path / "data.sql"
    schema_path.write_text(SCHEMA, encoding="utf-8")
    data_path.write_text(DATA, encoding="utf-8")
    database = tmp_path / "mini.sqlite"
    load_sqlite(schema_path, data_path, database)
    import sqlite3

    connection = sqlite3.connect(database)
    names = {row[0] for row in connection.execute('SELECT name FROM country')}
    assert "Cote d'Ivoire" in names
    assert connection.execute("SELECT capital FROM country WHERE code = 'FR'").fetchone()[0] == "Paris"
    assert connection.execute("SELECT population FROM city WHERE name = 'Paris'").fetchone()[0] is None
    search = dict(connection.execute("SELECT search_id, relation FROM semstruct_search_id"))
    assert search[3] == "city"
    assert search[6] == "river"
    connection.close()


def test_benchmark_schemas_when_present():
    raw = project_root() / "data" / "raw"
    expected = {
        "mondial-schema.pgsql.bz2": {"country", "city", "borders"},
        "imdb2009_subset-schema.pgsql.bz2": {"title", "name", "cast_info"},
        "wikipedia2008_subset-schema.pgsql.bz2": {"page", "revision", "pagecontent"},
    }
    for filename, tables in expected.items():
        path = raw / filename
        if not path.exists():
            continue
        schema = read_schema(path)
        assert tables <= set(schema.relations)
        assert any(relation.foreign_keys for relation in schema.relations.values())
