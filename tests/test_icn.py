from semstruct.data.schema import schema_graph
from semstruct.networks.icn import (
    _JoinCache,
    _value_index,
    answer_is_relevant,
    answer_quality,
    joined_answers,
    order_networks,
    retrieval_metrics,
)
from tests.test_candidate_networks import _database


def _network(score: float, size: int, network_id: str) -> dict:
    return {"score": score, "size": size, "network_id": network_id, "nodes": []}


def test_the_join_returns_the_bern_rhine_row(tmp_path):
    schema, connection = _database(tmp_path)
    nodes = [
        {
            "relation": "city",
            "keywords": [{"keyword": "bern", "role": "value", "attribute": "name", "match_count": 1}],
        },
        {"relation": "country", "keywords": []},
        {
            "relation": "river",
            "keywords": [{"keyword": "rhine", "role": "value", "attribute": "name", "match_count": 1}],
        },
    ]
    _cells, row_ids = _value_index(connection, schema, [{"nodes": nodes}], 80)
    rows = joined_answers(connection, schema, schema_graph(schema), nodes, row_ids)
    connection.close()
    assert rows == [[3, 1, 6]]
    assert answer_is_relevant(rows[0], [[1, 3, 6]])
    assert not answer_is_relevant(rows[0], [[3, 6]])
    assert not answer_is_relevant([9], [[1, 3, 6]])


def test_retrieval_metrics_match_a_hand_ranked_list():
    measured = retrieval_metrics([[0, 1, 1], [1]])
    assert measured["top1"] == 1
    assert measured["mrr"] == 0.75
    assert abs(measured["map"] - ((0.5 + 2 / 3) / 2 + 1) / 2) < 1e-9
    assert retrieval_metrics([[]]) == {"top1": 0, "mrr": 0.0, "map": 0.0}


def test_low_scores_stay_in_order_and_collection_stops_at_k(tmp_path):
    ordered = order_networks(
        [
            _network(0.1, 3, "a"),
            _network(0.9, 2, "b"),
            _network(0.4, 1, "c"),
        ]
    )
    assert [item["network_id"] for item in ordered] == ["b", "c", "a"]
    schema, connection = _database(tmp_path)
    city = {
        "relation": "city",
        "keywords": [{"keyword": "bern", "role": "value", "attribute": "name", "match_count": 1}],
    }
    river = {
        "relation": "river",
        "keywords": [{"keyword": "rhine", "role": "value", "attribute": "name", "match_count": 1}],
    }
    _cells, row_ids = _value_index(connection, schema, [{"nodes": [city]}, {"nodes": [river]}], 80)
    from semstruct.networks.icn import _collect_answers

    ranked, evaluated, _sql_ms = _collect_answers(
        connection,
        schema,
        schema_graph(schema),
        [
            {"score": 0.9, "size": 1, "network_id": "city", "nodes": [city]},
            {"score": 0.1, "size": 1, "network_id": "river", "nodes": [river]},
        ],
        row_ids,
        [[3]],
        limit=1,
    )
    connection.close()
    assert evaluated == 1
    assert [item["network_id"] for item in ranked] == ["city"]


def test_a_shared_prefix_is_joined_once(tmp_path):
    schema, connection = _database(tmp_path)
    city = {
        "relation": "city",
        "keywords": [{"keyword": "bern", "role": "value", "attribute": "name", "match_count": 1}],
    }
    country = {"relation": "country", "keywords": []}
    river = {
        "relation": "river",
        "keywords": [{"keyword": "rhine", "role": "value", "attribute": "name", "match_count": 1}],
    }
    _cells, row_ids = _value_index(
        connection,
        schema,
        [{"nodes": [city, country]}, {"nodes": [city, country, river]}],
        80,
    )
    cache = _JoinCache()
    graph = schema_graph(schema)
    cache.combinations(connection, schema, graph, [city, country], row_ids)
    first = cache.extensions
    cache.combinations(connection, schema, graph, [city, country, river], row_ids)
    connection.close()
    assert cache.extensions == first + 1


def test_answer_quality_uses_exact_matches():
    precision, recall, f1 = answer_quality([False, True], 2)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5
    assert answer_quality([], 1) == (0.0, 0.0, 0.0)
    assert answer_quality([True], 0) == (1.0, None, None)


def test_zero_scores_order_by_size_then_id():
    ordered = order_networks(
        [
            _network(0.0, 3, "c"),
            _network(0.0, 1, "b"),
            _network(0.0, 1, "a"),
        ]
    )
    assert [item["network_id"] for item in ordered] == ["a", "b", "c"]
