from semstruct.data.schema import schema_graph
from semstruct.networks.icn import (
    _value_index,
    answer_is_relevant,
    joined_answers,
    retrieval_metrics,
    select_networks,
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


def test_selection_keeps_the_threshold_and_the_best_network():
    networks = [
        _network(0.1, 3, "a"),
        _network(0.9, 2, "b"),
        _network(0.4, 1, "c"),
    ]
    chosen = select_networks(networks, 0.3, limit=1)
    assert [item["network_id"] for item in chosen] == ["b"]
    fallback = select_networks([_network(0.05, 4, "a"), _network(0.02, 1, "b")], 0.5)
    assert [item["network_id"] for item in fallback] == ["a"]
    tied = select_networks([_network(0.8, 3, "b"), _network(0.8, 1, "a")], 0.5)
    assert [item["network_id"] for item in tied] == ["a", "b"]
