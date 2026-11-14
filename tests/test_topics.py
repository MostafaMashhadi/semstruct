from semstruct.data.topics import parse_topic_file, topic_query_norms
from semstruct.data.elements import normalize_text


def test_topic_file_skips_comments_and_metadata(tmp_path):
    path = tmp_path / "001.txt"
    path.write_text(
        "\n".join(
            [
                "# [reference standard]",
                "# imdb2009",
                "# denzel washington",
                "#([999], [])",
                "([39927668], [])",
                "([1, 2], [(1, 2)]) # note",
            ]
        ),
        encoding="utf-8",
    )
    topic = parse_topic_file(path)
    assert topic.topic_id == "1"
    assert topic.query == "denzel washington"
    assert topic.search_ids == [39927668, 1, 2]
    assert normalize_text(topic.query) in topic_query_norms([topic])
