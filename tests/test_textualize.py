from semstruct.data.elements import (
    attribute_text,
    conflicts_with_keywords,
    keyword_from_value,
    relation_text,
    value_text,
)
from semstruct.data.schema import Attribute, ForeignKey, Relation


def test_keyword_normalizes_underscores_and_drops_long_values():
    assert keyword_from_value("New_York", 80) == "new york"
    assert keyword_from_value("The History of the French Revolution in Europe", 80) == "history french"
    assert keyword_from_value("x" * 90, 80) is None
    assert keyword_from_value("12", 80) is None


def test_attribute_text_drops_keyword_examples():
    relation = Relation(
        "city",
        [Attribute("name", "text")],
        foreign_keys=[ForeignKey(["country"], "country", ["code"])],
    )
    attribute = relation.attributes[0]
    assert conflicts_with_keywords("Bern", ["bern"])
    rendered = attribute_text(relation, attribute, ["Bern", "Paris", "Zurich"], ["bern"])
    assert "example values: Paris; Zurich" in rendered
    assert "Bern" not in rendered.split("example values: ", 1)[-1]
    plain = relation_text(relation)
    assert "[RELATION] name: city" in plain
    assert "Bern" not in plain


def test_value_text_keeps_the_cell():
    rendered = value_text("page", "page_title", "New_York", 80)
    assert '"New_York"' in rendered
    assert "normalized: new york" in rendered
    assert "attribute: page.page_title" in rendered
