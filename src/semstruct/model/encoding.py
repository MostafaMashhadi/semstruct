"""Pack a keyword, a query, and an element into one DistilBERT sequence."""

from __future__ import annotations


def encode_segments(tokenizer, segments: list[str], max_length: int) -> dict[str, list[int]]:
    """Build `[CLS] segment [SEP] segment [SEP] ...`, truncated to `max_length`.

    Earlier segments are kept ahead of later ones, so a keyword is not dropped
    to make room for a long element description.
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pieces = [tokenizer.encode(segment or "", add_special_tokens=False) for segment in segments]
    budget = max_length - (1 + len(pieces))
    if budget < 1:
        budget = 1
    kept: list[list[int]] = []
    remaining = budget
    for index, piece in enumerate(pieces):
        later = len(pieces) - index - 1
        reserve = min(later, max(0, remaining - 1)) if later else 0
        allowance = remaining if later == 0 else max(0, remaining - reserve)
        taken = piece[:allowance]
        kept.append(taken)
        remaining -= len(taken)
    ids = [cls_id]
    for piece in kept:
        ids.extend(piece)
        ids.append(sep_id)
    if len(ids) > max_length:
        ids = ids[: max_length - 1] + [sep_id]
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def encode_keyword_query_element(tokenizer, keyword: str, query: str, element: str, max_length: int) -> dict[str, list[int]]:
    return encode_segments(tokenizer, [keyword, query, element], max_length)


def encode_keyword_query(tokenizer, keyword: str, query: str, max_length: int) -> dict[str, list[int]]:
    return encode_segments(tokenizer, [keyword, query], max_length)


def encode_element(tokenizer, element: str, max_length: int) -> dict[str, list[int]]:
    return encode_segments(tokenizer, [element], max_length)
