"""Lexical overlap and frozen DistilBERT [CLS] cosine."""

from __future__ import annotations

import torch

from semstruct.data.elements import content_tokens
from semstruct.model.encoding import encode_element, encode_keyword_query


def overlap_score(query: str, element_text: str) -> float:
    query_tokens = content_tokens(query)
    element_tokens = content_tokens(element_text)
    if not query_tokens or not element_tokens:
        return 0.0
    return len(query_tokens & element_tokens) / len(query_tokens)


def overlap_scores(queries: list[str], elements: list[str]) -> list[float]:
    return [overlap_score(query, element) for query, element in zip(queries, elements)]


def pretrained_cosine(
    model,
    tokenizer,
    keywords: list[str],
    queries: list[str],
    elements: list[str],
    *,
    batch_size: int,
    device,
    max_length: int,
) -> list[float]:
    """Cosine of frozen [CLS] vectors for `[CLS] k [SEP] q [SEP]` and `[CLS] Te [SEP]`."""
    if not keywords:
        return []
    left = [encode_keyword_query(tokenizer, keyword, query, max_length) for keyword, query in zip(keywords, queries)]
    right = [encode_element(tokenizer, element, max_length) for element in elements]
    left_vectors = _cls_vectors(model, tokenizer, left, batch_size, device)
    right_vectors = _cls_vectors(model, tokenizer, right, batch_size, device)
    left_vectors = torch.nn.functional.normalize(left_vectors, dim=-1)
    right_vectors = torch.nn.functional.normalize(right_vectors, dim=-1)
    return [float(value) for value in (left_vectors * right_vectors).sum(dim=-1).tolist()]


def _cls_vectors(model, tokenizer, rows: list[dict[str, list[int]]], batch_size: int, device) -> torch.Tensor:
    model.eval()
    model.to(device)
    vectors = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.sep_token_id
        input_ids = []
        attention = []
        for row in batch:
            length = len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * (width - length))
            attention.append(row["attention_mask"] + [0] * (width - length))
        encoded = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        }
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state[:, 0]
        vectors.append(hidden.detach().cpu())
    return torch.cat(vectors, dim=0)
