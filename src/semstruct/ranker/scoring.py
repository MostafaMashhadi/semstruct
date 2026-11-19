"""Sigmoid probabilities for one keyword, its query, and one element."""

from __future__ import annotations

import torch

from semstruct.ranker.encoding import encode_keyword_query_element


def relevance_probabilities(
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
    if not (len(keywords) == len(queries) == len(elements)):
        raise ValueError("keywords, queries, and elements must have the same length")
    if not keywords:
        return []
    model.eval()
    model.to(device)
    scores: list[float] = []
    for start in range(0, len(keywords), batch_size):
        batch = [
            encode_keyword_query_element(tokenizer, keyword, query, element, max_length)
            for keyword, query, element in zip(
                keywords[start : start + batch_size],
                queries[start : start + batch_size],
                elements[start : start + batch_size],
            )
        ]
        encoded = _pad(tokenizer, batch)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.view(-1)
            probabilities = torch.sigmoid(logits)
        scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
    return scores


def keyword_relevance(model, tokenizer, query: str, element_text: str, *, batch_size: int, device, max_length: int) -> dict:
    """Score each whitespace-separated keyword of one query against one element."""
    keywords = [token for token in query.split() if token]
    if not keywords:
        return {"keywords": {}}
    scores = relevance_probabilities(
        model,
        tokenizer,
        keywords,
        [query] * len(keywords),
        [element_text] * len(keywords),
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    return {"keywords": dict(zip(keywords, scores))}


def query_class_score(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    return float(sum(probabilities) / len(probabilities))


def _pad(tokenizer, rows: list[dict[str, list[int]]]) -> dict[str, "torch.Tensor"]:
    width = max(len(row["input_ids"]) for row in rows)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.sep_token_id
    input_ids = []
    attention = []
    for row in rows:
        length = len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * (width - length))
        attention.append(row["attention_mask"] + [0] * (width - length))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
