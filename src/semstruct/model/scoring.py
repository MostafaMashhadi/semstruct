"""Full-query scores, per-keyword scores, and query-class aggregation.

Keyword scores reuse the same classifier. They are an inference-time signal
for a later structural ranker. They are not separate training targets.
"""

from __future__ import annotations

import re


def relevance_probabilities(model, tokenizer, queries: list[str], elements: list[str], *, batch_size: int, device, max_length: int) -> list[float]:
    import torch

    if len(queries) != len(elements):
        raise ValueError("queries and elements must have the same length")
    if not queries:
        return []
    model.eval()
    model.to(device)
    scores: list[float] = []
    for start in range(0, len(queries), batch_size):
        batch_queries = queries[start : start + batch_size]
        batch_elements = elements[start : start + batch_size]
        encoded = tokenizer(
            batch_queries,
            batch_elements,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
        scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
    return scores


def keyword_relevance(model, tokenizer, query: str, element_text: str, *, batch_size: int, device, max_length: int) -> dict:
    keywords = [token for token in re.split(r"\s+", query.strip()) if token]
    full = relevance_probabilities(
        model,
        tokenizer,
        [query],
        [element_text],
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )[0]
    if not keywords:
        return {"full_query": full, "keywords": {}}
    per_keyword = relevance_probabilities(
        model,
        tokenizer,
        keywords,
        [element_text] * len(keywords),
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    return {"full_query": full, "keywords": dict(zip(keywords, per_keyword))}


def query_class_score(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    return float(sum(probabilities) / len(probabilities))
