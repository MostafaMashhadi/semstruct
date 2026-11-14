"""Lexical overlap and frozen DistilBERT embedding baselines."""

from __future__ import annotations

from semstruct.data.elements import content_tokens


def overlap_score(query: str, element_text: str) -> float:
    query_tokens = content_tokens(query)
    element_tokens = content_tokens(element_text)
    if not query_tokens or not element_tokens:
        return 0.0
    return len(query_tokens & element_tokens) / len(query_tokens)


def overlap_scores(queries: list[str], elements: list[str]) -> list[float]:
    return [overlap_score(query, element) for query, element in zip(queries, elements)]


def encode_texts(model, tokenizer, texts: list[str], *, batch_size: int, device, max_length: int):
    import torch

    if not texts:
        return torch.empty(0, 0)
    model.eval()
    model.to(device)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        vectors.append(pooled.cpu())
    return torch.cat(vectors, dim=0)


def cosine_from_texts(model, tokenizer, queries: list[str], elements: list[str], *, batch_size: int, device, max_length: int) -> list[float]:
    import torch

    unique = list(dict.fromkeys([*queries, *elements]))
    vectors = encode_texts(model, tokenizer, unique, batch_size=batch_size, device=device, max_length=max_length)
    index = {text: position for position, text in enumerate(unique)}
    query_index = torch.tensor([index[text] for text in queries], dtype=torch.long)
    element_index = torch.tensor([index[text] for text in elements], dtype=torch.long)
    dots = (vectors[query_index] * vectors[element_index]).sum(dim=-1)
    return [float(value) for value in dots.tolist()]
