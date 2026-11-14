"""Classification and ranking metrics for query–element relevance."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def average_precision(labels_in_rank_order: list[int]) -> float | None:
    hits = 0
    total = 0.0
    relevant = 0
    for rank, label in enumerate(labels_in_rank_order, start=1):
        if label:
            relevant += 1
            hits += 1
            total += hits / rank
    if relevant == 0:
        return None
    return total / relevant


def reciprocal_rank(labels_in_rank_order: list[int]) -> float | None:
    if not any(labels_in_rank_order):
        return None
    for rank, label in enumerate(labels_in_rank_order, start=1):
        if label:
            return 1.0 / rank
    return None


def recall_at_k(labels_in_rank_order: list[int], k: int) -> float | None:
    relevant = sum(labels_in_rank_order)
    if relevant == 0:
        return None
    return sum(labels_in_rank_order[:k]) / relevant


def mean_metric(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None and not math.isnan(value)]
    if not kept:
        return None
    return float(sum(kept) / len(kept))


def ranking_report(grouped_scores: dict[str, list[tuple[float, int]]]) -> dict[str, float | int | None]:
    average_precisions = []
    reciprocal_ranks = []
    recall5 = []
    recall10 = []
    for pairs in grouped_scores.values():
        ordered = [label for _score, label in sorted(pairs, key=lambda item: (-item[0],))]
        average_precisions.append(average_precision(ordered))
        reciprocal_ranks.append(reciprocal_rank(ordered))
        recall5.append(recall_at_k(ordered, 5))
        recall10.append(recall_at_k(ordered, 10))
    return {
        "map": mean_metric(average_precisions),
        "mrr": mean_metric(reciprocal_ranks),
        "recall@5": mean_metric(recall5),
        "recall@10": mean_metric(recall10),
        "queries": sum(value is not None for value in average_precisions),
    }


def classification_report(labels: list[int], scores: list[float], threshold: float) -> dict[str, float | None]:
    if not labels:
        return {"precision": None, "recall": None, "f1": None, "accuracy": None, "roc_auc": None, "pr_auc": None, "threshold": threshold}
    array_labels = np.asarray(labels)
    array_scores = np.asarray(scores, dtype=float)
    predictions = (array_scores >= threshold).astype(int)
    report: dict[str, float | None] = {
        "precision": float(precision_score(array_labels, predictions, zero_division=0)),
        "recall": float(recall_score(array_labels, predictions, zero_division=0)),
        "f1": float(f1_score(array_labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(array_labels, predictions)),
        "threshold": float(threshold),
        "roc_auc": None,
        "pr_auc": None,
    }
    if len(set(array_labels.tolist())) > 1:
        report["roc_auc"] = float(roc_auc_score(array_labels, array_scores))
        report["pr_auc"] = float(average_precision_score(array_labels, array_scores))
    return report


def best_f1_threshold(labels: list[int], scores: list[float]) -> float:
    if not scores:
        return 0.5
    grid = np.quantile(np.asarray(scores, dtype=float), np.linspace(0.0, 1.0, 101))
    best_threshold = 0.5
    best_f1 = -1.0
    array_labels = np.asarray(labels)
    for threshold in grid:
        predictions = (np.asarray(scores) >= threshold).astype(int)
        score = float(f1_score(array_labels, predictions, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def scores_by_query(pairs: list[dict], scores: list[float], label_of) -> dict[str, list[tuple[float, int]]]:
    grouped: dict[str, list[tuple[float, int]]] = {}
    for pair, score in zip(pairs, scores):
        grouped.setdefault(pair["query_id"], []).append((float(score), int(label_of(pair))))
    return grouped
