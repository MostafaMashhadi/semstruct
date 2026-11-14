from semstruct.training.metrics import (
    average_precision,
    best_f1_threshold,
    recall_at_k,
    reciprocal_rank,
)


def test_ranking_metrics():
    assert average_precision([1, 1, 0]) == 1.0
    assert average_precision([0, 1, 0]) == 0.5
    assert average_precision([0, 0]) is None
    assert reciprocal_rank([0, 1]) == 0.5
    assert recall_at_k([1, 0, 1], 1) == 0.5


def test_threshold_separates_scores():
    threshold = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 <= threshold <= 0.8
