from semstruct.runlog import format_pooled, format_train_start
from semstruct.training.metrics import (
    average_precision,
    best_f1_threshold,
    pool_retrieval,
    recall_at_k,
    reciprocal_rank,
    retrieval_counts,
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


def test_precision_and_recall_count_retrieved_hits():
    report = retrieval_counts([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert report["tp"] == 1
    assert report["fp"] == 1
    assert report["fn"] == 1


def test_pool_retrieval_covers_every_fold():
    pooled = pool_retrieval(
        [
            {"tp": 1, "fp": 1, "fn": 0, "topics": 10, "query_ms": [4.0, 6.0]},
            {"tp": 1, "fp": 0, "fn": 1, "topics": 10, "query_ms": [10.0]},
        ]
    )
    assert pooled["precision"] == 2 / 3
    assert pooled["recall"] == 2 / 3
    assert pooled["topics"] == 20
    assert pooled["scoring_ms"] == 20 / 3


def test_log_lines_name_the_fold_and_the_metrics():
    start = format_train_start("mondial", 2, 5, "mps", 42, 100, 32, 8, 10, 3, 16, "outputs/models/mondial/fold-2")
    assert "mondial" in start
    assert "fold 2 of 5" in start
    assert "32 train" in start
    pooled = format_pooled("mondial", "fine-tuned", 5, 5, 0.5, 0.4, 0.44, 12.5)
    assert "mondial" in pooled
    assert "precision 0.500" in pooled
    assert "recall 0.400" in pooled
    assert "f1 0.440" in pooled
    assert "ranker results" in pooled
