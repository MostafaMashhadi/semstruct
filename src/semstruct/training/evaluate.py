"""Score each fold's held-out Coffman topics and pool precision, recall, and F1."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from semstruct.config import load_config
from semstruct.data.instances import folds_for_topics, select_fold_pairs
from semstruct.data.prepare import read_jsonl
from semstruct.device import release_device_cache
from semstruct.model.classifier import load_ranker
from semstruct.model.scoring import relevance_probabilities
from semstruct.paths import fold_model_dir, fold_output_dir, output_dir, processed_dir, project_root
from semstruct.runlog import emit, format_eval, format_pooled
from semstruct.training.baselines import overlap_scores, pretrained_cosine
from semstruct.training.metrics import best_f1_threshold, pool_retrieval, retrieval_counts


def evaluate_dataset(
    dataset: str,
    config: dict | None = None,
    root: Path | None = None,
    batch_size: int | None = None,
    fold: int | str = "all",
) -> dict:
    root = root or project_root()
    config = dict(config or load_config(dataset, root=root))
    if batch_size is not None:
        config["batch_size"] = batch_size
    directory = processed_dir(dataset, root)
    pairs = read_jsonl(directory / "pairs.jsonl")
    queries = read_jsonl(directory / "queries.jsonl")
    topic_ids = [str(query["topic_id"]) for query in queries if query.get("source") == "coffman" and query.get("topic_id")]
    n_folds = int(config.get("n_folds", 5))
    available = folds_for_topics(topic_ids, n_folds)
    chosen = available if fold == "all" else [int(fold)]
    pooled = None
    for fold_index in chosen:
        if fold_index not in available:
            raise ValueError(f"Fold {fold_index} is outside {available}")
        _evaluate_fold(dataset, config, root, pairs, topic_ids, fold_index, n_folds)
        release_device_cache()
        pooled = write_cv_metrics(dataset, root, n_folds)
        from semstruct.reports import refresh_reports

        refresh_reports(root)
    if pooled is None:
        pooled = write_cv_metrics(dataset, root, n_folds)
    return pooled


def write_cv_metrics(dataset: str, root: Path | None = None, n_folds: int = 5) -> dict:
    root = root or project_root()
    completed = []
    fold_payloads = {}
    for fold in range(n_folds):
        path = fold_output_dir(dataset, fold, root) / "metrics.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed.append(fold)
        fold_payloads[str(fold)] = payload
    systems = {}
    for system in ("finetuned_distilbert", "pretrained_distilbert"):
        blocks = [fold_payloads[str(fold)][system] for fold in completed]
        pooled = pool_retrieval(blocks) if blocks else pool_retrieval([])
        pooled["label"] = "ranker results"
        systems[system] = pooled
    report = {
        "dataset": dataset,
        "folds_completed": completed,
        "folds_expected": n_folds,
        "ranker_results": systems,
        "folds": fold_payloads,
    }
    destination = output_dir(dataset, root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cv-metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_path = destination / "run.log"
    names = {"finetuned_distilbert": "fine-tuned", "pretrained_distilbert": "pre-trained"}
    for system, label in names.items():
        block = systems[system]
        emit(
            format_pooled(
                dataset,
                label,
                len(completed),
                n_folds,
                float(block["precision"]),
                float(block["recall"]),
                float(block["f1"]),
                float(block["scoring_ms"]),
            ),
            log_path,
        )
    return report


def _evaluate_fold(dataset, config, root, pairs, topic_ids, fold, n_folds) -> dict:
    _train, validation, test = select_fold_pairs(
        pairs,
        topic_ids,
        fold,
        int(config["seed"]),
        int(config.get("value_negative_cap", 10)),
        n_folds,
    )
    if not validation or not test:
        raise ValueError(f"{dataset} fold {fold} is missing validation or test pairs")
    checkpoint = fold_model_dir(dataset, fold, root)
    if not (checkpoint / "scorer.pt").is_file():
        raise FileNotFoundError(f"No fine-tuned model at {checkpoint}. Run train --fold {fold} first.")
    batch = int(config["batch_size"])
    max_length = int(config["max_length"])
    log_path = fold_output_dir(dataset, fold, root) / "run.log"
    finetuned = _score_loaded(
        "fine-tuned",
        dataset,
        fold,
        n_folds,
        validation,
        test,
        log_path,
        *_finetuned_scores(checkpoint, validation, test, batch, max_length),
    )
    release_device_cache()
    pretrained = _score_loaded(
        "pre-trained",
        dataset,
        fold,
        n_folds,
        validation,
        test,
        log_path,
        *_pretrained_scores(config["model_id"], validation, test, batch, max_length),
    )
    release_device_cache()
    overlap = _overlap_block(validation, test)
    test_topics = sorted({str(row.get("topic_id")) for row in test if row.get("topic_id")})
    payload = {
        "dataset": dataset,
        "fold": fold,
        "test_topics": test_topics,
        "finetuned_distilbert": finetuned,
        "pretrained_distilbert": pretrained,
        "internal": {"token_overlap": overlap},
    }
    record = fold_output_dir(dataset, fold, root)
    record.mkdir(parents=True, exist_ok=True)
    (record / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _score_loaded(name, dataset, fold, n_folds, validation, test, log_path, validation_scores, test_scores, query_ms) -> dict:
    threshold = best_f1_threshold([int(row["label"]) for row in validation], validation_scores)
    report = retrieval_counts([int(row["label"]) for row in test], test_scores, threshold)
    topics = len({row["query_id"] for row in test})
    scoring_ms = float(sum(query_ms) / len(query_ms)) if query_ms else 0.0
    report.update({"topics": topics, "scoring_ms": scoring_ms, "query_ms": query_ms, "label": "ranker results"})
    emit(
        format_eval(
            dataset,
            fold,
            n_folds,
            name,
            float(report["threshold"]),
            topics,
            float(report["precision"]),
            float(report["recall"]),
            float(report["f1"]),
            scoring_ms,
        ),
        log_path,
    )
    return report


def _finetuned_scores(checkpoint: Path, validation: list[dict], test: list[dict], batch_size: int, max_length: int):
    from semstruct.device import select_device

    tokenizer, model = load_ranker(str(checkpoint))
    device = select_device()
    try:
        validation_scores = _probabilities(model, tokenizer, validation, batch_size, device, max_length)
        test_scores, query_ms = _timed_probabilities(model, tokenizer, test, batch_size, device, max_length)
        return validation_scores, test_scores, query_ms
    finally:
        del model, tokenizer


def _pretrained_scores(model_id: str, validation: list[dict], test: list[dict], batch_size: int, max_length: int):
    from transformers import AutoModel, AutoTokenizer

    from semstruct.device import select_device

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    device = select_device()
    try:
        validation_scores = _cosine(model, tokenizer, validation, batch_size, device, max_length)
        test_scores, query_ms = _timed_cosine(model, tokenizer, test, batch_size, device, max_length)
        return validation_scores, test_scores, query_ms
    finally:
        del model, tokenizer


def _probabilities(model, tokenizer, rows, batch_size, device, max_length) -> list[float]:
    return relevance_probabilities(
        model,
        tokenizer,
        [row["keyword"] for row in rows],
        [row["query"] for row in rows],
        [row["element_text"] for row in rows],
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )


def _cosine(model, tokenizer, rows, batch_size, device, max_length) -> list[float]:
    return pretrained_cosine(
        model,
        tokenizer,
        [row["keyword"] for row in rows],
        [row["query"] for row in rows],
        [row["element_text"] for row in rows],
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )


def _timed_probabilities(model, tokenizer, rows, batch_size, device, max_length):
    return _time_groups(rows, lambda group: _probabilities(model, tokenizer, group, batch_size, device, max_length))


def _timed_cosine(model, tokenizer, rows, batch_size, device, max_length):
    return _time_groups(rows, lambda group: _cosine(model, tokenizer, group, batch_size, device, max_length))


def _time_groups(pairs: list[dict], score_group) -> tuple[list[float], list[float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        groups[pair["query_id"]].append(index)
    scores = [0.0] * len(pairs)
    query_ms = []
    for indices in groups.values():
        rows = [pairs[index] for index in indices]
        started = time.perf_counter()
        batch_scores = score_group(rows)
        query_ms.append((time.perf_counter() - started) * 1000)
        for index, score in zip(indices, batch_scores):
            scores[index] = float(score)
    return scores, query_ms


def _overlap_block(validation: list[dict], test: list[dict]) -> dict:
    validation_scores = overlap_scores(
        [f"{row['keyword']} {row['query']}" for row in validation],
        [row["element_text"] for row in validation],
    )
    test_scores = overlap_scores(
        [f"{row['keyword']} {row['query']}" for row in test],
        [row["element_text"] for row in test],
    )
    threshold = best_f1_threshold([int(row["label"]) for row in validation], validation_scores)
    report = retrieval_counts([int(row["label"]) for row in test], test_scores, threshold)
    report["label"] = "internal check"
    return report
