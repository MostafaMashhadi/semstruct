"""Fine-tune one DistilBERT ranker per fold of a database."""

from __future__ import annotations

import inspect
import json
import logging
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from semstruct.config import load_config
from semstruct.data.instances import folds_for_topics, fold_roles, select_fold_pairs
from semstruct.data.prepare import read_jsonl
from semstruct.device import release_device_cache, select_device
from semstruct.model.classifier import load_ranker, save_ranker
from semstruct.model.encoding import encode_keyword_query_element
from semstruct.paths import fold_model_dir, fold_output_dir, processed_dir, project_root
from semstruct.runlog import (
    emit,
    format_early_stop,
    format_epoch,
    format_step,
    format_train_finish,
    format_train_start,
)
from semstruct.training.metrics import best_f1_threshold, retrieval_counts

logger = logging.getLogger(__name__)


def train_dataset(
    dataset: str,
    config: dict | None = None,
    root: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit_pairs: int | None = None,
    fold: int | str = "all",
) -> dict:
    root = root or project_root()
    config = dict(config or load_config(dataset, root=root))
    if epochs is not None:
        config["num_epochs"] = epochs
    if batch_size is not None:
        config["batch_size"] = batch_size
    pairs, queries = _load_rows(dataset, root, limit_pairs)
    topic_ids = _topic_ids(queries)
    n_folds = int(config.get("n_folds", 5))
    folds = _requested_folds(fold, topic_ids, n_folds)
    runs = []
    for fold_index in folds:
        runs.append(_train_fold(dataset, config, root, pairs, queries, topic_ids, fold_index, n_folds))
        release_device_cache()
        from semstruct.reports import refresh_reports

        refresh_reports(root)
    return {"dataset": dataset, "folds": runs}


def _train_fold(dataset, config, root, pairs, queries, topic_ids, fold, n_folds) -> dict:
    import torch
    from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

    logging.getLogger("transformers").setLevel(logging.ERROR)
    seed = int(config["seed"])
    cap = int(config.get("value_negative_cap", 10))
    train_pairs, validation_pairs, _test_pairs = select_fold_pairs(pairs, topic_ids, fold, seed, cap, n_folds)
    if not train_pairs or not validation_pairs:
        raise ValueError(f"{dataset} fold {fold} is missing train or validation pairs. Run prepare first.")
    roles = fold_roles(topic_ids, fold, seed, n_folds)
    train_topics = sum(role == "train" for role in roles.values())
    validation_topics = sum(role == "validation" for role in roles.values())
    test_topics = sum(role == "test" for role in roles.values())
    destination = fold_model_dir(dataset, fold, root)
    log_path = fold_output_dir(dataset, fold, root) / "run.log"
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_ranker(config["model_id"])
    revision = getattr(model.config, "_commit_hash", None)
    collator = DataCollatorWithPadding(tokenizer)
    train_dataset = PairDataset(train_pairs, tokenizer, int(config["max_length"]))
    validation_dataset = PairDataset(validation_pairs, tokenizer, int(config["max_length"]))
    steps = math.ceil(len(train_pairs) / int(config["batch_size"])) * int(config["num_epochs"])
    warmup_steps = max(0, int(steps * float(config["warmup_ratio"])))
    arguments = _training_arguments(
        TrainingArguments,
        output_dir=str(destination),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        num_train_epochs=int(config["num_epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        seed=seed,
        dataloader_num_workers=int(config["dataloader_num_workers"]),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    callback = _make_log_callback(
        TrainerCallback,
        dataset,
        fold,
        n_folds,
        int(config["early_stopping_patience"]),
        log_path,
        model,
        destination,
        torch,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping_patience"])),
            callback,
        ],
    )
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    device = str(trainer.args.device)
    emit(
        format_train_start(
            dataset,
            fold,
            n_folds,
            device,
            seed,
            len(train_pairs),
            train_topics,
            validation_topics,
            test_topics,
            int(config["num_epochs"]),
            int(config["batch_size"]),
            destination,
        ),
        log_path,
    )
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    best_path = destination / "best.pt"
    if best_path.is_file():
        state = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        best_path.unlink()
    save_ranker(model, tokenizer, destination)
    best_f1 = callback.best_f1
    best_step = callback.best_step
    emit(
        format_train_finish(dataset, fold, n_folds, best_f1 if best_f1 is not None else 0.0, best_step, destination, elapsed),
        log_path,
    )
    history = {
        "best_metric": best_f1,
        "best_step": best_step,
        "global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history,
    }
    run = {
        "dataset": dataset,
        "fold": fold,
        "n_folds": n_folds,
        "model_id": config["model_id"],
        "revision": revision,
        "device": device,
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "seed": seed,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "train_topics": train_topics,
        "validation_topics": validation_topics,
        "test_topics": test_topics,
        "queries": len(queries),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "warmup_steps": warmup_steps,
        "epochs": int(config["num_epochs"]),
        "batch_size": int(config["batch_size"]),
        "max_length": int(config["max_length"]),
        "value_negative_cap": cap,
        "best_validation_f1": best_f1,
        "best_step": best_step,
        "elapsed_seconds": elapsed,
    }
    record = fold_output_dir(dataset, fold, root)
    record.mkdir(parents=True, exist_ok=True)
    (record / "run_config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (record / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (destination / "run_config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    del trainer, model, tokenizer
    return run


def overfit_check(dataset: str, config: dict | None = None, root: Path | None = None, batch_size: int = 16) -> dict:
    """Train on one mixed batch until the head can reduce its loss."""
    import torch

    root = root or project_root()
    config = dict(config or load_config(dataset, root=root))
    pairs, queries = _load_rows(dataset, root, limit_pairs=None)
    topic_ids = _topic_ids(queries)
    train_pairs, _validation, _test = select_fold_pairs(
        pairs,
        topic_ids,
        0,
        int(config["seed"]),
        int(config.get("value_negative_cap", 10)),
        int(config.get("n_folds", 5)),
    )
    batch_rows = _mixed_batch(train_pairs, batch_size)
    if len(batch_rows) < 2:
        raise ValueError("Overfit check needs at least two training pairs")
    tokenizer, model = load_ranker(config["model_id"])
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    device = select_device()
    model.to(device)
    model.train()
    view = PairDataset(batch_rows, tokenizer, int(config["max_length"]))
    batch = _pad_batch(tokenizer, [view[index] for index in range(len(view))])
    batch = {key: value.to(device) for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for _step in range(30):
        optimizer.zero_grad(set_to_none=True)
        labels = batch.pop("labels")
        outputs = model(**batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits.view(-1), labels.float().view(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        batch["labels"] = labels
    result = {
        "dataset": dataset,
        "device": str(device),
        "batch_size": len(batch_rows),
        "learning_rate": 1e-4,
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "decreased": losses[-1] < losses[0],
    }
    destination = fold_output_dir(dataset, 0, root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "overfit_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("overfit check %s", result)
    del model, tokenizer, optimizer, batch
    release_device_cache()
    return result


def _make_log_callback(callback_base, dataset, fold, n_folds, patience, log_path, model, destination, torch):
    class PlainLogCallback(callback_base):
        def __init__(self):
            super().__init__()
            self.best_f1 = -1.0
            self.best_step = 0
            self.bad_epochs = 0
            self.stopped = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs or "eval_loss" in logs:
                return
            emit(
                format_step(
                    dataset,
                    fold,
                    n_folds,
                    int(state.global_step),
                    int(state.max_steps),
                    float(logs.get("epoch", 0.0)),
                    float(logs["loss"]),
                ),
                log_path,
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics or "eval_f1" not in metrics:
                return
            f1 = float(metrics["eval_f1"])
            best = f1 > self.best_f1
            if best:
                self.best_f1 = f1
                self.best_step = int(state.global_step)
                self.bad_epochs = 0
                cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                destination.mkdir(parents=True, exist_ok=True)
                torch.save(cpu_state, destination / "best.pt")
            else:
                self.bad_epochs += 1
            emit(
                format_epoch(
                    dataset,
                    fold,
                    n_folds,
                    float(metrics.get("epoch", state.epoch or 0.0)),
                    float(metrics.get("eval_precision", 0.0)),
                    float(metrics.get("eval_recall", 0.0)),
                    f1,
                    best,
                ),
                log_path,
            )
            if self.bad_epochs >= patience and not self.stopped:
                self.stopped = True
                emit(
                    format_early_stop(
                        dataset,
                        fold,
                        n_folds,
                        float(metrics.get("epoch", state.epoch or 0.0)),
                        patience,
                    ),
                    log_path,
                )

    return PlainLogCallback()


def _compute_metrics(eval_prediction):
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    probabilities = _sigmoid(np.asarray(logits).reshape(-1))
    label_list = [int(label) for label in np.asarray(labels).reshape(-1).tolist()]
    threshold = best_f1_threshold(label_list, probabilities.tolist())
    report = retrieval_counts(label_list, probabilities.tolist(), threshold)
    return {
        "f1": float(report["f1"]),
        "precision": float(report["precision"]),
        "recall": float(report["recall"]),
    }


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -50, 50)
    return 1.0 / (1.0 + np.exp(-clipped))


class PairDataset:
    def __init__(self, pairs: list[dict], tokenizer, max_length: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        row = self.pairs[index]
        encoded = encode_keyword_query_element(
            self.tokenizer,
            row["keyword"],
            row["query"],
            row["element_text"],
            self.max_length,
        )
        encoded["labels"] = int(row["label"])
        return encoded


def _pad_batch(tokenizer, rows: list[dict]) -> dict:
    import torch

    width = max(len(row["input_ids"]) for row in rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.sep_token_id
    input_ids = []
    attention = []
    labels = []
    for row in rows:
        length = len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * (width - length))
        attention.append(row["attention_mask"] + [0] * (width - length))
        labels.append(int(row["labels"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float),
    }


def _training_arguments(argument_cls, **kwargs):
    parameters = inspect.signature(argument_cls.__init__).parameters
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return argument_cls(**accepted)


def _mixed_batch(pairs: list[dict], batch_size: int) -> list[dict]:
    positives = [pair for pair in pairs if int(pair["label"]) == 1]
    negatives = [pair for pair in pairs if int(pair["label"]) == 0]
    half = max(1, batch_size // 2)
    return positives[:half] + negatives[:half]


def _topic_ids(queries: list[dict]) -> list[str]:
    return [str(query["topic_id"]) for query in queries if query.get("source") == "coffman" and query.get("topic_id")]


def _requested_folds(fold: int | str, topic_ids: list[str], n_folds: int) -> list[int]:
    available = folds_for_topics(topic_ids, n_folds)
    if fold == "all":
        return available
    chosen = int(fold)
    if chosen not in available:
        raise ValueError(f"Fold {chosen} is outside {available}")
    return [chosen]


def _load_rows(dataset: str, root: Path, limit_pairs: int | None) -> tuple[list[dict], list[dict]]:
    directory = processed_dir(dataset, root)
    pairs = read_jsonl(directory / "pairs.jsonl")
    queries = read_jsonl(directory / "queries.jsonl")
    if limit_pairs is not None:
        queries, pairs = limit_training_rows(queries, pairs, limit_pairs)
    return pairs, queries


def limit_training_rows(queries: list[dict], pairs: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["query_id"]].append(pair)
    kept: list[dict] = []
    total = 0
    for query in queries:
        size = len(grouped.get(query["query_id"], []))
        if total + size > limit and kept:
            continue
        kept.append(query)
        total += size
    kept_ids = {query["query_id"] for query in kept}
    return kept, [pair for pair in pairs if pair["query_id"] in kept_ids]
