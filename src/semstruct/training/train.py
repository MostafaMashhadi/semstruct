"""Fine-tune one DistilBERT relevance classifier for a database."""

from __future__ import annotations

import inspect
import json
import logging
import math
import platform
from collections import defaultdict
from pathlib import Path

from semstruct.config import load_config
from semstruct.data.prepare import read_jsonl
from semstruct.device import release_device_cache
from semstruct.model.classifier import load_classifier
from semstruct.paths import model_dir, output_dir, processed_dir, project_root

logger = logging.getLogger(__name__)


def train_dataset(
    dataset: str,
    config: dict | None = None,
    root: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit_pairs: int | None = None,
) -> dict:
    import torch
    from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

    root = root or project_root()
    config = dict(config or load_config(dataset, root=root))
    if epochs is not None:
        config["num_epochs"] = epochs
    if batch_size is not None:
        config["batch_size"] = batch_size
    pairs, queries = _load_training_rows(dataset, root, limit_pairs)
    train_pairs = [row for row in pairs if row["split"] == "train"]
    validation_pairs = [row for row in pairs if row["split"] == "validation"]
    if not train_pairs or not validation_pairs:
        raise ValueError(f"{dataset} is missing train or validation pairs. Run prepare first.")

    tokenizer, model = load_classifier(config["model_id"])
    revision = getattr(model.config, "_commit_hash", None)
    trainer = None
    try:
        return _run_trainer(
            dataset,
            config,
            root,
            queries,
            train_pairs,
            validation_pairs,
            tokenizer,
            model,
            revision,
            torch,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    finally:
        trainer = None
        model = None
        tokenizer = None
        release_device_cache()


def _run_trainer(
    dataset,
    config,
    root,
    queries,
    train_pairs,
    validation_pairs,
    tokenizer,
    model,
    revision,
    torch,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
):
    weights = _class_weights(train_pairs)
    collator = DataCollatorWithPadding(tokenizer)
    train_dataset = PairDataset(train_pairs, tokenizer, int(config["max_length"]))
    validation_dataset = PairDataset(validation_pairs, tokenizer, int(config["max_length"]))
    steps = math.ceil(len(train_pairs) / int(config["batch_size"])) * int(config["num_epochs"])
    warmup_steps = max(0, int(steps * float(config["warmup_ratio"])))
    destination = model_dir(dataset, root)
    destination.mkdir(parents=True, exist_ok=True)
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
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        logging_strategy="steps",
        logging_steps=50,
        seed=int(config["seed"]),
        dataloader_num_workers=int(config["dataloader_num_workers"]),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        overwrite_output_dir=True,
        disable_tqdm=True,
    )
    trainer = _make_weighted_trainer(Trainer)(
        class_weights=weights,
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(config["early_stopping_patience"]))],
    )
    logger.info("training %s on %s pairs using %s", dataset, len(train_pairs), arguments.device)
    trainer.train()
    trainer.save_model(destination)
    tokenizer.save_pretrained(destination)
    history = {
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history,
    }
    run = {
        "dataset": dataset,
        "model_id": config["model_id"],
        "revision": revision,
        "device": str(arguments.device),
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "seed": int(config["seed"]),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "queries": len(queries),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "warmup_steps": warmup_steps,
        "epochs": int(config["num_epochs"]),
        "batch_size": int(config["batch_size"]),
        "max_length": int(config["max_length"]),
        "best_validation_f1": trainer.state.best_metric,
    }
    report_dir = output_dir(dataset, root)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "run_config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (report_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (destination / "run_config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info("saved %s model to %s", dataset, destination)
    return run


def overfit_check(dataset: str, config: dict | None = None, root: Path | None = None, batch_size: int = 16) -> dict:
    """Train on one mixed batch until the head can reduce its loss."""
    import torch

    root = root or project_root()
    config = dict(config or load_config(dataset, root=root))
    pairs, _queries = _load_training_rows(dataset, root, limit_pairs=None)
    batch_rows = _mixed_batch([row for row in pairs if row["split"] == "train"], batch_size)
    if len(batch_rows) < 2:
        raise ValueError("Overfit check needs at least two training pairs")
    from semstruct.device import select_device
    from transformers import DataCollatorWithPadding

    tokenizer, model = load_classifier(config["model_id"])
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    device = select_device()
    model.to(device)
    model.train()
    collator = DataCollatorWithPadding(tokenizer)
    dataset_view = PairDataset(batch_rows, tokenizer, int(config["max_length"]))
    batch = collator([dataset_view[index] for index in range(len(dataset_view))])
    batch = {key: value.to(device) for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for _step in range(30):
        optimizer.zero_grad(set_to_none=True)
        labels = batch["labels"]
        outputs = model(**{key: value for key, value in batch.items() if key != "labels"})
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
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
    destination = output_dir(dataset, root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "overfit_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("overfit check %s", result)
    del model, tokenizer, optimizer, batch
    release_device_cache()
    return result


class PairDataset:
    def __init__(self, pairs: list[dict], tokenizer, max_length: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        row = self.pairs[index]
        encoded = self.tokenizer(
            row["query"],
            row["element_text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = int(row["label"])
        return encoded


def _make_weighted_trainer(trainer_cls):
    import torch

    class WeightedTrainer(trainer_cls):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                labels,
                weight=self.class_weights.to(outputs.logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _training_arguments(argument_cls, **kwargs):
    parameters = inspect.signature(argument_cls.__init__).parameters
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return argument_cls(**accepted)


def _compute_metrics(eval_prediction):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    predictions = (probabilities[:, 1] >= 0.5).astype(int)
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def _class_weights(pairs: list[dict]):
    import torch

    labels = [int(pair["label"]) for pair in pairs]
    total = len(labels)
    positives = sum(labels)
    negatives = total - positives
    return torch.tensor(
        [
            total / (2 * max(negatives, 1)),
            total / (2 * max(positives, 1)),
        ],
        dtype=torch.float,
    )


def _mixed_batch(pairs: list[dict], batch_size: int) -> list[dict]:
    positives = [pair for pair in pairs if int(pair["label"]) == 1]
    negatives = [pair for pair in pairs if int(pair["label"]) == 0]
    half = max(1, batch_size // 2)
    return positives[:half] + negatives[:half]


def _load_training_rows(dataset: str, root: Path, limit_pairs: int | None) -> tuple[list[dict], list[dict]]:
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
    by_split: dict[str, list[dict]] = defaultdict(list)
    for query in sorted(queries, key=lambda item: item["query_id"]):
        by_split[query["split"]].append(query)
    kept: list[dict] = []
    total = 0
    for split in ("train", "validation", "test"):
        if not by_split[split]:
            continue
        query = by_split[split].pop(0)
        kept.append(query)
        total += len(grouped[query["query_id"]])
    for split in ("train", "validation", "test"):
        for query in by_split[split]:
            size = len(grouped[query["query_id"]])
            if total + size > limit:
                continue
            kept.append(query)
            total += size
    kept_ids = {query["query_id"] for query in kept}
    return kept, [pair for pair in pairs if pair["query_id"] in kept_ids]

