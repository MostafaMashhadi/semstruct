"""Load the English uncased DistilBERT cross-encoder."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_LABELS = {0: "irrelevant", 1: "relevant"}


def load_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_classifier(model_name_or_path: str, *, num_labels: int = 2):
    tokenizer = load_tokenizer(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=MODEL_LABELS,
        label2id={label: index for index, label in MODEL_LABELS.items()},
    )
    return tokenizer, model
