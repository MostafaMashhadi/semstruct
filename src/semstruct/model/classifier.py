"""DistilBERT encoder plus one linear unit for keyword-element relevance."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class RelevanceModel(nn.Module):
    """Fine-tuned score `p(y = 1 | k, q, e) = sigmoid(w · h_CLS + b)`."""

    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        hidden = getattr(encoder.config, "hidden_size", None) or encoder.config.dim
        self.scorer = nn.Linear(int(hidden), 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        del kwargs
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.scorer(hidden).squeeze(-1)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits.unsqueeze(-1))


def load_ranker(model_name_or_path: str):
    """Load a hub checkpoint, or a fold directory that contains `scorer.pt`."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    encoder = AutoModel.from_pretrained(model_name_or_path)
    model = RelevanceModel(encoder)
    path = Path(model_name_or_path)
    scorer = path / "scorer.pt" if path.is_dir() else None
    if scorer is not None and scorer.is_file():
        state = torch.load(scorer, map_location="cpu", weights_only=True)
        model.scorer.load_state_dict(state)
    return tokenizer, model


def save_ranker(model: RelevanceModel, tokenizer, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    torch.save(model.scorer.state_dict(), destination / "scorer.pt")


def load_classifier(model_name_or_path: str, *, num_labels: int = 1):
    """Compatibility name for the one-unit ranker. `num_labels` is unused."""
    del num_labels
    return load_ranker(model_name_or_path)
