"""One plain terminal line per training event, copied into the fold log."""

from __future__ import annotations

from pathlib import Path


def emit(message: str, path: Path | None = None) -> str:
    print(message, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    return message


def format_prepare(
    dataset: str,
    reused: bool,
    synthetic_queries: int,
    synthetic_pairs: int,
    topics: int,
    coffman_pairs: int,
    train_topics: int,
    validation_topics: int,
    test_topics: int,
    pair_path: Path | str,
) -> str:
    sqlite = "reused sqlite" if reused else "loaded sqlite"
    return (
        f"{dataset} | prepare | {sqlite} | synthetic queries {synthetic_queries} | "
        f"synthetic pairs {synthetic_pairs} | coffman topics {topics} | "
        f"coffman pairs {coffman_pairs} | roles {train_topics} train, "
        f"{validation_topics} validation, {test_topics} test | {pair_path}"
    )


def format_train_start(
    dataset: str,
    fold: int,
    n_folds: int,
    device: str,
    seed: int,
    train_pairs: int,
    train_topics: int,
    validation_topics: int,
    test_topics: int,
    epochs: int,
    batch_size: int,
    destination: Path | str,
) -> str:
    return (
        f"{dataset} | fold {fold} of {n_folds} | train | device {device} | seed {seed} | "
        f"train pairs {train_pairs} | topics {train_topics} train, {validation_topics} validation, "
        f"{test_topics} test | epochs {epochs} | batch {batch_size} | {destination}"
    )


def format_step(dataset: str, fold: int, n_folds: int, step: int, total: int, epoch: float, loss: float) -> str:
    return (
        f"{dataset} | fold {fold} of {n_folds} | step {step}/{total} | "
        f"epoch {epoch:.2f} | loss {loss:.4f}"
    )


def format_epoch(
    dataset: str,
    fold: int,
    n_folds: int,
    epoch: float,
    precision: float,
    recall: float,
    f1: float,
    best: bool,
) -> str:
    marker = " | best" if best else ""
    return (
        f"{dataset} | fold {fold} of {n_folds} | epoch {epoch:.0f} | "
        f"validation precision {precision:.3f} | recall {recall:.3f} | f1 {f1:.3f}{marker}"
    )


def format_early_stop(dataset: str, fold: int, n_folds: int, epoch: float, patience: int) -> str:
    return (
        f"{dataset} | fold {fold} of {n_folds} | early stop | "
        f"epoch {epoch:.0f} | patience {patience}"
    )


def format_train_finish(
    dataset: str,
    fold: int,
    n_folds: int,
    best_f1: float,
    step: int,
    destination: Path | str,
    elapsed_seconds: float,
) -> str:
    return (
        f"{dataset} | fold {fold} of {n_folds} | saved | best validation f1 {best_f1:.3f} | "
        f"step {step} | {destination} | elapsed {elapsed_seconds:.0f}s"
    )


def format_eval(
    dataset: str,
    fold: int,
    n_folds: int,
    system: str,
    threshold: float,
    topics: int,
    precision: float,
    recall: float,
    f1: float,
    scoring_ms: float,
) -> str:
    return (
        f"{dataset} | fold {fold} of {n_folds} | evaluate | {system} | "
        f"threshold {threshold:.3f} | test topics {topics} | "
        f"precision {precision:.3f} | recall {recall:.3f} | f1 {f1:.3f} | "
        f"scoring {scoring_ms:.1f} ms"
    )


def format_pooled(
    dataset: str,
    system: str,
    folds_completed: int,
    n_folds: int,
    precision: float,
    recall: float,
    f1: float,
    scoring_ms: float,
) -> str:
    return (
        f"{dataset} | pooled | folds {folds_completed}/{n_folds} | {system} | "
        f"precision {precision:.3f} | recall {recall:.3f} | f1 {f1:.3f} | "
        f"scoring {scoring_ms:.1f} ms | ranker results"
    )
