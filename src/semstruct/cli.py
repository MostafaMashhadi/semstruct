"""Command line interface for download, prepare, train, and evaluate."""

from __future__ import annotations

import argparse
import logging

from semstruct import __version__
from semstruct.config import load_config
from semstruct.data.download import download_dataset
from semstruct.data.prepare import prepare_dataset
from semstruct.paths import resolve_datasets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="semstruct", description="SemStruct DistilBERT domain adaptation")
    parser.add_argument("--version", action="version", version=f"semstruct {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download benchmark subsets into data/raw")
    _add_dataset(download)

    prepare = subparsers.add_parser("prepare", help="Build SQLite databases and training pairs")
    _add_dataset(prepare)
    prepare.add_argument("--config", default=None)
    prepare.add_argument("--rebuild", action="store_true")

    train = subparsers.add_parser("train", help="Fine-tune one DistilBERT classifier per database")
    _add_dataset(train)
    train.add_argument("--config", default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--limit-pairs", type=int, default=None)
    train.add_argument("--fold", default="all", help="Fold 0-4, or all")
    train.add_argument(
        "--overfit-check",
        action="store_true",
        help="Memorize one batch before training. The run stops if the loss does not fall.",
    )

    evaluate = subparsers.add_parser("evaluate", help="Score the fine-tuned model and the baselines")
    _add_dataset(evaluate)
    evaluate.add_argument("--config", default=None)
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.add_argument("--fold", default="all", help="Fold 0-4, or all")

    cns = subparsers.add_parser("cns", help="Generate primary candidate networks for the Coffman topics")
    _add_dataset(cns)
    cns.add_argument("--max-relations", type=int, default=5)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    datasets = resolve_datasets(args.dataset)
    if args.command == "download":
        for dataset in datasets:
            download_dataset(dataset)
        return
    if args.command == "prepare":
        for dataset in datasets:
            config = load_config(dataset, args.config)
            prepare_dataset(dataset, config, rebuild=args.rebuild)
        return
    if args.command == "train":
        from semstruct.ranker.train import overfit_check, train_dataset

        for dataset in datasets:
            config = load_config(dataset, args.config)
            if args.overfit_check:
                result = overfit_check(dataset, config, batch_size=args.batch_size or 16)
                if not result["decreased"]:
                    raise SystemExit(f"Overfit check failed for {dataset}: loss did not decrease")
            train_dataset(
                dataset,
                config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                limit_pairs=args.limit_pairs,
                fold=_parse_fold(args.fold),
            )
        return
    if args.command == "evaluate":
        from semstruct.ranker.evaluate import evaluate_dataset

        for dataset in datasets:
            config = load_config(dataset, args.config)
            evaluate_dataset(dataset, config, batch_size=args.batch_size, fold=_parse_fold(args.fold))
        return
    if args.command == "cns":
        from semstruct.networks.primary import generate_dataset

        for dataset in datasets:
            generate_dataset(dataset, max_relations=args.max_relations)
        return
    raise SystemExit(f"Unknown command {args.command}")


def _add_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=["mondial", "imdb", "wikipedia", "all"])


def _parse_fold(value: str) -> int | str:
    if value == "all":
        return "all"
    if value in {"0", "1", "2", "3", "4"}:
        return int(value)
    raise SystemExit("Fold must be 0, 1, 2, 3, 4, or all.")


if __name__ == "__main__":
    main()
