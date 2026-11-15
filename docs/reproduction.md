# Reproduction

The commands are the same on macOS, Windows, and Linux. Paths in the code use `pathlib`. Dataset names are lowercase. Text files are UTF-8. PostgreSQL does not need to be installed. The loader reads the plain dumps itself.

Tested on macOS with Python 3.12, PyTorch 2.15.0, Transformers 5.23.0, and Accelerate 1.15.0. The Trainer imports Accelerate, which is listed in `requirements.txt`. Windows and Linux use the same code. Install a CUDA build of PyTorch on NVIDIA machines. An Intel Mac, or any machine without CUDA or MPS, trains on CPU. Only the device and the runtime change.

## macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Apple Silicon the default PyTorch wheel uses MPS. On Linux without an NVIDIA GPU it uses CPU. After each database the CLI releases the CUDA or MPS cache, so `train --dataset all` does not keep every model resident. That matters on an 8 GB machine.

## Windows

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## NVIDIA CUDA

Install PyTorch from the CUDA wheel index that matches the driver, then install the rest of the requirements. The training arguments enable pinned memory only when CUDA is visible. Example for a current CUDA 12 wheel:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
```

Confirm the device before a long run:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

## Experiment

```bash
semstruct download --dataset all
semstruct prepare --dataset all
semstruct train --dataset mondial --overfit-check
semstruct train --dataset imdb
semstruct train --dataset wikipedia
semstruct evaluate --dataset all
```

`train --overfit-check` memorizes one batch of 16 pairs for 30 steps at learning rate `1e-4`, with dropout disabled, and writes `outputs/<dataset>/overfit_check.json`. The loss has to fall. The check then continues into the real fine-tune, which uses the learning rate in `configs/default.yaml`.

`prepare --rebuild` reloads SQLite from the dumps. Without `--rebuild`, an existing `data/processed/<dataset>/database.sqlite` is reused. Prepare writes relation, attribute, and value counts, with positives and negatives separate, to `prepare_stats.json`. It also assigns the 50 Coffman topics: 17 to test, 29 to train, and 4 to validation.

## Outputs

| Path | Contents |
| --- | --- |
| `data/processed/<dataset>/database.sqlite` | Rows, plus `semstruct_search_id` |
| `data/processed/<dataset>/schema.json` | Relations, keys, types |
| `data/processed/<dataset>/pairs.jsonl` | Training pairs and splits |
| `data/processed/<dataset>/queries.jsonl` | Generated queries and their classes |
| `data/processed/<dataset>/prepare_stats.json` | Pair and split counts |
| `outputs/models/<dataset>/` | Fine-tuned tokenizer and model |
| `outputs/<dataset>/run_config.json` | Model id, revision, device, platform, hyperparameters |
| `outputs/<dataset>/train_history.json` | Trainer log and best validation F1 |
| `outputs/<dataset>/metrics.json` | Test metrics, baselines, and Coffman report |

`run_config.json` records the Hub model id, the downloaded revision, the selected device, and `platform.platform()`. The cleaned terminal log of the reference run, with the per-step trainer history, is [logs/2026-11-14-reference-run.md](logs/2026-11-14-reference-run.md). `outputs/` is gitignored, so that file is the committed copy of the log.

## Tests

The unit tests do not download a model. They cover the dump parser, foreign-key classes, label rules, the example-value leakage filter, topic files, and the ranking metrics.

```bash
python -m pytest
```

If the benchmark schema files are already in `data/raw/`, the parser test also checks that Mondial, IMDb, and Wikipedia expose their expected tables and at least one foreign key.

## Manual download

The Dataverse page is <https://dataverse.lib.virginia.edu/dataset.xhtml?persistentId=doi:10.18130/V3/KEVCF8>. Place the nine evaluation files named in [datasets.md](datasets.md) in `data/raw/` and run `prepare`.
