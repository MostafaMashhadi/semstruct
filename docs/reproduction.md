# Reproduction

The commands are the same on macOS, Windows, and Linux. Paths in the code use `pathlib`. Dataset names are lowercase. Text files are UTF-8. PostgreSQL does not need to be installed. The loader reads the plain dumps itself.

Tested on macOS with Python 3.12. The Trainer imports Accelerate, which is listed in `requirements.txt`. Windows and Linux use the same code. Install a CUDA build of PyTorch on NVIDIA machines. An Intel Mac, or any machine without CUDA or MPS, trains on CPU.

## macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Apple Silicon the default PyTorch wheel uses MPS. On Linux without an NVIDIA GPU it uses CPU. After each database the CLI releases the CUDA or MPS cache. That matters on an 8 GB machine.

## Windows

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## NVIDIA CUDA

Install PyTorch from the CUDA wheel index that matches the driver, then install the rest of the requirements. The training arguments enable pinned memory only when CUDA is visible.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Experiment

```bash
semstruct download --dataset all
semstruct prepare --dataset mondial
semstruct train --dataset mondial --fold 0
semstruct evaluate --dataset mondial --fold 0
semstruct train --dataset mondial --fold all
semstruct evaluate --dataset mondial --fold all
```

Repeat for `imdb` and `wikipedia`, one database at a time. `--fold all` runs folds 0 through 4. Each fold writes `outputs/<dataset>/fold-<k>/run.log`, `run_config.json`, and `train_history.json`. Evaluation writes `metrics.json` in that directory and pools finished folds into `outputs/<dataset>/cv-metrics.json`.

`prepare` reuses SQLite unless `--rebuild` is set. It rewrites the pair files, stamps each Coffman topic with its test fold, and copies `prepare_stats.json` to `outputs/<dataset>/`.

## Outputs

| Path | Contents |
| --- | --- |
| `data/processed/<dataset>/pairs.jsonl` | Keyword instances |
| `data/processed/<dataset>/prepare_stats.json` | Counts, categories, and fold roles |
| `outputs/<dataset>/prepare_stats.json` | The same prepare record |
| `outputs/models/<dataset>/fold-<k>/` | Fine-tuned encoder and `scorer.pt` |
| `outputs/<dataset>/fold-<k>/run.log` | Plain lines from that fold |
| `outputs/<dataset>/fold-<k>/metrics.json` | Threshold, precision, recall, F1, scoring time |
| `outputs/<dataset>/cv-metrics.json` | Pooled ranker numbers for the folds that have finished |
| `exports/training-data/` | Current pair export |

## Tests

The unit tests do not download a model. They cover the keyword instance, the 10/8/32 fold roles, retrieval precision and recall, and the plain log lines.

```bash
python -m pytest
```
