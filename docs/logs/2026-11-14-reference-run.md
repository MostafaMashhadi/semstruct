# Reference run log, 14 November 2026

This is the log of the 14 November run. A later prepare step adds the Coffman topics to the pairs and changes the counts below. The commands and metrics in this file are the earlier corpus, in which those 50 topics were held out of the pairs. The machine was an Apple Silicon Mac with 8 GB of memory, macOS 27.2, Python 3.12.12, PyTorch 2.15.0, Transformers 5.23.0, and Accelerate 1.15.0. The device was MPS. The checkpoint id is `distilbert/distilbert-base-uncased`, Hub revision `12040accade4e8a0f71eabdb258fecc2e7e948be`. Commands were run from the repository root as `PYTHONPATH=src .venv/bin/python -m semstruct ...` because the package was not installed editable in that shell.

Progress bars that rewrite a single terminal line are omitted. They contain no extra measurements. The per-step trainer records for Mondial and IMDb were inside those bars, so the tables below are copied from `outputs/<dataset>/train_history.json`, which is the same series the Trainer printed every 50 steps. `outputs/` is gitignored. This file is the copy that stays in the repository.

Times below are UTC.

## Prepare

Started 13:29:18. Finished in 103 s. Exit code 0.

```text
PYTHONPATH=src .venv/bin/python -m semstruct prepare --dataset all
INFO semstruct.data.prepare: loading mondial into data/processed/mondial/database.sqlite
INFO semstruct.data.dump: finished loading 17115 rows from mondial-data.pgsql.bz2
INFO semstruct.data.dump: loaded 28 relations into data/processed/mondial/database.sqlite
INFO semstruct.data.instances: generated queries from 50/175 classes
INFO semstruct.data.instances: generated queries from 100/175 classes
INFO semstruct.data.instances: generated queries from 150/175 classes
INFO semstruct.data.instances: prepared {'classes': 175, 'queries': 1178, 'pairs': 29996, 'skipped_topic_collisions': 2, 'skipped_empty_rows': 4, 'splits': {'train': 942, 'validation': 117, 'test': 119}}
INFO semstruct.data.prepare: wrote 29996 pairs for mondial
INFO semstruct.data.prepare: loading imdb into data/processed/imdb/database.sqlite
INFO semstruct.data.dump: finished loading 1673076 rows from imdb2009_subset-data.pgsql.bz2
INFO semstruct.data.dump: loaded 6 relations into data/processed/imdb/database.sqlite
INFO semstruct.data.instances: prepared {'classes': 18, 'queries': 266, 'pairs': 6380, 'skipped_topic_collisions': 0, 'skipped_empty_rows': 28, 'splits': {'train': 212, 'validation': 26, 'test': 28}}
INFO semstruct.data.prepare: wrote 6380 pairs for imdb
INFO semstruct.data.prepare: loading wikipedia into data/processed/wikipedia/database.sqlite
INFO semstruct.data.dump: finished loading 206318 rows from wikipedia2008_subset-data.pgsql.bz2
INFO semstruct.data.dump: loaded 6 relations into data/processed/wikipedia/database.sqlite
INFO semstruct.data.instances: prepared {'classes': 16, 'queries': 178, 'pairs': 3832, 'skipped_topic_collisions': 0, 'skipped_empty_rows': 30, 'splits': {'train': 142, 'validation': 17, 'test': 19}}
INFO semstruct.data.prepare: wrote 3832 pairs for wikipedia
```

IMDb logged `loaded N rows` every 50,000 rows from 50,000 through 1,650,000, ending in `char_name`, then `name`, `title`, `cast_info`, and `movie_info`. Wikipedia logged the same counter at 50,000, 100,000, 150,000, and 200,000, all while inserting `pagelinks`. The finished totals above are the load results.

## Mondial overfit check

Started 13:31:15. Failed after 169 s, after the check itself had passed.

```text
PYTHONPATH=src .venv/bin/python -m semstruct train --dataset mondial --overfit-check
INFO semstruct.training.train: overfit check {'dataset': 'mondial', 'device': 'mps', 'batch_size': 16, 'learning_rate': 0.0001, 'steps': 30, 'initial_loss': 0.7044544219970703, 'final_loss': 0.0010363305918872356, 'decreased': True}
ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=1.1.0`
```

The head memorized one mixed batch of 16 pairs. Loss fell from 0.704 to 0.001, and `outputs/mondial/overfit_check.json` was written. The process then stopped while constructing `TrainingArguments`, before any fine-tuning step. `accelerate` 1.15.0 was installed and added to `requirements.txt` and `pyproject.toml`. The same overfit file is the smoke check for this reference run. It was not repeated.

Loading the sequence-classification head printed the expected Hugging Face report: `vocab_transform`, `vocab_layer_norm`, and `vocab_projector` were unexpected, and `pre_classifier` plus `classifier` were newly initialized. The same report appeared at the start of every training and frozen-encoder load. It is not a failed checkpoint.

## Mondial and IMDb fine-tunes

Started 13:34:24.

```text
PYTHONPATH=src .venv/bin/python -m semstruct train --dataset all
INFO semstruct.training.train: training mondial on 24064 pairs using mps
INFO semstruct.training.train: saved mondial model to outputs/models/mondial
INFO semstruct.training.train: training imdb on 5088 pairs using mps
INFO semstruct.training.train: saved imdb model to outputs/models/imdb
INFO semstruct.training.train: training wikipedia on 3060 pairs using mps
```

Mondial and IMDb are the checkpoints from this process. Both ran all 3 epochs. Validation F1 rose at every epoch, so early stopping did not cut the run. The best checkpoint is the last epoch for both.

The Wikipedia portion of this same process was stopped by hand at about step 156 of 576. Step time had grown from about 1 s to about 19 s, the process footprint was 6.4 GB, and the 8 GB machine was swapping. No Wikipedia checkpoint was saved from that attempt. The CLI now releases the MPS or CUDA cache after each database so the next database does not keep the previous allocator cache.

### Mondial validation by epoch

| epoch | step | eval loss | F1 | precision | recall | accuracy | runtime (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1504 | 0.1957 | 0.8583 | 0.8246 | 0.8947 | 0.9261 | 10.00 |
| 2 | 3008 | 0.1735 | 0.8667 | 0.8403 | 0.8947 | 0.9312 | 9.15 |
| 3 | 4512 | 0.1843 | 0.8734 | 0.8458 | 0.9028 | 0.9345 | 7.88 |

Best validation F1 0.8734 at step 4512. Train runtime 940.2 s. 76.78 samples/s. 4.799 steps/s. Mean train loss 0.2211. Warmup was 451 steps. The saved model is `outputs/models/mondial`.

### IMDb validation by epoch

| epoch | step | eval loss | F1 | precision | recall | accuracy | runtime (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 318 | 0.0422 | 0.9689 | 0.9524 | 0.9859 | 0.9842 | 2.90 |
| 2 | 636 | 0.0309 | 0.9759 | 0.9530 | 1.0000 | 0.9877 | 2.51 |
| 3 | 954 | 0.0299 | 0.9793 | 0.9595 | 1.0000 | 0.9894 | 2.48 |

Best validation F1 0.9793 at step 954. Train runtime 332.5 s. 45.90 samples/s. 2.869 steps/s. Mean train loss 0.1325. Warmup was 95 steps. The saved model is `outputs/models/imdb`.

## Wikipedia fine-tune

Started 14:07:51. Finished 14:11:25. Exit code 0. Elapsed 213 s. This is the saved Wikipedia checkpoint. tqdm was disabled for this run, so the terminal printed the trainer dicts directly.

```text
PYTHONPATH=src .venv/bin/python -m semstruct train --dataset wikipedia
INFO semstruct.training.train: training wikipedia on 3060 pairs using mps
{'loss': '0.615', 'grad_norm': '11.59', 'learning_rate': '1.719e-05', 'epoch': '0.2604'}
{'loss': '0.3367', 'grad_norm': '7.256', 'learning_rate': '1.838e-05', 'epoch': '0.5208'}
{'loss': '0.1777', 'grad_norm': '0.2497', 'learning_rate': '1.645e-05', 'epoch': '0.7812'}
{'eval_loss': '0.2291', 'eval_f1': '0.8727', 'eval_precision': '0.878', 'eval_recall': '0.8675', 'eval_accuracy': '0.9367', 'eval_runtime': '2.933', 'eval_samples_per_second': '113.2', 'eval_steps_per_second': '7.16', 'epoch': '1'}
{'loss': '0.1723', 'grad_norm': '0.1836', 'learning_rate': '1.453e-05', 'epoch': '1.042'}
{'loss': '0.1414', 'grad_norm': '0.8361', 'learning_rate': '1.26e-05', 'epoch': '1.302'}
{'loss': '0.07493', 'grad_norm': '0.3167', 'learning_rate': '1.067e-05', 'epoch': '1.562'}
{'loss': '0.1344', 'grad_norm': '0.09383', 'learning_rate': '8.748e-06', 'epoch': '1.823'}
{'eval_loss': '0.1322', 'eval_f1': '0.924', 'eval_precision': '0.8977', 'eval_recall': '0.9518', 'eval_accuracy': '0.9608', 'eval_runtime': '1.564', 'eval_samples_per_second': '212.3', 'eval_steps_per_second': '13.43', 'epoch': '2'}
{'loss': '0.09623', 'grad_norm': '18.77', 'learning_rate': '6.821e-06', 'epoch': '2.083'}
{'loss': '0.05581', 'grad_norm': '0.1267', 'learning_rate': '4.894e-06', 'epoch': '2.344'}
{'loss': '0.08004', 'grad_norm': '0.06333', 'learning_rate': '2.967e-06', 'epoch': '2.604'}
{'loss': '0.06472', 'grad_norm': '0.09951', 'learning_rate': '1.04e-06', 'epoch': '2.865'}
{'eval_loss': '0.1045', 'eval_f1': '0.9425', 'eval_precision': '0.9011', 'eval_recall': '0.988', 'eval_accuracy': '0.9699', 'eval_runtime': '1.714', 'eval_samples_per_second': '193.7', 'eval_steps_per_second': '12.25', 'epoch': '3'}
{'train_runtime': '202.8', 'train_samples_per_second': '45.27', 'train_steps_per_second': '2.841', 'train_loss': '0.1732', 'epoch': '3'}
INFO semstruct.training.train: saved wikipedia model to outputs/models/wikipedia
```

Best validation F1 0.9425 at step 576. Train runtime 202.8 s. 45.27 samples/s. 2.841 steps/s. Mean train loss 0.1732. Warmup was 57 steps.

`run_config.json` records `revision: null` for all three saves. The sequence-classification config object did not expose `_commit_hash` after the head was added. The download log for the same weights names revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.

## Evaluation

Started 14:11:33. Finished 14:14:01. Exit code 0. Elapsed 148 s.

```text
PYTHONPATH=src .venv/bin/python -m semstruct evaluate --dataset all
INFO semstruct.training.evaluate: scoring mondial with distilbert
INFO semstruct.training.evaluate: scoring mondial with token_overlap
INFO semstruct.training.evaluate: scoring mondial with frozen_distilbert
INFO semstruct.training.evaluate: wrote outputs/mondial/metrics.json
INFO semstruct.training.evaluate: scoring imdb with distilbert
INFO semstruct.training.evaluate: scoring imdb with token_overlap
INFO semstruct.training.evaluate: scoring imdb with frozen_distilbert
INFO semstruct.training.evaluate: wrote outputs/imdb/metrics.json
INFO semstruct.training.evaluate: scoring wikipedia with distilbert
INFO semstruct.training.evaluate: scoring wikipedia with token_overlap
INFO semstruct.training.evaluate: scoring wikipedia with frozen_distilbert
INFO semstruct.training.evaluate: wrote outputs/wikipedia/metrics.json
```

Every Coffman topic resolved. `unresolved_search_ids` is 0 and `topics_without_resolved_tuples` is 0 for all three databases, 50 topics each. Pair metrics below are the test split. Schema MAP ranks relations and keyword attributes. Class MRR ranks the generating query class by the mean probability of its relations and text attributes. Coffman MAP ranks relations of the held-out topics.

| dataset | system | F1 | P | R | ROC-AUC | PR-AUC | pair MAP | pair MRR | R@5 | R@10 | schema MAP | class MRR | class R@1 | Coffman MAP | Coffman MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mondial | distilbert | 0.876 | 0.854 | 0.899 | 0.984 | 0.949 | 0.950 | 0.987 | 0.753 | 0.988 | 0.257 | 0.032 | 0.008 | 0.367 | 0.508 |
| mondial | token_overlap | 0.400 | 0.250 | 1.000 | 0.553 | 0.305 | 0.694 | 1.000 | 0.473 | 0.698 | 0.181 | 0.093 | 0.034 | 0.400 | 0.486 |
| mondial | frozen_distilbert | 0.400 | 0.250 | 0.999 | 0.543 | 0.291 | 0.438 | 0.737 | 0.314 | 0.478 | 0.150 | 0.115 | 0.050 | 0.149 | 0.204 |
| imdb | distilbert | 0.968 | 0.938 | 1.000 | 0.998 | 0.991 | 0.992 | 1.000 | 0.774 | 1.000 | 0.587 | 0.122 | 0.000 | 0.723 | 0.753 |
| imdb | token_overlap | 0.351 | 0.425 | 0.298 | 0.587 | 0.405 | 0.798 | 1.000 | 0.538 | 0.805 | 0.467 | 0.195 | 0.000 | 0.535 | 0.633 |
| imdb | frozen_distilbert | 0.304 | 0.315 | 0.293 | 0.535 | 0.316 | 0.462 | 0.787 | 0.335 | 0.488 | 0.302 | 0.213 | 0.036 | 0.575 | 0.633 |
| wikipedia | distilbert | 0.926 | 0.884 | 0.973 | 0.992 | 0.971 | 0.969 | 1.000 | 0.796 | 1.000 | 0.612 | 0.149 | 0.053 | 0.512 | 0.595 |
| wikipedia | token_overlap | 0.420 | 0.561 | 0.336 | 0.617 | 0.412 | 0.805 | 1.000 | 0.566 | 0.886 | 0.509 | 0.255 | 0.105 | 0.540 | 0.513 |
| wikipedia | frozen_distilbert | 0.401 | 0.251 | 0.991 | 0.481 | 0.263 | 0.403 | 0.638 | 0.287 | 0.458 | 0.383 | 0.195 | 0.053 | 0.356 | 0.323 |

## Tests

After the training code was in its final form, `python -m pytest` reported 15 passed in 0.93 s.

## Trainer step log

Loss is the mean over the logging window of 50 steps, except the last window of an epoch, which can be shorter. Learning rate is the rate at that step.

### Mondial

| step | epoch | loss | grad norm | learning rate | eval loss | eval F1 | precision | recall | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.033 | 0.6886 | 3.278 | 2.1729e-06 |  |  |  |  |  |
| 100 | 0.066 | 0.6315 | 2.630 | 4.3902e-06 |  |  |  |  |  |
| 150 | 0.100 | 0.4055 | 2.204 | 6.6075e-06 |  |  |  |  |  |
| 200 | 0.133 | 0.4287 | 2.870 | 8.8248e-06 |  |  |  |  |  |
| 250 | 0.166 | 0.3259 | 3.336 | 1.1042e-05 |  |  |  |  |  |
| 300 | 0.199 | 0.2913 | 2.726 | 1.3259e-05 |  |  |  |  |  |
| 350 | 0.233 | 0.2519 | 1.511 | 1.5477e-05 |  |  |  |  |  |
| 400 | 0.266 | 0.3236 | 9.745 | 1.7694e-05 |  |  |  |  |  |
| 450 | 0.299 | 0.2515 | 1.620 | 1.9911e-05 |  |  |  |  |  |
| 500 | 0.332 | 0.2803 | 1.728 | 1.9764e-05 |  |  |  |  |  |
| 550 | 0.366 | 0.3050 | 3.393 | 1.9517e-05 |  |  |  |  |  |
| 600 | 0.399 | 0.2739 | 2.801 | 1.9271e-05 |  |  |  |  |  |
| 650 | 0.432 | 0.2619 | 6.291 | 1.9025e-05 |  |  |  |  |  |
| 700 | 0.465 | 0.2210 | 2.876 | 1.8779e-05 |  |  |  |  |  |
| 750 | 0.499 | 0.3546 | 2.047 | 1.8532e-05 |  |  |  |  |  |
| 800 | 0.532 | 0.2769 | 21.072 | 1.8286e-05 |  |  |  |  |  |
| 850 | 0.565 | 0.2669 | 1.127 | 1.8040e-05 |  |  |  |  |  |
| 900 | 0.598 | 0.2789 | 3.257 | 1.7794e-05 |  |  |  |  |  |
| 950 | 0.632 | 0.2193 | 1.645 | 1.7547e-05 |  |  |  |  |  |
| 1000 | 0.665 | 0.2115 | 2.174 | 1.7301e-05 |  |  |  |  |  |
| 1050 | 0.698 | 0.2042 | 0.346 | 1.7055e-05 |  |  |  |  |  |
| 1100 | 0.731 | 0.2653 | 4.589 | 1.6809e-05 |  |  |  |  |  |
| 1150 | 0.765 | 0.2300 | 1.496 | 1.6562e-05 |  |  |  |  |  |
| 1200 | 0.798 | 0.1973 | 2.404 | 1.6316e-05 |  |  |  |  |  |
| 1250 | 0.831 | 0.2436 | 7.011 | 1.6070e-05 |  |  |  |  |  |
| 1300 | 0.864 | 0.2465 | 0.913 | 1.5824e-05 |  |  |  |  |  |
| 1350 | 0.898 | 0.2044 | 4.149 | 1.5577e-05 |  |  |  |  |  |
| 1400 | 0.931 | 0.1657 | 4.349 | 1.5331e-05 |  |  |  |  |  |
| 1450 | 0.964 | 0.2057 | 0.976 | 1.5085e-05 |  |  |  |  |  |
| 1500 | 0.997 | 0.1779 | 2.297 | 1.4839e-05 |  |  |  |  |  |
| 1504 | 1.000 |  |  |  | 0.1957 | 0.8583 | 0.8246 | 0.8947 | 0.9261 |
| 1550 | 1.031 | 0.1941 | 3.513 | 1.4592e-05 |  |  |  |  |  |
| 1600 | 1.064 | 0.1916 | 4.656 | 1.4346e-05 |  |  |  |  |  |
| 1650 | 1.097 | 0.1719 | 2.183 | 1.4100e-05 |  |  |  |  |  |
| 1700 | 1.130 | 0.2121 | 1.629 | 1.3854e-05 |  |  |  |  |  |
| 1750 | 1.164 | 0.2467 | 2.042 | 1.3607e-05 |  |  |  |  |  |
| 1800 | 1.197 | 0.1729 | 0.157 | 1.3361e-05 |  |  |  |  |  |
| 1850 | 1.230 | 0.2428 | 5.366 | 1.3115e-05 |  |  |  |  |  |
| 1900 | 1.263 | 0.2011 | 0.994 | 1.2869e-05 |  |  |  |  |  |
| 1950 | 1.297 | 0.2212 | 5.044 | 1.2623e-05 |  |  |  |  |  |
| 2000 | 1.330 | 0.1762 | 4.440 | 1.2376e-05 |  |  |  |  |  |
| 2050 | 1.363 | 0.1953 | 6.796 | 1.2130e-05 |  |  |  |  |  |
| 2100 | 1.396 | 0.1762 | 3.326 | 1.1884e-05 |  |  |  |  |  |
| 2150 | 1.430 | 0.2184 | 6.724 | 1.1638e-05 |  |  |  |  |  |
| 2200 | 1.463 | 0.2009 | 1.883 | 1.1391e-05 |  |  |  |  |  |
| 2250 | 1.496 | 0.2325 | 1.956 | 1.1145e-05 |  |  |  |  |  |
| 2300 | 1.529 | 0.1827 | 3.602 | 1.0899e-05 |  |  |  |  |  |
| 2350 | 1.562 | 0.1643 | 4.457 | 1.0653e-05 |  |  |  |  |  |
| 2400 | 1.596 | 0.2280 | 2.632 | 1.0406e-05 |  |  |  |  |  |
| 2450 | 1.629 | 0.2136 | 2.050 | 1.0160e-05 |  |  |  |  |  |
| 2500 | 1.662 | 0.1796 | 0.562 | 9.9138e-06 |  |  |  |  |  |
| 2550 | 1.695 | 0.2238 | 8.968 | 9.6676e-06 |  |  |  |  |  |
| 2600 | 1.729 | 0.2034 | 14.297 | 9.4213e-06 |  |  |  |  |  |
| 2650 | 1.762 | 0.1824 | 5.111 | 9.1751e-06 |  |  |  |  |  |
| 2700 | 1.795 | 0.1755 | 7.346 | 8.9288e-06 |  |  |  |  |  |
| 2750 | 1.828 | 0.2119 | 9.766 | 8.6826e-06 |  |  |  |  |  |
| 2800 | 1.862 | 0.1911 | 6.595 | 8.4363e-06 |  |  |  |  |  |
| 2850 | 1.895 | 0.2085 | 3.123 | 8.1901e-06 |  |  |  |  |  |
| 2900 | 1.928 | 0.1749 | 0.827 | 7.9439e-06 |  |  |  |  |  |
| 2950 | 1.961 | 0.2070 | 1.707 | 7.6976e-06 |  |  |  |  |  |
| 3000 | 1.995 | 0.1793 | 2.179 | 7.4514e-06 |  |  |  |  |  |
| 3008 | 2.000 |  |  |  | 0.1735 | 0.8667 | 0.8403 | 0.8947 | 0.9312 |
| 3050 | 2.028 | 0.1746 | 1.096 | 7.2051e-06 |  |  |  |  |  |
| 3100 | 2.061 | 0.1445 | 0.059 | 6.9589e-06 |  |  |  |  |  |
| 3150 | 2.094 | 0.1906 | 19.006 | 6.7126e-06 |  |  |  |  |  |
| 3200 | 2.128 | 0.1720 | 1.366 | 6.4664e-06 |  |  |  |  |  |
| 3250 | 2.161 | 0.2237 | 2.755 | 6.2201e-06 |  |  |  |  |  |
| 3300 | 2.194 | 0.1709 | 2.354 | 5.9739e-06 |  |  |  |  |  |
| 3350 | 2.227 | 0.1728 | 1.186 | 5.7277e-06 |  |  |  |  |  |
| 3400 | 2.261 | 0.1648 | 5.165 | 5.4814e-06 |  |  |  |  |  |
| 3450 | 2.294 | 0.1866 | 1.131 | 5.2352e-06 |  |  |  |  |  |
| 3500 | 2.327 | 0.1742 | 2.219 | 4.9889e-06 |  |  |  |  |  |
| 3550 | 2.360 | 0.1774 | 1.938 | 4.7427e-06 |  |  |  |  |  |
| 3600 | 2.394 | 0.1549 | 10.546 | 4.4964e-06 |  |  |  |  |  |
| 3650 | 2.427 | 0.1941 | 3.154 | 4.2502e-06 |  |  |  |  |  |
| 3700 | 2.460 | 0.1331 | 4.381 | 4.0039e-06 |  |  |  |  |  |
| 3750 | 2.493 | 0.2183 | 16.818 | 3.7577e-06 |  |  |  |  |  |
| 3800 | 2.527 | 0.2183 | 2.302 | 3.5115e-06 |  |  |  |  |  |
| 3850 | 2.560 | 0.1571 | 3.653 | 3.2652e-06 |  |  |  |  |  |
| 3900 | 2.593 | 0.1704 | 2.116 | 3.0190e-06 |  |  |  |  |  |
| 3950 | 2.626 | 0.1611 | 7.558 | 2.7727e-06 |  |  |  |  |  |
| 4000 | 2.660 | 0.1836 | 1.483 | 2.5265e-06 |  |  |  |  |  |
| 4050 | 2.693 | 0.1803 | 0.595 | 2.2802e-06 |  |  |  |  |  |
| 4100 | 2.726 | 0.1528 | 4.127 | 2.0340e-06 |  |  |  |  |  |
| 4150 | 2.759 | 0.1576 | 2.182 | 1.7877e-06 |  |  |  |  |  |
| 4200 | 2.793 | 0.1751 | 2.974 | 1.5415e-06 |  |  |  |  |  |
| 4250 | 2.826 | 0.1685 | 1.759 | 1.2952e-06 |  |  |  |  |  |
| 4300 | 2.859 | 0.1736 | 32.764 | 1.0490e-06 |  |  |  |  |  |
| 4350 | 2.892 | 0.1381 | 5.006 | 8.0276e-07 |  |  |  |  |  |
| 4400 | 2.926 | 0.2079 | 1.850 | 5.5651e-07 |  |  |  |  |  |
| 4450 | 2.959 | 0.1447 | 3.251 | 3.1027e-07 |  |  |  |  |  |
| 4500 | 2.992 | 0.1850 | 0.647 | 6.4024e-08 |  |  |  |  |  |
| 4512 | 3.000 |  |  |  | 0.1843 | 0.8734 | 0.8458 | 0.9028 | 0.9345 |

### IMDb

| step | epoch | loss | grad norm | learning rate | eval loss | eval F1 | precision | recall | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.157 | 0.6732 | 2.213 | 1.0316e-05 |  |  |  |  |  |
| 100 | 0.314 | 0.4283 | 1.347 | 1.9907e-05 |  |  |  |  |  |
| 150 | 0.472 | 0.3029 | 3.279 | 1.8743e-05 |  |  |  |  |  |
| 200 | 0.629 | 0.1145 | 1.264 | 1.7579e-05 |  |  |  |  |  |
| 250 | 0.786 | 0.1188 | 0.138 | 1.6414e-05 |  |  |  |  |  |
| 300 | 0.943 | 0.0985 | 19.947 | 1.5250e-05 |  |  |  |  |  |
| 318 | 1.000 |  |  |  | 0.0422 | 0.9689 | 0.9524 | 0.9859 | 0.9842 |
| 350 | 1.101 | 0.0616 | 0.115 | 1.4086e-05 |  |  |  |  |  |
| 400 | 1.258 | 0.0589 | 0.081 | 1.2922e-05 |  |  |  |  |  |
| 450 | 1.415 | 0.0698 | 0.130 | 1.1758e-05 |  |  |  |  |  |
| 500 | 1.572 | 0.0629 | 0.098 | 1.0594e-05 |  |  |  |  |  |
| 550 | 1.730 | 0.1003 | 0.078 | 9.4296e-06 |  |  |  |  |  |
| 600 | 1.887 | 0.0620 | 0.260 | 8.2654e-06 |  |  |  |  |  |
| 636 | 2.000 |  |  |  | 0.0309 | 0.9759 | 0.9530 | 1.0000 | 0.9877 |
| 650 | 2.044 | 0.0508 | 0.898 | 7.1013e-06 |  |  |  |  |  |
| 700 | 2.201 | 0.0519 | 5.886 | 5.9371e-06 |  |  |  |  |  |
| 750 | 2.358 | 0.0402 | 1.465 | 4.7730e-06 |  |  |  |  |  |
| 800 | 2.516 | 0.0683 | 0.190 | 3.6088e-06 |  |  |  |  |  |
| 850 | 2.673 | 0.0547 | 0.084 | 2.4447e-06 |  |  |  |  |  |
| 900 | 2.830 | 0.0566 | 0.146 | 1.2806e-06 |  |  |  |  |  |
| 950 | 2.987 | 0.0531 | 0.115 | 1.1641e-07 |  |  |  |  |  |
| 954 | 3.000 |  |  |  | 0.0299 | 0.9793 | 0.9595 | 1.0000 | 0.9894 |

### Wikipedia

| step | epoch | loss | grad norm | learning rate | eval loss | eval F1 | precision | recall | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.260 | 0.6150 | 11.587 | 1.7193e-05 |  |  |  |  |  |
| 100 | 0.521 | 0.3367 | 7.256 | 1.8382e-05 |  |  |  |  |  |
| 150 | 0.781 | 0.1777 | 0.250 | 1.6455e-05 |  |  |  |  |  |
| 192 | 1.000 |  |  |  | 0.2291 | 0.8727 | 0.8780 | 0.8675 | 0.9367 |
| 200 | 1.042 | 0.1723 | 0.184 | 1.4528e-05 |  |  |  |  |  |
| 250 | 1.302 | 0.1414 | 0.836 | 1.2601e-05 |  |  |  |  |  |
| 300 | 1.562 | 0.0749 | 0.317 | 1.0674e-05 |  |  |  |  |  |
| 350 | 1.823 | 0.1344 | 0.094 | 8.7476e-06 |  |  |  |  |  |
| 384 | 2.000 |  |  |  | 0.1322 | 0.9240 | 0.8977 | 0.9518 | 0.9608 |
| 400 | 2.083 | 0.0962 | 18.770 | 6.8208e-06 |  |  |  |  |  |
| 450 | 2.344 | 0.0558 | 0.127 | 4.8940e-06 |  |  |  |  |  |
| 500 | 2.604 | 0.0800 | 0.063 | 2.9672e-06 |  |  |  |  |  |
| 550 | 2.865 | 0.0647 | 0.100 | 1.0405e-06 |  |  |  |  |  |
| 576 | 3.000 |  |  |  | 0.1045 | 0.9425 | 0.9011 | 0.9880 | 0.9699 |
