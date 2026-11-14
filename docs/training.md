# Training

The model is `distilbert/distilbert-base-uncased`, loaded from the Hugging Face Hub. The multilingual checkpoint is not used. Weights are cached by Transformers and are not stored in git. After fine-tuning, each database has its own checkpoint in `outputs/models/<dataset>/`, in Hugging Face format. The run does not upload that checkpoint.

## Query classes

Let the schema graph be an undirected graph whose nodes are relations and whose edges are foreign keys. A self-key does not connect two relations. A query class of size \(k \in \{1, 2, 3\}\) is a connected set of \(k\) relations.

Classes are enumerated exhaustively and then capped. All single-relation classes are kept. If the total exceeds `max_classes` (400), the larger classes are shuffled with seed 42 and truncated. The class id is `class:` plus the sorted relation names joined by `|`.

## Keyword queries

For a class, the loader samples up to `queries_per_class` (20) joined rows. The SQL join follows a spanning tree of the foreign keys inside the class and stops after that limit. A join that exceeds `join_timeout_seconds` (15) is skipped.

From each row the generator reads text cells of at most `max_value_chars` (80) characters that contain a letter. Underscores become spaces. A cell of at most four tokens is one keyword. A longer cell contributes its first two non-stopword tokens. When the row has two or more usable cells, the query contains two keywords. Otherwise it contains one.

The same normalized query string is kept once. If it equals a Coffman topic, it is dropped.

## Labels

For a query generated from class \(c\) with keyword attributes \(A_c\) and keyword cells \(V_c\):

- Positive relations are the relations in \(c\).
- Positive attributes are the members of \(A_c\).
- Positive values are the cells in \(V_c\).
- Negatives are other relations, attributes, and values. Half of the negative budget, `hard_negative_ratio` 0.5, prefers elements whose tokens overlap the query. The other half prefers values that do not contain a query keyword.

There are `negatives_per_positive` (3) negatives for each positive, drawn without replacement and never from the positive set. Labels follow class membership. A cell that merely contains the same string is negative when it did not generate the query.

The total number of pairs is capped at `max_pairs` (30000) by dropping whole queries after a seeded shuffle. The surviving queries are split 80/10/10 by query id, so no query appears in two splits. The split itself is a fresh shuffle with the same seed.

## Element text

Relation text lists the name, attributes, primary key, foreign keys, and a comment when the dump has one. It does not list cell values.

Attribute text lists `relation.attribute`, the SQL type, the referenced column, and up to five example values drawn once per attribute. For a given pair, an example is omitted when it equals a query keyword or contains one. Content-pool samples (`max_values_per_attribute`, 200) are a separate, larger draw used as negative values.

Value text quotes the cell, adds a space-normalized form when it differs, and names the attribute and relation.

Columns named `__search_id`, non-text types, and columns whose names contain `password`, `token`, or `email` are not keyword sources.

## Objective

The classifier is a cross-encoder. The input is the query, a separator, and the element text, truncated at 128 tokens. A linear layer on the classifier token has two logits, `irrelevant` and `relevant`.

\[
\mathcal{L} = -\sum_{(q,e,y)} \Big( y \log p_\theta(y=1 \mid q, e) + (1-y) \log p_\theta(y=0 \mid q, e) \Big)
\]

Class weights \(N / (2 N_c)\) balance the three-to-one negative ratio. The optimizer is AdamW with learning rate \(2 \times 10^{-5}\), weight decay 0.01, and a warmup of 10 percent of the planned steps (`warmup_steps`, because this Transformers release has no `warmup_ratio` field). Training runs for at most 3 epochs and keeps the checkpoint with the best validation F1. Early stopping has patience 2. The decision threshold used at evaluation is tuned for F1 on the validation split. F1 at 0.5 is also recorded.

The device is CUDA when it is available, otherwise Apple MPS, otherwise CPU. Data loader workers are 0 so Windows does not need a spawn workaround. The training entry point is `semstruct.cli.main`.

At inference the same model returns:

- \(p(y=1 \mid q, e)\) for the full query
- the same probability for each whitespace-separated keyword
- the query-class score, the mean probability of the class's relations and text attributes

Keyword probabilities are not trained as separate targets. A scored Mondial test query, `montevideo uruguay`, is in the README section "Score a query".

## Metrics

On the synthetic validation and test splits, for the fine-tuned model, for token overlap, and for cosine similarity of frozen DistilBERT mean-pooled embeddings:

- Precision, recall, F1, accuracy, ROC-AUC, and PR-AUC of the pair labels. F1 uses the validation-tuned threshold. F1 at 0.5 is reported beside it.
- The same classification metrics by element type.
- MAP, MRR, Recall@5, and Recall@10 of the elements paired with each query.
- The same ranking metrics over every relation and text attribute. A relation is relevant when it belongs to the generating class. An attribute is relevant when it supplied a keyword.
- MRR and Recall@1 of the generating query class among the capped class list. Ties break by class id.

Token overlap is the fraction of query content-tokens that also occur in the element text. The frozen baseline encodes the query and the element separately with the original DistilBERT encoder and takes their cosine.

On the Coffman topics, gold relations are the relations of tuples named by `__search_id`. MAP, MRR, Recall@5, and Recall@10 compare those relations with every other relation. Topics whose ids cannot be resolved are counted and omitted from that average. The top five relations and attributes, plus per-keyword scores for the top relations, are stored in `outputs/<dataset>/metrics.json` under `coffman.qualitative`. The assessments do not label attributes, so attribute lists stay qualitative.

## Configuration

Shared defaults live in [configs/default.yaml](../configs/default.yaml). Per-database files in `configs/datasets/` currently set the dataset name and description. Override any key with `semstruct train --config path.yaml` or with `--epochs`, `--batch-size`, and `--limit-pairs`.
