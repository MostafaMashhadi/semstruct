# Training

The model is `distilbert/distilbert-base-uncased`. After fine-tuning, each database has five checkpoints in `outputs/models/<dataset>/fold-<k>/`. The run does not upload those checkpoints.

## Keyword instance

A row is one keyword `k`, the full query `q`, one element `e`, and a label. The model reads `[CLS] k [SEP] q [SEP] Te [SEP]`, truncated at 128 tokens. The pooled vector is the `[CLS]` hidden state. One linear unit and a sigmoid produce the relevance probability. The loss is the unweighted mean binary cross-entropy.

Element text follows the draft. A relation contributes its name, its comment when the dump has one, its attribute names, and the relations joined to it. An attribute contributes its name, type, key role, comment when present, and its relation. A value contributes the cell, its attribute, and its relation. These dumps have no relation comments, so that line is absent.

## Labels

The label is 1 when `e` is an index hit that the intended interpretation assigns to `k`.

For a synthetic query, the intended hits are the cell, the attribute, and the relation that supplied `k`. For a Coffman topic, a value is intended when it lies in a judged tuple and contains `k`. An attribute is intended when a judged tuple's cell in that attribute contains `k`, or when `k` matches the attribute name and the attribute belongs to a judged relation. A relation is intended when `k` matches the relation name and that relation is judged.

The other index hits of the same keyword are negatives. Every schema hit is kept. In the training loss only, other value hits are capped at 10 per keyword, drawn with seed 42. Validation and test score every index hit. The threshold is chosen on the full validation list.

A synthetic string that equals any Coffman topic is dropped. Every remaining synthetic instance trains every fold.

## Cross-validation

Seed 42 shuffles the 50 topic ids into 5 folds of 10. Fold `k` tests those 10 topics. Of the other 40, 8 choose the checkpoint and the decision threshold, and 32 train. A test topic never updates that fold's weights.

## Article numbers

The query texts are the synthetic queries plus the 50 Coffman topics. A label is positive only for the index hit the topic or the generator assigns to that keyword. Training keeps every schema hit and at most 10 other value hits per keyword, and the reported precision and recall score every index hit. The pre-trained numbers are the frozen [CLS] cosine.

The cells filled here are ranker results for pre-trained DistilBERT and fine-tuned DistilBERT, plus the mean time to score one test query. ICN, Key2Vec, network counts, and end-to-end time stay with the online stage. Wikipedia is measured and kept beside the Mondial and IMDb table.

| Database | System | Precision | Recall | F1 | Scoring |
| --- | --- | ---: | ---: | ---: | ---: |
| Mondial | Fine-tuned | 0.455 | 0.787 | 0.576 | 57 ms |
| Mondial | Pre-trained | 0.257 | 0.667 | 0.371 | 90 ms |
| IMDb | Fine-tuned | 0.038 | 0.709 | 0.072 | 42.6 s |
| IMDb | Pre-trained | 0.000 | 0.420 | 0.001 | 46.5 s |
| Wikipedia | Fine-tuned | 0.171 | 0.415 | 0.243 | 6.4 s |
| Wikipedia | Pre-trained | 0.015 | 0.425 | 0.030 | 6.9 s |

| Database | Synthetic queries | Synthetic pairs | Coffman topics | Coffman pairs |
| --- | ---: | ---: | ---: | ---: |
| Mondial | 984 | 30,000 | 50 | 1,724 |
| IMDb | 268 | 5,334 | 50 | 786,125 |
| Wikipedia | 171 | 3,013 | 50 | 107,865 |

| Database | Category | Positive | Negative | Total |
| --- | --- | ---: | ---: | ---: |
| Mondial | Relation | 1,925 | 0 | 1,925 |
| Mondial | Attribute | 2,057 | 13,495 | 15,552 |
| Mondial | Value | 2,081 | 12,166 | 14,247 |
| Mondial | All | 6,063 | 25,661 | 31,724 |
| IMDb | Relation | 534 | 0 | 534 |
| IMDb | Attribute | 687 | 2,089 | 2,776 |
| IMDb | Value | 709 | 787,440 | 788,149 |
| IMDb | All | 1,930 | 789,529 | 791,459 |
| Wikipedia | Relation | 287 | 0 | 287 |
| Wikipedia | Attribute | 457 | 630 | 1,087 |
| Wikipedia | Value | 607 | 108,897 | 109,504 |
| Wikipedia | All | 1,351 | 109,527 | 110,878 |
