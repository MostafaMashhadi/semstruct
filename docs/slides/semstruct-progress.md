---
marp: true
theme: semstruct
size: 16:9
paginate: true
footer: SemStruct
---

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "" -->

# SemStruct

One DistilBERT ranker per database, scored on Coffman topics with 5-fold cross-validation.

---

<p class="eyebrow">The task</p>

# A keyword still has to land in the schema

A person types a few words. The model scores one keyword, in the context of the whole query, against one relation, attribute, or cell.

---

<p class="eyebrow">This version</p>

# The offline ranker

<div class="split">
  <div class="card">
    <p class="kicker">Built</p>
    <p>Keyword instances, a sigmoid head, and five folds on Mondial, IMDb, and Wikipedia.</p>
  </div>
  <div class="card">
    <p class="kicker">Later</p>
    <p>Candidate networks, SQL execution, and ranking of the joined answers.</p>
  </div>
</div>

---

<p class="eyebrow">Folds</p>

# Ten topics out, eight to choose, thirty-two to train

Each fold tests 10 Coffman topics. Eight of the rest choose the checkpoint and the threshold. The other 32 train, with every synthetic instance. A test topic never updates that fold.

---

<p class="eyebrow">Ranker results</p>

# Precision, recall, and F1

| Database | System | Precision | Recall | F1 | Scoring |
| --- | --- | ---: | ---: | ---: | ---: |
| Mondial | Fine-tuned | 0.455 | 0.787 | 0.576 | 57 ms |
| Mondial | Pre-trained | 0.257 | 0.667 | 0.371 | 90 ms |
| IMDb | Fine-tuned | 0.038 | 0.709 | 0.072 | 42.6 s |
| IMDb | Pre-trained | 0.000 | 0.420 | 0.001 | 46.5 s |
| Wikipedia | Fine-tuned | 0.171 | 0.415 | 0.243 | 6.4 s |
| Wikipedia | Pre-trained | 0.015 | 0.425 | 0.030 | 6.9 s |

<p class="note">These are keyword-element decisions, not end-to-end ICN retrieval. Wikipedia sits beside the Mondial and IMDb article rows.</p>
