---
marp: true
theme: semstruct
size: 16:9
paginate: true
footer: SemStruct
---

<!--
Export from the repository root:

npx @marp-team/marp-cli docs/slides/semstruct-progress.md --theme docs/slides/semstruct.css --html -o docs/slides/semstruct-progress.html

Use --pdf or --pptx for the other formats. The HTML file is a preview, not part of the repository.
-->

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "" -->

# SemStruct

One DistilBERT model per database, trained to say which schema element a keyword query refers to.

<p class="date">14 November 2026</p>

---

<p class="eyebrow">The task</p>

# A keyword still has to land in the schema

A person types a few words. The eventual answer is a joined tuple. Before that, the system has to name the relation, the attribute, and the cell those words refer to.

---

<p class="eyebrow">This version</p>

# The offline half is in place

<div class="split">
  <div class="card">
    <p class="kicker">Built</p>
    <p>One cross-encoder for Mondial, one for IMDb, and one for Wikipedia.</p>
  </div>
  <div class="card">
    <p class="kicker">Later</p>
    <p>Candidate networks, SQL execution, and ranking of the joined answers.</p>
  </div>
</div>

---

<p class="eyebrow">Data</p>

# The Coffman and Weaver benchmark

<div class="grid two">
  <div class="card">
    <p class="kicker">Paper</p>
    <p class="cite">CIKM 2010</p>
    <p class="meta">doi:10.1145/1871437.1871531</p>
  </div>
  <div class="card">
    <p class="kicker">Files</p>
    <p class="cite">Virginia Dataverse, 2017</p>
    <p class="meta">doi:10.18130/V3/KEVCF8</p>
  </div>
</div>

<p class="note">Evaluation subsets only. Not the full IMDb or Wikipedia archives.</p>

---

<!-- _class: cards -->

<p class="eyebrow">The three subsets</p>

# Training pairs come from the databases

<div class="grid">
  <div class="card">
    <p class="kicker">Mondial</p>
    <p class="num">29,996</p>
    <p class="meta">pairs, 28 relations</p>
  </div>
  <div class="card">
    <p class="kicker">IMDb 2009</p>
    <p class="num">6,380</p>
    <p class="meta">pairs, 6 relations</p>
  </div>
  <div class="card">
    <p class="kicker">Wikipedia 2008</p>
    <p class="num">3,832</p>
    <p class="meta">pairs, 6 relations</p>
  </div>
</div>

<p class="note">Fifty human topics per database are held out of training.</p>

---

<p class="eyebrow">Supervision</p>

# A query class makes the label

<div class="steps">
  <div>
    <span>1</span>
    <p>Take one to three relations joined by foreign keys.</p>
  </div>
  <div>
    <span>2</span>
    <p>Read one or two short cells and turn them into keywords.</p>
  </div>
  <div>
    <span>3</span>
    <p>Those relations, attributes, and cells are relevant.</p>
  </div>
  <div>
    <span>4</span>
    <p>Elements outside the class are the negatives.</p>
  </div>
</div>

---

<p class="eyebrow">Model</p>

# A cross-encoder, not a new architecture

- Checkpoint `distilbert/distilbert-base-uncased`, English and uncased.
- The input is the query beside one element card, cut at 128 tokens.
- The output is the probability that the element is relevant.
- Each database keeps its own model under `outputs/models`. Nothing is uploaded.

---

<p class="eyebrow">The run</p>

# Prepare, fine-tune, then compare

<div class="steps">
  <div>
    <span>1</span>
    <p>Parse the PostgreSQL dumps into one SQLite database each.</p>
  </div>
  <div>
    <span>2</span>
    <p>Fine-tune on Apple Silicon, device MPS, for at most three epochs.</p>
  </div>
  <div>
    <span>3</span>
    <p>Score the held-out pairs with token overlap.</p>
  </div>
  <div>
    <span>4</span>
    <p>Score the same pairs with frozen DistilBERT.</p>
  </div>
</div>

---

<!-- _class: cards -->

<p class="eyebrow">Results</p>

# The fine-tuned models separate the pairs

<div class="grid">
  <div class="card">
    <p class="kicker">Mondial</p>
    <p class="num">0.876</p>
    <p class="meta">test F1, ROC-AUC 0.984</p>
  </div>
  <div class="card">
    <p class="kicker">IMDb</p>
    <p class="num">0.968</p>
    <p class="meta">test F1, ROC-AUC 0.998</p>
  </div>
  <div class="card">
    <p class="kicker">Wikipedia</p>
    <p class="num">0.926</p>
    <p class="meta">test F1, ROC-AUC 0.992</p>
  </div>
</div>

<p class="note">Validation F1 was 0.873, 0.979, and 0.943. The baselines stay near 0.30 to 0.42 F1.</p>

---

<!-- _class: rank -->

<p class="eyebrow">One query</p>

# montevideo uruguay

<p class="query">Held-out Mondial test query. Seven of 28 elements are relevant.</p>

<ul class="scores">
  <li class="hit"><span class="score">0.996</span><span>relation city</span><span class="tag">relevant</span></li>
  <li class="hit"><span class="score">0.996</span><span>relation country</span><span class="tag">relevant</span></li>
  <li class="hit"><span class="score">0.995</span><span>relation borders</span><span class="tag">relevant</span></li>
  <li class="hit"><span class="score">0.994</span><span>attribute city.name</span><span class="tag">relevant</span></li>
  <li class="hit"><span class="score">0.989</span><span>attribute country.province</span><span class="tag">relevant</span></li>
  <li><span class="score">0.908</span><span>value country.capital = montevideo</span><span class="tag">overlap</span></li>
  <li><span class="score">0.061</span><span>values with neither keyword</span><span class="tag">irrelevant</span></li>
</ul>

---

<p class="eyebrow">Use a checkpoint</p>

# Load the model for that database

- `outputs/models/mondial`
- `outputs/models/imdb`
- `outputs/models/wikipedia`

A Mondial model is not used on IMDb. The code is MIT. The dumps are not redistributed.
