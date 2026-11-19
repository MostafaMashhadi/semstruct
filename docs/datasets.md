# Datasets

Training and evaluation use the evaluation subsets of the Coffman and Weaver relational keyword-search benchmark.

> Joel Coffman and Alfred C. Weaver. A Framework for Evaluating Database Keyword Search Strategies. CIKM 2010. <https://doi.org/10.1145/1871437.1871531>

The files are distributed by the University of Virginia Dataverse:

> Coffman, Joel; Weaver, Alfred C., 2017. Benchmark for Relational Keyword Search. <https://doi.org/10.18130/V3/KEVCF8>

`semstruct download` asks the Dataverse API for these evaluation files and checks their MD5 sums.

| Dataset | Schema | Data | Topics |
| --- | --- | --- | --- |
| Mondial | `mondial-schema.pgsql.bz2` | `mondial-data.pgsql.bz2` | `Mondial-topics-qrels.tar.bz2` |
| IMDb | `imdb2009_subset-schema.pgsql.bz2` | `imdb2009_subset-data.pgsql.bz2` | `IMDb-topics-qrels-V2.tar.bz2` |
| Wikipedia | `wikipedia2008_subset-schema.pgsql.bz2` | `wikipedia2008_subset-data.pgsql.bz2` | `Wikipedia-topics-qrels.tar.bz2` |

The full IMDb dump and the full Wikipedia dump are not used. IMDb topics are the corrected V2 assessments.

## Terms

The benchmark may be used for academic, personal, and non-commercial work. It must not be resold or repurposed. IMDb rows were obtained with IMDbPY and remain courtesy of IMDb. This repository's MIT license covers the code only. Do not commit or republish the dumps. `data/raw/` and `data/processed/` are gitignored.

If the API download fails, save the files above into `data/raw/` and run `semstruct prepare`. The loader accepts the filenames exactly as published.

## Databases

Mondial is a geographical schema: countries, cities, provinces, rivers, organizations, and the foreign keys among them. The data file is a sequence of `INSERT` statements.

The IMDb subset is a small slice of the 2009 IMDbPY schema: `title`, `name`, `cast_info`, `char_name`, `movie_info`, and `role_type`. Rows arrive as PostgreSQL `COPY` data. Topics refer to a person relation; in this schema that relation is `name`.

The Wikipedia subset is a MediaWiki schema: `page`, `revision`, `pagecontent`, `pagelinks`, `mwuser`, and `user_groups`. Columns whose names contain `password`, `token`, or `email` are not used as keyword text.

Every table carries `__search_id`. The topic files identify relevant tuples by that integer. The loader stores `semstruct_search_id(search_id, relation)` so a judged tuple can be tied to its relation.

## Topics

Each numbered file in a topic archive has the form:

```text
# keyword query
([search_id, ...], [(source_id, target_id), ...])
```

Lines that start with `#` are comments. A commented-out tuple line is not a relevance judgment. The query is the first comment that is not a dataset banner such as `imdb2009` or `[reference standard]`.

Each database has 50 of these queries. They are dealt into 5 folds of 10. A fold tests its 10 topics, holds out 8 for the checkpoint and the threshold, and trains on the other 32. A synthetic query whose normalized text equals a topic is dropped.

The query texts are the synthetic queries plus the 50 Coffman topics. A label is positive only for the index hit the topic or the generator assigns to that keyword. Training keeps every schema hit and at most 10 other value hits per keyword, and the reported precision and recall score every index hit. The pre-trained numbers are the frozen [CLS] cosine.

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
