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

Every table carries `__search_id`. The topic files identify relevant tuples by that integer. The loader stores `semstruct_search_id(search_id, relation)` so evaluation can recover the relation of a judged tuple without guessing.

## Topics

Each numbered file in a topic archive has the form:

```text
# keyword query
([search_id, ...], [(source_id, target_id), ...])
```

Lines that start with `#` are comments. A commented-out tuple line is not a relevance judgment. The query is the first comment that is not a dataset banner such as `imdb2009` or `[reference standard]`.

Each database has 50 of these queries. One third, 17 queries, is reserved for the test split. The other 33 are used as training data: 29 in train and 4 in validation. A synthetic query whose normalized text equals a topic is dropped, so the human wording is not learned from one split and scored in another.

The judged tuple supplies the labels. Its relation is positive. A cell in that tuple is a positive attribute and a positive value when the cell contains a keyword from the topic. The pair counts for relations, attributes, and values, with positives and negatives separate, are in [training.md](training.md).

On the 15 November training split the synthetic queries were 943 on Mondial (24,080 pairs), 212 on IMDb (5,088 pairs), and 142 on Wikipedia (3,060 pairs). The relation, attribute, and value breakdown is in the same section of [training.md](training.md).
