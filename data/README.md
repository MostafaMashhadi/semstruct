# Data directories

`raw/` holds the Coffman and Weaver dump files. `processed/` holds one SQLite database and the generated JSONL pairs per dataset. Each pair is a relation, an attribute, or a cell value, and it is labeled positive or negative. `prepare_stats.json` stores those counts. One third of the Coffman topics are test queries, and the other two thirds are training and validation queries. `outputs/` at the repository root holds checkpoints and metrics.

Those files are gitignored. The benchmark license does not allow redistributing the dumps. See `docs/datasets.md` for filenames, the Dataverse DOI, and the citation.
