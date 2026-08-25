# AlgoFinder everything data warehouse

`private/db/algofinder.duckdb` is a derived query layer over recorded
experiment evidence. It does not replace source ledgers, summaries, manifests,
Parquet files, or campaign roots; those remain authoritative and the database
can be rebuilt from them at any time. `private/db/analytics.duckdb` is left
untouched as the legacy benchmark database.

The warehouse uses five DuckDB schemas:

- `catalog` inventories roots, batches, artifacts, all observed locations,
  canonical JSON records, digests, and integrity conflicts.
- `experiment` projects summaries, plans, schedules, model identities,
  profiles, arms, matched pairs, and checkpoints.
- `campaign` projects immutable campaign ledger entities.
- `learning` projects backfills, data sets, Parquet rows, policies, memories,
  runs, distributions, and outer-evaluation records.
- `analytics` contains views. Its default evidence views exclude incomplete,
  diagnostic, and scratch comparisons; the raw schemas retain all of them.

Small stable text evidence is deduplicated by SHA-256 at
`private/corpus/algofinder/sha256/`. Large or binary files remain in their
original roots and are verified through catalog locations. Files that look like
secrets or prohibited challenge/sealed payloads are cataloged as rejected and
are never copied into the corpus.

From the repository root:

```bash
# no-write inventory
python -m algofinder.data discover \
  --experiments-root /path/to/recorded-experiments

# idempotent development ingest (includes incomplete/diagnostic evidence)
python -m algofinder.data ingest --mode dev \
  --experiments-root /path/to/recorded-experiments

# production finalization imports only completed comparison roots
python -m algofinder.data ingest --mode prod \
  --experiments-root /path/to/recorded-experiments

python -m algofinder.data verify
python -m algofinder.data overview

# rebuild only the typed projections after a projection/schema upgrade
python -m algofinder.data reproject
```

`run_learned_policy_comparison` accepts `warehouse_mode="dev"` or
`"prod"`. The module CLI defaults to `dev`: each persisted summary checkpoint
triggers a best-effort incremental ingest. Production mode performs no
synchronous DuckDB work during the search loop and finalizes once the
comparison is complete. Ingestion failures only defer the derived index; they
do not invalidate an experiment because its source artifacts remain intact.
