# Local data

Raw LID-DS files are intentionally not included in the public bundle. Place a
user-authorized local dataset under the path configured by
`paths.raw_data_root`, and use a small local scenario for the first ingest
smoke run. The ingestion code treats raw files as read-only and writes all
derived tables under `artifacts/`.
