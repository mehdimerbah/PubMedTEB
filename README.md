# PubMedTEB

A biomedical text embedding benchmark built as an [MTEB](https://github.com/embeddings-benchmark/mteb) extension, using datasets curated from the full PubMed abstract corpus (38.2M articles).

## Project Status

This project is under active development as part of my Master's thesis at the University of Tuebingen.

**Current stage**: Preprocessing: the PubMed XML-to-Parquet ETL pipeline and data filtering are complete. Benchmark task construction is next.

## Repository Structure

```
PubMedTEB/
├── preprocessing/     # XML-to-Parquet ETL + filtering notebook
├── main.py            # Pipeline entry point
├── docs/              # Design documents and plans
├── mesh_explorer/     # MeSH hierarchy visualization
└── pyproject.toml     # Project config (uv)
```

See [`preprocessing/README.md`](preprocessing/README.md) for details on the data pipeline.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data

The raw PubMed Baseline 2025 XML files and processed Parquet files.