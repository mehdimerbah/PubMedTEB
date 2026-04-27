# PubMedTEB

A biomedical text embedding benchmark built as an [MTEB](https://github.com/embeddings-benchmark/mteb) extension, using datasets curated from the full PubMed abstract corpus (38.2M articles).

## Project Status

This project is under active development as part of my Master's thesis at the University of Tuebingen.

**Current stage**: All 8 task builders are implemented. Datasets and dense-model smoke evaluations exist for every task. 

## Repository Structure

```
PubMedTEB/
├── pubmedteb/                       # Main package: builders, MTEB tasks, models, infra, analysis
├── preprocessing/                    # XML-to-Parquet ETL + filtering notebook
├── datasets/                         # Built benchmark datasets
├── results/                          # MTEB evaluation outputs
├── reports/mesh_investigation/       # MeSH investigation report, figures, tables
├── notebooks/                        # Analysis notebooks
├── docs/                             # Progress tracker, project report, design specs
├── scripts/galvani/                  # SLURM deployment for cluster eval
├── tools/mesh_explorer/              # Standalone MeSH hierarchy viewer
├── archive/                          # Historical corpus characterization outputs
├── literature/                       # Reference PDFs
├── main.py                           # ETL pipeline entry point
└── pyproject.toml                    # Project config (uv)
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data

The raw PubMed Baseline 2025 XML files and processed Parquet files.