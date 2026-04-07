# Preprocessing

Two-stage preprocessing pipeline that converts raw PubMed Baseline 2025 XML into a clean, filtered Parquet file ready for benchmark dataset construction.

## Stage 1: XML-to-Parquet ETL (`pipeline`)

Parses 1,219 PubMed XML files (~38.2M articles) in parallel and writes a single Parquet file with 14 columns:

| Column | Type | Description |
|--------|------|-------------|
| `pmid` | string | PubMed article identifier |
| `title` | string | Article title |
| `abstract_text` | string | Full abstract (concatenated sections) |
| `abstract_sections` | list\<struct\> | Structured sections: `{label, text}` |
| `journal` | string | Journal full title |
| `year` | int16 | Publication year |
| `month` | string | Publication month |
| `mesh_descriptors` | list\<struct\> | MeSH tags: `{text, uid, is_major}` |
| `cited_pmids` | list\<string\> | PMIDs cited by the article |
| `publication_types` | list\<string\> | Article types (e.g. "Journal Article", "Review") |
| `language` | string | Language code (e.g. "eng") |
| `doi` | string | Digital Object Identifier |
| `country` | string | Journal country of origin |
| `semantic_category` | string | Top-level MeSH branch (majority vote) |

### MeSH categorization

Each article's MeSH descriptors are mapped to the 16 NLM tree branches (Anatomy, Diseases, Chemicals and Drugs, etc.) using a precomputed UID-to-category mapping from `desc2025.xml`. The branch with the most votes across an article's descriptors becomes its `semantic_category`. Ties are broken alphabetically.

### Running the ETL

```bash
uv run python main.py run \
    --xml-dir /path/to/pubmed/xml \
    --output /path/to/output.parquet \
    --mesh-desc /path/to/desc2025.xml

uv run python main.py validate --parquet /path/to/output.parquet
```

## Stage 2: Filtering (`pubmed_filtering.ipynb`)

The notebook analyzes data quality, quantifies filter impact, and writes a filtered Parquet. Six filters are applied:

| Filter | Rationale | Removed |
|--------|-----------|---------|
| Has abstract | Embeddings need text | 30.0% |
| Has title | Required for pair tasks | 0.1% |
| English only | Model/evaluation consistency | 12.8% |
| Year >= 1970 | Removes missing/pre-modern dates | 0.04% |
| Abstract >= 150 chars | Removes stubs | 0.03% |
| Exclude non-research types | Retractions, errata, comments, etc. | 7.1% |

**Result**: ~24.8M articles retained (65% of 38.2M), with 81% MeSH coverage and 16 semantic categories represented.

### Filters intentionally not applied

- **No MeSH descriptors** (16%) -- only needed for MeSH-based tasks
- **No citations** (73%) -- only needed for citation prediction
- **No DOI** (23%) -- not needed for any benchmark task

### Running the notebook

```bash
uv run jupyter lab preprocessing/pubmed_filtering.ipynb
```

## Module structure

```
preprocessing/
├── cli.py                 # CLI entry point (run / validate / mesh subcommands)
├── orchestrator.py        # Parallel XML processing + deduplication
├── parser.py              # PubMed XML extraction (streaming, memory-efficient)
├── schema.py              # Arrow schema, ArticleRecord dataclass, batch conversion
├── mesh_categories.py     # MeSH UID -> top-level category mapping
├── mesh_tree.py           # Full MeSH hierarchy builder (for visualization)
├── transform.py           # Post-parse transformations (semantic category assignment)
├── validate.py            # Post-build validation suite
├── writer.py              # Incremental Parquet writer (zstd, 100k row groups)
├── pubmed_filtering.ipynb # Filter analysis + filtered Parquet output
└── data/
    └── mesh_uid_categories.json  # Precomputed MeSH mapping cache
```
