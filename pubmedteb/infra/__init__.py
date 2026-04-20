"""Shared infrastructure for PubMedTEB dataset builders.

Provides reusable indices and lookups that multiple builders depend on:

- **citation_graph**: Forward and reverse citation lookups as DuckDB views.
- **bm25_index**: BM25 full-text search via DuckDB FTS extension.
"""
