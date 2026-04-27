"""Ephemeral BM25 retrieval over a dataset's corpus.

For the leaderboard we score each query against *the task's corpus*
(e.g. the 5 k documents in PubMedCitationRetrieval), not the global
24.8 M index — BM25 IDF is corpus-dependent. This module builds a
DuckDB FTS index in-memory from the dataset's ``corpus.jsonl`` and
returns a MTEB-style run dict ``{qid: {docid: score}}``.
"""

from __future__ import annotations

import logging
import re

import duckdb

logger = logging.getLogger(__name__)


def _sanitize_bm25_query(text: str, max_tokens: int = 200) -> str:
    """Sanitize and truncate text for use as a DuckDB FTS query.

    Duplicated from :mod:`pubmedteb.infra.bm25_index` to keep this module
    free of the ``builders`` package import chain (which would otherwise
    create a cycle through ``negative_sampling``).
    """
    text = text.replace("'", "''")
    text = re.sub(r'[+\-~*"(){}[\]^:!&|\\/@#$%]', " ", text)
    words = text.split()[:max_tokens]
    return " ".join(words)


def bm25_retrieve(
    queries: dict[str, str],
    corpus: dict[str, str],
    top_k: int = 1000,
) -> dict[str, dict[str, float]]:
    """Score *queries* against *corpus* with DuckDB FTS BM25.

    Args:
        queries: ``{qid: query_text}``.
        corpus: ``{docid: doc_text}``.
        top_k: Number of documents to return per query.

    Returns:
        ``{qid: {docid: score}}`` — the top-*k* by BM25 score per query.
        Queries whose sanitized text is empty map to an empty dict.
    """
    con = duckdb.connect()
    try:
        con.execute("INSTALL fts")
        con.execute("LOAD fts")
        con.execute("SET scalar_subquery_error_on_multiple_rows = false")
        con.execute("CREATE TABLE docs(id VARCHAR, text VARCHAR)")
        con.executemany(
            "INSERT INTO docs VALUES (?, ?)",
            [(did, text) for did, text in corpus.items()],
        )
        con.execute("PRAGMA create_fts_index('docs', 'id', 'text')")
        logger.info("BM25 ephemeral index: %d docs", len(corpus))

        results: dict[str, dict[str, float]] = {}
        for qid, qtext in queries.items():
            sanitized = _sanitize_bm25_query(qtext)
            if not sanitized.strip():
                results[qid] = {}
                continue
            rows = con.execute(f"""
                SELECT id, score
                FROM (
                    SELECT id, fts_main_docs.match_bm25(id, '{sanitized}') AS score
                    FROM docs
                )
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT {top_k}
            """).fetchall()
            results[qid] = {did: float(score) for did, score in rows}
        return results
    finally:
        con.close()
