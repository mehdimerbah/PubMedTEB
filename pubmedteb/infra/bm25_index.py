"""BM25 full-text search index over PubMed abstracts.

Uses DuckDB's built-in FTS extension to create a BM25-scored full-text
index. The index is persisted as a DuckDB database file and ATTACHed
read-only by builders that need BM25 retrieval.

Storage::

    {cache_dir}/
        bm25_abstracts.duckdb   # persistent DuckDB database with FTS index

Usage in builders::

    from pubmedteb.infra.bm25_index import attach_bm25_index, query_bm25

    class MyBuilder(DatasetBuilder):
        def construct(self):
            attach_bm25_index(self.con)
            results = query_bm25(
                self.con, query_text="cancer immunotherapy",
                top_k=50, category_filter="Diseases",
            )
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import duckdb

from pubmedteb.builders.base import PARQUET_PATH

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "preprocessing" / "data" / "bm25_index"


def _sanitize_bm25_query(text: str, max_tokens: int = 200) -> str:
    """Sanitize and truncate text for use as a DuckDB FTS query.

    - Escapes single quotes
    - Removes FTS special operators
    - Truncates to *max_tokens* words
    """
    text = text.replace("'", "''")
    text = re.sub(r'[+\-~*"(){}[\]^:!&|\\/@#$%]', " ", text)
    words = text.split()[:max_tokens]
    return " ".join(words)


def build_bm25_index(
    parquet_path: Path = PARQUET_PATH,
    cache_dir: Path = CACHE_DIR,
    sample_size: int | None = None,
    seed: int = 42,
) -> Path:
    """Build a DuckDB FTS index over abstracts.

    Creates a persistent DuckDB database with a table ``abstracts``
    containing ``pmid``, ``abstract_text``, and ``semantic_category``,
    plus a BM25 full-text index on ``abstract_text``.

    Args:
        parquet_path: Source Parquet file.
        cache_dir: Directory for the DuckDB database file.
        sample_size: If set, only index this many abstracts (for testing).
            ``None`` indexes the full corpus.
        seed: Seed for deterministic sampling when *sample_size* is set.

    Returns:
        Path to the DuckDB database file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "bm25_abstracts.duckdb"

    # Remove existing database to rebuild cleanly
    if db_path.exists():
        db_path.unlink()

    t0 = time.time()
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL fts")
    con.execute("LOAD fts")

    limit_clause = ""
    if sample_size:
        limit_clause = f"ORDER BY hash(pmid || '{seed}') LIMIT {sample_size}"

    logger.info(
        "Building BM25 index%s from %s ...",
        f" (sample_size={sample_size})" if sample_size else " (full corpus)",
        parquet_path,
    )

    con.execute(f"""
        CREATE TABLE abstracts AS
        SELECT pmid, abstract_text, semantic_category
        FROM '{parquet_path}'
        WHERE length(abstract_text) >= 150
        {limit_clause}
    """)

    n_rows = con.execute("SELECT count(*) FROM abstracts").fetchone()[0]
    logger.info("Loaded %d abstracts into BM25 table, building FTS index ...", n_rows)

    con.execute("PRAGMA create_fts_index('abstracts', 'pmid', 'abstract_text')")

    elapsed = time.time() - t0
    con.close()

    logger.info("BM25 index built in %.1fs: %d documents -> %s", elapsed, n_rows, db_path)
    return db_path


def open_bm25_index(
    cache_dir: Path = CACHE_DIR,
) -> duckdb.DuckDBPyConnection:
    """Open a read-only connection to the BM25 DuckDB database.

    DuckDB FTS functions only work within the database that owns the
    index, so BM25 queries must use this dedicated connection rather
    than ATTACHing to a builder's in-memory connection.

    Args:
        cache_dir: Directory containing the BM25 database.

    Returns:
        A read-only DuckDB connection with FTS loaded.

    Raises:
        FileNotFoundError: If the BM25 database has not been built yet.
    """
    db_path = cache_dir / "bm25_abstracts.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {db_path}. "
            "Build it first: uv run python -m pubmedteb.run build-infra --component bm25"
        )
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("LOAD fts")
    # FTS match_bm25 is a scalar macro that internally uses a subquery.
    # When combined with NOT IN clauses, DuckDB may raise a false-positive
    # "scalar subquery returned multiple rows" error. This setting suppresses it.
    con.execute("SET scalar_subquery_error_on_multiple_rows = false")
    logger.info("BM25 index opened (read-only) from %s", db_path)
    return con


def query_bm25(
    con: duckdb.DuckDBPyConnection,
    query_text: str,
    top_k: int = 50,
    category_filter: str | None = None,
    exclude_pmids: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Run a BM25 query against the index.

    Args:
        con: DuckDB connection from :func:`open_bm25_index`.
        query_text: The query text (e.g., an abstract).
        top_k: Number of results to return.
        category_filter: If set, restrict to this ``semantic_category``.
        exclude_pmids: PMIDs to exclude from results.

    Returns:
        List of ``(pmid, bm25_score)`` tuples, ordered by score descending.
    """
    sanitized = _sanitize_bm25_query(query_text)
    if not sanitized.strip():
        return []

    where_clauses = ["score IS NOT NULL"]
    if category_filter:
        safe_cat = category_filter.replace("'", "''")
        where_clauses.append(f"a.semantic_category = '{safe_cat}'")
    if exclude_pmids:
        pmid_list = ", ".join(f"'{p}'" for p in exclude_pmids)
        where_clauses.append(f"a.pmid NOT IN ({pmid_list})")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT a.pmid, fts_main_abstracts.match_bm25(a.pmid, '{sanitized}') AS score
        FROM abstracts a
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT {top_k}
    """

    return con.execute(sql).fetchall()
