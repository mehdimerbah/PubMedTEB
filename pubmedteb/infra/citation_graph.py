"""Citation graph index: forward and reverse citation lookups.

Build once from the source Parquet, then load into any DuckDB connection
for efficient SQL-based citation queries.

Storage layout::

    {cache_dir}/
        reverse_citations.parquet   # cited_pmid -> list[citing_pmid]
        citation_stats.json         # metadata: counts, build date

Usage in builders::

    from pubmedteb.infra.citation_graph import load_citation_graph

    class MyBuilder(DatasetBuilder):
        def construct(self):
            load_citation_graph(self.con, self.parquet_path)
            # Now available:
            #   forward_citations(pmid, cited_pmids)
            #   reverse_citations(cited_pmid, citing_pmids)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import duckdb

from pubmedteb.builders.base import PARQUET_PATH

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "preprocessing" / "data" / "citation_graph"


def build_citation_graph(
    parquet_path: Path = PARQUET_PATH,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Build the reverse citation index from the source Parquet.

    The forward index (pmid -> cited_pmids) is already in the Parquet,
    so we only build the reverse index (cited_pmid -> citing_pmids).

    Args:
        parquet_path: Path to the filtered PubMed Parquet file.
        cache_dir: Directory to write the reverse citations Parquet and stats.

    Returns:
        Stats dict with edge counts, timings, etc.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    reverse_path = cache_dir / "reverse_citations.parquet"
    stats_path = cache_dir / "citation_stats.json"

    con = duckdb.connect()
    t0 = time.time()

    # Count forward citation stats
    logger.info("Counting citation statistics from %s ...", parquet_path)
    row = con.execute(f"""
        SELECT
            count(*) AS total_articles,
            count(*) FILTER (WHERE len(cited_pmids) > 0) AS articles_with_citations,
            sum(len(cited_pmids)) AS total_forward_edges
        FROM '{parquet_path}'
    """).fetchone()
    total_articles, articles_with_citations, total_forward_edges = row

    # Build reverse index: cited_pmid -> list of citing pmids
    logger.info("Building reverse citation index ...")
    con.execute(f"""
        COPY (
            SELECT cited_pmid, LIST(pmid ORDER BY pmid) AS citing_pmids
            FROM (
                SELECT pmid, UNNEST(cited_pmids) AS cited_pmid
                FROM '{parquet_path}'
                WHERE len(cited_pmids) > 0
            )
            GROUP BY cited_pmid
        ) TO '{reverse_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Count reverse stats
    reverse_row = con.execute(f"""
        SELECT count(*) AS distinct_cited_pmids
        FROM '{reverse_path}'
    """).fetchone()
    distinct_cited_pmids = reverse_row[0]

    elapsed = time.time() - t0
    con.close()

    stats = {
        "build_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_parquet": str(parquet_path),
        "total_articles": total_articles,
        "articles_with_citations": articles_with_citations,
        "total_forward_edges": total_forward_edges,
        "distinct_cited_pmids": distinct_cited_pmids,
        "build_time_seconds": round(elapsed, 1),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Citation graph built in %.1fs: %d citing articles, %d forward edges, "
        "%d distinct cited PMIDs -> %s",
        elapsed,
        articles_with_citations,
        total_forward_edges,
        distinct_cited_pmids,
        reverse_path,
    )
    return stats


def ensure_citation_graph(
    cache_dir: Path = CACHE_DIR,
    parquet_path: Path = PARQUET_PATH,
) -> Path:
    """Ensure the reverse citations Parquet exists, building if needed.

    Returns:
        Path to reverse_citations.parquet.
    """
    reverse_path = cache_dir / "reverse_citations.parquet"
    if not reverse_path.exists():
        logger.info("Reverse citation index not found — building ...")
        build_citation_graph(parquet_path, cache_dir)
    return reverse_path


def load_citation_graph(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path = PARQUET_PATH,
    cache_dir: Path = CACHE_DIR,
) -> None:
    """Register citation graph views in a DuckDB connection.

    After this call, the connection has two views:

    - ``forward_citations(pmid, cited_pmids)`` — from the source Parquet
    - ``reverse_citations(cited_pmid, citing_pmids)`` — from the cached index

    If the cache does not exist, it is built automatically.

    Args:
        con: An open DuckDB connection (typically the builder's ``self.con``).
        parquet_path: Path to the filtered PubMed Parquet.
        cache_dir: Directory containing (or to contain) the citation cache.
    """
    reverse_path = ensure_citation_graph(cache_dir, parquet_path)

    con.execute(f"""
        CREATE OR REPLACE VIEW forward_citations AS
        SELECT pmid, cited_pmids
        FROM '{parquet_path}'
        WHERE len(cited_pmids) > 0
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW reverse_citations AS
        SELECT * FROM '{reverse_path}'
    """)

    logger.info("Citation graph views registered: forward_citations, reverse_citations")
