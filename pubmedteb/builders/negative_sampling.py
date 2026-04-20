"""Shared hard-negative samplers for PubMedTEB dataset builders.

All samplers are *pooled*: they take the union of per-query context
(descriptor UIDs, depth-3 labels, journals) and draw from the full
filtered Parquet. Each returns ``list[(pmid, abstract_text)]``.

The mix ratios come from the MeSH investigation, Decision 4
(see ``reports/mesh_investigation/tables/T3_hardneg_mix_per_task.csv``):

- ``sample_by_descriptor`` — shares ≥1 MeSH descriptor UID (strongest signal)
- ``sample_by_depth3`` — shares ≥1 depth-3 sub-subcategory (moderate)
- ``sample_by_bm25`` — BM25 top-k on query text, pooled then deduped
- ``sample_by_journal`` — same journal (moderate, sparse)
- ``sample_random`` — uniform random filler
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from pubmedteb.analysis.mesh import MeshMappings, register_mesh_tables
from pubmedteb.infra.bm25_index import query_bm25

logger = logging.getLogger(__name__)


class NegativeSampler:
    """Pooled negative-distractor sampling bound to a DuckDB connection.

    Construction registers ``uid_depth1 / uid_depth2 / uid_depth3`` views
    in the connection. Each sampler is independent and runs a single
    Parquet scan over the 24.8 M filtered corpus (~15 s on the shared
    GPFS volume).

    Args:
        con: Builder's main DuckDB connection (operates on the Parquet).
        parquet_path: Path to the filtered Parquet.
        seed: Deterministic seed for hash-based ORDER BY.
        mappings: MeSH depth-1/2/3 mapping dataclass.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        parquet_path: Path,
        seed: int,
        mappings: MeshMappings,
    ) -> None:
        self.con = con
        self.parquet = str(parquet_path)
        self.seed = seed
        self.mappings = mappings
        register_mesh_tables(con, mappings)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _register_temp(self, name: str, values: set[str], col: str = "v") -> None:
        """(Re-)create a single-column temp table from a set of strings."""
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {name}({col} VARCHAR)")
        if values:
            self.con.executemany(
                f"INSERT INTO {name} VALUES (?)", [(v,) for v in values]
            )

    # ── Samplers ────────────────────────────────────────────────────────

    def sample_by_descriptor(
        self,
        descriptor_uids: set[str],
        exclude_pmids: set[str],
        n: int,
        tag: str = "desc",
    ) -> list[tuple[str, str]]:
        """Sample n articles sharing ≥ 1 descriptor UID with the query set.

        Uses ``list_has_any`` on the pre-transformed UID list rather than
        ``UNNEST`` — avoids materializing a 300 M-row cartesian product.
        """
        if not descriptor_uids or n <= 0:
            return []
        self._register_temp("_ns_uids", descriptor_uids, col="uid")
        self._register_temp("_ns_excl", exclude_pmids, col="pmid")
        rows = self.con.execute(f"""
            SELECT p.pmid, p.abstract_text
            FROM '{self.parquet}' p
            WHERE length(p.abstract_text) >= 150
              AND p.pmid NOT IN (SELECT pmid FROM _ns_excl)
              AND list_has_any(
                    list_transform(p.mesh_descriptors, x -> x.uid),
                    (SELECT list(uid) FROM _ns_uids)
                  )
            ORDER BY hash(p.pmid || '{self.seed}_{tag}')
            LIMIT {n}
        """).fetchall()
        return [(r[0], r[1]) for r in rows]

    def sample_by_depth3(
        self,
        depth3_labels: set[str],
        exclude_pmids: set[str],
        n: int,
        tag: str = "d3",
    ) -> list[tuple[str, str]]:
        """Sample n articles sharing ≥ 1 depth-3 label with the query set.

        Reduces to descriptor sampling over the set of UIDs that map to
        any of the query depth-3 labels — evaluated in Python on the
        ~30 k-entry mapping, avoiding a SQL UNNEST + JOIN.
        """
        if not depth3_labels or n <= 0:
            return []
        eligible_uids = {
            uid for uid, labels in self.mappings.uid_to_depth3.items()
            if any(label in depth3_labels for label in labels)
        }
        return self.sample_by_descriptor(eligible_uids, exclude_pmids, n, tag=tag)

    def sample_by_journal(
        self,
        journals: set[str],
        exclude_pmids: set[str],
        n: int,
        tag: str = "journal",
    ) -> list[tuple[str, str]]:
        """Sample n articles from the pooled journal set."""
        if not journals or n <= 0:
            return []
        self._register_temp("_ns_journal", journals, col="journal")
        self._register_temp("_ns_excl", exclude_pmids, col="pmid")
        rows = self.con.execute(f"""
            SELECT p.pmid, p.abstract_text
            FROM '{self.parquet}' p
            WHERE p.journal IN (SELECT journal FROM _ns_journal)
              AND p.pmid NOT IN (SELECT pmid FROM _ns_excl)
              AND length(p.abstract_text) >= 150
            ORDER BY hash(p.pmid || '{self.seed}_{tag}')
            LIMIT {n}
        """).fetchall()
        return [(r[0], r[1]) for r in rows]

    def sample_random(
        self,
        exclude_pmids: set[str],
        n: int,
        tag: str = "random",
    ) -> list[tuple[str, str]]:
        """Sample n random articles uniformly from the filtered corpus."""
        if n <= 0:
            return []
        self._register_temp("_ns_excl", exclude_pmids, col="pmid")
        rows = self.con.execute(f"""
            SELECT p.pmid, p.abstract_text
            FROM '{self.parquet}' p
            WHERE p.pmid NOT IN (SELECT pmid FROM _ns_excl)
              AND length(p.abstract_text) >= 150
            ORDER BY hash(p.pmid || '{self.seed}_{tag}')
            LIMIT {n}
        """).fetchall()
        return [(r[0], r[1]) for r in rows]

    def sample_by_bm25(
        self,
        bm25_con: duckdb.DuckDBPyConnection,
        queries: list[tuple[str, str]],
        exclude_pmids: set[str],
        n: int,
        top_k_per_query: int | None = None,
    ) -> list[tuple[str, str]]:
        """Pooled BM25: union of top-k results across queries, deduped and capped.

        Args:
            bm25_con: Connection from :func:`pubmedteb.infra.bm25_index.open_bm25_index`.
            queries: List of ``(qid, query_text)`` — query text is typically the
                article title or abstract used by the builder.
            exclude_pmids: PMIDs to exclude (query pmids + already-selected distractors).
            n: Target number of distractors.
            top_k_per_query: BM25 top-k per query. Defaults to ``max(20, ceil(3n/|Q|))``.

        Returns:
            ``list[(pmid, abstract_text)]`` of length ≤ n. Abstracts are fetched
            from the main Parquet via ``self.con``.
        """
        if not queries or n <= 0:
            return []
        if top_k_per_query is None:
            top_k_per_query = max(20, (3 * n + len(queries) - 1) // len(queries))

        seen: set[str] = set(exclude_pmids)
        hits: list[tuple[str, int]] = []  # (pmid, rank for deterministic tie-break)
        for _, qtext in queries:
            res = query_bm25(
                bm25_con, qtext, top_k=top_k_per_query, exclude_pmids=seen,
            )
            for rank, (pmid, _score) in enumerate(res):
                if pmid in seen:
                    continue
                seen.add(pmid)
                hits.append((pmid, rank))
                if len(hits) >= 5 * n:  # cap pool to bound memory
                    break
            if len(hits) >= 5 * n:
                break

        if not hits:
            return []

        # Deterministic sample of n from the pool using the builder seed
        pool_pmids = [p for p, _ in hits][: max(n, len(hits))]
        self._register_temp("_ns_bm25_pool", set(pool_pmids), col="pmid")
        rows = self.con.execute(f"""
            SELECT p.pmid, p.abstract_text
            FROM '{self.parquet}' p
            WHERE p.pmid IN (SELECT pmid FROM _ns_bm25_pool)
              AND length(p.abstract_text) >= 150
            ORDER BY hash(p.pmid || '{self.seed}_bm25')
            LIMIT {n}
        """).fetchall()
        return [(r[0], r[1]) for r in rows]


# ── Mix helpers ─────────────────────────────────────────────────────────


def split_mix(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split *total* across buckets by *weights* (sum need not be 1.0).

    Rounding residue goes to the bucket whose fractional part is largest,
    so the counts always sum exactly to *total*.
    """
    s = sum(weights.values())
    if s == 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    floor = {k: int(v) for k, v in raw.items()}
    residue = total - sum(floor.values())
    # Distribute residue to the buckets with the largest fractional parts
    by_frac = sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
    for k, _ in by_frac[:residue]:
        floor[k] += 1
    return floor


def descriptor_uids_of(rows: list[tuple], desc_col_index: int) -> set[str]:
    """Flatten ``mesh_descriptors`` structs from a row set into a UID set."""
    out: set[str] = set()
    for row in rows:
        descs = row[desc_col_index] or []
        for d in descs:
            uid = d.get("uid") if isinstance(d, dict) else d["uid"]
            if uid:
                out.add(uid)
    return out


def depth3_labels_of(
    rows: list[tuple],
    desc_col_index: int,
    uid_to_depth3: dict[str, list[str]],
) -> set[str]:
    """Flatten ``mesh_descriptors`` structs into the union of depth-3 labels."""
    out: set[str] = set()
    for row in rows:
        descs = row[desc_col_index] or []
        for d in descs:
            uid = d.get("uid") if isinstance(d, dict) else d["uid"]
            if uid:
                out.update(uid_to_depth3.get(uid, []))
    return out
