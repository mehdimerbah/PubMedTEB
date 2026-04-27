"""Temporal retrieval dataset builder.

Query = abstract of a recent (year ≥ 2018) citing paper.
Relevant docs = abstracts of its in-corpus cited papers that are at least
*min_year_gap* years older than the citing paper.

Tests robustness to biomedical terminology drift across decades — can a
model connect modern vocabulary to older seminal work?

Hard-negative mix: 40 % same-descriptor, 15 % shared depth-3,
20 % BM25, 10 % same-journal, 15 % random.

V1 caveat: negatives are not year-restricted; the sampler can return
recent papers. If the BM25 gate indicates the model can shortcut via
publication year, iterate to V2 with ``year_max`` filters on the
negative samplers.
"""

from __future__ import annotations

import logging

from pubmedteb.builders.base import WHERE_HAS_DESCRIPTOR
from pubmedteb.builders.citation_retrieval import CitationRetrievalBuilder

logger = logging.getLogger(__name__)


class TemporalRetrievalBuilder(CitationRetrievalBuilder):
    """Build a temporal citation-retrieval dataset."""

    MIX: dict[str, float] = {
        "descriptor": 0.40,
        "depth3": 0.15,
        "bm25": 0.20,
        "journal": 0.10,
        "random": 0.15,
    }

    SIZES = {
        "mini": {
            "n_queries": 500,
            "n_corpus": 5_000,
            "min_pos": 3,
            "max_pos": 10,
            "query_year_min": 2018,
            "min_year_gap": 10,
        },
        "full": {
            "n_queries": 5_000,
            "n_corpus": 50_000,
            "min_pos": 5,
            "max_pos": 20,
            "query_year_min": 2018,
            "min_year_gap": 10,
        },
    }

    def _sample_query_articles(
        self,
        n: int,
        min_pos: int,
        max_pos: int,
    ) -> tuple[list[tuple], dict[str, list[str]]]:
        cfg = self.SIZES[self.size]
        year_min = cfg["query_year_min"]
        gap = cfg["min_year_gap"]
        oversample = int(n * 3)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _tp_candidates AS
            SELECT pmid, title, abstract_text, journal, mesh_descriptors, cited_pmids, year
            FROM '{self.parquet_path}'
            WHERE year >= {year_min}
              AND len(cited_pmids) >= {min_pos}
              AND length(abstract_text) >= 150
              AND {WHERE_HAS_DESCRIPTOR}
              AND array_length(string_split(title, ' ')) >= 5
            ORDER BY hash(pmid || '{self.seed}_tp_cand')
            LIMIT {oversample}
        """)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _tp_qualified AS
            SELECT ce.citing, ce.cited
            FROM (
                SELECT pmid AS citing, year AS citing_year, UNNEST(cited_pmids) AS cited
                FROM _tp_candidates
            ) ce
            JOIN '{self.parquet_path}' p ON p.pmid = ce.cited
            WHERE length(p.abstract_text) >= 150
              AND ce.citing_year - p.year >= {gap}
        """)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _tp_valid AS
            SELECT citing, count(*) AS n_cited
            FROM _tp_qualified
            GROUP BY citing
            HAVING count(*) >= {min_pos}
        """)

        rows = self.con.execute(f"""
            SELECT c.pmid, c.title, c.abstract_text, c.journal, c.mesh_descriptors
            FROM _tp_candidates c
            JOIN _tp_valid v ON v.citing = c.pmid
            ORDER BY hash(c.pmid || '{self.seed}_tp_pick')
            LIMIT {n}
        """).fetchall()

        query_pmids = {r[0] for r in rows}
        if not query_pmids:
            return [], {}

        self._register_pmid_temp("_tp_chosen", query_pmids)
        edge_rows = self.con.execute("""
            SELECT q.citing, q.cited
            FROM _tp_qualified q
            JOIN _tp_chosen ch ON ch.pmid = q.citing
            ORDER BY hash(q.citing || '_' || q.cited)
        """).fetchall()

        per_query_cited: dict[str, list[str]] = {p: [] for p in query_pmids}
        for citing, cited in edge_rows:
            lst = per_query_cited[citing]
            if len(lst) < max_pos:
                lst.append(cited)

        for tbl in ("_tp_candidates", "_tp_qualified", "_tp_valid", "_tp_chosen"):
            self.con.execute(f"DROP TABLE IF EXISTS {tbl}")

        return rows, per_query_cited
