"""Review-to-primary-research retrieval dataset builder.

Query = abstract of a Review or Systematic Review.
Relevant docs = abstracts of the primary-research papers it cites
(in-corpus, non-review).

Hard-negative mix (T3 recipe, see ``reports/mesh_investigation/tables/
T3_hardneg_mix_per_task.csv``): 40 % same-descriptor, 15 % shared depth-3,
15 % BM25, 5 % same-journal, 25 % random.
"""

from __future__ import annotations

import logging

from pubmedteb.builders.base import WHERE_HAS_DESCRIPTOR
from pubmedteb.builders.citation_retrieval import CitationRetrievalBuilder

logger = logging.getLogger(__name__)


class ReviewRetrievalBuilder(CitationRetrievalBuilder):
    """Build a review-to-primary-research retrieval dataset.

    Queries are filtered to review or systematic-review articles with
    ≥ *min_pos* in-corpus non-review citations. Relevant documents are
    the cited primary-research abstracts (capped at *max_pos*). The
    corpus is padded with hard-negative distractors per the T3 mix.
    """

    MIX: dict[str, float] = {
        "descriptor": 0.40,
        "depth3": 0.15,
        "bm25": 0.15,
        "journal": 0.05,
        "random": 0.25,
    }

    SIZES = {
        "mini": {
            "n_queries": 500,
            "n_corpus": 5_000,
            "min_pos": 3,
            "max_pos": 10,
        },
        "full": {
            "n_queries": 5_000,
            "n_corpus": 100_000,
            "min_pos": 10,
            "max_pos": 25,
        },
    }

    def _sample_query_articles(
        self,
        n: int,
        min_pos: int,
        max_pos: int,
    ) -> tuple[list[tuple], dict[str, list[str]]]:
        """Sample review articles with ≥ *min_pos* in-corpus non-review citations.

        Relies on the eligibility floor being ≥15 at the full preset; the
        *min_pos* here doubles as both the cited_pmids floor and the
        qualifying non-review-target count floor.
        """
        oversample = int(n * 3)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _rv_candidates AS
            SELECT pmid, title, abstract_text, journal, mesh_descriptors, cited_pmids
            FROM '{self.parquet_path}'
            WHERE len(cited_pmids) >= {min_pos}
              AND length(abstract_text) >= 150
              AND {WHERE_HAS_DESCRIPTOR}
              AND array_length(string_split(title, ' ')) >= 5
              AND (list_contains(publication_types, 'Review')
                   OR list_contains(publication_types, 'Systematic Review'))
            ORDER BY hash(pmid || '{self.seed}_rv_cand')
            LIMIT {oversample}
        """)

        # Qualifying citations: non-review, in-corpus, with usable abstract
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _rv_qualified AS
            SELECT ce.citing, ce.cited
            FROM (
                SELECT pmid AS citing, UNNEST(cited_pmids) AS cited
                FROM _rv_candidates
            ) ce
            JOIN '{self.parquet_path}' p ON p.pmid = ce.cited
            WHERE length(p.abstract_text) >= 150
              AND NOT list_contains(p.publication_types, 'Review')
        """)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _rv_valid AS
            SELECT citing, count(*) AS n_cited
            FROM _rv_qualified
            GROUP BY citing
            HAVING count(*) >= {min_pos}
        """)

        rows = self.con.execute(f"""
            SELECT c.pmid, c.title, c.abstract_text, c.journal, c.mesh_descriptors
            FROM _rv_candidates c
            JOIN _rv_valid v ON v.citing = c.pmid
            ORDER BY hash(c.pmid || '{self.seed}_rv_pick')
            LIMIT {n}
        """).fetchall()

        query_pmids = {r[0] for r in rows}
        if not query_pmids:
            return [], {}

        self._register_pmid_temp("_rv_chosen", query_pmids)
        edge_rows = self.con.execute("""
            SELECT q.citing, q.cited
            FROM _rv_qualified q
            JOIN _rv_chosen ch ON ch.pmid = q.citing
            ORDER BY hash(q.citing || '_' || q.cited)
        """).fetchall()

        per_query_cited: dict[str, list[str]] = {p: [] for p in query_pmids}
        for citing, cited in edge_rows:
            lst = per_query_cited[citing]
            if len(lst) < max_pos:
                lst.append(cited)

        for tbl in ("_rv_candidates", "_rv_qualified", "_rv_valid", "_rv_chosen"):
            self.con.execute(f"DROP TABLE IF EXISTS {tbl}")

        return rows, per_query_cited
