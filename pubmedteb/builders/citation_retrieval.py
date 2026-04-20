"""Citation retrieval dataset builder.

Query = abstract of a citing article. Relevant docs = abstracts of its
in-corpus cited papers.

Hard-negative mix (T3 recipe, ``reports/mesh_investigation/tables/
T3_hardneg_mix_per_task.csv``): 50 % same-descriptor, 10 % shared depth-3,
10 % BM25, 5 % same-journal, 25 % random.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pubmedteb.analysis.mesh import load_mesh_mappings
from pubmedteb.builders.base import DatasetBuilder
from pubmedteb.builders.negative_sampling import (
    NegativeSampler,
    depth3_labels_of,
    descriptor_uids_of,
    split_mix,
)
from pubmedteb.infra.bm25_index import open_bm25_index
from pubmedteb.infra.citation_graph import ensure_citation_graph

logger = logging.getLogger(__name__)

MIX = {
    "descriptor": 0.50,
    "depth3": 0.10,
    "bm25": 0.10,
    "journal": 0.05,
    "random": 0.25,
}


class CitationRetrievalBuilder(DatasetBuilder):
    """Build a citation retrieval dataset.

    Query is the abstract of a citing paper. The relevant documents are
    the abstracts of its in-corpus cited papers (bounded to *max_pos*).
    The corpus is padded with hard-negative distractors per the T3 mix.
    """

    SIZES = {
        "mini": {
            "n_queries": 500,
            "n_corpus": 5_000,
            "min_pos": 5,
            "max_pos": 5,
        },
        "full": {
            "n_queries": 5_000,
            "n_corpus": 100_000,
            "min_pos": 10,
            "max_pos": 30,
        },
    }

    def construct(
        self,
    ) -> tuple[dict[str, str], dict[str, dict], dict[str, dict[str, int]]]:
        cfg = self.SIZES[self.size]

        ensure_citation_graph(parquet_path=self.parquet_path)

        query_rows, per_query_cited = self._sample_query_articles(
            n=cfg["n_queries"],
            min_pos=cfg["min_pos"],
            max_pos=cfg["max_pos"],
        )
        logger.info(
            "Sampled %d query articles with %d total in-corpus citations",
            len(query_rows),
            sum(len(v) for v in per_query_cited.values()),
        )

        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, int]] = {}
        corpus: dict[str, dict] = {}

        for pmid, _title, abstract, _journal, _descs in query_rows:
            queries[pmid] = abstract

        cited_pmids_union: set[str] = set()
        for pmid in queries:
            cited_pmids_union.update(per_query_cited[pmid])

        cited_texts = self._fetch_abstracts(cited_pmids_union)
        logger.info("Fetched %d positive abstracts", len(cited_texts))

        for pmid in queries:
            relevant = {c: 1 for c in per_query_cited[pmid] if c in cited_texts}
            qrels[pmid] = relevant
        for pmid, abstract in cited_texts.items():
            corpus[pmid] = {"text": abstract}

        n_distractors = cfg["n_corpus"] - len(corpus)
        if n_distractors <= 0:
            logger.warning(
                "Positives already fill corpus (%d >= %d); no distractors added.",
                len(corpus), cfg["n_corpus"],
            )
            return queries, corpus, qrels

        mappings = load_mesh_mappings()
        sampler = NegativeSampler(self.con, self.parquet_path, self.seed, mappings)

        descriptor_uids = descriptor_uids_of(query_rows, desc_col_index=4)
        depth3_labels = depth3_labels_of(
            query_rows, desc_col_index=4, uid_to_depth3=mappings.uid_to_depth3,
        )
        journals = {row[3] for row in query_rows if row[3]}
        logger.info(
            "Negative context: %d descriptor UIDs, %d depth-3 labels, %d journals",
            len(descriptor_uids), len(depth3_labels), len(journals),
        )

        counts = split_mix(n_distractors, MIX)
        logger.info("Negative mix (%d distractors): %s", n_distractors, counts)

        # Exclude every positive (direct cite) so negatives are "topically similar
        # but not cited" — the whole point of the T3 recipe for this task.
        selected: set[str] = set(queries) | set(corpus)

        desc_neg = sampler.sample_by_descriptor(
            descriptor_uids, selected, counts["descriptor"],
        )
        selected.update(p for p, _ in desc_neg)
        logger.info("descriptor negatives: %d", len(desc_neg))

        d3_neg = sampler.sample_by_depth3(
            depth3_labels, selected, counts["depth3"],
        )
        selected.update(p for p, _ in d3_neg)
        logger.info("depth-3 negatives: %d", len(d3_neg))

        bm25_con = open_bm25_index()
        try:
            bm25_neg = sampler.sample_by_bm25(
                bm25_con,
                [(qid, queries[qid]) for qid in queries],
                selected,
                counts["bm25"],
            )
        finally:
            bm25_con.close()
        selected.update(p for p, _ in bm25_neg)
        logger.info("BM25 negatives: %d", len(bm25_neg))

        journal_neg = sampler.sample_by_journal(
            journals, selected, counts["journal"],
        )
        selected.update(p for p, _ in journal_neg)
        logger.info("journal negatives: %d", len(journal_neg))

        random_neg = sampler.sample_random(selected, counts["random"])
        logger.info("random negatives: %d", len(random_neg))

        for pmid, abstract in desc_neg + d3_neg + bm25_neg + journal_neg + random_neg:
            corpus[pmid] = {"text": abstract}

        return queries, corpus, qrels

    def _sample_query_articles(
        self,
        n: int,
        min_pos: int,
        max_pos: int,
    ) -> tuple[list[tuple], dict[str, list[str]]]:
        """Sample citing articles with ≥ *min_pos* in-corpus cited abstracts.

        Returns (query_rows, per_query_cited) where query_rows contains
        ``(pmid, title, abstract_text, journal, mesh_descriptors)`` and
        per_query_cited maps the query pmid to a capped list of cited pmids.
        """
        oversample = int(n * 3)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cr_candidates AS
            SELECT pmid, title, abstract_text, journal, mesh_descriptors, cited_pmids
            FROM '{self.parquet_path}'
            WHERE len(cited_pmids) >= {min_pos}
              AND length(abstract_text) >= 150
              AND len(mesh_descriptors) >= 1
              AND array_length(string_split(title, ' ')) >= 5
            ORDER BY hash(pmid || '{self.seed}_cr_cand')
            LIMIT {oversample}
        """)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cr_qualified AS
            SELECT ce.citing, ce.cited
            FROM (
                SELECT pmid AS citing, UNNEST(cited_pmids) AS cited
                FROM _cr_candidates
            ) ce
            JOIN '{self.parquet_path}' p ON p.pmid = ce.cited
            WHERE length(p.abstract_text) >= 150
        """)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cr_valid AS
            SELECT citing, count(*) AS n_cited
            FROM _cr_qualified
            GROUP BY citing
            HAVING count(*) >= {min_pos}
        """)

        rows = self.con.execute(f"""
            SELECT c.pmid, c.title, c.abstract_text, c.journal, c.mesh_descriptors
            FROM _cr_candidates c
            JOIN _cr_valid v ON v.citing = c.pmid
            ORDER BY hash(c.pmid || '{self.seed}_cr_pick')
            LIMIT {n}
        """).fetchall()

        query_pmids = {r[0] for r in rows}
        if not query_pmids:
            return [], {}

        self._register_pmid_temp("_cr_chosen", query_pmids)
        edge_rows = self.con.execute("""
            SELECT q.citing, q.cited
            FROM _cr_qualified q
            JOIN _cr_chosen ch ON ch.pmid = q.citing
            ORDER BY hash(q.citing || '_' || q.cited)
        """).fetchall()

        per_query_cited: dict[str, list[str]] = {p: [] for p in query_pmids}
        for citing, cited in edge_rows:
            lst = per_query_cited[citing]
            if len(lst) < max_pos:
                lst.append(cited)

        # Drop staging temp tables so the negative-sampling phase has a
        # clean connection and doesn't carry the ~340M-edge materialization.
        for tbl in ("_cr_candidates", "_cr_qualified", "_cr_valid", "_cr_chosen"):
            self.con.execute(f"DROP TABLE IF EXISTS {tbl}")

        return rows, per_query_cited

    def _fetch_abstracts(self, pmids: set[str]) -> dict[str, str]:
        """Fetch abstracts for a set of pmids from the Parquet."""
        if not pmids:
            return {}
        self._register_pmid_temp("_cr_fetch", pmids)
        rows = self.con.execute(f"""
            SELECT p.pmid, p.abstract_text
            FROM '{self.parquet_path}' p
            JOIN _cr_fetch f ON f.pmid = p.pmid
            WHERE length(p.abstract_text) >= 150
        """).fetchall()
        return {r[0]: r[1] for r in rows}

    def _register_pmid_temp(self, name: str, pmids: set[str]) -> None:
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {name}(pmid VARCHAR)")
        if pmids:
            self.con.executemany(
                f"INSERT INTO {name} VALUES (?)", [(p,) for p in pmids]
            )
