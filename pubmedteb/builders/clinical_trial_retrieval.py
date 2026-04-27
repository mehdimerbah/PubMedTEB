"""Clinical-trial retrieval dataset builder (Task 6).

Query = abstract of a non-trial paper that cites in-corpus clinical trials.
Relevant docs = the cited clinical-trial abstracts.
Corpus = clinical trials only (positives plus same-depth-2 / adjacent-depth-1
trial distractors).

Hard-negative mix (per design spec §Task 6): 60 % clinical trials in the
query's depth-2 disease area but NOT cited; 40 % clinical trials in an
adjacent disease area (same depth-1 branch, different depth-2).
"""

from __future__ import annotations

import logging

from pubmedteb.analysis.mesh import load_mesh_mappings, majority_label_with_margin
from pubmedteb.builders.base import WHERE_HAS_DESCRIPTOR
from pubmedteb.builders.citation_retrieval import CitationRetrievalBuilder
from pubmedteb.builders.negative_sampling import split_mix
from pubmedteb.infra.citation_graph import ensure_citation_graph

logger = logging.getLogger(__name__)

TRIAL_TYPES = (
    "Randomized Controlled Trial",
    "Clinical Trial",
    "Controlled Clinical Trial",
)


class ClinicalTrialRetrievalBuilder(CitationRetrievalBuilder):
    """Build a clinical-trial retrieval dataset.

    Inherits ``_register_pmid_temp`` / ``_fetch_abstracts`` from
    :class:`CitationRetrievalBuilder` and overrides ``construct`` and
    ``_sample_query_articles`` to enforce the trial-only corpus.
    """

    MIX: dict[str, float] = {
        "same_depth2": 0.60,
        "adjacent_depth1": 0.40,
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
            "n_corpus": 30_000,
            "min_pos": 5,
            "max_pos": 20,
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
            "Sampled %d non-trial query articles with %d total in-corpus trial citations",
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
        logger.info("Fetched %d positive trial abstracts", len(cited_texts))

        for pmid in queries:
            qrels[pmid] = {c: 1 for c in per_query_cited[pmid] if c in cited_texts}
        for pmid, abstract in cited_texts.items():
            corpus[pmid] = {"text": abstract}

        n_distractors = cfg["n_corpus"] - len(corpus)
        if n_distractors <= 0:
            logger.warning(
                "Positives already fill corpus (%d >= %d); no distractors added.",
                len(corpus), cfg["n_corpus"],
            )
            return queries, corpus, qrels

        # Compute query depth-2 / depth-1 label sets (majority over major UIDs).
        mappings = load_mesh_mappings()
        uid_to_d1 = mappings.uid_to_depth1
        uid_to_d2 = mappings.uid_to_depth2

        query_depth2: set[str] = set()
        query_depth1: set[str] = set()
        for row in query_rows:
            descs = row[4] or []
            major_uids = [
                d["uid"] for d in descs
                if (d.get("is_major") if isinstance(d, dict) else d["is_major"])
                and (d.get("uid") if isinstance(d, dict) else d["uid"])
            ]
            if not major_uids:
                continue
            d2_label, _, _ = majority_label_with_margin(major_uids, uid_to_d2)
            d1_label, _, _ = majority_label_with_margin(major_uids, uid_to_d1)
            if d2_label:
                query_depth2.add(d2_label)
            if d1_label:
                query_depth1.add(d1_label)

        logger.info(
            "Query coverage: %d depth-2 labels, %d depth-1 labels",
            len(query_depth2), len(query_depth1),
        )

        # Eligible UIDs:
        # - same-depth-2: any UID that maps to a query depth-2 label
        # - adjacent-depth-1: maps to a query depth-1 label but NOT to any query depth-2
        d2_eligible = {
            uid for uid, labels in uid_to_d2.items()
            if any(l in query_depth2 for l in labels)
        }
        d1_eligible = {
            uid for uid, labels in uid_to_d1.items()
            if any(l in query_depth1 for l in labels)
        } - d2_eligible
        logger.info(
            "Eligible UIDs: %d same-depth-2, %d adjacent-depth-1-only",
            len(d2_eligible), len(d1_eligible),
        )

        counts = split_mix(n_distractors, self.MIX)
        logger.info("Negative mix (%d distractors): %s", n_distractors, counts)

        selected: set[str] = set(queries) | set(corpus)

        same_d2_neg = self._sample_trials_by_uids(
            d2_eligible, selected, counts["same_depth2"], tag="ct_d2",
        )
        selected.update(p for p, _ in same_d2_neg)
        logger.info("same-depth-2 trial negatives: %d", len(same_d2_neg))

        adj_d1_neg = self._sample_trials_by_uids(
            d1_eligible, selected, counts["adjacent_depth1"], tag="ct_d1",
        )
        logger.info("adjacent-depth-1 trial negatives: %d", len(adj_d1_neg))

        for pmid, abstract in same_d2_neg + adj_d1_neg:
            corpus[pmid] = {"text": abstract}

        return queries, corpus, qrels

    def _sample_query_articles(
        self,
        n: int,
        min_pos: int,
        max_pos: int,
    ) -> tuple[list[tuple], dict[str, list[str]]]:
        """Non-trial papers citing ≥ min_pos in-corpus clinical trials."""
        oversample = int(n * 3)
        trial_list = "[" + ", ".join(f"'{t}'" for t in TRIAL_TYPES) + "]"

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _ct_candidates AS
            SELECT pmid, title, abstract_text, journal, mesh_descriptors, cited_pmids
            FROM '{self.parquet_path}'
            WHERE len(cited_pmids) >= {min_pos}
              AND length(abstract_text) >= 150
              AND {WHERE_HAS_DESCRIPTOR}
              AND array_length(string_split(title, ' ')) >= 5
              AND NOT list_has_any(publication_types, {trial_list})
            ORDER BY hash(pmid || '{self.seed}_ct_cand')
            LIMIT {oversample}
        """)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _ct_qualified AS
            SELECT ce.citing, ce.cited
            FROM (
                SELECT pmid AS citing, UNNEST(cited_pmids) AS cited
                FROM _ct_candidates
            ) ce
            JOIN '{self.parquet_path}' p ON p.pmid = ce.cited
            WHERE length(p.abstract_text) >= 150
              AND list_has_any(p.publication_types, {trial_list})
        """)

        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _ct_valid AS
            SELECT citing, count(*) AS n_cited
            FROM _ct_qualified
            GROUP BY citing
            HAVING count(*) >= {min_pos}
        """)

        rows = self.con.execute(f"""
            SELECT c.pmid, c.title, c.abstract_text, c.journal, c.mesh_descriptors
            FROM _ct_candidates c
            JOIN _ct_valid v ON v.citing = c.pmid
            ORDER BY hash(c.pmid || '{self.seed}_ct_pick')
            LIMIT {n}
        """).fetchall()

        query_pmids = {r[0] for r in rows}
        if not query_pmids:
            return [], {}

        self._register_pmid_temp("_ct_chosen", query_pmids)
        edge_rows = self.con.execute("""
            SELECT q.citing, q.cited
            FROM _ct_qualified q
            JOIN _ct_chosen ch ON ch.pmid = q.citing
            ORDER BY hash(q.citing || '_' || q.cited)
        """).fetchall()

        per_query_cited: dict[str, list[str]] = {p: [] for p in query_pmids}
        for citing, cited in edge_rows:
            lst = per_query_cited[citing]
            if len(lst) < max_pos:
                lst.append(cited)

        for tbl in ("_ct_candidates", "_ct_qualified", "_ct_valid", "_ct_chosen"):
            self.con.execute(f"DROP TABLE IF EXISTS {tbl}")

        return rows, per_query_cited

    def _sample_trials_by_uids(
        self,
        uids: set[str],
        exclude_pmids: set[str],
        n: int,
        tag: str,
    ) -> list[tuple[str, str]]:
        """Sample n clinical trials whose mesh UIDs intersect *uids*."""
        if not uids or n <= 0:
            return []

        uids_table = f"_ct_uids_{tag}"
        excl_table = f"_ct_excl_{tag}"
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {uids_table}(uid VARCHAR)")
        self.con.executemany(
            f"INSERT INTO {uids_table} VALUES (?)", [(u,) for u in uids],
        )
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {excl_table}(pmid VARCHAR)")
        if exclude_pmids:
            self.con.executemany(
                f"INSERT INTO {excl_table} VALUES (?)", [(p,) for p in exclude_pmids],
            )

        trial_list = "[" + ", ".join(f"'{t}'" for t in TRIAL_TYPES) + "]"
        rows = self.con.execute(f"""
            SELECT p.pmid, p.abstract_text
            FROM '{self.parquet_path}' p
            WHERE list_has_any(p.publication_types, {trial_list})
              AND length(p.abstract_text) >= 150
              AND p.pmid NOT IN (SELECT pmid FROM {excl_table})
              AND list_has_any(
                    list_transform(p.mesh_descriptors, x -> x.uid),
                    (SELECT list(uid) FROM {uids_table})
                  )
            ORDER BY hash(p.pmid || '{self.seed}_{tag}')
            LIMIT {n}
        """).fetchall()

        for tbl in (uids_table, excl_table):
            self.con.execute(f"DROP TABLE IF EXISTS {tbl}")
        return [(r[0], r[1]) for r in rows]
