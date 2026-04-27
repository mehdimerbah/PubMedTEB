"""Cross-specialty retrieval dataset builder (Task 4).


    A query-target pair counts as cross-specialty iff
        depth-1(query) != depth-1(target)
        AND |major_uids(query) ∩ major_uids(target)| >= 1.

The depth-1 branch is read from the precomputed ``semantic_category``
column. The major-UID intersection is the topical-overlap gate.

Hard-negative mix (per design spec §Task 4):
    60 % target-branch (semantic_category NOT IN query branches) + shared major UID
    40 % query-branch  (semantic_category IN query branches)     + shared major UID
"""

from __future__ import annotations

import logging

from pubmedteb.builders.base import WHERE_HAS_MAJOR
from pubmedteb.builders.citation_retrieval import CitationRetrievalBuilder
from pubmedteb.builders.negative_sampling import split_mix
from pubmedteb.infra.citation_graph import ensure_citation_graph

logger = logging.getLogger(__name__)


class CrossSpecialtyRetrievalBuilder(CitationRetrievalBuilder):
    """Build a cross-specialty retrieval dataset (D3 definition)."""

    MIX: dict[str, float] = {
        "target_branch": 0.60,
        "query_branch": 0.40,
    }

    SIZES = {
        "mini": {
            "n_queries": 500,
            "n_corpus": 5_000,
            "min_pos": 3,
            "max_pos": 10,
            "min_cites": 10,
            "oversample_factor": 12,
        },
        "full": {
            "n_queries": 5_000,
            "n_corpus": 50_000,
            "min_pos": 3,
            "max_pos": 20,
            "min_cites": 10,
            "oversample_factor": 12,
        },
    }

    def construct(
        self,
    ) -> tuple[dict[str, str], dict[str, dict], dict[str, dict[str, int]]]:
        cfg = self.SIZES[self.size]

        ensure_citation_graph(parquet_path=self.parquet_path)

        # 1. Sample candidate queries: ≥1 major UID + semantic_category + ≥min_cites citations.
        oversample = int(cfg["n_queries"] * cfg["oversample_factor"])
        candidates = self.con.execute(f"""
            SELECT pmid, title, abstract_text, journal, semantic_category,
                   mesh_descriptors, cited_pmids
            FROM '{self.parquet_path}'
            WHERE length(abstract_text) >= 150
              AND semantic_category != ''
              AND {WHERE_HAS_MAJOR}
              AND len(cited_pmids) >= {cfg["min_cites"]}
              AND array_length(string_split(title, ' ')) >= 5
            ORDER BY hash(pmid || '{self.seed}_cs_cand')
            LIMIT {oversample}
        """).fetchall()
        logger.info("Sampled %d candidate queries (pre-D3-filter)", len(candidates))

        # 2. Bulk-fetch cited articles' (semantic_category, major_uids, abstract).
        all_cited: set[str] = set()
        for row in candidates:
            all_cited.update(row[6])
        logger.info("Cited-pmid universe: %d", len(all_cited))

        if not all_cited:
            return {}, {}, {}

        self._register_pmid_temp("_cs_cited", all_cited)
        cited_rows = self.con.execute(f"""
            SELECT p.pmid, p.semantic_category,
                   list_transform(
                       list_filter(p.mesh_descriptors, x -> x.is_major),
                       x -> x.uid
                   ) AS major_uids,
                   p.abstract_text
            FROM '{self.parquet_path}' p
            JOIN _cs_cited c ON c.pmid = p.pmid
            WHERE length(p.abstract_text) >= 150
              AND p.semantic_category != ''
              AND len(list_filter(p.mesh_descriptors, x -> x.is_major)) >= 1
        """).fetchall()
        self.con.execute("DROP TABLE IF EXISTS _cs_cited")
        cited_info: dict[str, tuple[str, set[str], str]] = {
            r[0]: (r[1], set(r[2]), r[3]) for r in cited_rows
        }
        logger.info("Cited articles with usable info: %d", len(cited_info))

        # 3. Apply D3 per candidate: keep cross-branch citations sharing ≥1 major UID.
        valid_candidates: list[tuple] = []
        per_query_qualifying: dict[str, list[str]] = {}
        for row in candidates:
            qpmid, _t, _a, _j, qcat, qmesh, qcites = row
            qmajor: set[str] = {
                d["uid"] for d in (qmesh or [])
                if (d.get("is_major") if isinstance(d, dict) else d["is_major"])
                and (d.get("uid") if isinstance(d, dict) else d["uid"])
            }
            if not qmajor:
                continue
            qualifying: list[str] = []
            for cpmid in qcites:
                info = cited_info.get(cpmid)
                if info is None:
                    continue
                ccat, cmajor, _abs = info
                if ccat != qcat and (qmajor & cmajor):
                    qualifying.append(cpmid)
                    if len(qualifying) >= cfg["max_pos"]:
                        break
            if len(qualifying) >= cfg["min_pos"]:
                valid_candidates.append(row)
                per_query_qualifying[qpmid] = qualifying
        logger.info(
            "After D3: %d valid candidates with >= %d cross-branch citations",
            len(valid_candidates), cfg["min_pos"],
        )

        # 4. Deterministically sample n_queries from valid candidates.
        valid_candidates.sort(
            key=lambda r: hash((r[0], str(self.seed), "cs_pick"))
        )
        valid_candidates = valid_candidates[: cfg["n_queries"]]
        per_query_qualifying = {
            r[0]: per_query_qualifying[r[0]] for r in valid_candidates
        }
        logger.info("Selected %d queries for the dataset", len(valid_candidates))

        if not valid_candidates:
            return {}, {}, {}

        # 5. Build queries / corpus / qrels.
        queries: dict[str, str] = {r[0]: r[2] for r in valid_candidates}
        corpus: dict[str, dict] = {}
        qrels: dict[str, dict[str, int]] = {}
        for qpmid, cpmids in per_query_qualifying.items():
            qrels[qpmid] = {}
            for cpmid in cpmids:
                info = cited_info.get(cpmid)
                if info is None:
                    continue
                corpus[cpmid] = {"text": info[2]}
                qrels[qpmid][cpmid] = 1

        n_distractors = cfg["n_corpus"] - len(corpus)
        if n_distractors <= 0:
            logger.warning(
                "Positives already fill corpus (%d >= %d); no distractors added.",
                len(corpus), cfg["n_corpus"],
            )
            return queries, corpus, qrels

        # 6. Pool query branches and major UIDs for hard negatives.
        query_d1_pool: set[str] = {r[4] for r in valid_candidates if r[4]}
        query_major_uids: set[str] = set()
        for r in valid_candidates:
            for d in r[5] or []:
                is_major = d.get("is_major") if isinstance(d, dict) else d["is_major"]
                uid = d.get("uid") if isinstance(d, dict) else d["uid"]
                if is_major and uid:
                    query_major_uids.add(uid)
        logger.info(
            "Query pool: %d branches, %d major UIDs",
            len(query_d1_pool), len(query_major_uids),
        )

        counts = split_mix(n_distractors, self.MIX)
        logger.info("Negative mix (%d distractors): %s", n_distractors, counts)

        selected: set[str] = set(queries) | set(corpus)

        target_neg = self._sample_branch_filtered(
            query_major_uids, query_d1_pool, selected,
            counts["target_branch"], tag="cs_target", in_branches=False,
        )
        selected.update(p for p, _ in target_neg)
        logger.info("target-branch negatives: %d", len(target_neg))

        query_branch_neg = self._sample_branch_filtered(
            query_major_uids, query_d1_pool, selected,
            counts["query_branch"], tag="cs_query", in_branches=True,
        )
        logger.info("query-branch negatives: %d", len(query_branch_neg))

        for pmid, abstract in target_neg + query_branch_neg:
            corpus[pmid] = {"text": abstract}

        return queries, corpus, qrels

    def _sample_branch_filtered(
        self,
        uids: set[str],
        branches: set[str],
        exclude_pmids: set[str],
        n: int,
        tag: str,
        in_branches: bool,
    ) -> list[tuple[str, str]]:
        """Sample n articles whose mesh descriptor UIDs intersect *uids*,
        constrained to ``semantic_category {IN | NOT IN} branches``.
        """
        if not uids or not branches or n <= 0:
            return []

        uids_table = f"_cs_uids_{tag}"
        excl_table = f"_cs_excl_{tag}"
        br_table = f"_cs_br_{tag}"

        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {uids_table}(uid VARCHAR)")
        self.con.executemany(
            f"INSERT INTO {uids_table} VALUES (?)", [(u,) for u in uids],
        )
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {excl_table}(pmid VARCHAR)")
        if exclude_pmids:
            self.con.executemany(
                f"INSERT INTO {excl_table} VALUES (?)",
                [(p,) for p in exclude_pmids],
            )
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {br_table}(branch VARCHAR)")
        self.con.executemany(
            f"INSERT INTO {br_table} VALUES (?)", [(b,) for b in branches],
        )

        branch_clause = "IN" if in_branches else "NOT IN"
        rows = self.con.execute(f"""
            SELECT p.pmid, p.abstract_text
            FROM '{self.parquet_path}' p
            WHERE length(p.abstract_text) >= 150
              AND p.semantic_category != ''
              AND p.semantic_category {branch_clause} (SELECT branch FROM {br_table})
              AND p.pmid NOT IN (SELECT pmid FROM {excl_table})
              AND list_has_any(
                    list_transform(p.mesh_descriptors, x -> x.uid),
                    (SELECT list(uid) FROM {uids_table})
                  )
            ORDER BY hash(p.pmid || '{self.seed}_{tag}')
            LIMIT {n}
        """).fetchall()

        for tbl in (uids_table, excl_table, br_table):
            self.con.execute(f"DROP TABLE IF EXISTS {tbl}")
        return [(r[0], r[1]) for r in rows]
