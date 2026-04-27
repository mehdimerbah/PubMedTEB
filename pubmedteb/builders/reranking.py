"""Reranking dataset builder.

Given the abstract of a citing paper and 50 candidate abstracts, the task
is to rank the actually-cited ones above the BM25-retrieved topical
hard negatives.

Candidates per query (v1):
    3–10 positives (in-corpus citations)
  + 40–47 BM25 top-k with category (depth-1 semantic_category) filter
  = 50 total candidates.

This is the "shape" variant of T3 for reranking. If the BM25 calibration
gate shows this is too easy, switch to the full T3 mix
(20 descriptor / 20 depth-3 / 40 BM25 / 10 journal / 10 random) per query.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pubmedteb.builders.citation_retrieval import CitationRetrievalBuilder
from pubmedteb.infra.bm25_index import open_bm25_index, query_bm25
from pubmedteb.infra.citation_graph import ensure_citation_graph

logger = logging.getLogger(__name__)


class RerankingBuilder(CitationRetrievalBuilder):
    """Build a PubMed reranking dataset (v1: BM25 hard-neg shape)."""

    SIZES = {
        "mini": {
            "n_queries": 500,
            "n_candidates": 50,
            "min_pos": 3,
            "max_pos": 10,
        },
        "full": {
            "n_queries": 5_000,
            "n_candidates": 50,
            "min_pos": 3,
            "max_pos": 10,
        },
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._top_ranked: dict[str, list[str]] = {}

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
        for pmid, _title, abstract, _journal, _descs in query_rows:
            queries[pmid] = abstract

        # semantic_category (depth-1 MeSH branch) for per-query BM25 filtering
        self._register_pmid_temp("_rr_qpmids", set(queries))
        cat_rows = self.con.execute(f"""
            SELECT p.pmid, p.semantic_category
            FROM '{self.parquet_path}' p
            JOIN _rr_qpmids q ON q.pmid = p.pmid
        """).fetchall()
        query_category = {r[0]: r[1] for r in cat_rows}
        self.con.execute("DROP TABLE IF EXISTS _rr_qpmids")

        # Fetch positive abstracts (seed corpus)
        cited_pmids_union: set[str] = set()
        for pmid in queries:
            cited_pmids_union.update(per_query_cited[pmid])
        cited_texts = self._fetch_abstracts(cited_pmids_union)
        logger.info("Fetched %d positive abstracts", len(cited_texts))

        corpus: dict[str, dict] = {}
        for pmid, abstract in cited_texts.items():
            corpus[pmid] = {"text": abstract}

        for pmid in queries:
            relevant = {c: 1 for c in per_query_cited[pmid] if c in cited_texts}
            qrels[pmid] = relevant

        # Per-query BM25 hard negatives
        top_ranked: dict[str, list[str]] = {}
        all_neg_pmids: set[str] = set()
        bm25_con = open_bm25_index()
        try:
            for qid, qtext in queries.items():
                positives = list(qrels[qid])
                n_needed = cfg["n_candidates"] - len(positives)
                if n_needed <= 0:
                    top_ranked[qid] = positives[: cfg["n_candidates"]]
                    continue

                exclude = set(positives) | {qid}
                qcat = query_category.get(qid)
                # Over-fetch so we can survive pmid collisions and missing abstracts
                raw = query_bm25(
                    bm25_con,
                    qtext,
                    top_k=n_needed + 20,
                    category_filter=qcat,
                    exclude_pmids=exclude,
                )
                neg_pmids = [pmid for pmid, _ in raw[:n_needed]]
                top_ranked[qid] = positives + neg_pmids
                all_neg_pmids.update(neg_pmids)
        finally:
            bm25_con.close()

        # Fetch abstracts for BM25 negatives not already in corpus
        missing = all_neg_pmids - set(corpus)
        if missing:
            neg_texts = self._fetch_abstracts(missing)
            logger.info("Fetched %d BM25 negative abstracts", len(neg_texts))
            for pmid, text in neg_texts.items():
                corpus[pmid] = {"text": text}

        # Drop any candidates whose abstracts we couldn't fetch — keep top_ranked
        # consistent with the corpus so MTEB doesn't see dangling docids.
        dropped = 0
        for qid, docids in top_ranked.items():
            kept = [d for d in docids if d in corpus]
            if len(kept) != len(docids):
                dropped += len(docids) - len(kept)
            top_ranked[qid] = kept
        if dropped:
            logger.warning("Dropped %d dangling candidate pmids from top_ranked", dropped)

        logger.info(
            "Built reranking set: %d queries, %d corpus docs, "
            "avg %.1f candidates/query",
            len(queries), len(corpus),
            sum(len(v) for v in top_ranked.values()) / max(1, len(top_ranked)),
        )

        self._top_ranked = top_ranked
        return queries, corpus, qrels

    def write(
        self,
        queries: dict[str, str],
        corpus: dict[str, dict],
        qrels: dict[str, dict[str, int]],
        top_ranked: dict[str, list[str]] | None = None,
    ) -> None:
        super().write(queries, corpus, qrels, top_ranked=top_ranked or self._top_ranked)
