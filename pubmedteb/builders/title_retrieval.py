"""Title-to-abstract retrieval dataset builder.

Query = article title, relevant document = same article's abstract.

Hard-negative mix (T3 recipe, see ``reports/mesh_investigation/tables/
T3_hardneg_mix_per_task.csv``): 30 % same-descriptor, 20 % shared depth-3,
20 % BM25, 10 % same-journal, 20 % random.
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

logger = logging.getLogger(__name__)

MIX = {
    "descriptor": 0.30,
    "depth3": 0.20,
    "bm25": 0.20,
    "journal": 0.10,
    "random": 0.20,
}


class TitleRetrievalBuilder(DatasetBuilder):
    """Build a title -> abstract retrieval dataset from PubMed.

    Each query is an article title; the single relevant document is the
    same article's abstract (1:1 ground truth). The corpus is padded with
    hard-negative distractors per the T3 mix.
    """

    SIZES = {
        "mini": {"n_queries": 500, "n_corpus": 5_000},
        "full": {"n_queries": 10_000, "n_corpus": 100_000},
    }

    def construct(
        self,
    ) -> tuple[dict[str, str], dict[str, dict], dict[str, dict[str, int]]]:
        cfg = self.SIZES[self.size]
        n_queries = cfg["n_queries"]
        n_distractors = cfg["n_corpus"] - n_queries

        query_rows = self._sample_query_articles(n_queries)
        logger.info("Sampled %d query articles", len(query_rows))

        queries: dict[str, str] = {}
        corpus: dict[str, dict] = {}
        qrels: dict[str, dict[str, int]] = {}
        for pmid, title, abstract, _journal, _descs in query_rows:
            queries[pmid] = title
            corpus[pmid] = {"text": abstract}
            qrels[pmid] = {pmid: 1}

        mappings = load_mesh_mappings()
        sampler = NegativeSampler(self.con, self.parquet_path, self.seed, mappings)

        query_pmids: set[str] = set(queries)
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

        selected: set[str] = set(query_pmids)

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

    def _sample_query_articles(self, n: int) -> list[tuple]:
        """Sample articles usable as queries.

        Filters: abstract ≥ 150 chars, title ≥ 5 words, has ≥ 1 descriptor.
        Titles deduplicated case-insensitively.
        """
        oversample = int(n * 1.1)
        rows = self.query(f"""
            SELECT pmid, title, abstract_text, journal, mesh_descriptors
            FROM {{parquet}}
            WHERE length(abstract_text) >= 150
              AND len(mesh_descriptors) >= 1
              AND array_length(string_split(title, ' ')) >= 5
            ORDER BY hash(pmid || '{self.seed}')
            LIMIT {oversample}
        """)

        seen_titles: set[str] = set()
        result: list[tuple] = []
        for row in rows:
            title_lower = row[1].lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                result.append(row)
                if len(result) >= n:
                    break
        return result
