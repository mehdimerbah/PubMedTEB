"""Title-to-abstract retrieval dataset builder.

Query = article title, relevant document = same article's abstract.
Hard negatives stratified by semantic category (60%), journal (20%), random (20%).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pubmedteb.builders.base import DatasetBuilder

logger = logging.getLogger(__name__)


class TitleRetrievalBuilder(DatasetBuilder):
    """Build a title -> abstract retrieval dataset from PubMed.

    Each query is an article title; the single relevant document is the
    same article's abstract (1:1 ground truth). The corpus is padded with
    hard-negative distractors sampled by semantic category, journal, and
    random selection.
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

        # Phase 1: Sample query articles
        query_articles = self._sample_query_articles(n_queries)
        logger.info("Sampled %d query articles", len(query_articles))

        # Phase 2: Build queries, corpus targets, qrels
        queries: dict[str, str] = {}
        corpus: dict[str, dict] = {}
        qrels: dict[str, dict[str, int]] = {}

        for pmid, title, abstract, category, journal in query_articles:
            queries[pmid] = title
            corpus[pmid] = {"text": abstract}
            qrels[pmid] = {pmid: 1}

        # Collect metadata for hard negative sampling
        query_pmids = set(queries.keys())
        query_categories = {cat for _, _, _, cat, _ in query_articles if cat}
        query_journals = {journal for _, _, _, _, journal in query_articles if journal}

        # Phase 3: Sample distractors
        n_cat = int(n_distractors * 0.6)
        n_journal = int(n_distractors * 0.2)
        n_random = n_distractors - n_cat - n_journal

        selected_pmids = set(query_pmids)

        cat_distractors = self._sample_by_category(
            query_categories, selected_pmids, n_cat,
        )
        selected_pmids.update(d[0] for d in cat_distractors)
        logger.info("Sampled %d category distractors", len(cat_distractors))

        journal_distractors = self._sample_by_journal(
            query_journals, selected_pmids, n_journal,
        )
        selected_pmids.update(d[0] for d in journal_distractors)
        logger.info("Sampled %d journal distractors", len(journal_distractors))

        random_distractors = self._sample_random(selected_pmids, n_random)
        logger.info("Sampled %d random distractors", len(random_distractors))

        # Phase 4: Assemble corpus
        for pmid, abstract in cat_distractors + journal_distractors + random_distractors:
            corpus[pmid] = {"text": abstract}

        return queries, corpus, qrels

    def _sample_query_articles(self, n: int) -> list[tuple]:
        """Sample articles suitable as queries.

        Filters: title >= 5 words, has semantic_category, abstract >= 150 chars.
        Uses deterministic hash-based ordering for reproducibility.
        """
        # Over-sample to account for duplicate title filtering
        oversample = int(n * 1.1)
        rows = self.query(f"""
            SELECT pmid, title, abstract_text, semantic_category, journal
            FROM {{parquet}}
            WHERE length(abstract_text) >= 150
              AND semantic_category != ''
              AND array_length(string_split(title, ' ')) >= 5
            ORDER BY hash(pmid || '{self.seed}')
            LIMIT {oversample}
        """)

        # Deduplicate titles
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

    def _sample_by_category(
        self,
        categories: set[str],
        exclude_pmids: set[str],
        n: int,
    ) -> list[tuple[str, str]]:
        """Sample distractors from the same semantic categories as query articles."""
        if not categories or n <= 0:
            return []

        cat_list = ", ".join(f"'{c}'" for c in categories)
        exclude_list = ", ".join(f"'{p}'" for p in exclude_pmids)

        rows = self.query(f"""
            SELECT pmid, abstract_text
            FROM {{parquet}}
            WHERE semantic_category IN ({cat_list})
              AND pmid NOT IN ({exclude_list})
              AND length(abstract_text) >= 150
            ORDER BY hash(pmid || '{self.seed}_cat')
            LIMIT {n}
        """)
        return [(r[0], r[1]) for r in rows]

    def _sample_by_journal(
        self,
        journals: set[str],
        exclude_pmids: set[str],
        n: int,
    ) -> list[tuple[str, str]]:
        """Sample distractors from the same journals as query articles."""
        if not journals or n <= 0:
            return []

        journal_list = ", ".join(f"'{j.replace(chr(39), chr(39)+chr(39))}'" for j in journals)
        exclude_list = ", ".join(f"'{p}'" for p in exclude_pmids)

        rows = self.query(f"""
            SELECT pmid, abstract_text
            FROM {{parquet}}
            WHERE journal IN ({journal_list})
              AND pmid NOT IN ({exclude_list})
              AND length(abstract_text) >= 150
            ORDER BY hash(pmid || '{self.seed}_journal')
            LIMIT {n}
        """)
        return [(r[0], r[1]) for r in rows]

    def _sample_random(
        self,
        exclude_pmids: set[str],
        n: int,
    ) -> list[tuple[str, str]]:
        """Sample random distractors from the full corpus."""
        if n <= 0:
            return []

        exclude_list = ", ".join(f"'{p}'" for p in exclude_pmids)

        rows = self.query(f"""
            SELECT pmid, abstract_text
            FROM {{parquet}}
            WHERE pmid NOT IN ({exclude_list})
              AND length(abstract_text) >= 150
            ORDER BY hash(pmid || '{self.seed}_random')
            LIMIT {n}
        """)
        return [(r[0], r[1]) for r in rows]
