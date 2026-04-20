"""PubMed Citation Retrieval task for MTEB.

Given the abstract of a citing paper, retrieve the abstracts of the papers
it references from a corpus that includes topically-similar non-cited
hard negatives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from pubmedteb.tasks import load_corpus_jsonl, load_queries_jsonl, load_qrels

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets/pubmed_citation_retrieval")


class PubMedCitationRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PubMedCitationRetrieval",
        description=(
            "Given the abstract of a citing PubMed article, retrieve the "
            "abstracts of the papers it references. Corpus is padded with "
            "topically-similar non-cited hard negatives drawn per the T3 "
            "recipe (50% descriptor, 10% depth-3, 10% BM25, 5% journal, "
            "25% random)."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "pubmedteb/citation_retrieval",
            "revision": "1.0.0",
        },
        domains=["Medical", "Academic"],
        license="cc-by-4.0",
        date=("1970-01-01", "2025-12-31"),
        annotations_creators="derived",
        sample_creation="found",
        prompt={
            "query": "Given a biomedical article abstract, retrieve the abstracts of papers it cites",
        },
    )

    def __init__(self, dataset_dir: Path | str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dataset_dir = Path(dataset_dir) if dataset_dir else DATASETS_DIR

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        corpus = load_corpus_jsonl(self._dataset_dir / "corpus.jsonl")
        queries = load_queries_jsonl(self._dataset_dir / "queries.jsonl")
        relevant_docs = load_qrels(self._dataset_dir / "qrels.tsv")

        self.dataset = {
            "default": {
                "test": {
                    "corpus": corpus,
                    "queries": queries,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }
        self.data_loaded = True
        logger.info(
            "Loaded PubMedCitationRetrieval: %d queries, %d corpus docs",
            len(queries), len(corpus),
        )
