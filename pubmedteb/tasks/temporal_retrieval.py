"""PubMed Temporal Retrieval task for MTEB.

Given the abstract of a recent (year ≥ 2018) paper, retrieve the older
(≥10 y earlier) papers it cites. Tests robustness to biomedical
terminology drift across decades.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from pubmedteb.tasks import load_corpus_jsonl, load_queries_jsonl, load_qrels

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets/pubmed_temporal_retrieval")


class PubMedTemporalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PubMedTemporalRetrieval",
        description=(
            "Given the abstract of a recent (year ≥ 2018) biomedical "
            "article, retrieve the older (≥10 y earlier) papers it "
            "cites. Tests robustness to terminology drift over decades."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "pubmedteb/temporal_retrieval",
            "revision": "1.0.0",
        },
        domains=["Medical", "Academic"],
        license="cc-by-4.0",
        date=("1970-01-01", "2025-12-31"),
        annotations_creators="derived",
        sample_creation="found",
        prompt={
            "query": "Given a biomedical article abstract, retrieve the older papers it cites",
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
            "Loaded PubMedTemporalRetrieval: %d queries, %d corpus docs",
            len(queries), len(corpus),
        )
