"""PubMed Clinical-Trial Retrieval task for MTEB.

Given the abstract of a non-trial paper that cites clinical trials,
retrieve the cited trial abstracts from a trial-only corpus seeded with
same-depth-2 / adjacent-depth-1 hard negatives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from pubmedteb.tasks import load_corpus_jsonl, load_qrels, load_queries_jsonl

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets/pubmed_clinical_trial_retrieval")


class PubMedClinicalTrialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PubMedClinicalTrialRetrieval",
        description=(
            "Given the abstract of a non-trial PubMed article that cites "
            "clinical trials, retrieve the cited trial abstracts. Corpus is "
            "trials-only, padded with same-depth-2 (60%) and adjacent-"
            "depth-1 (40%) trial hard negatives."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "pubmedteb/clinical_trial_retrieval",
            "revision": "1.0.0",
        },
        domains=["Medical", "Academic"],
        license="cc-by-4.0",
        date=("1970-01-01", "2025-12-31"),
        annotations_creators="derived",
        sample_creation="found",
        prompt={
            "query": "Given a biomedical article abstract, retrieve abstracts of clinical trials it cites",
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
            "Loaded PubMedClinicalTrialRetrieval: %d queries, %d corpus docs",
            len(queries), len(corpus),
        )
