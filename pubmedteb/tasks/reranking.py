"""PubMed Reranking task for MTEB.

Given a paper abstract and a pre-selected candidate set of 50 abstracts,
rerank so that the actually-cited papers appear at the top. Candidates
are a mix of cited positives and BM25-retrieved hard negatives from the
same MeSH branch.

Implemented via ``AbsTaskRetrieval`` with ``top_ranked`` populated — the
MTEB v2.12+ pattern after ``AbsTaskReranking`` was deprecated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from pubmedteb.tasks import load_corpus_jsonl, load_queries_jsonl, load_qrels

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets/pubmed_reranking")


def load_top_ranked(path: Path) -> dict[str, list[str]]:
    """Load top_ranked.jsonl into ``{qid: [docid, ...]}``."""
    top_ranked: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            top_ranked[obj["qid"]] = list(obj["docids"])
    return top_ranked


class PubMedReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PubMedReranking",
        description=(
            "Given a biomedical article abstract and a candidate set of 50 "
            "abstracts (the cited references plus BM25-retrieved hard "
            "negatives from the same MeSH branch), rerank so the cited "
            "papers appear at the top."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "pubmedteb/reranking",
            "revision": "1.0.0",
        },
        domains=["Medical", "Academic"],
        license="cc-by-4.0",
        date=("1970-01-01", "2025-12-31"),
        annotations_creators="derived",
        sample_creation="found",
        prompt={
            "query": "Given a biomedical article abstract, rerank the candidate abstracts so that cited papers appear first",
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
        top_ranked = load_top_ranked(self._dataset_dir / "top_ranked.jsonl")

        self.dataset = {
            "default": {
                "test": {
                    "corpus": corpus,
                    "queries": queries,
                    "relevant_docs": relevant_docs,
                    "top_ranked": top_ranked,
                }
            }
        }
        self.data_loaded = True
        logger.info(
            "Loaded PubMedReranking: %d queries, %d corpus docs, %d top_ranked lists",
            len(queries), len(corpus), len(top_ranked),
        )
