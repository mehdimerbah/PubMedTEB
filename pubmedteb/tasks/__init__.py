"""MTEB task registry and data loading helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_corpus_jsonl(path: Path) -> Dataset:
    """Load a corpus JSONL file into a HuggingFace Dataset.

    Each line must be ``{"_id": str, "text": str}``.

    Returns:
        Dataset with columns ``id`` and ``text``.
    """
    ids, texts = [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["_id"])
            texts.append(obj["text"])
    return Dataset.from_dict({"id": ids, "text": texts})


def load_queries_jsonl(path: Path) -> Dataset:
    """Load a queries JSONL file into a HuggingFace Dataset.

    Each line must be ``{"_id": str, "text": str}``.

    Returns:
        Dataset with columns ``id`` and ``text``.
    """
    ids, texts = [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["_id"])
            texts.append(obj["text"])
    return Dataset.from_dict({"id": ids, "text": texts})


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load a qrels TSV file into a nested dict.

    Expected format: ``query_id\\tdoc_id\\tscore`` (no header).

    Returns:
        ``{query_id: {doc_id: score}}``.
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, doc_id, score = parts
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = int(score)
    return qrels


# Task registry — import here to keep it centralized
from pubmedteb.tasks.citation_retrieval import PubMedCitationRetrieval  # noqa: E402
from pubmedteb.tasks.classification import PubMedMeSHClassification  # noqa: E402
from pubmedteb.tasks.clinical_trial_retrieval import PubMedClinicalTrialRetrieval  # noqa: E402
from pubmedteb.tasks.cross_specialty_retrieval import PubMedCrossSpecialtyRetrieval  # noqa: E402
from pubmedteb.tasks.reranking import PubMedReranking  # noqa: E402
from pubmedteb.tasks.review_retrieval import PubMedReviewRetrieval  # noqa: E402
from pubmedteb.tasks.temporal_retrieval import PubMedTemporalRetrieval  # noqa: E402
from pubmedteb.tasks.title_retrieval import PubMedTitleRetrieval  # noqa: E402

ALL_TASKS: list[type] = [
    PubMedTitleRetrieval,
    PubMedCitationRetrieval,
    PubMedReranking,
    PubMedReviewRetrieval,
    PubMedTemporalRetrieval,
    PubMedMeSHClassification,
    PubMedClinicalTrialRetrieval,
    PubMedCrossSpecialtyRetrieval,
]

_TASK_MAP: dict[str, type] = {t.metadata.name: t for t in ALL_TASKS}


def get_task(name: str, **kwargs):
    """Instantiate a task by its metadata name.

    Args:
        name: Task name as defined in TaskMetadata (e.g., "PubMedTitleRetrieval").
        **kwargs: Passed to the task constructor (e.g., dataset_dir).

    Returns:
        An instantiated MTEB task object.
    """
    if name not in _TASK_MAP:
        available = list(_TASK_MAP.keys())
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return _TASK_MAP[name](**kwargs)
