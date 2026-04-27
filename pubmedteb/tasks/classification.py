"""PubMed MeSH Classification task for MTEB.

Predict an article's depth-2 MeSH subcategory from its abstract. Labels
are derived by majority vote over the article's major MeSH descriptors
(P2 policy, support >= 0.60). The benchmark uses the 50 largest depth-2
classes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)

DATASETS_DIR = Path("datasets/pubmed_classification")


def _load_classification_jsonl(path: Path) -> Dataset:
    """Load ``{_id, text, label}`` JSONL into a HuggingFace Dataset."""
    ids, texts, labels = [], [], []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["_id"])
            texts.append(obj["text"])
            labels.append(obj["label"])
    return Dataset.from_dict({"id": ids, "text": texts, "label": labels})


class PubMedMeSHClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PubMedMeSHClassification",
        description=(
            "Predict the depth-2 MeSH subcategory of a PubMed article from "
            "its abstract. Labels are derived by majority vote over the "
            "article's major MeSH descriptors (P2 policy, descriptor "
            "support >= 0.60). The 50 largest depth-2 classes are kept, "
            "balanced to a target per-class count."
        ),
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        dataset={
            "path": "pubmedteb/mesh_classification",
            "revision": "1.0.0",
        },
        domains=["Medical", "Academic"],
        license="cc-by-4.0",
        date=("1970-01-01", "2025-12-31"),
        annotations_creators="derived",
        sample_creation="found",
    )

    def __init__(self, dataset_dir: Path | str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._dataset_dir = Path(dataset_dir) if dataset_dir else DATASETS_DIR

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        train = _load_classification_jsonl(self._dataset_dir / "train.jsonl")
        test = _load_classification_jsonl(self._dataset_dir / "test.jsonl")
        self.dataset = {"default": {"train": train, "test": test}}
        self.data_loaded = True
        logger.info(
            "Loaded PubMedMeSHClassification: %d train, %d test",
            len(train), len(test),
        )
