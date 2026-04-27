"""MeSH depth-2 classification dataset builder.

P2 = major-only majority vote at depth-2, requiring ≥ 0.60 descriptor support
on the winning label. The 50 largest depth-2 classes are kept and balanced
to ~1,000 articles per class (mini: ~200/class). 80/20 stratified train/test
split, deterministic via hash on PMID.

Output: ``train.jsonl`` and ``test.jsonl`` with ``{_id, text, label}`` rows,
plus ``labels.json`` documenting the policy and class list.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict

from pubmedteb.analysis.mesh import load_mesh_mappings, majority_label_with_margin
from pubmedteb.builders.base import DatasetBuilder, write_jsonl

logger = logging.getLogger(__name__)

SUPPORT_THRESHOLD = 0.60
N_CLASSES = 50
TRAIN_FRAC = 0.80


class ClassificationBuilder(DatasetBuilder):
    """Build a depth-2 MeSH classification dataset using policy P2."""

    SIZES = {
        "mini": {"per_class": 200, "candidate_sample": 500_000},
        "full": {"per_class": 1_000, "candidate_sample": 3_000_000},
    }

    def construct(self):
        # Classification has its own output format; build() does the work.
        raise NotImplementedError("ClassificationBuilder overrides build() directly.")

    def build(self) -> None:
        t0 = time.time()
        cfg = self.SIZES[self.size]

        logger.info(
            "Building classification dataset: size=%s, seed=%d, output=%s",
            self.size, self.seed, self.output_dir,
        )

        mappings = load_mesh_mappings()
        uid_to_d2 = mappings.uid_to_depth2

        # 1) Reservoir-sample candidate articles with usable abstract + ≥1 major UID.
        logger.info("Sampling %d candidates from Parquet", cfg["candidate_sample"])
        rows = self.con.execute(f"""
            SELECT pmid, abstract_text,
                   list_transform(
                       list_filter(mesh_descriptors, x -> x.is_major),
                       x -> x.uid
                   ) AS major_uids
            FROM '{self.parquet_path}'
            WHERE length(abstract_text) >= 150
              AND len(list_filter(mesh_descriptors, x -> x.is_major)) >= 1
            USING SAMPLE {cfg["candidate_sample"]} ROWS (reservoir, {self.seed})
        """).fetchall()
        logger.info("Sampled %d candidates", len(rows))

        # 2) Apply P2: major-only majority vote at depth-2, support ≥ 0.60.
        labeled: list[tuple[str, str, str]] = []
        for pmid, abstract, uids in rows:
            if not uids:
                continue
            label, support, _total = majority_label_with_margin(list(uids), uid_to_d2)
            if label and support >= SUPPORT_THRESHOLD:
                labeled.append((pmid, abstract, label))
        logger.info(
            "After P2 (support >= %.2f): %d labeled articles",
            SUPPORT_THRESHOLD, len(labeled),
        )

        # 3) Pick the N_CLASSES largest depth-2 labels.
        label_counts = Counter(l for _, _, l in labeled)
        top_labels = [l for l, _ in label_counts.most_common(N_CLASSES)]
        logger.info(
            "Top %d classes (sizes: max=%d, min=%d): %s ...",
            len(top_labels),
            label_counts[top_labels[0]] if top_labels else 0,
            label_counts[top_labels[-1]] if top_labels else 0,
            top_labels[:5],
        )

        top_set = set(top_labels)
        eligible = [(p, a, l) for p, a, l in labeled if l in top_set]
        by_class: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for row in eligible:
            by_class[row[2]].append(row)

        # 4) Stratified per-class capping with deterministic ordering by hash(pmid).
        per_class = cfg["per_class"]
        sampled: list[tuple[str, str, str]] = []
        for label in top_labels:
            articles = by_class[label]
            articles.sort(key=lambda r: hash((r[0], str(self.seed), "cls")))
            sampled.extend(articles[:per_class])
        logger.info(
            "Sampled %d articles after per-class capping (target %d/class)",
            len(sampled), per_class,
        )

        # 5) Stratified 80/20 train/test split per class.
        train: list[tuple[str, str, str]] = []
        test: list[tuple[str, str, str]] = []
        for label in top_labels:
            articles = [r for r in sampled if r[2] == label]
            articles.sort(key=lambda r: hash((r[0], str(self.seed), "split")))
            n_train = int(round(len(articles) * TRAIN_FRAC))
            train.extend(articles[:n_train])
            test.extend(articles[n_train:])
        logger.info(
            "Split: %d train, %d test across %d classes",
            len(train), len(test), len(top_labels),
        )

        # 6) Write JSONL outputs.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.output_dir / "train.jsonl"
        test_path = self.output_dir / "test.jsonl"
        n_tr = write_jsonl(
            train_path,
            ({"_id": p, "text": a, "label": l} for p, a, l in train),
        )
        n_te = write_jsonl(
            test_path,
            ({"_id": p, "text": a, "label": l} for p, a, l in test),
        )
        logger.info("Wrote %d train rows -> %s", n_tr, train_path)
        logger.info("Wrote %d test rows -> %s", n_te, test_path)

        labels_meta = {
            "n_classes": len(top_labels),
            "labels": top_labels,
            "policy": (
                f"P2 — major-only majority at depth-2, "
                f"support >= {SUPPORT_THRESHOLD}"
            ),
            "per_class_target": per_class,
            "train_frac": TRAIN_FRAC,
        }
        with open(self.output_dir / "labels.json", "w") as f:
            json.dump(labels_meta, f, indent=2)

        logger.info("Done in %.1fs", time.time() - t0)
        self.con.close()
