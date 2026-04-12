"""Base class for PubMedTEB dataset builders.

Each builder connects to the filtered PubMed Parquet via DuckDB,
constructs (queries, corpus, qrels) data, and writes to disk
as JSONL/TSV files consumable by MTEB task wrappers.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

PARQUET_PATH = Path(
    "/gpfs01/berens/data/data/pubmed_processed/pubmed_teb_filtered.parquet"
)


class DatasetBuilder(ABC):
    """Abstract base class for all PubMedTEB dataset builders.

    Subclasses implement ``construct()`` to produce queries, corpus, and
    relevance judgments. The base class handles DuckDB connection management
    and writing the standard output files.

    Args:
        output_dir: Directory to write corpus.jsonl, queries.jsonl, qrels.tsv.
        seed: Random seed for reproducible sampling.
        size: Dataset size preset ("mini" or "full").
        parquet_path: Override path to the filtered Parquet file.
    """

    SIZES: dict[str, dict] = {}

    def __init__(
        self,
        output_dir: Path,
        seed: int = 42,
        size: str = "full",
        parquet_path: Path | None = None,
    ) -> None:
        if size not in self.SIZES:
            raise ValueError(f"Unknown size '{size}'. Choose from: {list(self.SIZES)}")
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.size = size
        self.parquet_path = parquet_path or PARQUET_PATH
        self.con = duckdb.connect()

    def build(self) -> None:
        """Build the dataset: construct data, write to disk, log stats."""
        t0 = time.time()
        logger.info(
            "Building dataset: size=%s, seed=%d, output=%s",
            self.size, self.seed, self.output_dir,
        )

        queries, corpus, qrels = self.construct()
        self.write(queries, corpus, qrels)

        elapsed = time.time() - t0
        logger.info(
            "Done in %.1fs — %d queries, %d corpus docs, %d qrels",
            elapsed, len(queries), len(corpus),
            sum(len(v) for v in qrels.values()),
        )
        self.con.close()

    @abstractmethod
    def construct(
        self,
    ) -> tuple[dict[str, str], dict[str, dict], dict[str, dict[str, int]]]:
        """Build queries, corpus, and relevance judgments.

        Returns:
            queries: ``{query_id: query_text}``
            corpus: ``{doc_id: {"text": str}}``
            qrels: ``{query_id: {doc_id: relevance_score}}``
        """
        ...

    def write(
        self,
        queries: dict[str, str],
        corpus: dict[str, dict],
        qrels: dict[str, dict[str, int]],
    ) -> None:
        """Write queries, corpus, and qrels to standard files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        queries_path = self.output_dir / "queries.jsonl"
        with open(queries_path, "w") as f:
            for qid, text in queries.items():
                f.write(json.dumps({"_id": qid, "text": text}) + "\n")
        logger.info("Wrote %d queries to %s", len(queries), queries_path)

        corpus_path = self.output_dir / "corpus.jsonl"
        with open(corpus_path, "w") as f:
            for doc_id, doc in corpus.items():
                f.write(json.dumps({"_id": doc_id, "text": doc["text"]}) + "\n")
        logger.info("Wrote %d corpus docs to %s", len(corpus), corpus_path)

        qrels_path = self.output_dir / "qrels.tsv"
        with open(qrels_path, "w") as f:
            for qid, docs in qrels.items():
                for doc_id, score in docs.items():
                    f.write(f"{qid}\t{doc_id}\t{score}\n")
        logger.info("Wrote qrels to %s", qrels_path)

    def query(self, sql: str) -> list[tuple]:
        """Execute a SQL query against the Parquet file.

        Use ``{parquet}`` as placeholder for the Parquet path in the SQL.
        """
        sql = sql.replace("{parquet}", f"'{self.parquet_path}'")
        return self.con.execute(sql).fetchall()
