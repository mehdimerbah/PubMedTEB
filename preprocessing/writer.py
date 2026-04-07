"""Incremental Parquet writer for the PubMed pipeline.

Wraps pyarrow.parquet.ParquetWriter to write RecordBatches incrementally,
avoiding loading the entire dataset into memory at once.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing.schema import ARROW_SCHEMA

logger = logging.getLogger(__name__)


class ParquetBatchWriter:
    """Incrementally writes Arrow RecordBatches to a Parquet file.

    Usage::

        with ParquetBatchWriter(output_path) as writer:
            writer.write_batch(batch1)
            writer.write_batch(batch2)

    The file uses zstd compression and a configurable row group size
    for efficient column-pruning reads by DuckDB.
    """

    def __init__(
        self,
        output_path: Path,
        compression: str = "zstd",
        row_group_size: int = 100_000,
    ) -> None:
        self._output_path = Path(output_path)
        self._compression = compression
        self._row_group_size = row_group_size
        self._writer: pq.ParquetWriter | None = None
        self._rows_written = 0
        self._batches_written = 0

    def __enter__(self) -> ParquetBatchWriter:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            str(self._output_path),
            schema=ARROW_SCHEMA,
            compression=self._compression,
        )
        logger.info("Opened Parquet writer: %s (compression=%s)",
                     self._output_path, self._compression)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        logger.info(
            "Closed Parquet writer: %d rows in %d batches -> %s",
            self._rows_written,
            self._batches_written,
            self._output_path,
        )

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """Write a single RecordBatch to the Parquet file."""
        if self._writer is None:
            raise RuntimeError("Writer not initialized. Use as context manager.")
        table = pa.Table.from_batches([batch])
        self._writer.write_table(table, row_group_size=self._row_group_size)
        self._rows_written += batch.num_rows
        self._batches_written += 1

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def batches_written(self) -> int:
        return self._batches_written
