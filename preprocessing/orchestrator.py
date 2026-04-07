"""Multiprocessing orchestrator for the PubMed ETL pipeline.

Coordinates parallel XML parsing across worker processes and sequential
Parquet writing in the main process. Each worker handles one XML file
end-to-end, returning parsed ArticleRecords to the main process for
transformation and writing.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa

from preprocessing.mesh_categories import init_mapping
from preprocessing.parser import parse_xml_file
from preprocessing.schema import ArticleRecord, records_to_arrow_batch
from preprocessing.writer import ParquetBatchWriter

logger = logging.getLogger(__name__)


def _worker_fn(xml_path_str: str) -> tuple[str, list[ArticleRecord]]:
    """Worker function: parse one XML file and return article records.

    Runs in a subprocess. Accepts a string path (for pickling) and
    returns the filename plus the list of parsed records.
    """
    path = Path(xml_path_str)
    records = parse_xml_file(path)
    return path.name, records


def discover_xml_files(xml_dir: Path) -> list[Path]:
    """Find all PubMed XML files in the given directory.

    Filters to files matching pubmed25n*.xml (excludes .md5, .gz).
    Returns sorted by name for deterministic ordering.
    """
    xml_files = sorted(
        p for p in xml_dir.iterdir()
        if p.suffix == ".xml" and p.name.startswith("pubmed25n")
    )
    logger.info("Discovered %d XML files in %s", len(xml_files), xml_dir)
    return xml_files


def _deduplicate_parquet(parquet_path: Path) -> int:
    """Remove duplicate PMIDs from a Parquet file, keeping the last occurrence.

    PubMed Baseline distributes revised articles in later-numbered files.
    Since files are processed in sorted order, later rows correspond to
    newer revisions — we keep the last row per PMID.

    Returns the number of duplicate rows removed.
    """
    con = duckdb.connect()
    dup_count = con.execute(f"""
        SELECT COUNT(*) - COUNT(DISTINCT pmid) FROM '{parquet_path}'
    """).fetchone()[0]

    if dup_count == 0:
        con.close()
        return 0

    tmp = parquet_path.with_suffix(".dedup.parquet")
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY pmid ORDER BY rowid DESC) as rn
                FROM '{parquet_path}'
            ) WHERE rn = 1
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    tmp.rename(parquet_path)
    return dup_count


def run_pipeline(
    xml_dir: Path,
    output_path: Path,
    mesh_desc_path: Path | None = None,
    mesh_cache_path: Path | None = None,
    n_workers: int | None = None,
    file_limit: int | None = None,
    resume_from: int = 0,
) -> dict:
    """Execute the full ETL pipeline.

    Steps:
        1. Discover XML files
        2. Initialize MeSH mapping
        3. Submit files to worker pool
        4. Collect and write Arrow batches (in completion order)
        5. Log progress and timing

    Args:
        xml_dir: Directory containing PubMed Baseline XML files.
        output_path: Destination Parquet file path.
        mesh_desc_path: Path to NLM desc2025.xml (optional if cache exists).
        mesh_cache_path: Path to JSON cache of the MeSH mapping.
        n_workers: Number of parallel workers (default: cpu_count - 1).
        file_limit: Process only the first N files (for testing).
        resume_from: Skip the first N files (for resuming).

    Returns:
        Dict with pipeline statistics.
    """
    workers = n_workers or max(1, mp.cpu_count() - 1)

    # Step 1: Discover files
    xml_files = discover_xml_files(xml_dir)
    if resume_from:
        xml_files = xml_files[resume_from:]
        logger.info("Resuming from file %d", resume_from)
    if file_limit:
        xml_files = xml_files[:file_limit]
        logger.info("Limited to %d files", file_limit)

    total_files = len(xml_files)
    logger.info("Pipeline starting: %d files, %d workers", total_files, workers)

    # Step 2: Initialize MeSH mapping
    uid_to_categories = init_mapping(mesh_desc_path, mesh_cache_path)
    logger.info("MeSH mapping ready: %d descriptors", len(uid_to_categories))

    # Step 3-4: Parse in parallel, write sequentially
    stats: dict = {
        "total_articles": 0,
        "total_files": 0,
        "total_files_expected": total_files,
        "errors": [],
        "start_time": time.time(),
    }

    with ParquetBatchWriter(output_path) as writer:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_worker_fn, str(path)): path
                for path in xml_files
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    filename, records = future.result()
                    batch = records_to_arrow_batch(records, uid_to_categories)
                    writer.write_batch(batch)

                    stats["total_articles"] += len(records)
                    stats["total_files"] += 1

                    elapsed = time.time() - stats["start_time"]
                    rate = stats["total_articles"] / max(elapsed, 0.1)
                    pct = stats["total_files"] / total_files * 100
                    logger.info(
                        "[%3.0f%%] %s: %d articles (total: %d | %.0f art/s)",
                        pct, filename, len(records),
                        stats["total_articles"], rate,
                    )

                except Exception as e:
                    logger.error("Failed processing %s: %s", path.name, e)
                    stats["errors"].append({"file": path.name, "error": str(e)})

    # Step 5: Deduplicate PMIDs (PubMed Baseline distributes revised
    # articles in later files, so the same PMID can appear multiple times)
    removed = _deduplicate_parquet(output_path)
    if removed:
        logger.info("Deduplication removed %d rows", removed)
        stats["total_articles"] -= removed
        stats["duplicates_removed"] = removed

    # Step 6: Finalize
    stats["elapsed_seconds"] = time.time() - stats["start_time"]
    stats["articles_per_second"] = (
        stats["total_articles"] / max(stats["elapsed_seconds"], 1)
    )

    logger.info(
        "Pipeline complete: %d articles from %d files in %.1f seconds (%.0f art/s)",
        stats["total_articles"],
        stats["total_files"],
        stats["elapsed_seconds"],
        stats["articles_per_second"],
    )

    if stats["errors"]:
        logger.warning("%d files had errors", len(stats["errors"]))

    return stats


def write_run_manifest(stats: dict, output_path: Path) -> None:
    """Write a JSON manifest documenting the pipeline run.

    Includes timestamp, git commit (if available), article/file counts,
    duration, and any errors encountered.
    """
    git_hash = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "pipeline_version": "0.1.0",
        "total_articles": stats["total_articles"],
        "total_files": stats["total_files"],
        "elapsed_seconds": round(stats["elapsed_seconds"], 1),
        "articles_per_second": round(stats["articles_per_second"], 1),
        "errors": stats["errors"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", output_path)
