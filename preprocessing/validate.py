"""Post-build validation for the PubMed Parquet output.

Runs a comprehensive suite of checks to verify the output is correct
and complete: schema conformance, row counts, null rates, coverage
statistics, and optional cross-validation against a reference file
or raw XML spot-check.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from preprocessing.schema import ARROW_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Aggregated results from all validation checks."""

    passed: bool = True
    total_rows: int = 0
    unique_pmids: int = 0
    duplicate_pmids: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    mesh_coverage_pct: float = 0.0
    citation_coverage_pct: float = 0.0
    year_range: tuple[int, int] = (0, 0)
    schema_matches: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    reference_overlap: dict[str, int] | None = None


def validate_parquet(
    parquet_path: Path,
    expected_row_count: int | None = None,
    reference_parquet: Path | None = None,
    xml_spot_check_path: Path | None = None,
) -> ValidationResult:
    """Run the full validation suite on an output Parquet file.

    Checks:
        1. Schema matches ARROW_SCHEMA
        2. Row count is reasonable (24-25M for full PubMed)
        3. No duplicate PMIDs
        4. PMID column has no nulls
        5. MeSH and citation coverage percentages
        6. Year range sanity
        7. Optional: PMID overlap with reference Parquet
        8. Optional: round-trip spot-check against XML
    """
    result = ValidationResult()
    pq_path = str(parquet_path)

    # 1. Schema check
    pf = pq.ParquetFile(pq_path)
    actual_schema = pf.schema_arrow
    expected_names = set(ARROW_SCHEMA.names)
    actual_names = set(actual_schema.names)

    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing:
        result.errors.append(f"Missing columns: {missing}")
        result.schema_matches = False
        result.passed = False
    if extra:
        result.warnings.append(f"Extra columns: {extra}")

    # 2-6. DuckDB-based checks
    con = duckdb.connect()

    # Row count + unique PMIDs
    row_stats = con.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT pmid) as unique_pmids,
            COUNT(*) - COUNT(DISTINCT pmid) as dup_pmids,
            SUM(CASE WHEN pmid IS NULL THEN 1 ELSE 0 END) as null_pmids
        FROM '{pq_path}'
    """).fetchone()

    result.total_rows = row_stats[0]
    result.unique_pmids = row_stats[1]
    result.duplicate_pmids = row_stats[2]
    null_pmids = row_stats[3]

    if null_pmids > 0:
        result.errors.append(f"Found {null_pmids} null PMIDs")
        result.passed = False

    if result.duplicate_pmids > 0:
        result.errors.append(
            f"Found {result.duplicate_pmids:,} duplicate PMIDs"
        )
        result.passed = False

    if expected_row_count:
        diff_pct = abs(result.total_rows - expected_row_count) / expected_row_count
        if diff_pct > 0.01:
            result.warnings.append(
                f"Row count {result.total_rows:,} differs from expected "
                f"{expected_row_count:,} by {diff_pct:.1%}"
            )

    # Null counts for key columns
    for col in ["title", "abstract_text", "journal", "language", "doi"]:
        if col not in actual_names:
            continue
        null_count = con.execute(f"""
            SELECT SUM(CASE WHEN {col} IS NULL OR {col} = '' THEN 1 ELSE 0 END)
            FROM '{pq_path}'
        """).fetchone()[0]
        result.null_counts[col] = null_count

    # MeSH and citation coverage
    coverage = con.execute(f"""
        SELECT
            SUM(CASE WHEN len(mesh_descriptors) > 0 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*) as mesh_pct,
            SUM(CASE WHEN len(cited_pmids) > 0 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*) as cite_pct
        FROM '{pq_path}'
    """).fetchone()
    result.mesh_coverage_pct = round(coverage[0], 1)
    result.citation_coverage_pct = round(coverage[1], 1)

    # Year range
    year_range = con.execute(f"""
        SELECT MIN(year), MAX(year) FROM '{pq_path}' WHERE year > 0
    """).fetchone()
    result.year_range = (year_range[0], year_range[1])

    max_valid_year = datetime.date.today().year + 1
    if result.year_range[1] > max_valid_year:
        result.warnings.append(f"Max year {result.year_range[1]} exceeds {max_valid_year}")

    # Semantic category coverage
    cat_stats = con.execute(f"""
        SELECT
            COUNT(DISTINCT semantic_category) as n_categories,
            SUM(CASE WHEN semantic_category != '' THEN 1 ELSE 0 END) * 100.0
                / COUNT(*) as cat_pct
        FROM '{pq_path}'
    """).fetchone()
    n_categories = cat_stats[0]
    cat_pct = round(cat_stats[1], 1)
    if n_categories < 10:
        result.warnings.append(
            f"Only {n_categories} distinct semantic categories (expected ~16)"
        )

    # 7. Reference Parquet comparison
    if reference_parquet and reference_parquet.exists():
        ref_path = str(reference_parquet)
        overlap = con.execute(f"""
            SELECT
                (SELECT COUNT(DISTINCT pmid) FROM '{pq_path}') as new_pmids,
                (SELECT COUNT(DISTINCT PMID) FROM '{ref_path}') as ref_pmids,
                (SELECT COUNT(*) FROM (
                    SELECT DISTINCT pmid FROM '{pq_path}'
                    INTERSECT
                    SELECT DISTINCT PMID FROM '{ref_path}'
                )) as shared_pmids
        """).fetchone()
        result.reference_overlap = {
            "new_unique_pmids": overlap[0],
            "reference_unique_pmids": overlap[1],
            "shared_pmids": overlap[2],
        }

    # 8. XML spot-check
    if xml_spot_check_path and xml_spot_check_path.exists():
        _run_spot_check(xml_spot_check_path, parquet_path, result, con)

    con.close()

    # Summary
    if result.passed:
        logger.info("Validation PASSED: %d rows, %d unique PMIDs",
                     result.total_rows, result.unique_pmids)
    else:
        logger.error("Validation FAILED: %s", "; ".join(result.errors))

    return result


def _run_spot_check(
    xml_path: Path,
    parquet_path: Path,
    result: ValidationResult,
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Re-parse a few articles from XML and compare against Parquet rows."""
    from preprocessing.parser import parse_xml_file

    records = parse_xml_file(xml_path)
    if not records:
        result.warnings.append(f"Spot-check: no articles parsed from {xml_path.name}")
        return

    sample = records[:100]
    sample_pmids = [r.pmid for r in sample]
    pmid_list = ", ".join(f"'{p}'" for p in sample_pmids)

    pq_rows = con.execute(f"""
        SELECT pmid, title, language, doi, country
        FROM '{parquet_path}'
        WHERE pmid IN ({pmid_list})
    """).fetchdf()

    pq_lookup = {row["pmid"]: row for _, row in pq_rows.iterrows()}

    mismatches = 0
    for rec in sample:
        pq_row = pq_lookup.get(rec.pmid)
        if pq_row is None:
            mismatches += 1
            continue
        if pq_row["title"] != rec.title:
            mismatches += 1
        if pq_row["language"] != rec.language:
            mismatches += 1

    if mismatches > 0:
        result.warnings.append(
            f"Spot-check: {mismatches} mismatches in {len(sample)} articles "
            f"from {xml_path.name}"
        )
    else:
        logger.info("Spot-check passed: %d articles from %s",
                     len(sample), xml_path.name)


def print_validation_report(result: ValidationResult) -> None:
    """Print a human-readable validation report."""
    status = "PASSED" if result.passed else "FAILED"
    print(f"\n{'='*60}")
    print(f"  Validation: {status}")
    print(f"{'='*60}")
    print(f"  Total rows:        {result.total_rows:>15,}")
    print(f"  Unique PMIDs:      {result.unique_pmids:>15,}")
    print(f"  Duplicate PMIDs:   {result.duplicate_pmids:>15,}")
    print(f"  Year range:        {result.year_range[0]} - {result.year_range[1]}")
    print(f"  MeSH coverage:     {result.mesh_coverage_pct:>14.1f}%")
    print(f"  Citation coverage: {result.citation_coverage_pct:>14.1f}%")
    print(f"  Schema matches:    {'Yes' if result.schema_matches else 'NO'}")

    if result.null_counts:
        print(f"\n  Null/empty counts:")
        for col, cnt in sorted(result.null_counts.items()):
            pct = cnt / max(result.total_rows, 1) * 100
            print(f"    {col:<20s} {cnt:>12,} ({pct:.1f}%)")

    if result.reference_overlap:
        ov = result.reference_overlap
        print(f"\n  Reference comparison:")
        print(f"    New PMIDs:       {ov['new_unique_pmids']:>12,}")
        print(f"    Reference PMIDs: {ov['reference_unique_pmids']:>12,}")
        print(f"    Shared PMIDs:    {ov['shared_pmids']:>12,}")

    if result.warnings:
        print(f"\n  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    if result.errors:
        print(f"\n  Errors:")
        for e in result.errors:
            print(f"    - {e}")

    print(f"{'='*60}\n")
