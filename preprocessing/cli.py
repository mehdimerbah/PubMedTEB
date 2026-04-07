"""Command-line interface for the PubMed ETL pipeline.

Provides three subcommands:
    run       — Execute the full ETL pipeline (XML -> Parquet)
    validate  — Validate an existing Parquet output file
    mesh      — Build or inspect the MeSH UID -> category mapping
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pubmed-etl",
        description="PubMed XML to Parquet preprocessing pipeline for BioMTEB.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run the ETL pipeline")
    run_p.add_argument(
        "--xml-dir", type=Path, required=True,
        help="Directory containing PubMed Baseline XML files",
    )
    run_p.add_argument(
        "--output", type=Path, required=True,
        help="Output Parquet file path",
    )
    run_p.add_argument(
        "--mesh-desc", type=Path, default=None,
        help="Path to NLM desc2025.xml for building MeSH mapping",
    )
    run_p.add_argument(
        "--mesh-cache", type=Path, default=None,
        help="Path to JSON cache of MeSH UID mapping",
    )
    run_p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    run_p.add_argument(
        "--file-limit", type=int, default=None,
        help="Process only first N files (for testing)",
    )
    run_p.add_argument(
        "--resume-from", type=int, default=0,
        help="Skip first N files (for resuming interrupted runs)",
    )
    run_p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # ── validate ─────────────────────────────────────────────────────
    val_p = sub.add_parser("validate", help="Validate output Parquet file")
    val_p.add_argument(
        "--parquet", type=Path, required=True,
        help="Parquet file to validate",
    )
    val_p.add_argument(
        "--reference", type=Path, default=None,
        help="Reference Parquet for PMID overlap comparison",
    )
    val_p.add_argument(
        "--xml-spot-check", type=Path, default=None,
        help="XML file for round-trip spot check",
    )
    val_p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # ── mesh ─────────────────────────────────────────────────────────
    mesh_p = sub.add_parser("mesh", help="Build or inspect MeSH mapping")
    mesh_p.add_argument(
        "--desc-xml", type=Path, required=True,
        help="Path to NLM desc2025.xml",
    )
    mesh_p.add_argument(
        "--output", type=Path, required=True,
        help="Output JSON mapping file",
    )
    mesh_p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = getattr(args, "log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "run":
        from preprocessing.orchestrator import run_pipeline, write_run_manifest

        stats = run_pipeline(
            xml_dir=args.xml_dir,
            output_path=args.output,
            mesh_desc_path=args.mesh_desc,
            mesh_cache_path=args.mesh_cache,
            n_workers=args.workers,
            file_limit=args.file_limit,
            resume_from=args.resume_from,
        )

        # Write manifest alongside the output
        manifest_path = args.output.with_suffix(".manifest.json")
        write_run_manifest(stats, manifest_path)

        if stats["errors"]:
            print(f"\nCompleted with {len(stats['errors'])} errors. See log.")
            return 1
        return 0

    elif args.command == "validate":
        from preprocessing.validate import validate_parquet, print_validation_report

        result = validate_parquet(
            parquet_path=args.parquet,
            reference_parquet=args.reference,
            xml_spot_check_path=args.xml_spot_check,
        )
        print_validation_report(result)
        return 0 if result.passed else 1

    elif args.command == "mesh":
        from preprocessing.mesh_categories import build_uid_to_categories, save_mapping

        mapping = build_uid_to_categories(args.desc_xml)
        save_mapping(mapping, args.output)
        print(f"MeSH mapping: {len(mapping)} descriptors -> {args.output}")
        return 0

    return 1
