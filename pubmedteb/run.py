"""CLI for PubMedTEB dataset building and model evaluation.

Usage:
    uv run python -m pubmedteb.run build --task title-retrieval --size mini
    uv run python -m pubmedteb.run evaluate --task title-retrieval --model pubmedbert e5-large
    uv run python -m pubmedteb.run build-infra --component all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pubmedteb",
        description="PubMedTEB: build benchmark datasets and evaluate embedding models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ────────────────────────────────────────────────────────
    build_p = sub.add_parser("build", help="Build a benchmark dataset from PubMed Parquet")
    build_p.add_argument(
        "--task", required=True,
        help="Task to build (e.g., title-retrieval)",
    )
    build_p.add_argument(
        "--size", default="mini", choices=["mini", "full"],
        help="Dataset size preset (default: mini)",
    )
    build_p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    build_p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: datasets/pubmed_{task})",
    )
    build_p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # ── evaluate ─────────────────────────────────────────────────────
    eval_p = sub.add_parser("evaluate", help="Evaluate models on a benchmark task")
    eval_p.add_argument(
        "--task", required=True,
        help="Task to evaluate (e.g., PubMedTitleRetrieval)",
    )
    eval_p.add_argument(
        "--model", nargs="+", default=None,
        help="Model names from registry",
    )
    eval_p.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to a local model checkpoint (this flag bypasses the registry)",
    )
    eval_p.add_argument(
        "--batch-size", type=int, default=32,
        help="Encoding batch size (default: 32)",
    )
    eval_p.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Results output directory (default: results/)",
    )
    eval_p.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Override dataset directory for the task",
    )
    eval_p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # ── evaluate-bm25 ────────────────────────────────────────────────
    bm25_p = sub.add_parser(
        "evaluate-bm25",
        help="Run the BM25 lexical baseline on a built dataset",
    )
    bm25_p.add_argument(
        "--task", required=True,
        help="Task name (e.g., PubMedCitationRetrieval)",
    )
    bm25_p.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Override dataset directory (default: datasets/pubmed_{slug})",
    )
    bm25_p.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Results output directory (default: results/)",
    )
    bm25_p.add_argument(
        "--top-k", type=int, default=1000,
        help="BM25 cutoff per query (default: 1000)",
    )
    bm25_p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # ── build-infra ──────────────────────────────────────────────────
    infra_p = sub.add_parser(
        "build-infra",
        help="Build infrastructure indices (MeSH depth-2, citation graph, BM25)",
    )
    infra_p.add_argument(
        "--component",
        choices=["mesh", "citations", "bm25", "all"],
        default="all",
        help="Which infrastructure component to build (default: all)",
    )
    infra_p.add_argument(
        "--desc-xml", type=Path, default=None,
        help="Path to desc2025.xml (required for mesh component)",
    )
    infra_p.add_argument(
        "--sample-size", type=int, default=None,
        help="BM25 index sample size (default: full corpus)",
    )
    infra_p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser


def _run_build(args: argparse.Namespace) -> int:
    """Execute the build subcommand."""
    from pubmedteb.builders import get_builder

    builder = get_builder(
        args.task,
        output_dir=args.output_dir,
        seed=args.seed,
        size=args.size,
    )
    builder.build()
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    """Execute the evaluate subcommand."""
    import mteb
    import torch

    from pubmedteb.models import MODELS, get_model
    from pubmedteb.tasks import get_task

    # Resolve which models to evaluate
    model_names: list[str] = []
    if args.model:
        model_names = args.model
    elif args.model_path:
        model_names = [str(args.model_path)]
    else:
        logger.error("Provide --model or --model-path")
        return 1

    # Instantiate the task
    task_kwargs = {}
    if args.dataset_dir:
        task_kwargs["dataset_dir"] = args.dataset_dir
    task = get_task(args.task, **task_kwargs)

    for model_name in model_names:
        logger.info("Evaluating model: %s", model_name)
        model = get_model(model_name)

        encode_kwargs: dict = {"batch_size": args.batch_size}
        if model_name in MODELS and "encode_kwargs" in MODELS[model_name]:
            encode_kwargs.update(MODELS[model_name]["encode_kwargs"])

        results = mteb.evaluate(
            model,
            tasks=[task],
            encode_kwargs=encode_kwargs,
        )

        # Save results to disk
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result_file = args.output_dir / f"{model_name}.json"
        results.to_disk(result_file)
        logger.info("Results saved to %s", result_file)

        _print_summary(model_name, results)

        # Free GPU memory before loading the next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return 0


def _run_evaluate_bm25(args: argparse.Namespace) -> int:
    """Execute the evaluate-bm25 subcommand."""
    from pubmedteb.eval_bm25 import evaluate_bm25

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        # Map MTEB task name (e.g. PubMedCitationRetrieval) → datasets/pubmed_citation_retrieval
        import re
        base = args.task[len("PubMed"):] if args.task.startswith("PubMed") else args.task
        slug = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()
        dataset_dir = Path("datasets") / f"pubmed_{slug}"

    metrics = evaluate_bm25(
        dataset_dir=dataset_dir,
        task_name=args.task,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    print(f"\n{'='*60}")
    print("Model: bm25")
    print(f"{'='*60}\n")
    print(f"Task: {args.task}")
    print(f"  test: main_score = {metrics.get('main_score', 0.0):.4f}")
    for key in [
        "ndcg_at_10", "ndcg_at_100",
        "map_at_10", "mrr_at_10",
        "recall_at_10", "recall_at_100",
        "accuracy",
    ]:
        if key in metrics:
            print(f"    {key}: {metrics[key]:.4f}")
    return 0


def _run_build_infra(args: argparse.Namespace) -> int:
    """Execute the build-infra subcommand."""
    components = (
        ["mesh", "citations", "bm25"] if args.component == "all"
        else [args.component]
    )

    for component in components:
        if component == "mesh":
            from preprocessing.mesh_categories import build_all_mappings, save_mapping

            desc_xml = args.desc_xml
            if desc_xml is None:
                # Try common location
                desc_xml = Path("/gpfs01/berens/data/data/pubmed/desc2025.xml")
            if not desc_xml.exists():
                logger.error("desc2025.xml not found at %s — use --desc-xml", desc_xml)
                return 1

            cache_path = (
                Path(__file__).resolve().parent.parent
                / "preprocessing" / "data" / "mesh_uid_categories.json"
            )
            (
                uid_to_categories,
                depth2_names,
                uid_to_subcategories,
                depth3_names,
                uid_to_subsubcategories,
            ) = build_all_mappings(desc_xml)
            save_mapping(
                uid_to_categories,
                cache_path,
                depth2_names,
                uid_to_subcategories,
                depth3_names,
                uid_to_subsubcategories,
            )
            print(
                f"MeSH: {len(uid_to_categories)} descriptors, "
                f"{len(depth2_names)} depth-2, "
                f"{len(depth3_names)} depth-3 -> {cache_path}"
            )

        elif component == "citations":
            from pubmedteb.infra.citation_graph import build_citation_graph

            stats = build_citation_graph()
            print(
                f"Citations: {stats['articles_with_citations']} citing articles, "
                f"{stats['total_forward_edges']} edges, "
                f"{stats['distinct_cited_pmids']} distinct cited PMIDs "
                f"({stats['build_time_seconds']}s)"
            )

        elif component == "bm25":
            from pubmedteb.infra.bm25_index import build_bm25_index

            db_path = build_bm25_index(sample_size=args.sample_size)
            print(f"BM25: index built at {db_path}")

    return 0


def _print_summary(model_name: str, results) -> None:
    """Print a concise results summary to stdout."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    for task_result in results.task_results:
        print(f"\nTask: {task_result.task_name}")
        for split, score_list in task_result.scores.items():
            for score_set in score_list:
                main = score_set.get("main_score")
                if isinstance(main, (int, float)):
                    print(f"  {split}: main_score = {main:.4f}")
                else:
                    print(f"  {split}: main_score = {main}")
                for key in [
                    "ndcg_at_10", "ndcg_at_100",
                    "map_at_10", "mrr_at_10",
                    "recall_at_10", "recall_at_100",
                    "accuracy",
                ]:
                    if key in score_set:
                        print(f"    {key}: {score_set[key]:.4f}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = getattr(args, "log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "build":
        return _run_build(args)
    elif args.command == "evaluate":
        return _run_evaluate(args)
    elif args.command == "evaluate-bm25":
        return _run_evaluate_bm25(args)
    elif args.command == "build-infra":
        return _run_build_infra(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
