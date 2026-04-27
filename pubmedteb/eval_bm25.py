"""BM25 lexical baseline evaluator for PubMedTEB tasks.

Given a built dataset (``queries.jsonl`` + ``corpus.jsonl`` + ``qrels.tsv``),
builds an ephemeral FTS index over the corpus, scores every query, and
computes MTEB-parity retrieval metrics via pytrec_eval. Writes the result
to ``{output_dir}/bm25.json`` using the same top-level schema as MTEB runs
so the downstream analysis notebook treats BM25 like any other model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytrec_eval

from pubmedteb.infra.bm25_retriever import bm25_retrieve

logger = logging.getLogger(__name__)

# pytrec_eval metric spec → the "at K" values we surface
K_VALUES = [1, 3, 5, 10, 20, 100, 1000]
METRIC_SPEC = {
    "ndcg_cut." + ",".join(str(k) for k in K_VALUES),
    "map_cut." + ",".join(str(k) for k in K_VALUES),
    "recall." + ",".join(str(k) for k in K_VALUES),
    "P." + ",".join(str(k) for k in K_VALUES),
    "recip_rank",
}


def _load_jsonl_dict(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            out[obj["_id"]] = obj["text"]
    return out


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def _load_top_ranked(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            out[obj["qid"]] = set(obj["docids"])
    return out


def _rename_metric(key: str) -> str:
    """pytrec_eval key → MTEB-style key (e.g. ``ndcg_cut_10`` → ``ndcg_at_10``)."""
    if key.startswith("ndcg_cut_"):
        return "ndcg_at_" + key[len("ndcg_cut_"):]
    if key.startswith("map_cut_"):
        return "map_at_" + key[len("map_cut_"):]
    if key.startswith("recall_"):
        return "recall_at_" + key[len("recall_"):]
    if key.startswith("P_"):
        return "precision_at_" + key[len("P_"):]
    if key == "recip_rank":
        return "mrr_at_10"
    return key


def evaluate_bm25(
    dataset_dir: Path,
    task_name: str,
    output_dir: Path = Path("results"),
    top_k: int = 1000,
) -> dict[str, float]:
    """Run BM25 over a built dataset and save MTEB-style results.

    Args:
        dataset_dir: Directory containing ``queries.jsonl``, ``corpus.jsonl``,
            ``qrels.tsv`` (e.g. ``datasets/pubmed_citation_retrieval``).
        task_name: MTEB task name, used as the JSON ``task_name`` field
            (e.g. ``"PubMedCitationRetrieval"``).
        output_dir: Where to write ``bm25.json``.
        top_k: BM25 cutoff per query.

    Returns:
        Aggregated metrics keyed in MTEB style (``ndcg_at_10`` etc).
    """
    dataset_dir = Path(dataset_dir)
    queries = _load_jsonl_dict(dataset_dir / "queries.jsonl")
    corpus = _load_jsonl_dict(dataset_dir / "corpus.jsonl")
    qrels = _load_qrels(dataset_dir / "qrels.tsv")

    top_ranked_path = dataset_dir / "top_ranked.jsonl"
    top_ranked: dict[str, set[str]] | None = None
    if top_ranked_path.exists():
        top_ranked = _load_top_ranked(top_ranked_path)
        logger.info(
            "Loaded %d queries, %d corpus docs, %d qrels, %d top_ranked lists "
            "(reranking mode — scoring restricted to each query's candidates)",
            len(queries), len(corpus), len(qrels), len(top_ranked),
        )
    else:
        logger.info(
            "Loaded %d queries, %d corpus docs, %d qrels",
            len(queries), len(corpus), len(qrels),
        )

    run = bm25_retrieve(queries, corpus, top_k=top_k)

    if top_ranked is not None:
        run = {
            qid: {d: s for d, s in scores.items() if d in top_ranked.get(qid, set())}
            for qid, scores in run.items()
        }

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, METRIC_SPEC)
    per_query = evaluator.evaluate(run)
    if not per_query:
        raise RuntimeError("pytrec_eval returned no scores — check qrels/run keys.")

    # Aggregate (mean over queries)
    sums: dict[str, float] = {}
    for qid_scores in per_query.values():
        for metric, value in qid_scores.items():
            sums[metric] = sums.get(metric, 0.0) + value
    n = len(per_query)
    aggregated = {_rename_metric(k): round(v / n, 6) for k, v in sums.items()}
    aggregated["main_score"] = aggregated.get("ndcg_at_10", 0.0)
    aggregated["accuracy"] = aggregated.get("precision_at_1", 0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset_revision": "1.0.0",
        "task_name": task_name,
        "model_name": "bm25",
        "scores": {"test": [aggregated]},
    }
    out_path = output_dir / "bm25.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("BM25 results saved to %s", out_path)
    return aggregated
