"""Build the full MeSH descriptor tree for interactive visualization.

Parses desc2025.xml to extract the complete hierarchical tree structure
(not just top-level branches), optionally merges article counts from
the pipeline Parquet output, and exports a nested JSON suitable for
D3.js sunburst and 3d-force-graph visualizations.

Usage:
    uv run python -m preprocessing.mesh_tree \
        --desc-xml /path/to/desc2025.xml \
        --output outputs/mesh_tree.json \
        [--parquet /path/to/pubmed.parquet]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from preprocessing.mesh_categories import TREE_BRANCH_NAMES

logger = logging.getLogger(__name__)


def build_full_tree(desc_xml_path: Path) -> dict:
    """Parse desc2025.xml and build a nested tree dict.

    Each DescriptorRecord has one or more TreeNumbers (e.g., D03.633.100).
    We build a nested dict where each tree-number segment is a node:

        root -> A (Anatomy) -> A01 (Body Regions) -> A01.047 (Abdomen)

    Descriptors with multiple tree numbers appear in multiple positions
    (poly-hierarchy). The descriptor's name and UID are attached at the
    leaf position corresponding to each tree number.

    Returns:
        A dict with:
        - "tree": nested hierarchy rooted at "MeSH 2025"
        - "descriptors": flat dict of UID -> {name, tree_numbers}
        - "stats": summary statistics
    """
    # Track all descriptors and their tree positions
    descriptors: dict[str, dict] = {}
    # Nested tree structure: code -> node
    tree_nodes: dict[str, dict] = {}

    for _, elem in ET.iterparse(str(desc_xml_path), events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue

        uid_el = elem.find("DescriptorUI")
        name_el = elem.find("DescriptorName/String")
        if uid_el is None or name_el is None:
            elem.clear()
            continue

        uid = uid_el.text.strip()
        name = name_el.text.strip()
        tree_numbers_el = elem.find("TreeNumberList")

        tree_nums = []
        if tree_numbers_el is not None:
            for tn in tree_numbers_el.findall("TreeNumber"):
                if tn.text:
                    tree_nums.append(tn.text.strip())

        descriptors[uid] = {"name": name, "tree_numbers": tree_nums}

        # Insert into tree structure
        for tn in tree_nums:
            _insert_tree_node(tree_nodes, tn, uid, name)

        elem.clear()

    # Build the final nested dict
    root = {
        "name": "MeSH 2025",
        "code": "",
        "uid": "",
        "article_count": 0,
        "subtree_article_count": 0,
        "children": [],
    }

    for branch_code, branch_name in sorted(TREE_BRANCH_NAMES.items()):
        if branch_code in tree_nodes:
            branch_node = _build_subtree(tree_nodes[branch_code], branch_code)
            branch_node["name"] = branch_name
            root["children"].append(branch_node)

    # Extract poly-hierarchy links
    poly_links = _extract_polyhierarchy(descriptors)

    total_paths = sum(len(d["tree_numbers"]) for d in descriptors.values())
    stats = {
        "total_descriptors": len(descriptors),
        "total_tree_paths": total_paths,
        "max_depth": _max_depth(root),
        "branches": len(root["children"]),
    }

    logger.info(
        "Built MeSH tree: %d descriptors, %d tree paths, %d branches, max depth %d",
        stats["total_descriptors"], stats["total_tree_paths"],
        stats["branches"], stats["max_depth"],
    )

    return {
        "tree": root,
        "polyhierarchy_links": poly_links,
        "metadata": {
            "mesh_version": "2025",
            **stats,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
    }


def _insert_tree_node(
    tree_nodes: dict[str, dict],
    tree_number: str,
    uid: str,
    name: str,
) -> None:
    """Insert a descriptor into the tree at the position given by tree_number.

    Tree numbers like "D03.633.100" are split into segments:
    ["D", "D03", "D03.633", "D03.633.100"]. Each segment becomes a node
    in the nested structure.
    """
    parts = tree_number.split(".")
    # Build cumulative codes: D, D03, D03.633, D03.633.100
    # But the first part is the branch letter
    branch = parts[0][0]  # "D" from "D03"

    if branch not in tree_nodes:
        tree_nodes[branch] = {"children": {}, "uid": "", "name": ""}

    current = tree_nodes[branch]
    full_code = parts[0]

    for i, part in enumerate(parts):
        if i == 0:
            full_code = part
        else:
            full_code = full_code + "." + part

        if i < len(parts) - 1:
            # Intermediate node
            if full_code not in current["children"]:
                current["children"][full_code] = {
                    "children": {},
                    "uid": "",
                    "name": "",
                }
            current = current["children"][full_code]
        else:
            # Leaf for this tree number — attach descriptor info
            if full_code not in current["children"]:
                current["children"][full_code] = {
                    "children": {},
                    "uid": uid,
                    "name": name,
                }
            else:
                # Node already exists (created as intermediate); set its info
                current["children"][full_code]["uid"] = uid
                current["children"][full_code]["name"] = name


def _build_subtree(node: dict, code: str) -> dict:
    """Convert the flat-keyed tree structure into a nested children list."""
    result = {
        "name": node.get("name", code),
        "code": code,
        "uid": node.get("uid", ""),
        "article_count": 0,
        "subtree_article_count": 0,
        "children": [],
    }

    for child_code in sorted(node.get("children", {})):
        child_node = node["children"][child_code]
        result["children"].append(_build_subtree(child_node, child_code))

    return result


def _extract_polyhierarchy(descriptors: dict[str, dict]) -> list[dict]:
    """Find descriptors appearing in multiple top-level branches."""
    links = []
    for uid, info in descriptors.items():
        branches = set()
        for tn in info["tree_numbers"]:
            branches.add(tn[0])  # First character = branch letter
        if len(branches) > 1:
            links.append({
                "uid": uid,
                "name": info["name"],
                "positions": info["tree_numbers"],
                "branches": sorted(branches),
            })
    logger.info("Found %d poly-hierarchy descriptors (multi-branch)", len(links))
    return links


def _max_depth(node: dict, depth: int = 0) -> int:
    """Find the maximum depth of the tree."""
    if not node.get("children"):
        return depth
    return max(_max_depth(child, depth + 1) for child in node["children"])


def merge_article_counts(data: dict, parquet_path: Path) -> None:
    """Merge article counts from the Parquet file into the tree.

    Queries the Parquet to get per-MeSH-UID article counts, then walks
    the tree to set article_count on each node. Parent nodes accumulate
    the sum of their children's counts.
    """
    import duckdb

    con = duckdb.connect()
    counts = con.execute(f"""
        SELECT uid, COUNT(*) as cnt
        FROM (
            SELECT UNNEST(mesh_descriptors).uid as uid
            FROM '{parquet_path}'
        )
        GROUP BY uid
    """).fetchdf()

    unique_articles = con.execute(f"""
        SELECT COUNT(*) as cnt FROM '{parquet_path}'
    """).fetchone()[0]
    con.close()

    uid_counts = dict(zip(counts["uid"], counts["cnt"]))
    logger.info("Loaded article counts for %d MeSH UIDs", len(uid_counts))

    # Walk the tree and set counts
    _set_counts_recursive(data["tree"], uid_counts)

    # Update metadata
    total_with_counts = sum(1 for c in uid_counts.values() if c > 0)
    data["metadata"]["uids_with_articles"] = total_with_counts
    data["metadata"]["unique_articles"] = int(unique_articles)
    data["metadata"]["corpus_articles"] = int(counts["cnt"].sum())


def _set_counts_recursive(node: dict, uid_counts: dict[str, int]) -> int:
    """Set article_count and subtree_article_count on each node.

    - article_count: articles tagged with this exact descriptor UID
    - subtree_article_count: article_count + sum of children's subtree counts

    Returns the node's subtree total.
    """
    own_count = int(uid_counts.get(node.get("uid", ""), 0))
    node["article_count"] = own_count

    if not node.get("children"):
        node["subtree_article_count"] = own_count
        return own_count

    children_total = sum(
        _set_counts_recursive(child, uid_counts)
        for child in node["children"]
    )

    node["subtree_article_count"] = own_count + children_total
    return node["subtree_article_count"]


def save_tree_json(data: dict, output_path: Path) -> None:
    """Write the tree data to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Saved tree JSON to %s (%.1f MB)", output_path, size_mb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build MeSH tree JSON for interactive visualization."
    )
    parser.add_argument("--desc-xml", type=Path, required=True,
                        help="Path to NLM desc2025.xml")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON file path")
    parser.add_argument("--parquet", type=Path, default=None,
                        help="Parquet file for article counts (optional)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data = build_full_tree(args.desc_xml)

    if args.parquet and args.parquet.exists():
        merge_article_counts(data, args.parquet)
    else:
        logger.info("No Parquet file provided; article counts will be 0")

    save_tree_json(data, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
