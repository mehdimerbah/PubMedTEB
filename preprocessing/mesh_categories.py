"""MeSH descriptor UID to category and subcategory mappings.

The mappings are derived from the NLM MeSH descriptor XML file (desc2025.xml),
which maps each descriptor UID to one or more tree numbers.

- **Depth-1 (categories):** The first character of the tree number identifies
  one of 16 top-level NLM branches (e.g., ``C`` = "Diseases").
- **Depth-2 (subcategories):** The first segment of the tree number identifies
  a subcategory (e.g., ``C04`` = "Neoplasms").
- **Depth-3 (sub-subcategories):** The first two segments of the tree number
  identify a sub-subcategory (e.g., ``C04.557`` = "Neoplasms by Histologic Type").

Source:
    NLM MeSH 2025 Descriptors
    https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml

Citation:
    U.S. National Library of Medicine. Medical Subject Headings (MeSH), 2025.
    https://www.nlm.nih.gov/mesh/meshhome.html

Many descriptors have multiple tree numbers spanning different branches.
For example, "Lung Neoplasms" appears under both C (Diseases) and A (Anatomy).
In such cases, the descriptor maps to ALL its categories, and per-article
category assignment uses majority vote (see :func:`majority_vote`).
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# 16 top-level MeSH tree branches
TREE_BRANCH_NAMES: dict[str, str] = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals and Drugs",
    "E": "Techniques and Equipment",
    "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes",
    "H": "Disciplines and Occupations",
    "I": "Anthropology, Education, Sociology, and Social Phenomena",
    "J": "Technology, Industry, and Agriculture",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "V": "Publication Characteristics",
    "Z": "Geographicals",
}


# ── Majority vote utility ───────────────────────────────────────────────


def majority_vote(
    uids: list[str],
    uid_to_labels: dict[str, list[str]],
) -> str:
    """Assign a single label via majority vote over MeSH descriptor UIDs.

    Each UID maps to zero or more labels (categories or subcategories).
    The label with the most votes wins; ties are broken alphabetically
    for determinism.

    Works for both depth-1 (uid_to_categories) and depth-2
    (uid_to_subcategories) mappings.

    Args:
        uids: MeSH descriptor UIDs for a single article.
        uid_to_labels: UID -> list of label names mapping.

    Returns:
        The winning label, or ``""`` if no UIDs map to any label.
    """
    if not uids:
        return ""

    votes: Counter[str] = Counter()
    for uid in uids:
        for label in uid_to_labels.get(uid, []):
            votes[label] += 1

    if not votes:
        return ""

    max_count = max(votes.values())
    winners = sorted(label for label, cnt in votes.items() if cnt == max_count)
    return winners[0]


# ── XML parsing ─────────────────────────────────────────────────────────


def build_all_mappings(
    desc_xml_path: Path,
) -> tuple[
    dict[str, list[str]],
    dict[str, str],
    dict[str, list[str]],
    dict[str, str],
    dict[str, list[str]],
]:
    """Parse desc2025.xml once and build depth-1, depth-2, and depth-3 mappings.

    **Depth-1 (categories):** First character of each tree number maps to
    one of the 16 NLM branches via :data:`TREE_BRANCH_NAMES`.

    **Depth-2 (subcategories):** First segment of each tree number (e.g.,
    ``C04`` from ``C04.588.149``) maps to a human-readable subcategory
    name. Depth-2 nodes are identified as descriptors whose tree numbers
    have no dots (a single segment like ``C04``).

    **Depth-3 (sub-subcategories):** First two segments of each tree number
    (e.g., ``C04.557`` from ``C04.557.337.252``) map to a human-readable
    sub-subcategory name. Depth-3 nodes are identified as descriptors whose
    tree numbers have exactly one dot (e.g., ``C04.557``).

    Args:
        desc_xml_path: Path to the NLM MeSH descriptor XML file.

    Returns:
        Tuple of:
        - uid_to_categories: UID -> sorted list of depth-1 category names
        - depth2_names: depth-2 code -> name (e.g., ``"C04"`` -> ``"Neoplasms"``)
        - uid_to_subcategories: UID -> sorted list of depth-2 subcategory names
        - depth3_names: depth-3 code -> name (e.g., ``"C04.557"`` -> ``"Neoplasms by Histologic Type"``)
        - uid_to_subsubcategories: UID -> sorted list of depth-3 sub-subcategory names
    """
    # Pass 1: collect all UIDs, their tree numbers, and descriptor names.
    # Simultaneously identify depth-2 and depth-3 nodes.
    uid_tree_numbers: dict[str, list[str]] = {}
    code_to_name: dict[str, str] = {}  # depth-2 code -> descriptor name
    depth3_code_to_name: dict[str, str] = {}  # depth-3 code -> descriptor name

    for _, elem in ET.iterparse(str(desc_xml_path), events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue

        uid_el = elem.find("DescriptorUI")
        if uid_el is None or uid_el.text is None:
            elem.clear()
            continue

        uid = uid_el.text.strip()

        name_el = elem.find("DescriptorName/String")
        desc_name = name_el.text.strip() if name_el is not None and name_el.text else ""

        tree_nums_el = elem.find("TreeNumberList")
        tree_numbers: list[str] = []

        if tree_nums_el is not None:
            for tn in tree_nums_el.findall("TreeNumber"):
                if tn.text:
                    tn_text = tn.text.strip()
                    tree_numbers.append(tn_text)
                    dots = tn_text.count(".")
                    # A tree number with no dots is a depth-2 node (e.g., "C04")
                    if dots == 0 and len(tn_text) >= 2:
                        code_to_name[tn_text] = desc_name
                    # A tree number with exactly one dot is a depth-3 node (e.g., "C04.557")
                    elif dots == 1:
                        depth3_code_to_name[tn_text] = desc_name

        uid_tree_numbers[uid] = tree_numbers
        elem.clear()

    logger.info(
        "Parsed desc2025.xml: %d descriptors, %d depth-2 codes, %d depth-3 codes",
        len(uid_tree_numbers),
        len(code_to_name),
        len(depth3_code_to_name),
    )

    # Pass 2 (in-memory): resolve tree numbers to category/subcategory/sub-subcategory names
    uid_to_categories: dict[str, list[str]] = {}
    uid_to_subcategories: dict[str, list[str]] = {}
    uid_to_subsubcategories: dict[str, list[str]] = {}

    for uid, tree_numbers in uid_tree_numbers.items():
        categories: set[str] = set()
        subcategories: set[str] = set()
        subsubcategories: set[str] = set()

        for tn in tree_numbers:
            # Depth-1: first character
            prefix = tn[0]
            if prefix in TREE_BRANCH_NAMES:
                categories.add(TREE_BRANCH_NAMES[prefix])

            segments = tn.split(".")

            # Depth-2: first segment (before the first dot, or full string)
            depth2_code = segments[0]
            if depth2_code in code_to_name:
                subcategories.add(code_to_name[depth2_code])
            elif depth2_code and len(depth2_code) >= 2:
                logger.debug("Unresolved depth-2 code: %s (UID: %s)", depth2_code, uid)

            # Depth-3: first two segments joined (e.g., "C04.557")
            if len(segments) >= 2:
                depth3_code = ".".join(segments[:2])
                if depth3_code in depth3_code_to_name:
                    subsubcategories.add(depth3_code_to_name[depth3_code])
                else:
                    logger.debug("Unresolved depth-3 code: %s (UID: %s)", depth3_code, uid)

        uid_to_categories[uid] = sorted(categories)
        uid_to_subcategories[uid] = sorted(subcategories)
        uid_to_subsubcategories[uid] = sorted(subsubcategories)

    logger.info(
        "Built mappings: %d UIDs -> categories, %d -> subcategories, %d -> sub-subcategories",
        len(uid_to_categories),
        len(uid_to_subcategories),
        len(uid_to_subsubcategories),
    )
    return (
        uid_to_categories,
        code_to_name,
        uid_to_subcategories,
        depth3_code_to_name,
        uid_to_subsubcategories,
    )


def build_uid_to_categories(desc_xml_path: Path) -> dict[str, list[str]]:
    """Parse NLM desc2025.xml and build UID -> list of category names.

    Thin wrapper around :func:`build_all_mappings` for backward compatibility.

    Args:
        desc_xml_path: Path to the NLM MeSH descriptor XML file.

    Returns:
        Dict mapping descriptor UID to a sorted list of unique category names.
    """
    uid_to_categories, *_ = build_all_mappings(desc_xml_path)
    return uid_to_categories


# ── Persistence ─────────────────────────────────────────────────────────


def save_mapping(
    uid_to_categories: dict[str, list[str]],
    output_path: Path,
    depth2_names: dict[str, str] | None = None,
    uid_to_subcategories: dict[str, list[str]] | None = None,
    depth3_names: dict[str, str] | None = None,
    uid_to_subsubcategories: dict[str, list[str]] | None = None,
) -> None:
    """Persist the MeSH mappings as JSON for reproducibility.

    The JSON includes metadata for provenance tracking, plus
    depth-1 and (optionally) depth-2 and depth-3 mappings.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "mesh_version": "2025",
        "source": "desc2025.xml",
        "n_descriptors": len(uid_to_categories),
        "tree_branches": TREE_BRANCH_NAMES,
        "uid_to_categories": uid_to_categories,
    }
    if depth2_names is not None:
        payload["depth2_names"] = depth2_names
    if uid_to_subcategories is not None:
        payload["uid_to_subcategories"] = uid_to_subcategories
    if depth3_names is not None:
        payload["depth3_names"] = depth3_names
    if uid_to_subsubcategories is not None:
        payload["uid_to_subsubcategories"] = uid_to_subsubcategories

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved MeSH mapping to %s (%d descriptors)", output_path, len(uid_to_categories))


def load_mapping(path: Path) -> dict:
    """Load a previously saved MeSH mapping.

    Returns:
        Full payload dict with keys ``uid_to_categories`` and optionally
        ``uid_to_subcategories``, ``depth2_names``, ``uid_to_subsubcategories``,
        and ``depth3_names``.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
    """
    with open(path) as f:
        payload = json.load(f)
    n = len(payload["uid_to_categories"])
    has_depth2 = "uid_to_subcategories" in payload
    has_depth3 = "uid_to_subsubcategories" in payload
    logger.info(
        "Loaded MeSH mapping: %d descriptors, depth-2=%s, depth-3=%s (version %s)",
        n,
        has_depth2,
        has_depth3,
        payload.get("mesh_version", "unknown"),
    )
    return payload


def init_mapping(
    desc_xml_path: Path | None = None,
    cache_path: Path | None = None,
) -> dict:
    """Initialize MeSH mappings (depth-1, depth-2, and depth-3).

    First attempts to load from *cache_path*. If the cache exists but
    lacks depth-3 data (or depth-2), rebuilds from *desc_xml_path*. If no
    cache exists, builds from XML and saves.

    Args:
        desc_xml_path: Path to desc2025.xml (required if cache is missing
            or incomplete).
        cache_path: Path to the JSON cache file.

    Returns:
        Dict with keys ``uid_to_categories``, ``uid_to_subcategories``,
        ``depth2_names``, ``uid_to_subsubcategories``, and ``depth3_names``.

    Raises:
        FileNotFoundError: If neither cache nor desc_xml_path is available.
    """
    # Try cache first
    if cache_path and cache_path.exists():
        payload = load_mapping(cache_path)
        if "uid_to_subcategories" in payload and "uid_to_subsubcategories" in payload:
            return payload
        logger.info("Cache lacks depth-2 or depth-3 data — rebuilding from XML")

    # Build from source XML
    if desc_xml_path is None or not desc_xml_path.exists():
        # If we loaded a cache without depth-3, still return it with empty stubs
        if cache_path and cache_path.exists():
            payload = load_mapping(cache_path)
            payload.setdefault("uid_to_subcategories", {})
            payload.setdefault("depth2_names", {})
            payload.setdefault("uid_to_subsubcategories", {})
            payload.setdefault("depth3_names", {})
            logger.warning("No desc2025.xml available — returning partial mapping")
            return payload
        raise FileNotFoundError(
            "MeSH mapping requires either a cached JSON file or desc2025.xml. "
            "Download desc2025.xml from: "
            "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml"
        )

    (
        uid_to_categories,
        depth2_names,
        uid_to_subcategories,
        depth3_names,
        uid_to_subsubcategories,
    ) = build_all_mappings(desc_xml_path)

    payload = {
        "uid_to_categories": uid_to_categories,
        "depth2_names": depth2_names,
        "uid_to_subcategories": uid_to_subcategories,
        "depth3_names": depth3_names,
        "uid_to_subsubcategories": uid_to_subsubcategories,
    }

    # Save cache for next run
    if cache_path:
        save_mapping(
            uid_to_categories,
            cache_path,
            depth2_names,
            uid_to_subcategories,
            depth3_names,
            uid_to_subsubcategories,
        )

    return payload
