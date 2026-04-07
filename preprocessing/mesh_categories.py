"""MeSH descriptor UID to top-level category mapping.

The mapping is derived from the NLM MeSH descriptor XML file (desc2025.xml),
which maps each descriptor UID to one or more tree numbers. The first letter
of the tree number identifies the top-level MeSH branch.

Source:
    NLM MeSH 2025 Descriptors
    https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml

Citation:
    U.S. National Library of Medicine. Medical Subject Headings (MeSH), 2025.
    https://www.nlm.nih.gov/mesh/meshhome.html

Many descriptors have multiple tree numbers spanning different branches.
For example, "Lung Neoplasms" appears under both C (Diseases) and A (Anatomy).
In such cases, the descriptor maps to ALL its categories, and per-article
category assignment uses majority vote (see schema._assign_category).
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
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


def build_uid_to_categories(desc_xml_path: Path) -> dict[str, list[str]]:
    """Parse NLM desc2025.xml and build UID -> list of category names.

    Each MeSH DescriptorRecord has a TreeNumberList with one or more
    TreeNumber entries. The first character of each tree number identifies
    the top-level branch.

    Args:
        desc_xml_path: Path to the NLM MeSH descriptor XML file.

    Returns:
        Dict mapping descriptor UID (e.g., "D006801") to a sorted list
        of unique category names (e.g., ["Named Groups"]).
    """
    mapping: dict[str, list[str]] = {}

    for _, elem in ET.iterparse(str(desc_xml_path), events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue

        uid_el = elem.find("DescriptorUI")
        if uid_el is None or uid_el.text is None:
            elem.clear()
            continue

        uid = uid_el.text.strip()
        tree_nums = elem.find("TreeNumberList")
        categories: set[str] = set()

        if tree_nums is not None:
            for tn in tree_nums.findall("TreeNumber"):
                if tn.text:
                    prefix = tn.text[0]
                    if prefix in TREE_BRANCH_NAMES:
                        categories.add(TREE_BRANCH_NAMES[prefix])

        mapping[uid] = sorted(categories)
        elem.clear()

    logger.info("Built MeSH mapping: %d UIDs -> categories", len(mapping))
    return mapping


def save_mapping(mapping: dict[str, list[str]], output_path: Path) -> None:
    """Persist the UID -> categories mapping as JSON for reproducibility.

    The JSON includes metadata for provenance tracking.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mesh_version": "2025",
        "source": "desc2025.xml",
        "n_descriptors": len(mapping),
        "tree_branches": TREE_BRANCH_NAMES,
        "uid_to_categories": mapping,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved MeSH mapping to %s (%d descriptors)", output_path, len(mapping))


def load_mapping(path: Path) -> dict[str, list[str]]:
    """Load a previously saved UID -> categories mapping.

    Returns:
        The uid_to_categories dict.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
    """
    with open(path) as f:
        payload = json.load(f)
    mapping = payload["uid_to_categories"]
    logger.info(
        "Loaded MeSH mapping: %d descriptors (version %s)",
        len(mapping),
        payload.get("mesh_version", "unknown"),
    )
    return mapping


def init_mapping(
    desc_xml_path: Path | None = None,
    cache_path: Path | None = None,
) -> dict[str, list[str]]:
    """Initialize the MeSH UID -> categories mapping.

    First attempts to load from cache_path. If unavailable, builds from
    desc_xml_path and saves to cache_path for future use.

    Args:
        desc_xml_path: Path to desc2025.xml (required if cache doesn't exist).
        cache_path: Path to the JSON cache file.

    Returns:
        The UID -> categories mapping dict.

    Raises:
        FileNotFoundError: If neither cache nor desc_xml_path is available.
    """
    # Try cache first
    if cache_path and cache_path.exists():
        return load_mapping(cache_path)

    # Build from source XML
    if desc_xml_path is None or not desc_xml_path.exists():
        raise FileNotFoundError(
            "MeSH mapping requires either a cached JSON file or desc2025.xml. "
            "Download desc2025.xml from: "
            "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml"
        )

    mapping = build_uid_to_categories(desc_xml_path)

    # Save cache for next run
    if cache_path:
        save_mapping(mapping, cache_path)

    return mapping
