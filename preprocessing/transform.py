"""Post-parse transformations for PubMed article records.

Currently handles semantic category assignment from MeSH descriptors.
Structured as a separate module to keep parsing pure and transformations
explicit and independently testable.
"""

from __future__ import annotations

from collections import Counter

from preprocessing.schema import ArticleRecord


def assign_semantic_category(
    record: ArticleRecord,
    uid_to_categories: dict[str, list[str]],
) -> str:
    """Derive a broad semantic category from an article's MeSH descriptors.

    Strategy: count how many of the article's MeSH descriptors map to each
    of the 16 NLM tree branches. The branch with the most votes wins.
    Ties are broken alphabetically for determinism.

    Args:
        record: A parsed ArticleRecord.
        uid_to_categories: MeSH UID -> list of category names.

    Returns:
        A category string (e.g., "Diseases") or "" if no mapping exists.
    """
    if not record.mesh_descriptors:
        return ""

    votes: Counter[str] = Counter()
    for desc in record.mesh_descriptors:
        cats = uid_to_categories.get(desc.uid, [])
        for cat in cats:
            votes[cat] += 1

    if not votes:
        return ""

    max_count = max(votes.values())
    winners = sorted(cat for cat, cnt in votes.items() if cnt == max_count)
    return winners[0]
