"""Canonical data structures and Arrow schema for the PubMed pipeline.

Defines the intermediate Python representation (ArticleRecord) and the
target Parquet schema. All modules communicate through these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

import pyarrow as pa


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class MeSHDescriptor:
    """A single MeSH descriptor assigned to an article."""

    text: str
    uid: str
    is_major: bool


@dataclass(slots=True)
class ArticleRecord:
    """Parsed representation of a single PubMed article.

    One instance per <PubmedArticle> element. Converted to Arrow via records_to_arrow_batch().
    """

    pmid: str
    title: str
    abstract_text: str
    abstract_sections: list[dict[str, str]] = field(default_factory=list)
    journal: str = ""
    year: int = 0
    month: str = ""
    mesh_descriptors: list[MeSHDescriptor] = field(default_factory=list)
    cited_pmids: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)
    language: str = ""
    doi: str = ""
    country: str = ""


# ── Arrow schema ─────────────────────────────────────────────────────────

_mesh_struct = pa.struct([
    ("text", pa.utf8()),
    ("uid", pa.utf8()),
    ("is_major", pa.bool_()),
])

_section_struct = pa.struct([
    ("label", pa.utf8()),
    ("text", pa.utf8()),
])

ARROW_SCHEMA = pa.schema([
    ("pmid", pa.utf8()),
    ("title", pa.utf8()),
    ("abstract_text", pa.utf8()),
    ("abstract_sections", pa.list_(_section_struct)),
    ("journal", pa.utf8()),
    ("year", pa.int16()),
    ("month", pa.utf8()),
    ("mesh_descriptors", pa.list_(_mesh_struct)),
    ("cited_pmids", pa.list_(pa.utf8())),
    ("publication_types", pa.list_(pa.utf8())),
    ("language", pa.utf8()),
    ("doi", pa.utf8()),
    ("country", pa.utf8()),
    ("semantic_category", pa.utf8()),
])


# ── Batch conversion ─────────────────────────────────────────────────────


def _assign_category(
    mesh_descs: list[MeSHDescriptor],
    uid_to_categories: dict[str, list[str]],
) -> str:
    """Derive semantic category from MeSH descriptors via majority vote.

    Each MeSH UID maps to one or more of the 16 NLM tree branches.
    The branch with the most votes wins; ties broken alphabetically.
    Returns "" if no descriptors map to any category.
    """
    if not mesh_descs:
        return ""

    votes: Counter[str] = Counter()
    for desc in mesh_descs:
        cats = uid_to_categories.get(desc.uid, [])
        for cat in cats:
            votes[cat] += 1

    if not votes:
        return ""

    max_count = max(votes.values())
    winners = sorted(cat for cat, cnt in votes.items() if cnt == max_count)
    return winners[0]


def records_to_arrow_batch(
    records: list[ArticleRecord],
    uid_to_categories: dict[str, list[str]],
) -> pa.RecordBatch:
    """Convert a list of ArticleRecord to a PyArrow RecordBatch.

    Args:
        records: Parsed article records from a single XML file.
        uid_to_categories: MeSH UID -> list of category names mapping.

    Returns:
        A RecordBatch conforming to ARROW_SCHEMA.
    """
    pmids = []
    titles = []
    abstracts = []
    sections_col = []
    journals = []
    years = []
    months = []
    mesh_col = []
    cited_col = []
    pubtypes_col = []
    languages = []
    dois = []
    countries = []
    categories = []

    for rec in records:
        pmids.append(rec.pmid)
        titles.append(rec.title)
        abstracts.append(rec.abstract_text)
        sections_col.append(
            [{"label": s["label"], "text": s["text"]} for s in rec.abstract_sections]
        )
        journals.append(rec.journal)
        years.append(rec.year)
        months.append(rec.month)
        mesh_col.append(
            [{"text": m.text, "uid": m.uid, "is_major": m.is_major}
             for m in rec.mesh_descriptors]
        )
        cited_col.append(rec.cited_pmids)
        pubtypes_col.append(rec.publication_types)
        languages.append(rec.language)
        dois.append(rec.doi)
        countries.append(rec.country)
        categories.append(_assign_category(rec.mesh_descriptors, uid_to_categories))

    arrays = [
        pa.array(pmids, type=pa.utf8()),
        pa.array(titles, type=pa.utf8()),
        pa.array(abstracts, type=pa.utf8()),
        pa.array(sections_col, type=pa.list_(_section_struct)),
        pa.array(journals, type=pa.utf8()),
        pa.array(years, type=pa.int16()),
        pa.array(months, type=pa.utf8()),
        pa.array(mesh_col, type=pa.list_(_mesh_struct)),
        pa.array(cited_col, type=pa.list_(pa.utf8())),
        pa.array(pubtypes_col, type=pa.list_(pa.utf8())),
        pa.array(languages, type=pa.utf8()),
        pa.array(dois, type=pa.utf8()),
        pa.array(countries, type=pa.utf8()),
        pa.array(categories, type=pa.utf8()),
    ]

    return pa.RecordBatch.from_arrays(arrays, schema=ARROW_SCHEMA)
