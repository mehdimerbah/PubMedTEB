"""PubMed XML parsing module.

Extracts article-level metadata from PubMed Baseline XML files using
the standard library ElementTree parser. Each file contains ~30,000
<PubmedArticle> elements wrapped in a <PubmedArticleSet> root.

Memory management: uses iterparse with elem.clear() to release each
article element after processing, keeping peak memory proportional to
a single article rather than the full file.
"""

from __future__ import annotations

import datetime
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from preprocessing.schema import ArticleRecord, MeSHDescriptor

logger = logging.getLogger(__name__)

_MIN_YEAR = 1800
_MAX_YEAR = datetime.date.today().year + 1


# ── Public API ────────────────────────────────────────────────────────────


def parse_xml_file(path: Path) -> list[ArticleRecord]:
    """Parse all articles from a single PubMed XML file.

    Args:
        path: Path to a PubMed XML file (e.g., pubmed25n0001.xml).

    Returns:
        List of ArticleRecord instances. Malformed articles (missing PMID)
        are logged at WARNING level and skipped.
    """
    records: list[ArticleRecord] = []
    n_skipped = 0

    for event, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag != "PubmedArticle":
            continue
        record = parse_article(elem)
        if record is not None:
            records.append(record)
        else:
            n_skipped += 1
        elem.clear()

    if n_skipped:
        logger.warning("%s: skipped %d articles without PMID", path.name, n_skipped)
    logger.info("%s: parsed %d articles", path.name, len(records))
    return records


def parse_article(article_elem: ET.Element) -> ArticleRecord | None:
    """Extract all fields from a single <PubmedArticle> element.

    Returns None if the article lacks a PMID (malformed record).
    """
    citation = article_elem.find("MedlineCitation")
    if citation is None:
        return None

    # PMID 
    pmid_el = citation.find("PMID")
    if pmid_el is None or pmid_el.text is None:
        return None
    pmid = pmid_el.text.strip()

    # Article sub-element
    article_el = citation.find("Article")

    # Title
    title = _extract_title(article_el)

    # Abstract
    abstract_text, abstract_sections = _extract_abstract(article_el)

    # Journal
    journal = _extract_journal(article_el)

    # Year, Month
    year, month = _extract_pub_date(article_el)

    # MeSH descriptors
    mesh_descriptors = _extract_mesh(citation)

    # Publication types
    publication_types = _extract_publication_types(article_el)

    # Language
    language = _extract_language(article_el)

    # Country 
    country = _extract_country(citation)

    # PubmedData sub-element (citations, DOI)
    pubmed_data = article_elem.find("PubmedData")

    # Cited PMIDs
    cited_pmids = _extract_citations(pubmed_data)

    # DOI
    doi = _extract_doi(pubmed_data)

    return ArticleRecord(
        pmid=pmid,
        title=title,
        abstract_text=abstract_text,
        abstract_sections=abstract_sections,
        journal=journal,
        year=year,
        month=month,
        mesh_descriptors=mesh_descriptors,
        cited_pmids=cited_pmids,
        publication_types=publication_types,
        language=language,
        doi=doi,
        country=country,
    )


# ── Extraction helpers ────────────────────────────────────────────────────


def _extract_title(article_el: ET.Element | None) -> str:
    """Extract article title, preserving inline markup text."""
    if article_el is None:
        return ""
    title_el = article_el.find("ArticleTitle")
    if title_el is None:
        return ""
    return "".join(title_el.itertext()).strip()


def _extract_abstract(
    article_el: ET.Element | None,
) -> tuple[str, list[dict[str, str]]]:
    """Extract abstract text and structured sections.

    Handles both plain and structured abstracts. For structured abstracts
    (with Label attributes like BACKGROUND, METHODS, RESULTS, CONCLUSIONS),
    sections are preserved individually and also concatenated into plain text.

    Uses itertext() to capture text inside inline elements (<i>, <sub>, etc.)
    that are common in biomedical abstracts for species names and formulas.
    """
    if article_el is None:
        return "", []

    abstract_el = article_el.find("Abstract")
    if abstract_el is None:
        return "", []

    sections: list[dict[str, str]] = []
    parts: list[str] = []

    for at in abstract_el.findall("AbstractText"):
        text = "".join(at.itertext()).strip()
        label = at.get("Label", "")
        sections.append({"label": label, "text": text})
        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)

    plain_text = " ".join(parts)
    return plain_text, sections


def _extract_journal(article_el: ET.Element | None) -> str:
    """Extract journal full title."""
    if article_el is None:
        return ""
    journal_el = article_el.find("Journal/Title")
    if journal_el is None or journal_el.text is None:
        return ""
    return journal_el.text.strip()


def _extract_pub_date(article_el: ET.Element | None) -> tuple[int, str]:
    """Extract publication year and month from Journal/JournalIssue/PubDate.

    Returns (year, month) where year=0 if missing and month="" if missing.
    Month is kept as a string since PubMed uses both numeric and textual forms.
    Years outside [1800, current_year+1] are treated as missing (0).
    """
    if article_el is None:
        return 0, ""

    pubdate = article_el.find("Journal/JournalIssue/PubDate")
    if pubdate is None:
        return 0, ""

    year_el = pubdate.find("Year")
    month_el = pubdate.find("Month")

    year = 0
    if year_el is not None and year_el.text:
        try:
            year = int(year_el.text.strip())
        except ValueError:
            pass

    month = ""
    if month_el is not None and month_el.text:
        month = month_el.text.strip()

    # Fallback: some articles use MedlineDate instead of Year/Month
    if year == 0:
        medline_date = pubdate.find("MedlineDate")
        if medline_date is not None and medline_date.text:
            # MedlineDate format: "1998 Dec-1999 Jan" or "2000 Spring"
            # Extract the first 4-digit year and use it for consistency
            parts = medline_date.text.strip().split()
            for part in parts:
                if len(part) == 4 and part.isdigit():
                    year = int(part)
                    break

    # Filter implausible years
    if year != 0 and (year < _MIN_YEAR or year > _MAX_YEAR):
        year = 0

    return year, month


def _extract_mesh(citation_el: ET.Element) -> list[MeSHDescriptor]:
    """Extract MeSH descriptors from MeshHeadingList.

    Each MeshHeading contains a DescriptorName with attributes:
      - UI: the descriptor unique identifier 
      - MajorTopicYN: "Y" if this is a major topic of the article
    """
    mesh_list = citation_el.find("MeshHeadingList")
    if mesh_list is None:
        return []

    descriptors: list[MeSHDescriptor] = []
    for heading in mesh_list.findall("MeshHeading"):
        desc_el = heading.find("DescriptorName")
        if desc_el is None:
            continue
        text = desc_el.text or ""
        uid = desc_el.get("UI", "")
        is_major = desc_el.get("MajorTopicYN", "N") == "Y"
        descriptors.append(MeSHDescriptor(text=text.strip(), uid=uid, is_major=is_major))

    return descriptors


def _extract_citations(pubmed_data: ET.Element | None) -> list[str]:
    """Extract cited PMIDs from ReferenceList.

    Returns deduplicated list preserving insertion order.
    """
    if pubmed_data is None:
        return []

    ref_list = pubmed_data.find("ReferenceList")
    if ref_list is None:
        return []

    seen: set[str] = set()
    pmids: list[str] = []

    for ref in ref_list.findall("Reference"):
        id_list = ref.find("ArticleIdList")
        if id_list is None:
            continue
        for aid in id_list.findall("ArticleId"):
            if aid.get("IdType") == "pubmed" and aid.text:
                pid = aid.text.strip()
                if pid not in seen:
                    seen.add(pid)
                    pmids.append(pid)

    return pmids


def _extract_publication_types(article_el: ET.Element | None) -> list[str]:
    """Extract publication type strings"""
    if article_el is None:
        return []

    pt_list = article_el.find("PublicationTypeList")
    if pt_list is None:
        return []

    return [
        pt.text.strip()
        for pt in pt_list.findall("PublicationType")
        if pt.text
    ]


def _extract_language(article_el: ET.Element | None) -> str:
    """Extract article language (e.g., 'eng', 'fre')."""
    if article_el is None:
        return ""
    lang_el = article_el.find("Language")
    if lang_el is None or lang_el.text is None:
        return ""
    return lang_el.text.strip()


def _extract_doi(pubmed_data: ET.Element | None) -> str:
    """Extract DOI from ArticleIdList."""
    if pubmed_data is None:
        return ""

    id_list = pubmed_data.find("ArticleIdList")
    if id_list is None:
        return ""

    for aid in id_list.findall("ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            return aid.text.strip()

    return ""


def _extract_country(citation_el: ET.Element) -> str:
    """Extract journal country from MedlineJournalInfo."""
    country_el = citation_el.find("MedlineJournalInfo/Country")
    if country_el is None or country_el.text is None:
        return ""
    return country_el.text.strip()
