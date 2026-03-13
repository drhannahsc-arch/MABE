"""
paper_extractor/pdf_reader.py -- PDF text and table extraction
===============================================================
Uses pdfplumber for both text and table extraction.
Falls back to pypdf if pdfplumber fails on a page.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


@dataclass
class ExtractedTable:
    """A table extracted from a PDF page."""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    raw_bbox: Optional[tuple] = None

    @property
    def n_cols(self) -> int:
        return len(self.headers) if self.headers else 0

    @property
    def n_rows(self) -> int:
        return len(self.rows)


@dataclass
class PageContent:
    """Content extracted from one PDF page."""
    page_number: int
    text: str
    tables: List[ExtractedTable] = field(default_factory=list)


@dataclass
class PaperContent:
    """All content from a PDF."""
    filename: str
    n_pages: int
    pages: List[PageContent] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def all_tables(self) -> List[ExtractedTable]:
        tables = []
        for p in self.pages:
            tables.extend(p.tables)
        return tables


def read_pdf(filepath: str) -> PaperContent:
    """
    Extract text and tables from a PDF.
    Uses pdfplumber (preferred) with pypdf fallback.
    """
    if pdfplumber is None and PdfReader is None:
        raise ImportError("Need pdfplumber or pypdf. Install: pip install pdfplumber")

    if pdfplumber is not None:
        return _read_with_pdfplumber(filepath)
    else:
        return _read_with_pypdf(filepath)


def _read_with_pdfplumber(filepath: str) -> PaperContent:
    """Extract using pdfplumber (handles tables well)."""
    import os
    pages = []

    with pdfplumber.open(filepath) as pdf:
        n_pages = len(pdf.pages)
        metadata = pdf.metadata or {}

        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            # Text
            text = page.extract_text() or ""

            # Tables
            extracted_tables = []
            try:
                raw_tables = page.extract_tables()
                for t_idx, raw_table in enumerate(raw_tables or []):
                    if not raw_table or len(raw_table) < 2:
                        continue
                    # Clean cells
                    cleaned = []
                    for row in raw_table:
                        cleaned.append([_clean_cell(c) for c in row])

                    # First row with content = headers
                    headers = cleaned[0]
                    data_rows = cleaned[1:]

                    # Skip tables that are mostly empty
                    total_cells = sum(len(r) for r in data_rows)
                    non_empty = sum(1 for r in data_rows for c in r if c.strip())
                    if total_cells > 0 and non_empty / total_cells < 0.2:
                        continue

                    extracted_tables.append(ExtractedTable(
                        page_number=page_num,
                        table_index=t_idx,
                        headers=headers,
                        rows=data_rows,
                    ))
            except Exception:
                pass  # table extraction can fail on complex layouts

            pages.append(PageContent(
                page_number=page_num,
                text=text,
                tables=extracted_tables,
            ))

    return PaperContent(
        filename=os.path.basename(filepath),
        n_pages=n_pages,
        pages=pages,
        metadata={str(k): str(v) for k, v in (metadata or {}).items()},
    )


def _read_with_pypdf(filepath: str) -> PaperContent:
    """Fallback: pypdf for text only (no table extraction)."""
    import os
    reader = PdfReader(filepath)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(PageContent(page_number=i+1, text=text))

    return PaperContent(
        filename=os.path.basename(filepath),
        n_pages=len(reader.pages),
        pages=pages,
        metadata={str(k): str(v) for k, v in (reader.metadata or {}).items()},
    )


def _clean_cell(cell) -> str:
    """Clean a table cell value."""
    if cell is None:
        return ""
    s = str(cell).strip()
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    # Common PDF artifacts
    s = s.replace('\x00', '').replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    return s


def extract_metadata_from_text(text: str) -> Dict[str, str]:
    """
    Try to extract title, authors, DOI from first page text.
    Heuristic — not reliable on all formats.
    """
    meta = {}

    # DOI
    doi_match = re.search(r'(?:doi|DOI)[:\s]*([10]\.\d{4,}/[^\s]+)', text)
    if doi_match:
        meta['doi'] = doi_match.group(1).rstrip('.')

    # Year from common patterns
    year_match = re.search(r'(?:19|20)\d{2}', text[:2000])
    if year_match:
        meta['year'] = year_match.group()

    # Title: often the first substantial line (heuristic)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:10]:
        if len(line) > 20 and not re.match(r'^(?:doi|DOI|http|www|\d)', line):
            if not any(w in line.lower() for w in ['copyright', 'received', 'accepted',
                                                     'published', 'journal', 'volume']):
                meta['title'] = line[:200]
                break

    return meta
