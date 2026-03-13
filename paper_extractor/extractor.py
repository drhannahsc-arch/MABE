"""
paper_extractor/extractor.py -- Main extraction pipeline
=========================================================

Orchestrates: PDF reading -> table classification -> data parsing -> DB storage.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from paper_extractor.db import (
    init_db, get_db_path, file_hash, is_paper_processed,
    register_paper, insert_binding, insert_physical_constant,
    insert_raw_table, add_review_item, get_stats,
)
from paper_extractor.pdf_reader import read_pdf, extract_metadata_from_text
from paper_extractor.patterns import (
    classify_table, parse_binding_table,
    extract_binding_from_text, extract_constants_from_text,
    parse_garbled_table_rows,
)


def process_folder(input_dir: str, db_dir: str = None, verbose: bool = True) -> Dict[str, int]:
    """
    Process all PDFs in input_dir. Store results in db_dir.

    Parameters
    ----------
    input_dir : str
        Folder containing PDF files.
    db_dir : str or None
        Where to put the database. Default: same as input_dir.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with counts of new papers, tables, binding entries, etc.
    """
    if db_dir is None:
        db_dir = input_dir

    db_path = get_db_path(db_dir)
    conn = init_db(db_path)

    pdfs = sorted(Path(input_dir).glob("*.pdf"))
    if not pdfs:
        if verbose:
            print(f"No PDF files found in {input_dir}")
        return {'new_papers': 0}

    counts = {
        'total_pdfs': len(pdfs),
        'skipped': 0,
        'processed': 0,
        'failed': 0,
        'tables_found': 0,
        'tables_classified': 0,
        'binding_entries': 0,
        'constant_entries': 0,
        'review_items': 0,
    }

    for pdf_path in pdfs:
        fhash = file_hash(str(pdf_path))

        if is_paper_processed(conn, fhash):
            counts['skipped'] += 1
            if verbose:
                print(f"  SKIP (already processed): {pdf_path.name}")
            continue

        if verbose:
            print(f"  Processing: {pdf_path.name}")

        try:
            result = process_single_paper(str(pdf_path), conn, verbose=verbose)
            counts['processed'] += 1
            counts['tables_found'] += result.get('tables_found', 0)
            counts['tables_classified'] += result.get('tables_classified', 0)
            counts['binding_entries'] += result.get('binding_entries', 0)
            counts['constant_entries'] += result.get('constant_entries', 0)
            counts['review_items'] += result.get('review_items', 0)
        except Exception as e:
            counts['failed'] += 1
            if verbose:
                print(f"    FAILED: {e}")

    conn.close()

    if verbose:
        print(f"\nDone. {counts['processed']} new papers processed, "
              f"{counts['skipped']} skipped, {counts['failed']} failed.")
        print(f"  Tables found: {counts['tables_found']}")
        print(f"  Binding entries: {counts['binding_entries']}")
        print(f"  Constants: {counts['constant_entries']}")
        if counts['review_items'] > 0:
            print(f"  Items needing review: {counts['review_items']}")

    return counts


def process_single_paper(filepath: str, conn, verbose: bool = False) -> Dict[str, int]:
    """Process one PDF file."""
    result = {
        'tables_found': 0,
        'tables_classified': 0,
        'binding_entries': 0,
        'constant_entries': 0,
        'review_items': 0,
    }

    # Read PDF
    content = read_pdf(filepath)
    if verbose:
        print(f"    Pages: {content.n_pages}, Tables: {len(content.all_tables)}")

    # Extract metadata
    meta = extract_metadata_from_text(content.pages[0].text if content.pages else "")
    meta.update({k: v for k, v in content.metadata.items() if v})

    # Register paper
    fhash = file_hash(filepath)
    paper_id = register_paper(
        conn,
        filename=content.filename,
        fhash=fhash,
        n_pages=content.n_pages,
        title=meta.get('title', ''),
        doi=meta.get('doi', ''),
        year=int(meta['year']) if meta.get('year', '').isdigit() else None,
    )

    # Process tables
    for table in content.all_tables:
        result['tables_found'] += 1

        # Classify
        classification, confidence = classify_table(
            table.headers,
            table.rows[:3] if table.rows else None,
        )

        # Store raw table
        raw_id = insert_raw_table(
            conn, paper_id,
            page_number=table.page_number,
            table_index=table.table_index,
            headers=table.headers,
            rows=table.rows,
            classified_as=classification,
            confidence=f"{confidence:.2f}",
        )

        if classification == 'other':
            continue

        result['tables_classified'] += 1

        if verbose:
            print(f"    Table p{table.page_number}#{table.table_index}: "
                  f"{classification} (conf={confidence:.2f}, "
                  f"{table.n_cols}cols x {table.n_rows}rows)")

        # Parse based on classification
        if classification == 'binding':
            entries = parse_binding_table(
                table.headers, table.rows,
                default_receptor=meta.get('title', ''),
            )
            for entry in entries:
                entry['table_ref'] = f"p{table.page_number} table {table.table_index}"
                bid = insert_binding(conn, paper_id, entry)
                if bid:
                    result['binding_entries'] += 1

            if not entries and confidence > 0.3:
                # Table looked like binding data but couldn't parse
                add_review_item(
                    conn, paper_id,
                    source_table='raw_tables', source_id=raw_id,
                    issue='binding_table_unparsed',
                    context=f"Headers: {table.headers[:5]}, {table.n_rows} rows",
                )
                result['review_items'] += 1

    # Garbled table parsing: try to extract from raw table text
    # that wasn't successfully parsed by the structured parser
    for table in content.all_tables:
        # Build raw text from all rows
        all_cells = []
        for row in table.rows:
            all_cells.extend(str(c) for c in row if c)
        raw_text = ' | '.join(all_cells)

        if len(raw_text) > 20:
            garbled_entries = parse_garbled_table_rows(raw_text)
            for entry in garbled_entries:
                entry['table_ref'] = f"garbled p{table.page_number}#{table.table_index}"
                bid = insert_binding(conn, paper_id, entry)
                if bid:
                    result['binding_entries'] += 1

    # Text-based extraction (lower confidence)
    # Process full document text as one block for better context
    full_text = content.full_text
    text_bindings = extract_binding_from_text(full_text)
    for entry in text_bindings:
        entry['table_ref'] = 'text (full doc)'
        bid = insert_binding(conn, paper_id, entry)
        if bid:
            result['binding_entries'] += 1

    text_constants = extract_constants_from_text(full_text)
    for entry in text_constants:
        entry['table_ref'] = 'text (full doc)'
        cid = insert_physical_constant(conn, paper_id, entry)
        if cid:
            result['constant_entries'] += 1

    return result
