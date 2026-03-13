"""
paper_extractor/db.py -- SQLite database for extracted data
============================================================
"""

import sqlite3
import hashlib
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


DB_NAME = "paper_data.db"


def get_db_path(base_dir: str) -> str:
    return os.path.join(base_dir, DB_NAME)


def init_db(db_path: str) -> sqlite3.Connection:
    """Create database and tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
    -- Papers we've processed
    CREATE TABLE IF NOT EXISTS papers (
        paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_hash TEXT UNIQUE NOT NULL,
        title TEXT,
        authors TEXT,
        year INTEGER,
        journal TEXT,
        doi TEXT,
        first_processed TEXT NOT NULL,
        last_processed TEXT NOT NULL,
        n_pages INTEGER,
        status TEXT DEFAULT 'processed'  -- 'processed', 'failed', 'needs_review'
    );

    -- Binding thermodynamics (ITC, SPR, fluorescence)
    CREATE TABLE IF NOT EXISTS binding_data (
        bind_id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id),
        receptor TEXT,           -- lectin name, host name, protein
        ligand TEXT,             -- sugar, guest molecule
        Ka REAL,                 -- association constant M^-1
        Kd REAL,                 -- dissociation constant M or mM
        Kd_unit TEXT DEFAULT 'M',
        dG REAL,                 -- kcal/mol (as published)
        dG_unit TEXT DEFAULT 'kcal/mol',
        dH REAL,                 -- kcal/mol
        dH_unit TEXT DEFAULT 'kcal/mol',
        TdS REAL,                -- kcal/mol
        TdS_unit TEXT DEFAULT 'kcal/mol',
        dCp REAL,                -- cal/mol/K
        n_sites REAL,            -- stoichiometry
        temperature_C REAL,
        pH REAL,
        buffer TEXT,
        method TEXT,             -- 'ITC', 'SPR', 'fluorescence', 'NMR', 'inhibition'
        conditions TEXT,         -- freeform notes
        table_ref TEXT,          -- 'Table 1', 'Table 3', etc.
        confidence TEXT DEFAULT 'medium',  -- 'high', 'medium', 'low'
        raw_text TEXT,           -- original text snippet for verification
        UNIQUE(paper_id, receptor, ligand, temperature_C, method)
    );

    -- Physical constants (desolvation, solvation, transfer energies)
    CREATE TABLE IF NOT EXISTS physical_constants (
        const_id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id),
        quantity TEXT NOT NULL,   -- 'dH_sol', 'dG_hyd', 'dCp', 'Ka', 'barrier_height'
        compound TEXT NOT NULL,
        value REAL NOT NULL,
        unit TEXT NOT NULL,
        temperature_C REAL,
        solvent TEXT DEFAULT 'water',
        conditions TEXT,
        table_ref TEXT,
        confidence TEXT DEFAULT 'medium',
        raw_text TEXT
    );

    -- Crystal structure contacts
    CREATE TABLE IF NOT EXISTS contacts (
        contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id),
        pdb_id TEXT,
        receptor TEXT,
        ligand TEXT,
        residue_position TEXT,   -- 'C3-OH', 'C4-OH', etc.
        contact_type TEXT,       -- 'hbond', 'ch_pi', 'vdw', 'water_bridge', 'metal'
        partner_residue TEXT,    -- 'Asn14', 'Trp62', 'Ca2+'
        distance_A REAL,
        angle_deg REAL,
        notes TEXT,
        confidence TEXT DEFAULT 'medium',
        raw_text TEXT
    );

    -- Raw table extractions (for manual review)
    CREATE TABLE IF NOT EXISTS raw_tables (
        table_id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id),
        page_number INTEGER,
        table_index INTEGER,     -- nth table on that page
        headers TEXT,            -- JSON list of column headers
        rows TEXT,               -- JSON list of row lists
        classified_as TEXT,      -- 'binding', 'physical', 'contacts', 'other', 'unclassified'
        confidence TEXT DEFAULT 'low',
        extracted INTEGER DEFAULT 0  -- 1 if data was successfully parsed into typed tables
    );

    -- Manual review queue
    CREATE TABLE IF NOT EXISTS review_queue (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id INTEGER NOT NULL REFERENCES papers(paper_id),
        source_table TEXT,       -- which table the issue is from
        source_id INTEGER,
        issue TEXT,              -- 'ambiguous_unit', 'missing_value', 'unusual_format'
        context TEXT,            -- surrounding text
        resolved INTEGER DEFAULT 0,
        resolution TEXT
    );
    """)

    conn.commit()
    return conn


def file_hash(filepath: str) -> str:
    """SHA256 hash of file contents for dedup."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def is_paper_processed(conn: sqlite3.Connection, fhash: str) -> bool:
    """Check if a paper has already been processed."""
    row = conn.execute("SELECT paper_id FROM papers WHERE file_hash = ?", (fhash,)).fetchone()
    return row is not None


def register_paper(conn: sqlite3.Connection, filename: str, fhash: str,
                   n_pages: int, title: str = "", authors: str = "",
                   year: int = None, journal: str = "", doi: str = "") -> int:
    """Register a new paper. Returns paper_id."""
    now = datetime.now().isoformat()
    cur = conn.execute("""
        INSERT INTO papers (filename, file_hash, title, authors, year, journal, doi,
                           first_processed, last_processed, n_pages, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processed')
    """, (filename, fhash, title, authors, year, journal, doi, now, now, n_pages))
    conn.commit()
    return cur.lastrowid


def insert_binding(conn: sqlite3.Connection, paper_id: int, data: Dict[str, Any]) -> Optional[int]:
    """Insert a binding data row. Returns bind_id or None on conflict."""
    cols = ['paper_id', 'receptor', 'ligand', 'Ka', 'Kd', 'Kd_unit',
            'dG', 'dG_unit', 'dH', 'dH_unit', 'TdS', 'TdS_unit',
            'dCp', 'n_sites', 'temperature_C', 'pH', 'buffer', 'method',
            'conditions', 'table_ref', 'confidence', 'raw_text']
    data['paper_id'] = paper_id
    vals = [data.get(c) for c in cols]
    placeholders = ','.join(['?'] * len(cols))
    colnames = ','.join(cols)
    try:
        cur = conn.execute(f"INSERT OR IGNORE INTO binding_data ({colnames}) VALUES ({placeholders})", vals)
        conn.commit()
        return cur.lastrowid if cur.rowcount > 0 else None
    except Exception as e:
        print(f"  Warning: binding insert failed: {e}")
        return None


def insert_physical_constant(conn: sqlite3.Connection, paper_id: int, data: Dict[str, Any]) -> Optional[int]:
    cols = ['paper_id', 'quantity', 'compound', 'value', 'unit',
            'temperature_C', 'solvent', 'conditions', 'table_ref',
            'confidence', 'raw_text']
    data['paper_id'] = paper_id
    vals = [data.get(c) for c in cols]
    placeholders = ','.join(['?'] * len(cols))
    colnames = ','.join(cols)
    try:
        cur = conn.execute(f"INSERT INTO physical_constants ({colnames}) VALUES ({placeholders})", vals)
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        print(f"  Warning: constant insert failed: {e}")
        return None


def insert_raw_table(conn: sqlite3.Connection, paper_id: int,
                     page_number: int, table_index: int,
                     headers: list, rows: list,
                     classified_as: str = 'unclassified',
                     confidence: str = 'low') -> int:
    cur = conn.execute("""
        INSERT INTO raw_tables (paper_id, page_number, table_index,
                               headers, rows, classified_as, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (paper_id, page_number, table_index,
          json.dumps(headers), json.dumps(rows),
          classified_as, confidence))
    conn.commit()
    return cur.lastrowid


def add_review_item(conn: sqlite3.Connection, paper_id: int,
                    source_table: str, source_id: int,
                    issue: str, context: str = ""):
    conn.execute("""
        INSERT INTO review_queue (paper_id, source_table, source_id, issue, context)
        VALUES (?, ?, ?, ?, ?)
    """, (paper_id, source_table, source_id, issue, context))
    conn.commit()


def get_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    """Get database statistics."""
    stats = {}
    for table in ['papers', 'binding_data', 'physical_constants', 'contacts',
                  'raw_tables', 'review_queue']:
        row = conn.execute(f"SELECT COUNT(*) as n FROM {table}").fetchone()
        stats[table] = row['n']
    # Unresolved reviews
    row = conn.execute("SELECT COUNT(*) as n FROM review_queue WHERE resolved=0").fetchone()
    stats['unresolved_reviews'] = row['n']
    return stats
