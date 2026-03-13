"""
paper_extractor/__main__.py -- CLI interface
=============================================

Usage:
  python -m paper_extractor run [folder]       Process PDFs in folder
  python -m paper_extractor export [folder]    Export to Excel
  python -m paper_extractor stats [folder]     Show database stats
  python -m paper_extractor review [folder]    Show items needing review

Default folder: ./papers/
"""

import sys
import os
from pathlib import Path


DEFAULT_FOLDER = "papers"


def main():
    args = sys.argv[1:]

    if not args or args[0] in ('-h', '--help', 'help'):
        print(__doc__)
        return

    command = args[0]
    folder = args[1] if len(args) > 1 else DEFAULT_FOLDER

    # Resolve to absolute path
    folder = str(Path(folder).resolve())

    if command == 'run':
        cmd_run(folder)
    elif command == 'export':
        cmd_export(folder)
    elif command == 'stats':
        cmd_stats(folder)
    elif command == 'review':
        cmd_review(folder)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


def cmd_run(folder: str):
    """Process all PDFs in folder."""
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        print(f"Create it and put PDFs inside, then run again.")
        sys.exit(1)

    pdfs = list(Path(folder).glob("*.pdf"))
    print(f"Paper Extractor v0.1")
    print(f"Folder: {folder}")
    print(f"PDFs found: {len(pdfs)}")
    print()

    from paper_extractor.extractor import process_folder
    counts = process_folder(folder, verbose=True)

    print()
    print("Run 'python -m paper_extractor export' to generate Excel output.")
    print("Run 'python -m paper_extractor review' to see items needing attention.")


def cmd_export(folder: str):
    """Export database to Excel."""
    from paper_extractor.db import get_db_path
    db_path = get_db_path(folder)
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}")
        print("Run 'python -m paper_extractor run' first.")
        sys.exit(1)

    from paper_extractor.export import export_excel
    output = export_excel(folder, verbose=True)
    print(f"\nOpen {output} to review extracted data.")


def cmd_stats(folder: str):
    """Show database statistics."""
    from paper_extractor.db import get_db_path, init_db, get_stats
    db_path = get_db_path(folder)
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}")
        return

    conn = init_db(db_path)
    stats = get_stats(conn)
    conn.close()

    print(f"Paper Extractor Database: {db_path}")
    print(f"  Papers:            {stats['papers']}")
    print(f"  Binding entries:   {stats['binding_data']}")
    print(f"  Physical constants:{stats['physical_constants']}")
    print(f"  Crystal contacts:  {stats['contacts']}")
    print(f"  Raw tables:        {stats['raw_tables']}")
    print(f"  Needs review:      {stats['unresolved_reviews']}")


def cmd_review(folder: str):
    """Show items needing manual review."""
    from paper_extractor.db import get_db_path, init_db
    db_path = get_db_path(folder)
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}")
        return

    conn = init_db(db_path)
    rows = conn.execute("""
        SELECT rq.review_id, p.filename, rq.issue, rq.context
        FROM review_queue rq
        JOIN papers p ON rq.paper_id = p.paper_id
        WHERE rq.resolved = 0
        ORDER BY rq.review_id
    """).fetchall()
    conn.close()

    if not rows:
        print("No items needing review.")
        return

    print(f"{len(rows)} items needing review:\n")
    for row in rows:
        print(f"  [{row['review_id']}] {row['filename']}")
        print(f"       Issue: {row['issue']}")
        if row['context']:
            print(f"       Context: {row['context'][:100]}")
        print()


if __name__ == '__main__':
    main()
