"""
paper_extractor/patterns.py -- Table classification and value extraction
=========================================================================

Rules-based pattern matching for identifying and parsing data from
extracted PDF tables and text. Each data type has:
  - Header patterns for table classification
  - Value extraction regex for parsing cells
  - Unit normalization
"""

import re
from typing import List, Dict, Any, Optional, Tuple


# =====================================================================
# TABLE CLASSIFICATION
# =====================================================================
# Score each table against data type patterns. Highest score wins.

# Header keywords that indicate binding thermodynamics
BINDING_KEYWORDS = [
    r'K[_\s]?[aAdD]', r'[Kk]_?(?:ass|dis)', r'association',
    r'dissociation', r'[\-−]?\s*[Δδ∆][GgHh]', r'delta\s*[GgHh]',
    r'[Tt][Δδ∆][Ss]',
    r'enthalpy', r'entropy', r'free\s*energy',
    r'[Kk]\s*\(?\s*M\s*[\-−]', r'kcal', r'kJ',
    r'IC\s*50', r'EC\s*50', r'binding',
    r'stoichiometry', r'\bn\b.*site',
]

# Header keywords for physical constants / dissolution / solvation
PHYSICAL_KEYWORDS = [
    r'[Δδ∆].*[Hh].*sol', r'dissolution', r'solvation',
    r'hydration', r'transfer', r'sublimation',
    r'[Cc][Pp]', r'heat\s*capacity', r'barrier',
    r'torsion', r'rotation', r'frequency',
    r'refractive', r'density', r'viscosity',
    r'solubility', r'activity\s*coefficient',
]

# Header keywords for crystal contacts
CONTACT_KEYWORDS = [
    r'hydrogen\s*bond', r'[Hh]-bond', r'contact',
    r'distance.*[ÅA]', r'angle.*deg',
    r'residue', r'atom', r'interaction',
    r'[Cc][Hh].?[ππ]', r'stacking', r'van\s*der\s*[Ww]aals',
    r'coordination', r'water.*bridge',
]


def classify_table(headers: List[str], first_rows: List[List[str]] = None) -> Tuple[str, float]:
    """
    Classify a table by its headers (and optionally first rows).

    Returns (classification, confidence) where classification is one of:
      'binding', 'physical', 'contacts', 'other'
    and confidence is 0-1.
    """
    header_text = ' '.join(h.lower() for h in headers if h)

    # Also check first data row for clues
    row_text = ''
    if first_rows and first_rows[0]:
        row_text = ' '.join(str(c).lower() for c in first_rows[0] if c)

    combined = header_text + ' ' + row_text

    scores = {
        'binding': _keyword_score(combined, BINDING_KEYWORDS),
        'physical': _keyword_score(combined, PHYSICAL_KEYWORDS),
        'contacts': _keyword_score(combined, CONTACT_KEYWORDS),
    }

    # Disambiguation: if both binding and physical score, check for
    # distinguishing terms
    if scores['binding'] > 0 and scores['physical'] > 0:
        # "sol", "dissolution", "hydration", "transfer" strongly favor physical
        phys_strong = len(re.findall(
            r'_sol\b|sol[uv]|dissolut|hydrat|transfer|sublim|barrier|torsion|rotat|heat.?cap',
            combined, re.IGNORECASE))
        # "Ka", "Kd", "IC50", "binding", "association" strongly favor binding
        bind_strong = len(re.findall(
            r'K[_\s]?[aAdD]|IC\s*50|associat|dissociat|binding\s*constant',
            combined, re.IGNORECASE))
        scores['physical'] += phys_strong * 2
        scores['binding'] += bind_strong * 2

    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score < 0.5:
        return 'other', best_score

    # Normalize to 0-1
    confidence = min(1.0, best_score / 3.0)
    return best, confidence


def _keyword_score(text: str, patterns: List[str]) -> float:
    """Count keyword matches."""
    score = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1
    return score


# =====================================================================
# VALUE EXTRACTION
# =====================================================================

def parse_number(s: str) -> Optional[float]:
    """
    Parse a number from a table cell. Handles:
      - Plain numbers: 1.23, -4.56
      - Scientific notation: 1.2e4, 1.2 x 10^4, 1.2 × 10^4
      - Parenthetical uncertainty: 1.23(5), 1.23 ± 0.05
      - Unicode minus: −4.56
      - Comma thousands: 1,234
    Returns None if unparseable.
    """
    if not s or not s.strip():
        return None

    s = s.strip()

    # Non-binding indicators
    if s.upper() in ('NB', 'N.B.', 'ND', 'N.D.', '-', '—', '–', 'NI', 'N.I.'):
        return None

    # Unicode minus
    s = s.replace('−', '-').replace('–', '-').replace('—', '-')

    # Remove ± and everything after
    s = re.sub(r'\s*[±]\s*[\d.]+', '', s)

    # Remove parenthetical uncertainty: 1.23(5)
    s = re.sub(r'\(\d+\)', '', s)

    # Remove commas in numbers
    s = s.replace(',', '')

    # Scientific notation: "1.2 x 10^4" or "1.2 × 10^4" or "1.2 × 10 4"
    sci_match = re.match(r'([-]?\d+\.?\d*)\s*[×xX]\s*10\s*[\^]?\s*([-]?\d+)', s)
    if sci_match:
        mantissa = float(sci_match.group(1))
        exponent = int(sci_match.group(2))
        return mantissa * (10 ** exponent)

    # Plain scientific: 1.2e4
    try:
        return float(s)
    except ValueError:
        pass

    # Try extracting first number
    num_match = re.search(r'([-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', s)
    if num_match:
        try:
            return float(num_match.group(1))
        except ValueError:
            pass

    return None


def detect_unit(header: str, cell: str = "") -> Optional[str]:
    """
    Detect unit from a column header or cell value.
    """
    combined = (header + ' ' + cell).lower()

    # Energy units
    if re.search(r'kj\s*/?\s*mol|kj\s*mol', combined):
        return 'kJ/mol'
    if re.search(r'kcal\s*/?\s*mol|kcal\s*mol', combined):
        return 'kcal/mol'
    if re.search(r'cal\s*/?\s*mol\s*/?\s*k|cal\s*mol.*k', combined):
        return 'cal/mol/K'

    # Concentration
    if re.search(r'\bm[−\-]1\b|m\s*[\-−]\s*1|\bm\^?\s*[\-−]\s*1', combined):
        return 'M^-1'
    if re.search(r'\bmm\b|millimol', combined):
        return 'mM'
    if re.search(r'[μµ]m\b|micromol', combined):
        return 'uM'

    # Distance
    if re.search(r'[Åå]|angstrom', combined):
        return 'A'
    if re.search(r'\bnm\b', combined):
        return 'nm'

    # Angle
    if re.search(r'deg|°', combined):
        return 'deg'

    # Temperature
    if re.search(r'[°]?\s*c\b|celsius', combined):
        return 'C'
    if re.search(r'\bk\b|kelvin', combined):
        return 'K'

    return None


def normalize_energy_to_kj(value: float, unit: str) -> float:
    """Convert energy to kJ/mol."""
    if unit == 'kcal/mol':
        return value * 4.184
    if unit == 'kJ/mol':
        return value
    if unit == 'cal/mol':
        return value * 4.184e-3
    return value  # unknown unit, return as-is


# =====================================================================
# BINDING TABLE PARSER
# =====================================================================

# Column header mapping: what header text maps to which field
BINDING_HEADER_MAP = {
    'Ka': [r'K\s*a', r'K\s*ass', r'association\s*constant'],
    'Kd': [r'K\s*d', r'K\s*dis', r'dissociation\s*constant', r'IC\s*50'],
    'dG': [r'[Δδ∆d]\s*G', r'free\s*energy', r'\-?\s*[Δδ∆d]G'],
    'dH': [r'[Δδ∆d]\s*H', r'enthalpy', r'\-?\s*[Δδ∆d]H'],
    'TdS': [r'T\s*[Δδ∆d]\s*S', r'[Δδ∆d]S', r'entropy', r'\-?\s*T[Δδ∆d]S'],
    'dCp': [r'[Δδ∆d]\s*C\s*p', r'heat\s*capacity'],
    'n_sites': [r'\bn\b', r'stoich', r'sites?\s*/?\s*mono'],
    'ligand': [r'ligand', r'sugar', r'saccharide', r'carbohydrate', r'compound', r'substrate'],
    'receptor': [r'lectin', r'protein', r'receptor', r'host'],
}


def map_headers_to_fields(headers: List[str]) -> Dict[int, str]:
    """
    Map column indices to field names.
    Returns {col_index: field_name}.
    """
    mapping = {}
    for i, header in enumerate(headers):
        if not header:
            continue
        h_lower = header.lower().strip()
        for field, patterns in BINDING_HEADER_MAP.items():
            for pattern in patterns:
                if re.search(pattern, h_lower, re.IGNORECASE):
                    if field not in mapping.values():  # don't double-assign
                        mapping[i] = field
                    break
    return mapping


def parse_binding_table(headers: List[str], rows: List[List[str]],
                        default_receptor: str = "",
                        default_method: str = "ITC") -> List[Dict[str, Any]]:
    """
    Parse a table classified as binding data.
    Returns list of dicts ready for db.insert_binding().
    """
    col_map = map_headers_to_fields(headers)
    if not col_map:
        return []

    # Detect units from headers
    units = {}
    for col_idx, field in col_map.items():
        if field in ('Ka', 'Kd', 'dG', 'dH', 'TdS', 'dCp'):
            unit = detect_unit(headers[col_idx])
            if unit:
                units[field] = unit

    results = []
    for row in rows:
        if not row or all(not c.strip() for c in row if c):
            continue

        entry = {
            'receptor': default_receptor,
            'method': default_method,
            'confidence': 'medium',
        }

        for col_idx, field in col_map.items():
            if col_idx >= len(row):
                continue
            cell = row[col_idx]

            if field in ('ligand', 'receptor'):
                entry[field] = cell.strip()
            elif field in ('Ka', 'Kd', 'dG', 'dH', 'TdS', 'dCp', 'n_sites'):
                val = parse_number(cell)
                if val is not None:
                    entry[field] = val
                    if field in units:
                        entry[f'{field}_unit'] = units[field]

        # Only keep if we got at least ligand + one numeric value
        has_numeric = any(isinstance(entry.get(f), (int, float))
                         for f in ('Ka', 'Kd', 'dG', 'dH', 'TdS'))
        has_ligand = bool(entry.get('ligand', '').strip())

        if has_numeric and has_ligand:
            entry['raw_text'] = ' | '.join(str(c) for c in row if c)
            results.append(entry)

    return results


# =====================================================================
# TEXT PREPROCESSING
# =====================================================================

def normalize_text(text: str) -> str:
    """
    Normalize PDF-extracted text for pattern matching.
    Fixes common PDF artifacts that break regex.
    """
    # Replace newlines between parts of the same token
    # e.g. "K\na\n=" -> "Ka ="
    t = text

    # Fix split Ka/Kd/dH/dG tokens
    t = re.sub(r'K\s*\n\s*a\b', 'Ka', t)
    t = re.sub(r'K\s*\n\s*d\b', 'Kd', t)
    t = re.sub(r'[Δδ∆]\s*\n\s*H\b', 'ΔH', t)
    t = re.sub(r'[Δδ∆]\s*\n\s*G\b', 'ΔG', t)
    t = re.sub(r'[Δδ∆]\s*\n\s*S\b', 'ΔS', t)
    t = re.sub(r'T\s*[Δδ∆]\s*\n?\s*S\b', 'TΔS', t)

    # Collapse runs of whitespace (but preserve paragraph breaks)
    t = re.sub(r'[ \t]+', ' ', t)
    # Collapse single newlines (keep double = paragraph break)
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)

    # Fix "minus" artifacts
    t = t.replace('−', '-').replace('–', '-')

    # Fix "x10" scientific notation: "4.9 × 10^5" -> "4.9e5"
    t = re.sub(r'(\d)\s*[×xX]\s*10\s*[\^]?\s*([\-]?\d+)', r'\1e\2', t)

    return t


def get_context(text: str, pos: int, window: int = 150) -> str:
    """Get surrounding text for context, respecting sentence boundaries."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    chunk = text[start:end].replace('\n', ' ').strip()

    # Try to trim to sentence containing the match
    # Find the sentence boundary before and after the match position
    relative_pos = pos - start
    # Look backward for sentence start
    sent_start = 0
    for delim_pos in range(relative_pos, 0, -1):
        if chunk[delim_pos] in '.!?' and delim_pos < relative_pos - 2:
            sent_start = delim_pos + 1
            break
    # Look forward for sentence end
    sent_end = len(chunk)
    for delim_pos in range(relative_pos, len(chunk)):
        if chunk[delim_pos] in '.!?' and delim_pos > relative_pos + 5:
            sent_end = delim_pos + 1
            break

    return chunk[sent_start:sent_end].strip()


# =====================================================================
# TEXT-BASED EXTRACTION (for data mentioned in prose, not tables)
# =====================================================================

def extract_binding_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract binding data mentioned in running text.
    Handles garbled PDF whitespace, split tokens, missing spaces.
    """
    results = []
    seen_values = set()  # dedup by (field, value)

    # Normalize text first
    t = normalize_text(text)

    # ── PATTERN SET 1: Ka/Kd = VALUE ──────────────────────────────────
    # Handles: "Ka = 18,600 M-1", "Ka=3981M-1", "Ka of 1.2 x 10^4"
    # Also: "Ka (M-1)" followed by value on next line
    ka_patterns = [
        # "Ka = VALUE M-1" or "Ka=VALUE" (with optional spaces)
        re.compile(
            r'Ka\s*(?:=|is|was|of|:)\s*'
            r'([\d.,]+(?:[eE][\-]?\d+|\s*[×xX]\s*10\s*[\^]?\s*[\-]?\d+)?)\s*'
            r'(M\s*[\-]\s*1|M-1)?',
            re.IGNORECASE
        ),
        # "Kd = VALUE mM/uM/nM"
        re.compile(
            r'Kd\s*(?:=|is|was|of|:)\s*'
            r'([\d.,]+(?:[eE][\-]?\d+|\s*[×xX]\s*10\s*[\^]?\s*[\-]?\d+)?)\s*'
            r'(mM|[μµu]M|nM|M)',
            re.IGNORECASE
        ),
        # "association constant" or "binding constant" followed by value
        re.compile(
            r'(?:association|binding)\s*constants?\s*(?:\([^)]*\))?\s*'
            r'(?:=|is|was|of|:)\s*'
            r'([\d.,]+(?:[eE][\-]?\d+|\s*[×xX]\s*10\s*[\^]?\s*[\-]?\d+)?)\s*'
            r'(M\s*[\-]\s*1)?',
            re.IGNORECASE
        ),
        # "Ka VALUE" with number following (no = sign, common in garbled text)
        re.compile(
            r'Ka\s+'
            r'([\d.,]+(?:[eE][\-]?\d+|\s*[×xX]\s*10\s*[\^]?\s*[\-]?\d+)?)\s*'
            r'(M\s*[\-]\s*1)?',
        ),
    ]

    for pattern in ka_patterns:
        for match in pattern.finditer(t):
            val = parse_number(match.group(1))
            if val is None or val == 0:
                continue
            # Skip if too small (likely not Ka) or already seen
            key = ('Ka', round(val, 1))
            if key in seen_values:
                continue
            seen_values.add(key)

            context = get_context(t, match.start())
            ligand, receptor = _extract_context_names(context)

            results.append({
                'receptor': receptor,
                'ligand': ligand,
                'Ka': val,
                'method': 'text_extraction',
                'confidence': 'low',
                'raw_text': match.group(0)[:200].strip(),
                'conditions': context[:200],
            })

    # ── PATTERN SET 2: ΔH = VALUE kcal/mol ────────────────────────────
    dh_patterns = [
        # "ΔH = -8.4 kcal/mol" or "ΔHwas-6.6kcal/mol" (no spaces)
        re.compile(
            r'[Δδ∆]H\s*(?:°\s*)?(?:=|is|was|of|:)\s*'
            r'([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol)',
            re.IGNORECASE
        ),
        # "enthalpy of -8.4 kcal/mol"
        re.compile(
            r'enthalpy\s+(?:of\s+)?(?:binding\s+)?'
            r'(?:=|is|was|of|:)?\s*'
            r'([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol)',
            re.IGNORECASE
        ),
        # "-ΔH = 8.4" or "-ΔH (kcal/mol)" patterns
        re.compile(
            r'[\-]?\s*[Δδ∆]H\s*(?:\([^)]*\))?\s*(?:=|:)\s*'
            r'([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol)?',
            re.IGNORECASE
        ),
    ]

    for pattern in dh_patterns:
        for match in pattern.finditer(t):
            val = parse_number(match.group(1))
            if val is None:
                continue
            unit = match.group(2) if match.lastindex >= 2 and match.group(2) else 'kcal/mol'
            key = ('dH', round(val, 2))
            if key in seen_values:
                continue
            seen_values.add(key)

            context = get_context(t, match.start())
            ligand, receptor = _extract_context_names(context)

            results.append({
                'receptor': receptor,
                'ligand': ligand,
                'dH': val,
                'dH_unit': unit.replace('/', '/'),
                'method': 'text_extraction',
                'confidence': 'low',
                'raw_text': match.group(0)[:200].strip(),
                'conditions': context[:200],
            })

    # ── PATTERN SET 3: ΔG = VALUE ─────────────────────────────────────
    dg_patterns = [
        re.compile(
            r'[Δδ∆]G\s*(?:°\s*)?(?:=|is|was|of|:)\s*'
            r'([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol)',
            re.IGNORECASE
        ),
        re.compile(
            r'free\s*energy\s+(?:of\s+)?(?:binding\s+)?'
            r'(?:=|is|was|of|:)?\s*'
            r'([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol)',
            re.IGNORECASE
        ),
    ]

    for pattern in dg_patterns:
        for match in pattern.finditer(t):
            val = parse_number(match.group(1))
            if val is None:
                continue
            unit = match.group(2) if match.lastindex >= 2 and match.group(2) else 'kcal/mol'
            key = ('dG', round(val, 2))
            if key in seen_values:
                continue
            seen_values.add(key)

            context = get_context(t, match.start())
            ligand, receptor = _extract_context_names(context)

            results.append({
                'receptor': receptor,
                'ligand': ligand,
                'dG': val,
                'dG_unit': unit,
                'method': 'text_extraction',
                'confidence': 'low',
                'raw_text': match.group(0)[:200].strip(),
                'conditions': context[:200],
            })

    # ── PATTERN SET 4: TΔS = VALUE ────────────────────────────────────
    tds_patterns = [
        re.compile(
            r'T\s*[Δδ∆]S\s*(?:°\s*)?(?:=|is|was|of|:)\s*'
            r'([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol)',
            re.IGNORECASE
        ),
    ]

    for pattern in tds_patterns:
        for match in pattern.finditer(t):
            val = parse_number(match.group(1))
            if val is None:
                continue
            unit = match.group(2) if match.lastindex >= 2 and match.group(2) else 'kcal/mol'
            key = ('TdS', round(val, 2))
            if key in seen_values:
                continue
            seen_values.add(key)

            context = get_context(t, match.start())
            ligand, receptor = _extract_context_names(context)

            results.append({
                'receptor': receptor,
                'ligand': ligand,
                'TdS': val,
                'TdS_unit': unit,
                'method': 'text_extraction',
                'confidence': 'low',
                'raw_text': match.group(0)[:200].strip(),
                'conditions': context[:200],
            })

    return results


# =====================================================================
# CONTEXT NAME EXTRACTION
# =====================================================================

# Known sugar names for context matching
KNOWN_SUGARS = [
    'glucose', 'mannose', 'galactose', 'fructose', 'fucose', 'xylose',
    'ribose', 'lactose', 'sucrose', 'maltose', 'cellobiose', 'chitobiose',
    'GlcNAc', 'GalNAc', 'LacNAc', 'ManNAc', 'Neu5Ac', 'sialic',
    'MeαMan', 'MeαGlc', 'MeαGal', 'MeβGal', 'MeβGlc',
    'MeRMan', 'MeRGlc', 'MeRGal', 'MebGal', 'MebGlc',
    'trimannoside', 'mannotriose', 'mannobiose', 'chitotriose',
    'D-glucose', 'D-mannose', 'D-galactose', 'D-fructose', 'D-xylose',
    'D-ribose', 'D-cellobiose',
    'methyl α-D-mannopyranoside', 'methyl α-D-glucopyranoside',
    '2-deoxy', '3-deoxy', '4-deoxy', '6-deoxy',
    'thiodigalactoside', 'dithiogalactoside',
    'T-antigen', 'blood group',
]

KNOWN_RECEPTORS = [
    'ConA', 'concanavalin', 'WGA', 'wheat germ',
    'DGL', 'Dioclea', 'galectin', 'PNA', 'peanut',
    'SBA', 'soybean', 'ECorL', 'Erythrina',
    'hevein', 'UDA', 'Urtica', 'GNA', 'Galanthus',
    'ASA', 'Allium', 'NPL', 'Narcissus',
    'artocarpin', 'banana lectin', 'RCA', 'Ricinus',
    'abrin', 'jacalin', 'MBP', 'mannose binding',
    'lysozyme', 'receptor', 'lectin',
    'TC14', 'selectin',
]


def _extract_context_names(context: str) -> tuple:
    """
    Extract ligand and receptor names from surrounding text context.
    Returns (ligand, receptor).
    """
    ligand = ''
    receptor = ''

    ctx_lower = context.lower()

    # Find sugar/ligand
    for sugar in KNOWN_SUGARS:
        if sugar.lower() in ctx_lower:
            ligand = sugar
            break

    # Find receptor
    for rec in KNOWN_RECEPTORS:
        if rec.lower() in ctx_lower:
            receptor = rec
            break

    return ligand, receptor


# =====================================================================
# GARBLED TABLE ROW PARSER
# =====================================================================
# Handles tables where pdfplumber jams rows together, like:
# "D-Glucose10 1 ... | 8,000 500 ... | 18,600 7900 5300 5800 725 180 140 220 60 30"

def parse_garbled_table_rows(raw_text: str, headers_hint: str = "") -> List[Dict[str, Any]]:
    """
    Parse garbled table text where structure is destroyed but data is present.

    Handles patterns like:
      "D-Glucose10 1 D-Galactose15 1 | 8,000 30 | 18,600 180"
    where column 1 has names+numbers jammed together, and other columns
    have the Ka values separated by spaces.
    """
    results = []

    # Split on pipe characters
    columns = [c.strip() for c in raw_text.split('|') if c.strip()]
    if len(columns) < 2:
        return results

    # ── Step 1: Find the column with sugar names ──
    name_col = None
    num_cols = []

    for i, col in enumerate(columns):
        has_sugar = any(s.lower() in col.lower() for s in
                       ['glucose', 'mannose', 'galactose', 'fructose', 'fucose',
                        'xylose', 'ribose', 'lactose', 'cellobiose', 'GlcNAc',
                        'GalNAc', 'LacNAc', 'manno', 'deoxy', 'Glc', 'Man',
                        'Gal', 'Fuc', 'Xyl'])
        if has_sugar:
            name_col = i
        else:
            # Check if this column is mostly numbers
            nums = re.findall(r'[\d,]+(?:\.\d+)?', col)
            if nums:
                num_cols.append(i)

    if name_col is None or not num_cols:
        return results

    # ── Step 2: Split name column into individual entries ──
    name_text = columns[name_col]

    # Pattern: "D-Glucose10" or "2-Deoxy-D-Glucose14" — name followed by
    # a compound number, then maybe a space and another small number
    # Split at transitions from letter/hyphen to digit-digit pattern
    # that marks a new compound number
    sugar_entries = re.split(
        r'(?<=\d)\s+(?=[A-Z2-9])',  # split at "...10 1 D-..." boundaries
        name_text
    )

    # Better: find all "Name + CompoundNumber" patterns
    sugar_names_list = [
        'Glucose', 'Mannose', 'Galactose', 'Fructose', 'Fucose',
        'Xylose', 'Ribose', 'Cellobiose', 'Lactose', 'Maltose',
        'Sucrose', 'Chitobiose', 'GlcNAc', 'GalNAc', 'LacNAc',
        'Glucoside', 'Mannoside', 'Galactoside', 'GlucuronicAcid',
    ]
    sugar_alts = '|'.join(sugar_names_list)
    name_pattern = re.compile(
        r'((?:Methyl|2-Deoxy|3-Deoxy|4-Deoxy|6-Deoxy|N-Acetyl|[DL]-|[αβß]-)*'
        r'(?:' + sugar_alts + r'))'
        r'\s*(\d{0,3})',
        re.IGNORECASE
    )

    sugar_names = []
    for match in name_pattern.finditer(name_text):
        name = match.group(1).strip()
        # Clean up concatenated names
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)  # camelCase split
        sugar_names.append(name)

    if not sugar_names:
        return results

    # ── Step 3: Extract numbers from the last numeric column (usually ITC) ──
    # Prefer the last numeric column (often ITC Ka values)
    best_num_col = num_cols[-1]
    num_text = columns[best_num_col]
    numbers = re.findall(r'[\d,]+(?:\.\d+)?', num_text)
    parsed_numbers = [parse_number(n) for n in numbers]
    parsed_numbers = [n for n in parsed_numbers if n is not None]

    # ── Step 4: Zip names with numbers ──
    n_pairs = min(len(sugar_names), len(parsed_numbers))
    for i in range(n_pairs):
        val = parsed_numbers[i]
        if val <= 0:
            continue
        results.append({
            'ligand': sugar_names[i],
            'Ka': val,
            'confidence': 'low',
            'method': 'garbled_table',
            'raw_text': f"{sugar_names[i]} -> {val} (col {best_num_col})",
        })

    return results


# =====================================================================
# PHYSICAL CONSTANT EXTRACTION FROM TEXT
# =====================================================================

def extract_constants_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract physical constants mentioned in text.
    Handles garbled whitespace and various formats.
    """
    results = []
    seen = set()
    t = normalize_text(text)

    patterns = [
        # "ΔH = VALUE kcal/mol" or "ΔHwas-6.6kcal/mol"
        (re.compile(
            r'[Δδ∆]H\s*(?:°\s*)?(?:sol|hyd|sub|vap|fus|mix)?\s*'
            r'(?:=|is|was|of|:)\s*([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol|cal/?mol)',
            re.IGNORECASE),
         lambda m: {'quantity': 'dH', 'compound': '',
                     'value': parse_number(m.group(1)), 'unit': m.group(2)}),

        # "ΔG = VALUE"
        (re.compile(
            r'[Δδ∆]G\s*(?:°\s*)?(?:sol|hyd|sub|vap|fus|mix)?\s*'
            r'(?:=|is|was|of|:)\s*([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol|cal/?mol)',
            re.IGNORECASE),
         lambda m: {'quantity': 'dG', 'compound': '',
                     'value': parse_number(m.group(1)), 'unit': m.group(2)}),

        # "ΔCp = VALUE cal/mol/K"
        (re.compile(
            r'[Δδ∆]\s*C\s*p\s*(?:°\s*)?'
            r'(?:=|is|was|of|:)\s*([\-]?[\d.,]+)\s*'
            r'(cal/?mol/?K|J/?mol/?K|kJ/?mol/?K|kcal/?mol/?K)',
            re.IGNORECASE),
         lambda m: {'quantity': 'dCp', 'compound': '',
                     'value': parse_number(m.group(1)), 'unit': m.group(2)}),

        # "enthalpy of solution" type patterns
        (re.compile(
            r'(?:enthalpy|heat)\s+of\s+(?:solution|dissolution|hydration|solvation|transfer)\s+'
            r'(?:of\s+(\S+(?:\s+\S+){0,2}?)\s+)?'
            r'(?:=|is|was|:)\s*([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol|cal/?mol)',
            re.IGNORECASE),
         lambda m: {'quantity': 'dH_sol', 'compound': m.group(1) or '',
                     'value': parse_number(m.group(2)), 'unit': m.group(3)}),

        # Barrier height patterns
        (re.compile(
            r'(?:barrier|activation\s*energy)\s*(?:=|is|was|of|:)\s*'
            r'([\-]?[\d.,]+)\s*'
            r'(kcal/?mol|kJ/?mol)',
            re.IGNORECASE),
         lambda m: {'quantity': 'barrier', 'compound': '',
                     'value': parse_number(m.group(1)), 'unit': m.group(2)}),

        # pKa patterns
        (re.compile(
            r'pK\s*a\s*(?:=|is|was|of|:)\s*([\d.,]+)',
            re.IGNORECASE),
         lambda m: {'quantity': 'pKa', 'compound': '',
                     'value': parse_number(m.group(1)), 'unit': ''}),
    ]

    for pattern, extractor in patterns:
        for match in pattern.finditer(t):
            data = extractor(match)
            if data.get('value') is None:
                continue
            key = (data['quantity'], round(data['value'], 3))
            if key in seen:
                continue
            seen.add(key)

            context = get_context(t, match.start())
            data['confidence'] = 'low'
            data['raw_text'] = match.group(0)[:200].strip()
            # Try to get compound name from context
            if not data.get('compound'):
                for sugar in KNOWN_SUGARS:
                    if sugar.lower() in context.lower():
                        data['compound'] = sugar
                        break
            results.append(data)

    return results
