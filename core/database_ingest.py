"""
core/database_ingest.py — Sprint 37: Bulk database parsers

Reads downloaded database files and converts to UniversalComplex entries.
Each parser handles format differences and normalizes to the universal schema.

Supported:
  - PDBbind refined/general index files
  - SupraBank CSV export
  - Rekharsky & Inoue 2007 tables (manual format)
  - NIST/Martell metal complex tables
  - Custom CSV (generic fallback)

All parsers return list[UniversalComplex].
"""
import csv
import os
import math

from core.universal_schema import UniversalComplex


# ═══════════════════════════════════════════════════════════════════════════
# PDBbind Parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_pdbbind_index(filepath, max_entries=None):
    """Parse PDBbind general/refined set index file.

    PDBbind index format (INDEX_general_PL_data.YEAR):
      Column layout (space-separated, header lines start with #):
      PDB_code  resolution  release_year  -logKd/Ki  Kd/Ki_value  reference  ligand_name

    Returns list of UniversalComplex with binding_mode='protein_ligand'.
    Guest SMILES must be enriched separately from SDF files.
    """
    entries = []
    if not os.path.exists(filepath):
        print(f"  ⚠ PDBbind index not found: {filepath}")
        return entries

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            pdb_id = parts[0]
            try:
                neg_log_kd = float(parts[3])  # -log(Kd) or -log(Ki)
            except (ValueError, IndexError):
                continue

            kd_str = parts[4] if len(parts) > 4 else ""
            ligand_name = parts[-1] if len(parts) > 6 else pdb_id

            uc = UniversalComplex(
                name=f"PDB:{pdb_id}",
                binding_mode="protein_ligand",
                log_Ka_exp=neg_log_kd,  # PDBbind gives -logKd = logKa
                host_name=pdb_id,
                host_type="protein",
                host_pdb_id=pdb_id,
                guest_name=ligand_name,
                source="pdbbind",
                source_id=pdb_id,
                series_id=f"pdbbind_{pdb_id[:2]}",
                phase="Phase7",
                confidence="high",  # PDBbind refined set is curated
                notes=f"Kd/Ki={kd_str}",
            )
            entries.append(uc)

            if max_entries and len(entries) >= max_entries:
                break

    print(f"  Parsed {len(entries)} PDBbind entries from {os.path.basename(filepath)}")
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# SupraBank Parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_suprabank_csv(filepath, max_entries=None):
    """Parse SupraBank CSV export.

    SupraBank fields include: host_name, guest_name, guest_smiles,
    Ka, DeltaG, DeltaH, TDeltaS, solvent, temperature, pH, reference.

    Returns list of UniversalComplex with binding_mode='host_guest_inclusion'.
    """
    entries = []
    if not os.path.exists(filepath):
        print(f"  ⚠ SupraBank CSV not found: {filepath}")
        return entries

    with open(filepath, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # SupraBank typically has log(Ka) or Ka directly
                ka_str = row.get("Ka", row.get("log_Ka", row.get("logKa", "")))
                if not ka_str:
                    continue

                # Determine if value is Ka or log(Ka)
                ka_val = float(ka_str)
                if ka_val > 100:  # Likely Ka, not log(Ka)
                    log_ka = math.log10(ka_val)
                else:
                    log_ka = ka_val

                host_name = row.get("host_name", row.get("Host", "unknown"))
                guest_name = row.get("guest_name", row.get("Guest", "unknown"))
                guest_smiles = row.get("guest_smiles", row.get("SMILES", ""))

                temp = float(row.get("temperature", row.get("T", 25.0)))
                ph = float(row.get("pH", 7.0))

                uc = UniversalComplex(
                    name=f"{host_name}:{guest_name}",
                    binding_mode="host_guest_inclusion",
                    log_Ka_exp=round(log_ka, 2),
                    temperature_C=temp,
                    ph=ph,
                    host_name=host_name,
                    host_type=_classify_host_type(host_name),
                    guest_name=guest_name,
                    guest_smiles=guest_smiles,
                    source="suprabank",
                    source_id=row.get("id", ""),
                    series_id=f"supra_{host_name}",
                    phase="Phase6",
                    confidence="medium",
                )
                entries.append(uc)

            except (ValueError, KeyError):
                continue

            if max_entries and len(entries) >= max_entries:
                break

    print(f"  Parsed {len(entries)} SupraBank entries from {os.path.basename(filepath)}")
    return entries


def _classify_host_type(name):
    """Infer host_type from host name string."""
    nl = name.lower()
    if "cyclodextrin" in nl or "-cd" in nl or "α-" in nl or "β-" in nl or "γ-" in nl:
        return "cyclodextrin"
    if "cucurbit" in nl or "cb[" in nl or "cb7" in nl or "cb6" in nl:
        return "cucurbituril"
    if "calix" in nl:
        return "calixarene"
    if "pillar" in nl:
        return "pillararene"
    if "crown" in nl:
        return "crown_ether"
    if "cryptand" in nl:
        return "cryptand"
    if "cyclophane" in nl:
        return "cyclophane"
    return "synthetic_receptor"


# ═══════════════════════════════════════════════════════════════════════════
# Generic CSV Parser (works for any tabular binding data)
# ═══════════════════════════════════════════════════════════════════════════

def parse_generic_csv(filepath, column_map=None, binding_mode="unknown", source="other",
                       max_entries=None):
    """Parse any CSV with binding data using a column mapping.

    column_map: dict mapping UniversalComplex field names to CSV column names.
    Minimum required: {"log_Ka_exp": "your_logKa_column", "name": "your_name_column"}
    """
    entries = []
    if not os.path.exists(filepath):
        print(f"  ⚠ CSV not found: {filepath}")
        return entries

    if column_map is None:
        column_map = {}

    with open(filepath, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row.get(column_map.get("name", "name"), "")
                log_ka_col = column_map.get("log_Ka_exp", "log_Ka")
                log_ka = float(row.get(log_ka_col, 0))

                if log_ka == 0 and "Ka" in column_map:
                    ka = float(row.get(column_map["Ka"], 0))
                    if ka > 0:
                        log_ka = math.log10(ka)

                uc = UniversalComplex(
                    name=name,
                    binding_mode=binding_mode,
                    log_Ka_exp=round(log_ka, 2),
                    source=source,
                    guest_smiles=row.get(column_map.get("guest_smiles", "SMILES"), ""),
                    host_name=row.get(column_map.get("host_name", "host"), ""),
                    guest_name=row.get(column_map.get("guest_name", "guest"), ""),
                )
                entries.append(uc)

            except (ValueError, KeyError):
                continue

            if max_entries and len(entries) >= max_entries:
                break

    print(f"  Parsed {len(entries)} entries from {os.path.basename(filepath)}")
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Metal Complex Converter (backward compatible with existing VALIDATION_LIBRARY)
# ═══════════════════════════════════════════════════════════════════════════

def convert_metal_validation_library(old_library):
    """Convert existing ExperimentalComplex list to UniversalComplex list.

    Preserves all metal-specific fields. Adds binding_mode and source metadata.
    """
    entries = []
    for c in old_library:
        uc = UniversalComplex(
            name=c.name,
            binding_mode="metal_coordination",
            log_Ka_exp=c.log_K_exp,
            host_name=c.metal_formula,
            host_type="metal_ion",
            host_charge=c.metal_charge,
            guest_name=c.name.split("-", 1)[-1] if "-" in c.name else c.name,
            # Metal-specific
            metal_formula=c.metal_formula,
            metal_charge=c.metal_charge,
            metal_d_electrons=c.metal_d_electrons,
            donor_atoms=c.donor_atoms,
            donor_subtypes=getattr(c, "donor_subtypes", []),
            chelate_rings=c.chelate_rings,
            ring_sizes=getattr(c, "ring_sizes", []),
            denticity=c.denticity,
            donor_type=c.donor_type,
            is_macrocyclic=getattr(c, "is_macrocyclic", False),
            cavity_radius_nm=getattr(c, "cavity_radius_nm", 0.0),
            is_cage=getattr(c, "is_cage", False),
            # Metadata
            scaffold_type=getattr(c, "scaffold_type", "free"),
            geometry=getattr(c, "geometry", "octahedral"),
            source=c.source if hasattr(c, "source") else "manual",
            notes=getattr(c, "notes", ""),
            confidence="high",
            phase="Phase1-5",
        )
        entries.append(uc)

    print(f"  Converted {len(entries)} metal complexes to UniversalComplex")
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Enrichment Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def enrich_all(entries):
    """Run host registry lookup + RDKit guest compute on all entries.

    Modifies entries in-place. Graceful if RDKit unavailable.
    """
    from core.host_registry import enrich_complex_host

    enriched_guest = 0
    enriched_host = 0

    for uc in entries:
        # Host enrichment (always available)
        before_cavity = uc.cavity_volume_A3
        enrich_complex_host(uc)
        if uc.cavity_volume_A3 != before_cavity:
            enriched_host += 1

        # Guest enrichment (needs RDKit)
        if uc.guest_smiles:
            try:
                from core.guest_compute import enrich_complex
                before_vol = uc.guest_volume_A3
                enrich_complex(uc)
                if uc.guest_volume_A3 != before_vol:
                    enriched_guest += 1
            except ImportError:
                pass

    print(f"  Enriched: {enriched_host} hosts, {enriched_guest} guests (RDKit)")
    return entries
