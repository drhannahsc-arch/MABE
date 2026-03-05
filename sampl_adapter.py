"""
MABE Data Adapter: SAMPL Host-Guest Binding Thermodynamics
============================================================
Parses experimental ITC data from SAMPL6, SAMPL7, SAMPL8, SAMPL9 challenge repos.
All values from published ITC measurements — no inference or interpolation.

Sources:
  SAMPL6: github.com/samplchallenges/SAMPL6 (OA, TEMOA, CB8 hosts)
  SAMPL7: github.com/samplchallenges/SAMPL7 (TrimerTrip, GDCC, CD derivatives)
  SAMPL9: github.com/samplchallenges/SAMPL9 (bCD, HbCD, WP6 hosts)

License: CC-BY (cite NIH R01GM124270)
"""

import csv
import io
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HostGuestMeasurement:
    """Single ITC-measured host-guest binding datum."""
    system_id: str              # e.g., "OA-G2"
    host_name: str              # e.g., "OA" (octa acid)
    guest_name: str             # e.g., "(s)-(-)-perillic acid"
    guest_smiles: str           # canonical SMILES
    dG_kcal: float              # binding free energy (kcal/mol)
    dG_err: float               # uncertainty in dG
    dH_kcal: Optional[float]    # binding enthalpy (kcal/mol)
    dH_err: Optional[float]
    TdS_kcal: Optional[float]   # entropy contribution (kcal/mol)
    TdS_err: Optional[float]
    Ka: Optional[float]         # association constant (M^-1)
    Ka_err: Optional[float]
    n_stoich: Optional[float]   # binding stoichiometry
    sampl_edition: int          # 6, 7, 8, or 9
    host_type: str              # "cavitand", "cucurbituril", "cyclodextrin", "pillar[n]arene", "clip"
    conditions: str             # pH, buffer, temperature
    source_doi: str             # DOI of the experimental paper


# Host classification
HOST_METADATA = {
    # SAMPL6
    "OA":       {"type": "cavitand",      "full_name": "Octa Acid",
                 "conditions": "10 mM NaPO4 pH 11.7, 298 K", "doi": "10.1007/s10822-018-0170-6"},
    "TEMOA":    {"type": "cavitand",      "full_name": "Tetra-endo-methyl Octa Acid",
                 "conditions": "10 mM NaPO4 pH 11.7, 298 K", "doi": "10.1007/s10822-018-0170-6"},
    "CB8":      {"type": "cucurbituril",  "full_name": "Cucurbit[8]uril",
                 "conditions": "20 mM NaPO4 pH 7.4, 298 K", "doi": "10.1007/s10822-018-0170-6"},
    # SAMPL7
    "clip":     {"type": "clip",          "full_name": "TrimerTrip (acyclic CB)",
                 "conditions": "20 mM NaPO4 pH 7.4, 298 K", "doi": "10.1039/C9NJ05336K"},
    "OA-sm":    {"type": "cavitand",      "full_name": "Octa Acid (SAMPL7)",
                 "conditions": "10 mM NaPO4 pH 11.5, 298 K", "doi": "10.1007/s10822-020-00363-5"},
    "exoOA":    {"type": "cavitand",      "full_name": "exo Octa Acid",
                 "conditions": "10 mM NaPO4 pH 11.5, 298 K", "doi": "10.1007/s10822-020-00363-5"},
    # SAMPL7 CD derivatives
    "bCD":      {"type": "cyclodextrin",  "full_name": "beta-Cyclodextrin",
                 "conditions": "25 mM NaPO4 pH 6.8, 298 K", "doi": "10.1007/s10822-020-00363-5"},
    "MGLab8":   {"type": "cyclodextrin",  "full_name": "MGLab8 (bCD derivative)",
                 "conditions": "25 mM NaPO4 pH 6.8, 298 K", "doi": "10.1007/s10822-020-00363-5"},
    # SAMPL9
    "WP6":      {"type": "pillararene",   "full_name": "Water-soluble Pillar[6]arene",
                 "conditions": "10 mM NaPO4, 298 K", "doi": "pending"},
    "HbCD":     {"type": "cyclodextrin",  "full_name": "Hydroxypropyl-beta-CD",
                 "conditions": "10 mM NaPO4, 298 K", "doi": "pending"},
}


def _safe_float(val: str) -> Optional[float]:
    """Convert string to float, returning None for empty/NaN."""
    if not val or val.strip() in ("", "nan", "NaN", "NA", "N/A", "-"):
        return None
    try:
        return float(val.strip())
    except ValueError:
        return None


def _classify_host(system_id: str) -> str:
    """Extract host name from system ID like 'OA-G2' or 'clip-g5'."""
    parts = system_id.split("-")
    if len(parts) >= 2:
        # Handle cases like "bCD-PMZ", "WP6-G1", "clip-g3"
        host = parts[0]
        # Also handle "MGLab8-g1" etc.
        if host in HOST_METADATA:
            return host
        # Try joining first parts for compound host names
        for i in range(1, len(parts)):
            candidate = "-".join(parts[:i])
            if candidate in HOST_METADATA:
                return candidate
    return parts[0]


def parse_sampl6(filepath: str) -> list[HostGuestMeasurement]:
    """Parse SAMPL6 experimental_measurements.csv."""
    results = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            host = _classify_host(row["ID"])
            meta = HOST_METADATA.get(host, {"type": "unknown", "conditions": "298 K", "doi": "unknown"})
            results.append(HostGuestMeasurement(
                system_id=row["ID"],
                host_name=host,
                guest_name=row.get("name", ""),
                guest_smiles=row.get("SMILES", ""),
                dG_kcal=float(row["DG"]),
                dG_err=float(row.get("dDG", 0)),
                dH_kcal=_safe_float(row.get("DH")),
                dH_err=_safe_float(row.get("dDH")),
                TdS_kcal=_safe_float(row.get("TDS")),
                TdS_err=_safe_float(row.get("dTDS")),
                Ka=_safe_float(row.get("Ka")),
                Ka_err=_safe_float(row.get("dKa")),
                n_stoich=_safe_float(row.get("n")),
                sampl_edition=6,
                host_type=meta["type"],
                conditions=meta["conditions"],
                source_doi=meta["doi"],
            ))
    return results


def parse_sampl7(filepath: str) -> list[HostGuestMeasurement]:
    """Parse SAMPL7 experimental_measurements.csv (same format as SAMPL6)."""
    results = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            host = _classify_host(row["ID"])
            meta = HOST_METADATA.get(host, {"type": "unknown", "conditions": "298 K", "doi": "unknown"})
            results.append(HostGuestMeasurement(
                system_id=row["ID"],
                host_name=host,
                guest_name=row.get("name", ""),
                guest_smiles=row.get("SMILES", ""),
                dG_kcal=float(row["DG"]),
                dG_err=float(row.get("dDG", 0)),
                dH_kcal=_safe_float(row.get("DH")),
                dH_err=_safe_float(row.get("dDH")),
                TdS_kcal=_safe_float(row.get("TDS")),
                TdS_err=_safe_float(row.get("dTDS")),
                Ka=_safe_float(row.get("Ka")),
                Ka_err=_safe_float(row.get("dKa")),
                n_stoich=_safe_float(row.get("n")),
                sampl_edition=7,
                host_type=meta["type"],
                conditions=meta.get("conditions", "298 K"),
                source_doi=meta.get("doi", "10.1007/s10822-020-00363-5"),
            ))
    return results


def parse_sampl9(filepath: str) -> list[HostGuestMeasurement]:
    """Parse SAMPL9 experimental_measurements.csv."""
    results = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            host = _classify_host(row["ID"])
            meta = HOST_METADATA.get(host, {"type": "unknown", "conditions": "298 K", "doi": "pending"})
            results.append(HostGuestMeasurement(
                system_id=row["ID"],
                host_name=host,
                guest_name=row.get("name", ""),
                guest_smiles=row.get("SMILES", ""),
                dG_kcal=float(row["DG"]),
                dG_err=float(row.get("dDG", 0)),
                dH_kcal=_safe_float(row.get("DH")),
                dH_err=_safe_float(row.get("dDH")),
                TdS_kcal=_safe_float(row.get("TDS")),
                TdS_err=_safe_float(row.get("dTDS")),
                Ka=_safe_float(row.get("Ka")),
                Ka_err=_safe_float(row.get("dKa")),
                n_stoich=_safe_float(row.get("n")),
                sampl_edition=9,
                host_type=meta["type"],
                conditions=meta.get("conditions", "298 K"),
                source_doi=meta.get("doi", "pending"),
            ))
    return results


def load_all_sampl(data_dir: str) -> list[HostGuestMeasurement]:
    """Load all available SAMPL experimental data from a directory."""
    all_data = []
    parsers = {
        "sampl6_exp.csv": parse_sampl6,
        "sampl7_exp.csv": parse_sampl7,
        "sampl9_exp.csv": parse_sampl9,
    }
    for filename, parser in parsers.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 50:
            try:
                entries = parser(filepath)
                all_data.extend(entries)
                print(f"  Loaded {len(entries)} entries from {filename}")
            except Exception as e:
                print(f"  ERROR parsing {filename}: {e}")
        else:
            print(f"  SKIP {filename} (not found or empty)")
    return all_data


def sampl_summary(data: list[HostGuestMeasurement]) -> dict:
    """Generate a summary of the SAMPL dataset for verification."""
    by_host_type = {}
    by_edition = {}
    for d in data:
        by_host_type.setdefault(d.host_type, []).append(d)
        by_edition.setdefault(d.sampl_edition, []).append(d)

    return {
        "total_complexes": len(data),
        "by_host_type": {k: len(v) for k, v in sorted(by_host_type.items())},
        "by_edition": {k: len(v) for k, v in sorted(by_edition.items())},
        "dG_range_kcal": (min(d.dG_kcal for d in data), max(d.dG_kcal for d in data)),
        "hosts_represented": sorted(set(d.host_name for d in data)),
        "has_enthalpy": sum(1 for d in data if d.dH_kcal is not None),
        "has_entropy": sum(1 for d in data if d.TdS_kcal is not None),
    }


# === Conversion utilities for MABE scorer ===

def to_kj_mol(kcal_val: float) -> float:
    """Convert kcal/mol to kJ/mol."""
    return kcal_val * 4.184

def to_mabe_format(m: HostGuestMeasurement) -> dict:
    """Convert a SAMPL measurement to MABE UniversalComplex-compatible dict."""
    return {
        "system_id": m.system_id,
        "host_smiles": None,  # Need to add host SMILES from structure files
        "guest_smiles": m.guest_smiles,
        "experimental_dG_kJ": to_kj_mol(m.dG_kcal),
        "experimental_dG_err_kJ": to_kj_mol(m.dG_err),
        "experimental_dH_kJ": to_kj_mol(m.dH_kcal) if m.dH_kcal is not None else None,
        "experimental_Ka": m.Ka,
        "temperature_K": 298.0,
        "host_type": m.host_type,
        "source": f"SAMPL{m.sampl_edition}",
        "source_doi": m.source_doi,
        "data_quality": {
            "tier": 1,
            "source_id": f"SAMPL{m.sampl_edition}/{m.system_id}",
            "independently_confirmed": True,
            "notes": "ITC measurement, blind challenge, peer-reviewed"
        }
    }


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/sampl"

    print("=" * 60)
    print("MABE SAMPL Host-Guest Data Adapter")
    print("=" * 60)

    data = load_all_sampl(data_dir)
    if not data:
        print("No data loaded. Check data directory path.")
        sys.exit(1)

    summary = sampl_summary(data)
    print(f"\nTotal complexes: {summary['total_complexes']}")
    print(f"By host type: {summary['by_host_type']}")
    print(f"By SAMPL edition: {summary['by_edition']}")
    print(f"ΔG range: {summary['dG_range_kcal'][0]:.1f} to {summary['dG_range_kcal'][1]:.1f} kcal/mol")
    print(f"Hosts: {summary['hosts_represented']}")
    print(f"Entries with ΔH: {summary['has_enthalpy']}")
    print(f"Entries with TΔS: {summary['has_entropy']}")

    print("\n--- Sample entries (MABE format) ---")
    for entry in data[:3]:
        mabe = to_mabe_format(entry)
        print(f"  {mabe['system_id']}: ΔG = {mabe['experimental_dG_kJ']:.1f} kJ/mol "
              f"(Ka = {mabe['experimental_Ka']:.0f} M⁻¹)" if mabe['experimental_Ka'] else
              f"  {mabe['system_id']}: ΔG = {mabe['experimental_dG_kJ']:.1f} kJ/mol")