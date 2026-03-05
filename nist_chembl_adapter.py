"""
MABE Data Adapter: NIST WebBook + ChEMBL API
=============================================
Programmatic access to NIST thermochemical data and ChEMBL binding data.

NIST WebBook: webbook.nist.gov (public domain)
ChEMBL: www.ebi.ac.uk/chembl (CC-BY-SA 3.0)
"""

import json
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Optional


# ================================================================
# NIST WebBook Adapter
# ================================================================

@dataclass
class NISTThermochemEntry:
    """Thermochemical data from NIST WebBook."""
    cas_number: str
    name: str
    formula: str
    property_name: str          # e.g., "Delta_f_H_gas", "S_gas", "Cp_gas"
    value: float
    units: str                  # e.g., "kJ/mol", "J/mol·K"
    temperature_K: float
    source: str                 # "NIST WebBook SRD 69"
    url: str


def nist_webbook_url(cas: str, mask: int = 1) -> str:
    """
    Build a NIST WebBook URL for a compound.

    Args:
        cas: CAS registry number (e.g., "7440-50-8" for copper)
        mask: Data type mask. Key values:
            1  = Thermochemistry (gas phase)
            2  = Condensed phase thermochemistry (JANAF)
            4  = Phase change data
            8  = IR spectrum
            20 = Gas phase ion energetics
            200 = Henry's law constants
    """
    cas_clean = cas.replace("-", "")
    return f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas_clean}&Mask={mask}&Units=KJ"


def fetch_nist_gas_phase_thermo(cas: str) -> list[NISTThermochemEntry]:
    """
    Fetch gas-phase thermochemistry for a compound from NIST WebBook.

    Returns ΔfH°, S°, and Cp from the gas-phase thermochemistry page.
    """
    url = nist_webbook_url(cas, mask=1)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MABE-DataAdapter/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  NIST WebBook fetch failed for CAS {cas}: {e}")
        return []

    results = []

    # Extract compound name
    name_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html)
    name = name_match.group(1).strip() if name_match else cas

    # Extract formula
    formula_match = re.search(r"Formula.*?<td[^>]*>([^<]+)</td>", html, re.DOTALL)
    formula = formula_match.group(1).strip() if formula_match else ""

    # Look for standard enthalpy of formation
    dfh_match = re.search(
        r"f</sub>H.gas.*?<td[^>]*>\s*([-\d.]+)\s*</td>", html, re.DOTALL
    )
    if dfh_match:
        results.append(NISTThermochemEntry(
            cas_number=cas, name=name, formula=formula,
            property_name="Delta_f_H_gas",
            value=float(dfh_match.group(1)),
            units="kJ/mol", temperature_K=298.15,
            source="NIST WebBook SRD 69", url=url,
        ))

    # Look for standard entropy
    s_match = re.search(
        r"S.gas.*?<td[^>]*>\s*([-\d.]+)\s*</td>", html, re.DOTALL
    )
    if s_match:
        results.append(NISTThermochemEntry(
            cas_number=cas, name=name, formula=formula,
            property_name="S_gas",
            value=float(s_match.group(1)),
            units="J/mol·K", temperature_K=298.15,
            source="NIST WebBook SRD 69", url=url,
        ))

    return results


def fetch_nist_ion_energetics(cas: str) -> list[NISTThermochemEntry]:
    """
    Fetch gas-phase ion energetics (IE, EA, PA) from NIST WebBook.
    Useful for HSAB hardness/softness calculations.
    """
    url = nist_webbook_url(cas, mask=20)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MABE-DataAdapter/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  NIST WebBook ion energetics fetch failed for CAS {cas}: {e}")
        return []

    results = []

    name_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html)
    name = name_match.group(1).strip() if name_match else cas

    # Ionization energy
    ie_match = re.search(r"Ionization energy.*?<td[^>]*>\s*([-\d.]+)\s*</td>", html, re.DOTALL)
    if ie_match:
        results.append(NISTThermochemEntry(
            cas_number=cas, name=name, formula="",
            property_name="ionization_energy",
            value=float(ie_match.group(1)),
            units="eV", temperature_K=0,
            source="NIST WebBook SRD 69", url=url,
        ))

    return results


# Commonly needed species for MABE calibration
MABE_SPECIES = {
    # Metal ions (as neutral atoms — IE gives hardness)
    "Cu":   "7440-50-8",
    "Zn":   "7440-66-6",
    "Ni":   "7440-02-0",
    "Fe":   "7439-89-6",
    "Pb":   "7439-92-1",
    "Cd":   "7440-43-9",
    "Hg":   "7439-97-6",
    "Co":   "7440-48-4",
    "Mn":   "7439-96-5",
    # Water and common solvents
    "H2O":  "7732-18-5",
    # Common ligand atoms / small molecules
    "NH3":  "7664-41-7",
    "CO2":  "124-38-9",
}


# ================================================================
# ChEMBL API Adapter (via www.ebi.ac.uk)
# ================================================================

@dataclass
class ChEMBLBindingEntry:
    """Binding data from ChEMBL."""
    chembl_id: str
    target_chembl_id: str
    molecule_name: str
    target_name: str
    standard_type: str          # "Ki", "Kd", "IC50", "EC50"
    standard_value: float       # in nM
    standard_units: str
    assay_type: str             # "B" (binding), "F" (functional)
    source: str


def chembl_search_molecule(name: str) -> Optional[str]:
    """Search ChEMBL for a molecule by name, return ChEMBL ID."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={urllib.parse.quote(name)}&limit=5"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MABE-DataAdapter/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        mols = data.get("molecules", [])
        if mols:
            return mols[0].get("molecule_chembl_id")
    except Exception as e:
        print(f"  ChEMBL search failed for '{name}': {e}")
    return None


def chembl_get_activities(target_chembl_id: str, activity_type: str = "Ki",
                          limit: int = 100) -> list[ChEMBLBindingEntry]:
    """
    Fetch binding activities for a ChEMBL target.

    Args:
        target_chembl_id: e.g., "CHEMBL2111370" (Concanavalin A)
        activity_type: "Ki", "Kd", "IC50", etc.
        limit: max results
    """
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
           f"?target_chembl_id={target_chembl_id}"
           f"&standard_type={activity_type}"
           f"&limit={limit}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MABE-DataAdapter/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  ChEMBL activity fetch failed: {e}")
        return []

    results = []
    for act in data.get("activities", []):
        if act.get("standard_value") is not None:
            results.append(ChEMBLBindingEntry(
                chembl_id=act.get("molecule_chembl_id", ""),
                target_chembl_id=target_chembl_id,
                molecule_name=act.get("molecule_pref_name", ""),
                target_name=act.get("target_pref_name", ""),
                standard_type=act.get("standard_type", ""),
                standard_value=float(act["standard_value"]),
                standard_units=act.get("standard_units", "nM"),
                assay_type=act.get("assay_type", ""),
                source=f"ChEMBL {act.get('assay_chembl_id', '')}",
            ))
    return results


def chembl_get_molecule_info(chembl_id: str) -> dict:
    """Get molecule details including SMILES."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MABE-DataAdapter/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        structs = data.get("molecule_structures", {}) or {}
        return {
            "chembl_id": chembl_id,
            "name": data.get("pref_name", ""),
            "smiles": structs.get("canonical_smiles", ""),
            "formula": data.get("molecule_properties", {}).get("full_molformula", ""),
        }
    except Exception as e:
        print(f"  ChEMBL molecule fetch failed for {chembl_id}: {e}")
        return {}


if __name__ == "__main__":
    print("=" * 60)
    print("MABE NIST WebBook + ChEMBL Adapter Test")
    print("=" * 60)

    print("\n--- NIST WebBook: Gas-phase thermochemistry ---")
    for species, cas in list(MABE_SPECIES.items())[:5]:
        entries = fetch_nist_gas_phase_thermo(cas)
        for e in entries:
            print(f"  {e.name}: {e.property_name} = {e.value} {e.units}")

    print("\n--- ChEMBL: Test molecule lookup ---")
    result = chembl_search_molecule("EDTA")
    if result:
        print(f"  EDTA → {result}")
        info = chembl_get_molecule_info(result)
        print(f"  SMILES: {info.get('smiles', 'N/A')[:60]}")