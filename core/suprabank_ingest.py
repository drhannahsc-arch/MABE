"""
core/suprabank_ingest.py — SupraBank host-guest data ingestion.

Converts extracted SupraBank JSON (CB[5-8] + CD[alpha/beta/gamma])
into UniversalComplex entries for calibration and benchmarking.

Source: suprabank.org, CC-BY-4.0
Extraction: 2026-03-17, 1712 entries at 25°C water
"""

import json
import os
import math
from collections import defaultdict
from core.universal_schema import UniversalComplex

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_DIR, "data")

# Guest SMILES cache (loaded once)
_SMILES_CACHE = None

# Host name mapping: SupraBank convention → HOST_REGISTRY key
_HOST_MAP = {
    "CB5": "CB5",
    "CB6": "CB6",
    "CB7": "CB7",
    "CB8": "CB8",
    "alpha-CD": "alpha-CD",
    "beta-CD": "beta-CD",
    "gamma-CD": "gamma-CD",
}

_HOST_TYPE = {
    "CB5": "cucurbituril", "CB6": "cucurbituril",
    "CB7": "cucurbituril", "CB8": "cucurbituril",
    "alpha-CD": "cyclodextrin", "beta-CD": "cyclodextrin",
    "gamma-CD": "cyclodextrin",
}

_CAVITY_VOL = {
    "CB5": 82.0, "CB6": 164.0, "CB7": 279.0, "CB8": 479.0,
    "alpha-CD": 174.0, "beta-CD": 262.0, "gamma-CD": 427.0,
}


def _load_smiles():
    """Load guest SMILES from mol_data JSON."""
    global _SMILES_CACHE
    if _SMILES_CACHE is not None:
        return _SMILES_CACHE
    path = os.path.join(_DATA, "suprabank_mol_data.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        _SMILES_CACHE = {int(k): v for k, v in raw.items() if v}
    else:
        _SMILES_CACHE = {}
    return _SMILES_CACHE


def _enrich_guest(uc, smiles):
    """Populate guest descriptor fields from SMILES using RDKit."""
    if not smiles:
        return
    # Strip counterions — take largest fragment
    frags = smiles.split(".")
    smiles_clean = max(frags, key=len) if frags else smiles
    uc.guest_smiles = smiles_clean
    
    try:
        from core.guest_compute import compute_guest_properties
        props = compute_guest_properties(smiles_clean)
        if props:
            for key, val in props.items():
                if hasattr(uc, key) and val is not None:
                    setattr(uc, key, val)
    except Exception:
        # Fallback: try RDKit directly for critical fields
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
            mol = Chem.MolFromSmiles(smiles_clean)
            if mol is None:
                return
            mol_h = Chem.AddHs(mol)
            uc.guest_mw = Descriptors.MolWt(mol)
            uc.guest_logP = Descriptors.MolLogP(mol)
            uc.guest_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            uc.guest_n_hbond_donors = rdMolDescriptors.CalcNumHBD(mol)
            uc.guest_n_hbond_acceptors = rdMolDescriptors.CalcNumHBA(mol)
            uc.guest_n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            uc.guest_tpsa = Descriptors.TPSA(mol)
            # Volume
            if AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=20) == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=100)
                except Exception:
                    pass
                try:
                    uc.guest_volume_A3 = AllChem.ComputeMolVolume(mol_h)
                except Exception:
                    pass
        except Exception:
            pass


def load_suprabank_cb(max_entries=None):
    """Load cucurbituril host-guest entries from SupraBank extraction.
    
    Returns list[UniversalComplex].
    """
    path = os.path.join(_DATA, "suprabank_cb_raw.json")
    if not os.path.exists(path):
        print(f"  SupraBank CB data not found: {path}")
        return []
    
    with open(path) as f:
        raw = json.load(f)
    
    smiles_db = _load_smiles()
    entries = []
    
    for e in raw["entries"]:
        host = e["host"]
        if host not in _HOST_MAP:
            continue
        
        logka = e["logKa"]
        guest_name = e["guest_name"]
        mol_id = e.get("guest_mol_id")
        smiles = smiles_db.get(mol_id) if mol_id else None
        
        uc = UniversalComplex(
            name=f"{host}:{guest_name}",
            binding_mode="host_guest_inclusion",
            log_Ka_exp=logka,
            dg_exp_kj=-logka * 5.708,  # -RT ln(10) at 25°C
            temperature_C=25.0,
            solvent="water",
            host_name=_HOST_MAP[host],
            host_type=_HOST_TYPE[host],
            is_macrocyclic=True,
            cavity_volume_A3=_CAVITY_VOL[host],
            guest_name=guest_name,
            source="suprabank",
            source_id=str(e.get("int_id", "")),
            series_id=f"suprabank_{host}",
            phase="Phase6",
            confidence="high",
        )
        
        _enrich_guest(uc, smiles)
        
        # Packing coefficient
        if uc.guest_volume_A3 > 0 and uc.cavity_volume_A3 > 0:
            uc.packing_coefficient = uc.guest_volume_A3 / uc.cavity_volume_A3
        
        entries.append(uc)
        
        if max_entries and len(entries) >= max_entries:
            break
    
    return entries


def load_suprabank_cd(max_entries=None):
    """Load cyclodextrin host-guest entries from SupraBank extraction.
    
    Returns list[UniversalComplex].
    """
    path = os.path.join(_DATA, "suprabank_cd_raw.json")
    if not os.path.exists(path):
        print(f"  SupraBank CD data not found: {path}")
        return []
    
    with open(path) as f:
        raw = json.load(f)
    
    smiles_db = _load_smiles()
    entries = []
    
    for e in raw["entries"]:
        host = e["host"]
        if host not in _HOST_MAP:
            continue
        
        logka = e["logKa"]
        guest_name = e["guest_name"]
        mol_id = e.get("guest_mol_id")
        smiles = smiles_db.get(mol_id) if mol_id else None
        
        uc = UniversalComplex(
            name=f"{host}:{guest_name}",
            binding_mode="host_guest_inclusion",
            log_Ka_exp=logka,
            dg_exp_kj=-logka * 5.708,
            temperature_C=25.0,
            solvent="water",
            host_name=_HOST_MAP[host],
            host_type=_HOST_TYPE[host],
            is_macrocyclic=True,
            cavity_volume_A3=_CAVITY_VOL[host],
            guest_name=guest_name,
            source="suprabank",
            source_id=str(e.get("int_id", "")),
            series_id=f"suprabank_{host}",
            phase="Phase6",
            confidence="high",
        )
        
        _enrich_guest(uc, smiles)
        
        if uc.guest_volume_A3 > 0 and uc.cavity_volume_A3 > 0:
            uc.packing_coefficient = uc.guest_volume_A3 / uc.cavity_volume_A3
        
        entries.append(uc)
        
        if max_entries and len(entries) >= max_entries:
            break
    
    return entries


def load_suprabank_all(max_entries=None):
    """Load all SupraBank entries (CB + CD).
    
    Returns list[UniversalComplex].
    """
    cb = load_suprabank_cb(max_entries=max_entries)
    remaining = (max_entries - len(cb)) if max_entries else None
    cd = load_suprabank_cd(max_entries=remaining)
    all_entries = cb + cd
    
    # Summary
    by_host = defaultdict(int)
    with_smiles = 0
    with_volume = 0
    for uc in all_entries:
        by_host[uc.host_name] += 1
        if uc.guest_smiles:
            with_smiles += 1
        if uc.guest_volume_A3 > 0:
            with_volume += 1
    
    print(f"  SupraBank library: {len(all_entries)} entries")
    for h in sorted(by_host):
        print(f"    {h}: {by_host[h]}")
    print(f"  With SMILES: {with_smiles}/{len(all_entries)}")
    print(f"  With volume: {with_volume}/{len(all_entries)}")
    
    return all_entries
