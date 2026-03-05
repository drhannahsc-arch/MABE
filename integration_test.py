"""
MABE Data Infrastructure — Integration Test
=============================================
Tests all connected data sources and reports what's available for MABE.
Run this to verify all adapters work and see the full inventory.
"""

import os
import sys
import json

# Add adapter directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_sampl():
    """Test SAMPL host-guest binding data."""
    from sampl_adapter import load_all_sampl, sampl_summary, to_mabe_format, to_kj_mol
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sampl")
    data = load_all_sampl(data_dir)
    
    if not data:
        return {"status": "FAIL", "error": "No data loaded"}
    
    summary = sampl_summary(data)
    
    # Breakdown by host type
    by_type = {}
    for d in data:
        by_type.setdefault(d.host_type, []).append(d)
    
    detail = []
    for htype, entries in sorted(by_type.items()):
        dG_vals = [to_kj_mol(e.dG_kcal) for e in entries]
        detail.append({
            "host_type": htype,
            "count": len(entries),
            "dG_range_kJ": f"{min(dG_vals):.1f} to {max(dG_vals):.1f}",
            "hosts": sorted(set(e.host_name for e in entries)),
        })
    
    return {
        "status": "OK",
        "total_complexes": summary["total_complexes"],
        "by_edition": summary["by_edition"],
        "has_enthalpy": summary["has_enthalpy"],
        "has_entropy": summary["has_entropy"],
        "host_types": detail,
        "mabe_use": "Host-guest scorer holdout validation. 104 ITC-measured ΔG values across 5 host types.",
    }


def test_refractive_index():
    """Test RefractiveIndex.INFO optical constants."""
    from refractive_index_adapter import parse_nk_yaml
    import numpy as np
    
    db_root = os.path.join(os.path.dirname(__file__), "..", "data", "refractive_index", "repo", "database", "data")
    
    if not os.path.exists(db_root):
        return {"status": "FAIL", "error": "Database not cloned"}
    
    # Count materials
    yml_count = sum(1 for r, d, f in os.walk(db_root) for fn in f if fn.endswith('.yml'))
    
    # Test key materials for structural color
    test_materials = {
        "Ag": "main/Ag/nk/Johnson.yml",
        "Au": "main/Au/nk/Johnson.yml",
        "H2O": "main/H2O/nk/Hale.yml",
        "SiO2": "main/SiO2/nk/Rodney.yml",
        "TiO2": "main/TiO2/nk/Devore-o.yml",
    }
    
    tested = {}
    for name, rel_path in test_materials.items():
        full_path = os.path.join(db_root, rel_path)
        if os.path.exists(full_path):
            data = parse_nk_yaml(full_path)
            if data:
                # Get n at 550nm (green)
                idx = np.argmin(np.abs(data.wavelength_um - 0.55))
                tested[name] = {
                    "n_at_550nm": round(float(data.n[idx]), 4),
                    "k_at_550nm": round(float(data.k[idx]), 4) if data.k is not None else None,
                    "wavelength_range_um": f"{data.wavelength_um[0]:.3f}–{data.wavelength_um[-1]:.3f}",
                    "n_points": len(data.wavelength_um),
                }
            else:
                tested[name] = {"status": "parse_failed"}
        else:
            tested[name] = {"status": "file_not_found"}
    
    return {
        "status": "OK",
        "total_yml_files": yml_count,
        "tested_materials": tested,
        "mabe_use": "Photonic module: n(λ), k(λ) for Mie scattering, transfer matrix, structural color prediction.",
    }


def test_chembl():
    """Test ChEMBL API access."""
    from nist_chembl_adapter import chembl_search_molecule, chembl_get_activities, chembl_get_molecule_info
    
    # Test 1: Look up EDTA
    edta_id = chembl_search_molecule("EDTA")
    edta_info = chembl_get_molecule_info(edta_id) if edta_id else {}
    
    # Test 2: Get ConA binding data
    cona_activities = chembl_get_activities("CHEMBL3559", activity_type="Ki", limit=50)
    
    return {
        "status": "OK" if edta_id else "PARTIAL",
        "edta_chembl_id": edta_id,
        "edta_smiles": edta_info.get("smiles", "N/A"),
        "cona_ki_entries": len(cona_activities),
        "cona_sample": [
            {"molecule": a.molecule_name or a.chembl_id, 
             "Ki_nM": a.standard_value}
            for a in cona_activities[:5]
        ],
        "mabe_use": "Metal-ligand binding data, lectin-glycan Ki values, SMILES lookup for any ChEMBL compound.",
    }


def test_nist_webbook():
    """Test NIST WebBook access."""
    from nist_chembl_adapter import fetch_nist_gas_phase_thermo, MABE_SPECIES
    
    # Test with water (most likely to have data)
    entries = fetch_nist_gas_phase_thermo(MABE_SPECIES["H2O"])
    
    return {
        "status": "OK" if entries else "PARTIAL",
        "water_thermo": [{"property": e.property_name, "value": e.value, "units": e.units} for e in entries],
        "species_available": list(MABE_SPECIES.keys()),
        "mabe_use": "Gas-phase ΔfH°, S°, IE for BDE calibration and HSAB hardness calculations.",
    }


def main():
    print("=" * 70)
    print("  MABE DATA INFRASTRUCTURE — INTEGRATION REPORT")
    print("=" * 70)
    
    tests = {
        "SAMPL Host-Guest Binding": test_sampl,
        "RefractiveIndex.INFO Optical": test_refractive_index,
        "ChEMBL Binding API": test_chembl,
        "NIST WebBook Thermochemistry": test_nist_webbook,
    }
    
    results = {}
    for name, func in tests.items():
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        try:
            result = func()
            results[name] = result
            status = result.get("status", "UNKNOWN")
            icon = "✓" if status == "OK" else "◐" if status == "PARTIAL" else "✗"
            print(f"  {icon} Status: {status}")
            for k, v in result.items():
                if k not in ("status", "mabe_use"):
                    if isinstance(v, (list, dict)):
                        print(f"    {k}:")
                        if isinstance(v, list):
                            for item in v[:5]:
                                print(f"      {item}")
                        else:
                            for sk, sv in v.items():
                                if isinstance(sv, dict):
                                    print(f"      {sk}: {sv}")
                                else:
                                    print(f"      {sk}: {sv}")
                    else:
                        print(f"    {k}: {v}")
            print(f"  → MABE use: {result.get('mabe_use', 'N/A')}")
        except Exception as e:
            results[name] = {"status": "ERROR", "error": str(e)}
            print(f"  ✗ ERROR: {e}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    ok = sum(1 for r in results.values() if r.get("status") == "OK")
    partial = sum(1 for r in results.values() if r.get("status") == "PARTIAL")
    fail = sum(1 for r in results.values() if r.get("status") in ("FAIL", "ERROR"))
    print(f"  ✓ Connected:  {ok}/{len(tests)}")
    print(f"  ◐ Partial:    {partial}/{len(tests)}")
    print(f"  ✗ Failed:     {fail}/{len(tests)}")
    
    print(f"\n  DATA AVAILABLE FOR MABE:")
    sampl = results.get("SAMPL Host-Guest Binding", {})
    ri = results.get("RefractiveIndex.INFO Optical", {})
    chembl = results.get("ChEMBL Binding API", {})
    print(f"    Host-guest complexes:    {sampl.get('total_complexes', 0)} (with ITC ΔG, ΔH, TΔS)")
    print(f"    Optical constant files:  {ri.get('total_yml_files', 0)} (n, k vs wavelength)")
    print(f"    ChEMBL ConA Ki entries:  {chembl.get('cona_ki_entries', 0)}")
    
    print(f"\n  BLOCKED (need domain allowlist update):")
    print(f"    NIST SRD 46 SQLite:      data.nist.gov (99,000 metal-ligand systems)")
    print(f"    Materials Project API:    materialsproject.org (150,000 inorganic materials)")
    print(f"    BindingDB:               bindingdb.org (403 Forbidden)")
    print(f"    COD Structures:          crystallography.net (500,000 crystal structures)")


if __name__ == "__main__":
    main()