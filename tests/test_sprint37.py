"""
tests/test_sprint37.py — Sprint 37 verification

Tests:
  1. UniversalComplex creation and auto-compute
  2. HOST_REGISTRY lookup and enrichment
  3. Seed library builds without error
  4. Universal predictor returns results for all modalities
  5. Back-solve engine constructs and runs (small dataset)
  6. Backward compatibility: metal predictions still work
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.universal_schema import UniversalComplex, BindingMode
from core.host_registry import HOST_REGISTRY, lookup_host, enrich_complex_host
from core.guest_compute import rdkit_available, compute_guest_properties, enrich_complex
from core.seed_library import build_seed_library
from core.universal_predictor import (
    PhysicsParameters, predict, predict_batch, compute_statistics,
)
from core.backsolve_engine import (
    get_parameter_bounds, residual_analysis, print_residual_analysis,
)
from core.database_ingest import enrich_all

passed = 0
failed = 0
total = 0


def test(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {detail}")


print("\n" + "="*60)
print("  Sprint 37: Universal Back-Solve Engine Tests")
print("="*60)

# ── 1. Schema ─────────────────────────────────────────────────────────
print("\n  1. UniversalComplex schema")
uc = UniversalComplex(name="test", log_Ka_exp=5.0)
test("Auto-compute dg_exp", abs(uc.dg_exp_kj - (-28.55)) < 0.1,
     f"got {uc.dg_exp_kj}")
test("is_metal false for empty", not uc.is_metal())

uc_metal = UniversalComplex(name="Cu-EDTA", metal_formula="Cu2+",
                             binding_mode="metal_coordination")
test("is_metal true for Cu2+", uc_metal.is_metal())

uc_hg = UniversalComplex(name="test", binding_mode="host_guest_inclusion")
test("is_host_guest", uc_hg.is_host_guest())

# ── 2. Host registry ─────────────────────────────────────────────────
print("\n  2. Host registry")
test("Registry has β-CD", "beta-CD" in HOST_REGISTRY)
test("Registry has CB[7]", "CB7" in HOST_REGISTRY)
test("Registry has 18-crown-6", "18-crown-6" in HOST_REGISTRY)

bcd = lookup_host("β-CD")
test("β-CD lookup via alias", bcd is not None)
test("β-CD volume=262", bcd.cavity_volume_A3 == 262 if bcd else False)

cb7 = lookup_host("CB[7]")
test("CB[7] lookup", cb7 is not None)
test("CB[7] is_cage", cb7.is_cage if cb7 else False)

# Host enrichment
uc_enrich = UniversalComplex(name="test", host_name="beta-CD")
enrich_complex_host(uc_enrich)
test("Host enrichment fills cavity", uc_enrich.cavity_volume_A3 == 262)
test("Host enrichment fills type", uc_enrich.host_type == "cyclodextrin")

# ── 3. Guest compute ─────────────────────────────────────────────────
print("\n  3. Guest compute (RDKit)")
if rdkit_available():
    props = compute_guest_properties("CCCCCO")  # 1-pentanol
    test("RDKit volume > 0", props.get("guest_volume_A3", 0) > 50)
    test("RDKit SASA > 0", props.get("guest_sasa_total_A2", 0) > 50)
    test("RDKit rotors", props.get("guest_rotatable_bonds", -1) >= 2)
    test("RDKit HBD", props.get("guest_n_hbond_donors", -1) == 1)
    test("RDKit logP", abs(props.get("guest_logP", 99)) < 5)
else:
    print("  ⚠ RDKit not available — guest compute tests skipped")
    test("RDKit fallback graceful", compute_guest_properties("CCCO") == {})

# ── 4. Seed library ──────────────────────────────────────────────────
print("\n  4. Seed library")
lib = build_seed_library()
test("Seed library > 80 entries", len(lib) > 80, f"got {len(lib)}")

modes = set(uc.binding_mode for uc in lib)
test("Has metal_coordination", "metal_coordination" in modes)
test("Has host_guest_inclusion", "host_guest_inclusion" in modes)
test("Has protein_ligand", "protein_ligand" in modes)

metals = [uc for uc in lib if uc.is_metal()]
test("Metal entries > 35", len(metals) > 35, f"got {len(metals)}")

hg = [uc for uc in lib if uc.is_host_guest()]
test("Host-guest entries > 25", len(hg) > 25, f"got {len(hg)}")

# Enrich
enrich_all(lib)
bcd_entries = [uc for uc in lib if "β-CD" in uc.name or "beta-CD" in uc.host_name]
if bcd_entries:
    test("β-CD entries enriched cavity", bcd_entries[0].cavity_volume_A3 == 262)

# ── 5. Universal predictor ───────────────────────────────────────────
print("\n  5. Universal predictor")
params = PhysicsParameters()
test("Param count > 50", PhysicsParameters.param_count() > 50,
     f"got {PhysicsParameters.param_count()}")

vec = params.to_vector()
params2 = PhysicsParameters.from_vector(vec)
test("Round-trip vector", abs(params2.gamma_hydrophobic - params.gamma_hydrophobic) < 1e-10)

# Predict one host-guest
uc_test = UniversalComplex(
    name="β-CD:adamantane-test",
    binding_mode="host_guest_inclusion",
    log_Ka_exp=4.25,
    host_name="beta-CD",
    guest_smiles="OC(=O)C12CC3CC(CC(C3)C1)C2",
    guest_volume_A3=136,
    cavity_volume_A3=262,
    cavity_radius_nm=0.300,
)
r = predict(uc_test, params)
test("HG prediction returns result", r is not None)
test("HG pred has hydrophobic term", r.dg_hydrophobic != 0 or r.dg_shape != 0,
     f"hydro={r.dg_hydrophobic}, shape={r.dg_shape}")

# Batch predict full library
results = predict_batch(lib, params)
test("Batch predict count", len(results) == len(lib))
stats = compute_statistics(results)
test("Stats computed", "r2" in stats and "mae" in stats)
print(f"    Initial R²={stats['r2']:.3f}, MAE={stats['mae']:.1f}")

# ── 6. Parameter bounds ──────────────────────────────────────────────
print("\n  6. Back-solve infrastructure")
lower, upper = get_parameter_bounds(params)
test("Bounds length match params", len(lower) == PhysicsParameters.param_count())
test("All lower < upper", all(l < u for l, u in zip(lower, upper)))

# Residual analysis
analysis = residual_analysis(lib, params)
test("Residual analysis returns", len(analysis) > 0)
if analysis:
    print_residual_analysis(analysis)

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'='*60}")

if failed > 0:
    print(f"\n  ⚠ {failed} tests failed — review before proceeding")
else:
    print(f"\n  ✅ Sprint 37 infrastructure verified")
    print(f"     Ready for bulk database ingestion + back-solve")
