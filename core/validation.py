"""
core/validation.py — Sprint 36: Validation with Deep Physics

Extended validation library with Sprint 36 schema additions:
  - ring_sizes: List[int] per chelate ring (5 or 6)
  - is_macrocyclic: bool
  - cavity_radius_nm: float (crown/cryptand cavity)
"""
from dataclasses import dataclass, field
import math

from core.physics_integration import compute_enhanced_thermodynamics
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)
from core.coordination_generator import (
    METAL_D_ELECTRONS, METAL_HSAB_SOFTNESS, _IONIC_RADII,
)


@dataclass
class ExperimentalComplex:
    """A known metal-ligand complex with measured formation constant."""
    name: str
    metal_formula: str
    metal_charge: int
    metal_d_electrons: int
    donor_atoms: list
    donor_type: str
    chelate_rings: int
    denticity: int
    log_K_exp: float
    conditions: str
    source: str
    geometry: str = "octahedral"
    scaffold_type: str = "free"
    donor_subtypes: list = field(default_factory=list)
    # Sprint 36 schema additions
    ring_sizes: list = field(default_factory=list)    # e.g. [5,5,5] for EDTA
    is_macrocyclic: bool = False
    cavity_radius_nm: float = 0.0                     # Crown/cryptand cavity radius
    is_cage: bool = False                              # 3D encapsulation (cryptand)
    notes: str = ""


VALIDATION_LIBRARY = [
    # === EDTA complexes (N2O4 hexadentate, 5 chelate rings, all 5-membered) ===
    ExperimentalComplex("Ca-EDTA", "Ca2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        10.7, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5], notes="Hard metal, weak EDTA"),
    ExperimentalComplex("Mg-EDTA", "Mg2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        8.7, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5], notes="Hard, even weaker than Ca"),
    ExperimentalComplex("Mn-EDTA", "Mn2+", 2, 5, ["N","N","O","O","O","O"], "mixed", 5, 6,
        13.9, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Fe2-EDTA", "Fe2+", 2, 6, ["N","N","O","O","O","O"], "mixed", 5, 6,
        14.3, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Co-EDTA", "Co2+", 2, 7, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.3, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Ni-EDTA", "Ni2+", 2, 8, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.6, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Cu-EDTA", "Cu2+", 2, 9, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.8, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Zn-EDTA", "Zn2+", 2, 10, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Pb-EDTA", "Pb2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Cd-EDTA", "Cd2+", 2, 10, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Fe3-EDTA", "Fe3+", 3, 5, ["N","N","O","O","O","O"], "mixed", 5, 6,
        25.1, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5], notes="Trivalent, very strong"),
    ExperimentalComplex("Al-EDTA", "Al3+", 3, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.1, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),

    # === Ammonia / ethylenediamine (N donors, 5-membered rings) ===
    ExperimentalComplex("Ni-en3", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 3, 6,
        18.3, "25°C, I=0.5M", "NIST 46.7", geometry="octahedral",
        donor_subtypes=["N_amine"]*6, ring_sizes=[5,5,5],
        notes="Tris(ethylenediamine)"),
    ExperimentalComplex("Cu-en2", "Cu2+", 2, 9, ["N","N","N","N"], "borderline", 2, 4,
        19.6, "25°C, I=0.5M", "NIST 46.7", geometry="square_planar",
        donor_subtypes=["N_amine"]*4, ring_sizes=[5,5],
        notes="Bis(en), Jahn-Teller"),
    ExperimentalComplex("Zn-en3", "Zn2+", 2, 10, ["N","N","N","N","N","N"], "borderline", 3, 6,
        12.1, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6, ring_sizes=[5,5,5]),
    ExperimentalComplex("Co-en3", "Co2+", 2, 7, ["N","N","N","N","N","N"], "borderline", 3, 6,
        13.9, "25°C, I=0.5M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6, ring_sizes=[5,5,5]),
    ExperimentalComplex("Ni-NH3_6", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 0, 6,
        8.6, "25°C, I=2M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6,
        notes="Hexaammine — no chelate rings"),

    # === Soft donors (S, thiol/thioether) ===
    ExperimentalComplex("Hg-cysteine2", "Hg2+", 2, 10, ["S","S","N","N"], "soft", 0, 4,
        38.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["S_thiolate","S_thiolate","N_amine","N_amine"],
        notes="Hg-thiolate, extremely strong"),
    ExperimentalComplex("Ag-thiosulfate2", "Ag+", 1, 10, ["S","S"], "soft", 0, 2,
        13.5, "25°C, I=1M", "NIST 46.7", geometry="linear",
        donor_subtypes=["S_thiosulfate","S_thiosulfate"]),
    ExperimentalComplex("Cd-cysteine", "Cd2+", 2, 10, ["S","N","O"], "mixed", 1, 3,
        10.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["S_thiolate","N_amine","O_carboxylate"],
        ring_sizes=[5]),
    ExperimentalComplex("Pb-cysteine", "Pb2+", 2, 0, ["S","N","O"], "mixed", 1, 3,
        12.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["S_thiolate","N_amine","O_carboxylate"],
        ring_sizes=[5]),

    # === Hard donors (O-only, hydroxamate, catechol) ===
    ExperimentalComplex("Fe3-catechol3", "Fe3+", 3, 5, ["O","O","O","O","O","O"], "hard", 3, 6,
        43.8, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["O_catecholate"]*6, ring_sizes=[5,5,5],
        notes="Tris(catecholate), siderophore-like"),
    ExperimentalComplex("Fe3-acetohydroxamate3", "Fe3+", 3, 5, ["O","O","O","O","O","O"], "hard", 3, 6,
        28.3, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["O_hydroxamate"]*6, ring_sizes=[5,5,5],
        notes="Tris(hydroxamate)"),
    ExperimentalComplex("Ca-citrate", "Ca2+", 2, 0, ["O","O","O"], "hard", 1, 3,
        3.5, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["O_carboxylate","O_carboxylate","O_hydroxyl"],
        ring_sizes=[5]),
    ExperimentalComplex("Al-catechol3", "Al3+", 3, 0, ["O","O","O","O","O","O"], "hard", 3, 6,
        36.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["O_catecholate"]*6, ring_sizes=[5,5,5]),

    # === Crown ethers (MACROCYCLIC, O-donors, cavity size-match) ===
    ExperimentalComplex("K-18crown6", "K+", 1, 0, ["O","O","O","O","O","O"], "hard", 0, 6,
        2.0, "25°C, MeOH", "NIST 46.7",
        donor_subtypes=["O_ether"]*6,
        is_macrocyclic=True, cavity_radius_nm=0.134,
        notes="Size match: K+ (138 pm) ≈ 18C6 cavity (134 pm)"),
    ExperimentalComplex("Na-18crown6", "Na+", 1, 0, ["O","O","O","O","O","O"], "hard", 0, 6,
        0.8, "25°C, MeOH", "NIST 46.7",
        donor_subtypes=["O_ether"]*6,
        is_macrocyclic=True, cavity_radius_nm=0.134,
        notes="Na+ (102 pm) too small for 18C6 cavity"),

    # === DTPA (N3O5, 8-dentate, all 5-membered rings) ===
    ExperimentalComplex("Gd-DTPA", "Gd3+", 3, 7, ["N","N","N","O","O","O","O","O"], "mixed", 6, 8,
        22.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5,5], notes="MRI contrast agent"),
    ExperimentalComplex("Cu-DTPA", "Cu2+", 2, 9, ["N","N","N","O","O","O","O","O"], "mixed", 6, 8,
        21.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5,5]),

    # === Bipyridine (5-membered chelate rings) ===
    ExperimentalComplex("Fe2-bipy3", "Fe2+", 2, 6, ["N","N","N","N","N","N"], "borderline", 3, 6,
        17.2, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_pyridine"]*6, ring_sizes=[5,5,5],
        notes="Tris(bipyridyl)iron(II), low-spin d6"),
    ExperimentalComplex("Ni-bipy3", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 3, 6,
        20.2, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_pyridine"]*6, ring_sizes=[5,5,5]),

    # === Mixed (glycinate, 5-membered rings) ===
    ExperimentalComplex("Cu-glycine2", "Cu2+", 2, 9, ["N","N","O","O"], "mixed", 2, 4,
        15.1, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5], notes="Bis(glycinate)"),
    ExperimentalComplex("Zn-glycine2", "Zn2+", 2, 10, ["N","N","O","O"], "mixed", 2, 4,
        9.0, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5]),
]


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Prediction vs experiment for one complex."""
    name: str
    metal_formula: str
    donor_type: str
    log_K_exp: float
    log_K_pred: float
    dg_pred_kj: float
    error: float
    abs_error: float


@dataclass
class ValidationReport:
    """Summary of validation across all complexes."""
    n_complexes: int
    results: list
    mean_abs_error: float
    r_squared: float
    systematic_bias: float
    hard_mae: float
    borderline_mae: float
    soft_mae: float
    mixed_mae: float
    calibration_slope: float
    calibration_intercept: float
    calibration_r2: float
    notes: list


def _predict_log_K(complex_entry):
    """Run MABE enhanced thermodynamics for a known complex."""
    c = complex_entry
    formula = c.metal_formula
    charge = c.metal_charge
    d_electrons = c.metal_d_electrons
    softness = METAL_HSAB_SOFTNESS.get(formula, 0.3)
    ionic_r = _IONIC_RADII.get(formula, 80)
    hydrated_r = (ionic_r + 140) / 1000.0

    if formula == "Gd3+":
        d_electrons = 7
        softness = 0.12
        ionic_r = 94
        hydrated_r = 0.234

    rec = RecognitionChemistry(
        name=c.name, type="generative",
        donor_atoms=c.donor_atoms, donor_type=c.donor_type,
        denticity=c.denticity, hsab_match=0.7,
        chelate_rings=c.chelate_rings)

    # Attach Sprint 35d donor_subtypes
    if c.donor_subtypes:
        rec.donor_subtypes = c.donor_subtypes

    # Attach Sprint 36 schema fields
    if c.ring_sizes:
        rec.ring_sizes = c.ring_sizes
    if c.is_macrocyclic:
        rec.is_macrocyclic = True
        rec.cavity_radius_nm = c.cavity_radius_nm
    if c.is_cage:
        rec.is_cage = True

    struct = StructuralConstraint(
        name="free", type=c.scaffold_type, geometry=c.geometry,
        pore_size_nm=0.0)

    interior = InteriorDesign(
        description="free ligand", num_binding_sites=1,
        self_binding=False)

    prob = Problem(
        target=TargetSpecies(
            identity=c.name, formula=formula,
            charge=charge, d_electrons=d_electrons,
            hsab_softness=softness, ionic_radius_pm=ionic_r,
            hydrated_radius_nm=hydrated_r,
            coordination_number=len(c.donor_atoms)),
        matrix=Matrix(ph=7.0, temperature_c=25.0, ionic_strength_mm=100.0))

    thermo = compute_enhanced_thermodynamics(rec, struct, interior, prob)
    dg = thermo.dg_net_kj
    log_K_pred = -dg / 5.71

    return log_K_pred, dg


def run_validation(library=None, apply_calibration=False, calibration=None):
    """Run full validation against experimental library."""
    if library is None:
        library = VALIDATION_LIBRARY

    results = []
    for c in library:
        try:
            log_K_pred, dg = _predict_log_K(c)
            if apply_calibration and calibration:
                log_K_pred = calibration["slope"] * log_K_pred + calibration["intercept"]
        except Exception as e:
            log_K_pred = 0.0
            dg = 0.0

        error = log_K_pred - c.log_K_exp
        results.append(ValidationResult(
            name=c.name, metal_formula=c.metal_formula,
            donor_type=c.donor_type,
            log_K_exp=c.log_K_exp, log_K_pred=round(log_K_pred, 1),
            dg_pred_kj=round(dg, 1),
            error=round(error, 1), abs_error=round(abs(error), 1)))

    n = len(results)
    if n == 0:
        return ValidationReport(0, [], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

    mae = sum(r.abs_error for r in results) / n
    mean_error = sum(r.error for r in results) / n

    exp_vals = [r.log_K_exp for r in results]
    pred_vals = [r.log_K_pred for r in results]
    exp_mean = sum(exp_vals) / n
    ss_res = sum((p - e)**2 for p, e in zip(pred_vals, exp_vals))
    ss_tot = sum((e - exp_mean)**2 for e in exp_vals)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    def class_mae(dtype):
        cls = [r for r in results if r.donor_type == dtype]
        return sum(r.abs_error for r in cls) / max(1, len(cls))

    hard_mae = class_mae("hard")
    border_mae = class_mae("borderline")
    soft_mae = class_mae("soft")
    mixed_mae = class_mae("mixed")

    if n > 2:
        sum_x = sum(exp_vals)
        sum_y = sum(pred_vals)
        sum_xy = sum(x*y for x, y in zip(exp_vals, pred_vals))
        sum_x2 = sum(x**2 for x in exp_vals)
        denom = n * sum_x2 - sum_x**2
        if denom > 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            pred_cal = [slope * e + intercept for e in exp_vals]
            ss_res_cal = sum((p - e)**2 for p, e in zip(pred_cal, exp_vals))
            cal_r2 = 1 - ss_res_cal / ss_tot if ss_tot > 0 else 0
        else:
            slope, intercept, cal_r2 = 1.0, 0.0, r2
    else:
        slope, intercept, cal_r2 = 1.0, 0.0, r2

    notes = []
    if abs(mean_error) > 3:
        direction = "over" if mean_error > 0 else "under"
        notes.append(f"Systematic {direction}prediction by {abs(mean_error):.1f} log K units.")
    if abs(slope - 1.0) > 0.2:
        notes.append(f"Calibration slope = {slope:.2f} (ideal = 1.0).")
    if hard_mae > 2 * border_mae and hard_mae > 3:
        notes.append(f"Hard-metal MAE={hard_mae:.1f}. Check electrostatic model.")
    if soft_mae > 2 * border_mae and soft_mae > 3:
        notes.append(f"Soft-metal MAE={soft_mae:.1f}. Check covalent/polarization.")
    if r2 > 0.7:
        notes.append(f"R²={r2:.3f} — model captures trend.")

    return ValidationReport(
        n_complexes=n, results=results,
        mean_abs_error=round(mae, 2), r_squared=round(r2, 4),
        systematic_bias=round(mean_error, 2),
        hard_mae=round(hard_mae, 2), borderline_mae=round(border_mae, 2),
        soft_mae=round(soft_mae, 2), mixed_mae=round(mixed_mae, 2),
        calibration_slope=round(slope, 3),
        calibration_intercept=round(intercept, 2),
        calibration_r2=round(cal_r2, 4),
        notes=notes)


def print_validation_report(report):
    """Pretty-print validation results."""
    print(f"\n  VALIDATION REPORT ({report.n_complexes} complexes)")
    print(f"  {'═'*64}")
    print(f"  R² = {report.r_squared:.4f}   MAE = {report.mean_abs_error:.1f} log K   "
          f"Bias = {'+' if report.systematic_bias > 0 else ''}{report.systematic_bias:.1f}")
    print(f"  Calibration: pred = {report.calibration_slope:.3f} × exp + {report.calibration_intercept:.1f}")
    print()
    print(f"  Per-class MAE:  Hard={report.hard_mae:.1f}  Borderline={report.borderline_mae:.1f}  "
          f"Soft={report.soft_mae:.1f}  Mixed={report.mixed_mae:.1f}")
    print()
    print(f"  {'Complex':25s} {'Exp':>6s} {'Pred':>6s} {'Error':>6s}  Class")
    print(f"  {'─'*60}")
    for r in sorted(report.results, key=lambda x: x.log_K_exp):
        flag = " ⚠" if r.abs_error > 5 else ""
        print(f"  {r.name:25s} {r.log_K_exp:6.1f} {r.log_K_pred:6.1f} "
              f"{'+' if r.error > 0 else ''}{r.error:5.1f}  {r.donor_type}{flag}")
    if report.notes:
        print(f"\n  CALIBRATION NOTES:")
        for n in report.notes:
            print(f"  → {n}")
    print()


def derive_calibration(library=None):
    """Derive calibration factors from experimental library."""
    if library is None:
        library = VALIDATION_LIBRARY

    raw = run_validation(library, apply_calibration=False)
    exp = [r.log_K_exp for r in raw.results]
    pred = [r.log_K_pred for r in raw.results]
    n = len(exp)

    if n < 3:
        return {"slope": 1.0, "intercept": 0.0, "class_offsets": {}}

    sum_x = sum(pred)
    sum_y = sum(exp)
    sum_xy = sum(x*y for x, y in zip(pred, exp))
    sum_x2 = sum(x**2 for x in pred)
    denom = n * sum_x2 - sum_x**2

    if abs(denom) > 1e-10:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
    else:
        slope = 1.0
        intercept = 0.0

    class_offsets = {}
    for dtype in ("hard", "borderline", "soft", "mixed"):
        cls = [(r.log_K_exp, r.log_K_pred) for r in raw.results if r.donor_type == dtype]
        if cls:
            residuals = [e - (slope * p + intercept) for e, p in cls]
            class_offsets[dtype] = round(sum(residuals) / len(residuals), 2)

    return {
        "slope": round(slope, 4),
        "intercept": round(intercept, 2),
        "class_offsets": class_offsets,
        "raw_r2": raw.r_squared,
        "raw_mae": raw.mean_abs_error,
    }


def apply_calibration_to_log_K(raw_log_K, donor_type="mixed", calibration=None):
    """Apply calibration to a raw predicted log K."""
    if calibration is None:
        return raw_log_K
    cal = calibration["slope"] * raw_log_K + calibration["intercept"]
    cal += calibration.get("class_offsets", {}).get(donor_type, 0.0)
    return cal

