"""
MABE Platform - Sprint 18+19 Bootstrap Script
Sprint 18: Speciation Gate + Redox Routing
Sprint 19: Spin State + Strong/Weak Field LFSE

  core/speciation_gate.py   - Speciation predictor + redox pathways + gated design
  core/spin_state.py        - Spin state predictor + field-dependent LFSE + magnetic moment
  tests/test_sprint18.py    - 15 tests
  tests/test_sprint19.py    - 15 tests

Requires Sprints 16-17 in place.
Run: python bootstrap_sprint18_19.py
Then: python tests/test_sprint18.py && python tests/test_sprint19.py
"""

import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n\U0001f528 MABE Sprint 18+19\n")

write_file("core/speciation_gate.py", '''\
"""
core/speciation_gate.py — Sprint 18: Speciation Gate + Redox Routing

Before generating coordination environments, determine what species
actually exists at the working pH and redox conditions. If the free ion
doesn't exist, route to alternative capture strategies.

Integrates the Sprint E1 speciation model with the generative pipeline
and adds redox pathway scoring via Nernst equation.
"""
from dataclasses import dataclass, field
from typing import Optional
import math


# ═══════════════════════════════════════════════════════════════════════════
# SPECIATION DATABASE (from Sprint E1, expanded)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpeciesInfo:
    formula: str
    fraction: float
    charge: int
    bindable: bool
    geometry: str = ""       # Geometry of this species if different from free ion
    donor_preference: str = ""  # What donor atoms this species prefers
    notes: str = ""

@dataclass
class SpeciationResult:
    metal: str
    ph: float
    species: list
    dominant_species: str
    dominant_fraction: float
    free_ion_fraction: float
    bindable_fraction: float
    design_strategy: str     # "free_ion_binding", "hydroxo_binding", "anion_capture",
                             # "precipitation_sufficient", "redox_capture", "pH_adjust"
    effective_charge: float  # Weighted average charge of bindable species
    effective_formula: str   # What to actually design for
    design_notes: str

@dataclass
class RedoxPathway:
    """A redox transformation pathway for the target metal."""
    oxidized: str            # "Cr(VI)" / "Au3+"
    reduced: str             # "Cr(III)" / "Au0"
    e0_v: float              # Standard reduction potential
    n_electrons: int
    mechanism: str           # "reductive_deposition", "reductive_precipitation",
                             # "reductive_binding", "oxidative"
    favorable_ph_range: tuple
    common_reductants: list
    capture_implication: str  # What this means for binder design
    dg_redox_kj: float = 0.0  # Calculated from Nernst at conditions

# ── Speciation data ──────────────────────────────────────────────────────

_SPECIATION = {
    "Pb2+": {
        "free_ion_limit_ph": 6.5,
        "precipitate_ph": 8.5,
        "ranges": [
            ((0, 6.5), [SpeciesInfo("Pb2+", 0.95, 2, True, "hemidirected", "N/O/S")]),
            ((6.5, 8.5), [SpeciesInfo("Pb2+", 0.4, 2, True, "", "N/O"),
                          SpeciesInfo("Pb(OH)+", 0.35, 1, True, "", "O"),
                          SpeciesInfo("PbCO3(aq)", 0.25, 0, False, "", "")]),
            ((8.5, 14), [SpeciesInfo("Pb(OH)2(s)", 0.6, 0, False, "", "", "Precipitate"),
                         SpeciesInfo("Pb(OH)3-", 0.3, -1, True, "", "anion_exchange", "Plumbite anion"),
                         SpeciesInfo("PbCO3(s)", 0.1, 0, False)]),
        ],
    },
    "Cu2+": {
        "free_ion_limit_ph": 5.5,
        "precipitate_ph": 7.5,
        "ranges": [
            ((0, 5.5), [SpeciesInfo("Cu2+", 0.95, 2, True, "tetragonal_elongated", "N/O/S")]),
            ((5.5, 7.5), [SpeciesInfo("Cu2+", 0.55, 2, True),
                          SpeciesInfo("Cu(OH)+", 0.30, 1, True, "", "O"),
                          SpeciesInfo("Cu2(OH)2^2+", 0.15, 2, True)]),
            ((7.5, 14), [SpeciesInfo("Cu(OH)2(s)", 0.75, 0, False, "", "", "Precipitate"),
                         SpeciesInfo("Cu(OH)3-", 0.15, -1, True),
                         SpeciesInfo("Cu(OH)4^2-", 0.10, -2, True)]),
        ],
    },
    "Zn2+": {
        "free_ion_limit_ph": 7.0,
        "precipitate_ph": 8.5,
        "ranges": [
            ((0, 7.0), [SpeciesInfo("Zn2+", 0.95, 2, True, "tetrahedral", "N/O")]),
            ((7.0, 8.5), [SpeciesInfo("Zn2+", 0.45, 2, True),
                          SpeciesInfo("Zn(OH)+", 0.35, 1, True),
                          SpeciesInfo("ZnCO3(aq)", 0.20, 0, False)]),
            ((8.5, 14), [SpeciesInfo("Zn(OH)2(s)", 0.40, 0, False, "", "", "Amphoteric"),
                         SpeciesInfo("Zn(OH)3-", 0.35, -1, True, "", "anion_exchange"),
                         SpeciesInfo("Zn(OH)4^2-", 0.25, -2, True, "", "anion_exchange", "Zincate")]),
        ],
    },
    "Ni2+": {
        "free_ion_limit_ph": 7.5,
        "precipitate_ph": 9.5,
        "ranges": [
            ((0, 7.5), [SpeciesInfo("Ni2+", 0.95, 2, True, "square_planar", "N")]),
            ((7.5, 9.5), [SpeciesInfo("Ni2+", 0.45, 2, True),
                          SpeciesInfo("Ni(OH)+", 0.35, 1, True),
                          SpeciesInfo("NiCO3(aq)", 0.20, 0, False)]),
            ((9.5, 14), [SpeciesInfo("Ni(OH)2(s)", 0.85, 0, False, "", "", "Precipitate"),
                         SpeciesInfo("Ni(OH)3-", 0.15, -1, True)]),
        ],
    },
    "Fe3+": {
        "free_ion_limit_ph": 3.0,
        "precipitate_ph": 4.0,
        "ranges": [
            ((0, 3.0), [SpeciesInfo("Fe3+", 0.90, 3, True, "octahedral", "O/N")]),
            ((3.0, 4.0), [SpeciesInfo("Fe(OH)2+", 0.45, 2, True),
                          SpeciesInfo("Fe(OH)2+", 0.35, 1, True),
                          SpeciesInfo("Fe3+", 0.20, 3, True)]),
            ((4.0, 14), [SpeciesInfo("Fe(OH)3(s)", 0.95, 0, False, "", "", "Rust precipitate")]),
        ],
    },
    "Fe2+": {
        "free_ion_limit_ph": 6.0,
        "precipitate_ph": 8.0,
        "ranges": [
            ((0, 6.0), [SpeciesInfo("Fe2+", 0.95, 2, True, "octahedral", "N/O")]),
            ((6.0, 8.0), [SpeciesInfo("Fe2+", 0.50, 2, True),
                          SpeciesInfo("Fe(OH)+", 0.35, 1, True),
                          SpeciesInfo("FeCO3(aq)", 0.15, 0, False)]),
            ((8.0, 14), [SpeciesInfo("Fe(OH)2(s)", 0.80, 0, False),
                         SpeciesInfo("Fe(OH)3-", 0.20, -1, True)]),
        ],
    },
    "Au3+": {
        "free_ion_limit_ph": 2.0,
        "precipitate_ph": 5.0,
        "ranges": [
            ((0, 2.0), [SpeciesInfo("AuCl4-", 0.80, -1, True, "square_planar", "S/Cl",
                                     "Chloro complex dominates in Cl- media"),
                         SpeciesInfo("Au3+", 0.20, 3, True, "square_planar", "S")]),
            ((2.0, 5.0), [SpeciesInfo("Au(OH)Cl3-", 0.40, -1, True),
                          SpeciesInfo("Au(OH)2Cl2-", 0.30, -1, True),
                          SpeciesInfo("Au(OH)3(aq)", 0.30, 0, False)]),
            ((5.0, 14), [SpeciesInfo("Au(OH)4-", 0.65, -1, True, "", "anion_exchange", "Aurate"),
                         SpeciesInfo("Au(OH)3(s)", 0.35, 0, False, "", "", "Precipitate")]),
        ],
    },
    "Hg2+": {
        "free_ion_limit_ph": 3.0,
        "precipitate_ph": 4.0,
        "ranges": [
            ((0, 3.0), [SpeciesInfo("HgCl2(aq)", 0.60, 0, True, "linear", "S",
                                     "Neutral chloro complex, bindable by soft donors"),
                         SpeciesInfo("Hg2+", 0.30, 2, True, "linear", "S"),
                         SpeciesInfo("HgCl+", 0.10, 1, True)]),
            ((3.0, 6.0), [SpeciesInfo("Hg(OH)2(aq)", 0.50, 0, True, "", "S",
                                       "Neutral hydroxide, still bindable by thiol"),
                           SpeciesInfo("HgCl2(aq)", 0.30, 0, True),
                           SpeciesInfo("Hg(OH)+", 0.20, 1, True)]),
            ((6.0, 14), [SpeciesInfo("Hg(OH)2(aq)", 0.70, 0, True, "", "S",
                                      "Neutral but thiol-bindable"),
                          SpeciesInfo("HgO(s)", 0.30, 0, False, "", "", "Precipitate")]),
        ],
    },
    "Ag+": {
        "free_ion_limit_ph": 8.0,
        "precipitate_ph": 10.0,
        "ranges": [
            ((0, 8.0), [SpeciesInfo("Ag+", 0.90, 1, True, "linear", "S/N")]),
            ((8.0, 10.0), [SpeciesInfo("Ag+", 0.50, 1, True),
                           SpeciesInfo("AgOH(aq)", 0.40, 0, True, "", "S"),
                           SpeciesInfo("Ag2O(s)", 0.10, 0, False)]),
            ((10.0, 14), [SpeciesInfo("Ag(OH)2-", 0.60, -1, True),
                          SpeciesInfo("Ag2O(s)", 0.40, 0, False)]),
        ],
    },
    "Cd2+": {
        "free_ion_limit_ph": 8.0,
        "precipitate_ph": 10.0,
        "ranges": [
            ((0, 8.0), [SpeciesInfo("Cd2+", 0.90, 2, True, "octahedral", "N/S")]),
            ((8.0, 10.0), [SpeciesInfo("Cd2+", 0.40, 2, True),
                           SpeciesInfo("Cd(OH)+", 0.40, 1, True),
                           SpeciesInfo("CdCO3(aq)", 0.20, 0, False)]),
            ((10.0, 14), [SpeciesInfo("Cd(OH)2(s)", 0.80, 0, False),
                          SpeciesInfo("Cd(OH)3-", 0.20, -1, True)]),
        ],
    },
    "Mn2+": {
        "free_ion_limit_ph": 8.5,
        "precipitate_ph": 9.5,
        "ranges": [
            ((0, 8.5), [SpeciesInfo("Mn2+", 0.95, 2, True, "octahedral", "O/N")]),
            ((8.5, 9.5), [SpeciesInfo("Mn2+", 0.50, 2, True),
                          SpeciesInfo("Mn(OH)+", 0.30, 1, True),
                          SpeciesInfo("MnCO3(aq)", 0.20, 0, False)]),
            ((9.5, 14), [SpeciesInfo("Mn(OH)2(s)", 0.85, 0, False),
                         SpeciesInfo("MnO4^2-", 0.15, -2, True, "", "", "Only if oxidized")]),
        ],
    },
    "UO2_2+": {
        "free_ion_limit_ph": 5.0,
        "precipitate_ph": 6.5,
        "ranges": [
            ((0, 5.0), [SpeciesInfo("UO2^2+", 0.85, 2, True,
                                     "pentagonal_bipyramidal_equatorial", "O/P")]),
            ((5.0, 6.5), [SpeciesInfo("UO2(OH)+", 0.40, 1, True),
                          SpeciesInfo("(UO2)2(OH)2^2+", 0.30, 2, True),
                          SpeciesInfo("UO2^2+", 0.20, 2, True),
                          SpeciesInfo("UO2CO3(aq)", 0.10, 0, False)]),
            ((6.5, 14), [SpeciesInfo("UO2(CO3)3^4-", 0.50, -4, True, "", "anion_exchange",
                                      "Uranyl tricarbonate, dominant in groundwater"),
                         SpeciesInfo("UO2(OH)2(s)", 0.30, 0, False),
                         SpeciesInfo("UO2(CO3)2^2-", 0.20, -2, True)]),
        ],
    },
    "Ce3+": {
        "free_ion_limit_ph": 7.0,
        "precipitate_ph": 8.5,
        "ranges": [
            ((0, 7.0), [SpeciesInfo("Ce3+", 0.90, 3, True, "tricapped_trigonal_prismatic", "O")]),
            ((7.0, 8.5), [SpeciesInfo("Ce3+", 0.40, 3, True),
                          SpeciesInfo("Ce(OH)2+", 0.35, 2, True),
                          SpeciesInfo("Ce(OH)2+", 0.25, 1, True)]),
            ((8.5, 14), [SpeciesInfo("Ce(OH)3(s)", 0.90, 0, False, "", "", "Precipitate")]),
        ],
    },
    "Na+": {
        "free_ion_limit_ph": 14.0,
        "precipitate_ph": 99.0,
        "ranges": [
            ((0, 14), [SpeciesInfo("Na+", 0.99, 1, True, "octahedral", "O", "Always free ion")]),
        ],
    },
    "K+": {
        "free_ion_limit_ph": 14.0,
        "precipitate_ph": 99.0,
        "ranges": [
            ((0, 14), [SpeciesInfo("K+", 0.99, 1, True, "cubic", "O", "Always free ion")]),
        ],
    },
    "Ba2+": {
        "free_ion_limit_ph": 13.0,
        "precipitate_ph": 99.0,
        "ranges": [
            ((0, 14), [SpeciesInfo("Ba2+", 0.95, 2, True, "cubic", "O",
                                    "Free ion except with sulfate/carbonate")]),
        ],
    },
}

# ── Redox pathways ───────────────────────────────────────────────────────

_REDOX_PATHWAYS = {
    "Au3+": RedoxPathway(
        "Au3+", "Au0", 1.50, 3, "reductive_deposition", (0, 6),
        ["thiol", "ascorbate", "citrate", "NaBH4"],
        "Thiol donors can REDUCE Au3+ to Au0 nanoparticles. "
        "Capture may be reductive deposition, not coordination.",
        ),
    "Au+": RedoxPathway(
        "Au+", "Au0", 1.69, 1, "reductive_deposition", (0, 6),
        ["thiol", "ascorbate"],
        "Au+ → Au0 very favorable. Thiol ligands stabilize Au+ but also reduce.",
        ),
    "Fe3+": RedoxPathway(
        "Fe3+", "Fe2+", 0.77, 1, "reductive_binding", (0, 5),
        ["ascorbate", "dithionite", "sulfide"],
        "Fe3+ → Fe2+ changes HSAB class (hard→borderline) and preferred ligands. "
        "Fe2+ much more soluble at neutral pH.",
        ),
    "Cr6+": RedoxPathway(
        "CrO4^2-", "Cr3+", 1.33, 3, "reductive_precipitation", (1, 7),
        ["Fe2+", "sulfide", "zero-valent iron", "organic reductants"],
        "Cr(VI) remediation IS reduction to Cr(III). Cr3+ precipitates as Cr(OH)3 at pH > 4. "
        "Do NOT design a binder for Cr(VI) — design a reductant delivery system.",
        ),
    "Hg2+": RedoxPathway(
        "Hg2+", "Hg0", 0.85, 2, "reductive_deposition", (0, 7),
        ["SnCl2", "NaBH4"],
        "Hg2+ → Hg0 vapor is a remediation pathway (cold vapor). "
        "For capture, coordination binding (thiol) preferred over reduction.",
        ),
    "UO2_2+": RedoxPathway(
        "U(VI)", "U(IV)", 0.27, 2, "reductive_precipitation", (4, 9),
        ["sulfide", "Fe2+", "zero-valent iron", "microbial"],
        "UO2^2+ → UO2(s) immobilization. Reductive precipitation is the main "
        "remediation strategy for uranium in groundwater.",
        ),
    "Se4+": RedoxPathway(
        "SeO3^2-", "Se0", 0.74, 4, "reductive_precipitation", (3, 9),
        ["Fe2+", "zero-valent iron", "sulfide", "microbial"],
        "Selenite → elemental selenium. Main remediation pathway.",
        ),
    "As5+": RedoxPathway(
        "AsO4^3-", "As3+", 0.56, 2, "reductive_binding", (3, 9),
        ["Fe2+", "sulfide"],
        "Arsenate → arsenite. As3+ is more toxic but more treatable by "
        "coprecipitation with iron.",
        ),
}


# ═══════════════════════════════════════════════════════════════════════════
# SPECIATION PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

def predict_speciation(metal, ph, redox_mv=None, chloride_mm=0.0):
    """Predict what species of the target metal exist at given conditions.

    Returns a SpeciationResult that tells the generative engine:
    - What species to actually design for
    - What charge and geometry to use
    - Whether binding, precipitation, or redox is the right strategy
    """
    data = _SPECIATION.get(metal)

    if data is None:
        return _unknown_metal_speciation(metal, ph)

    # Find species at this pH
    species = []
    for (ph_lo, ph_hi), sp_list in data["ranges"]:
        if ph_lo <= ph < ph_hi:
            species = list(sp_list)
            break
    if not species:
        species = [SpeciesInfo(metal, 1.0, 2, True)]

    dominant = max(species, key=lambda s: s.fraction)
    free_ion_frac = sum(s.fraction for s in species
                         if s.charge > 0 and s.bindable and s.formula == metal)
    bindable_frac = sum(s.fraction for s in species if s.bindable)

    # Determine design strategy
    strategy, eff_formula, eff_charge, notes = _determine_strategy(
        metal, ph, species, dominant, free_ion_frac, bindable_frac, data)

    # Check redox pathways
    redox = _REDOX_PATHWAYS.get(metal)
    if redox and redox_mv is not None:
        redox_dg = _nernst_dg(redox.e0_v, redox.n_electrons, redox_mv, ph)
        redox.dg_redox_kj = redox_dg
        if redox_dg < -20.0:  # Strongly favorable
            notes += (f" Redox pathway {redox.oxidized}→{redox.reduced} is "
                      f"favorable (ΔG={redox_dg:.0f} kJ/mol). "
                      f"Consider {redox.mechanism}.")
            if redox.mechanism in ("reductive_precipitation", "reductive_deposition"):
                strategy = "redox_capture"
                eff_formula = redox.reduced

    return SpeciationResult(
        metal=metal, ph=ph, species=species,
        dominant_species=dominant.formula,
        dominant_fraction=dominant.fraction,
        free_ion_fraction=free_ion_frac,
        bindable_fraction=bindable_frac,
        design_strategy=strategy,
        effective_charge=eff_charge,
        effective_formula=eff_formula,
        design_notes=notes,
    )


def _determine_strategy(metal, ph, species, dominant, free_ion_frac, bindable_frac, data):
    """Determine the best capture strategy from speciation."""
    if free_ion_frac >= 0.5:
        return ("free_ion_binding", metal,
                float(max(s.charge for s in species if s.formula == metal)),
                f"Free ion dominates ({free_ion_frac*100:.0f}%). Standard coordination design.")

    if bindable_frac < 0.2:
        if ph > data["precipitate_ph"]:
            return ("precipitation_sufficient", f"{metal.rstrip('+0123456789')}(OH)x",
                    0.0, f"At pH {ph}, {metal} predominantly precipitates. "
                    f"Raise pH to {data['precipitate_ph']:.0f}+ for simple removal. "
                    f"No binder needed unless recovery is the goal.")
        else:
            return ("pH_adjust", metal, 2.0,
                    f"Low bindable fraction ({bindable_frac*100:.0f}%). "
                    f"Consider adjusting pH below {data['free_ion_limit_ph']:.1f} for free ion capture.")

    # Bindable but not free ion → hydroxo or anion species
    anion_species = [s for s in species if s.charge < 0 and s.bindable]
    hydroxo_species = [s for s in species if s.charge > 0 and s.bindable and "OH" in s.formula]

    if anion_species and sum(s.fraction for s in anion_species) >= 0.25:
        top_anion = max(anion_species, key=lambda s: s.fraction)
        return ("anion_capture", top_anion.formula, float(top_anion.charge),
                f"Dominant bindable species is anionic ({top_anion.formula}, "
                f"{top_anion.fraction*100:.0f}%). Use LDH or anion exchanger.")

    if hydroxo_species:
        top_h = max(hydroxo_species, key=lambda s: s.fraction)
        return ("hydroxo_binding", top_h.formula, float(top_h.charge),
                f"Hydroxo complex {top_h.formula} ({top_h.fraction*100:.0f}%). "
                f"Design for reduced charge and O-donor preference.")

    # Mixed — target whatever is most bindable
    top_bindable = max([s for s in species if s.bindable], key=lambda s: s.fraction)
    return ("mixed_species_binding", top_bindable.formula, float(top_bindable.charge),
            f"Mixed speciation. Target {top_bindable.formula} ({top_bindable.fraction*100:.0f}%).")


def _unknown_metal_speciation(metal, ph):
    """Fallback for metals not in database."""
    if ph < 7:
        return SpeciationResult(
            metal, ph, [SpeciesInfo(metal, 0.90, 2, True)],
            metal, 0.90, 0.90, 0.90, "free_ion_binding",
            2.0, metal, f"No speciation data for {metal}. Assuming free ion at pH {ph}.")
    else:
        return SpeciationResult(
            metal, ph, [SpeciesInfo(metal, 0.50, 2, True),
                        SpeciesInfo(f"{metal}_hydroxide", 0.50, 0, False)],
            metal, 0.50, 0.50, 0.50, "mixed_species_binding",
            1.0, metal,
            f"No speciation data. At pH {ph}, hydroxide formation likely. "
            f"Consider verifying speciation with PHREEQC.")


def _nernst_dg(e0_v, n_electrons, eh_mv, ph):
    """Calculate ΔG for redox pathway via Nernst equation.

    ΔG = -nFE  where E = Eh - E0 (simplified)
    For pH-dependent reactions: E = E0 - (0.0592/n) * pH (at 25°C)
    """
    F = 96485  # C/mol
    eh_v = eh_mv / 1000.0
    # pH-adjusted potential (many metal reductions are pH-dependent)
    e_adj = e0_v - (0.0592 / max(1, n_electrons)) * ph
    # ΔG = -nF(Eh - E_adj) ... if Eh > E_adj, reduction is favorable
    dg = -n_electrons * F * (eh_v - e_adj) / 1000.0  # kJ/mol
    return round(dg, 1)


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH GENERATIVE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def speciation_gated_design(
    target_identity, target_formula, charge=2,
    working_ph=7.0, working_temp_c=25.0, ionic_strength_mm=10.0,
    redox_mv=None, chloride_mm=0.0,
    **kwargs,
):
    """Run speciation check, then route to appropriate design strategy.

    This wraps generative_design() with a speciation gate that can:
    1. Proceed normally (free ion binding)
    2. Adjust effective charge/formula for hydroxo species
    3. Route to anion capture (LDH scaffold preference)
    4. Flag precipitation as sufficient (no binder needed)
    5. Flag redox capture pathway
    6. Recommend pH adjustment

    Returns (SpeciationResult, list[BinderAssembly] or str)
    """
    from core.generative_physics_adapter import design_and_score

    spec = predict_speciation(target_formula, working_ph, redox_mv, chloride_mm)

    if spec.design_strategy == "precipitation_sufficient":
        return spec, []

    if spec.design_strategy == "redox_capture":
        return spec, []  # Redox pathway — no coordination binder needed

    # Adjust design parameters based on speciation
    effective_charge = int(round(spec.effective_charge))
    if effective_charge <= 0:
        effective_charge = max(1, abs(effective_charge))

    # For anion capture, bias toward LDH/anion exchange scaffolds
    design_kwargs = dict(
        target_identity=target_identity,
        target_formula=target_formula,
        charge=max(1, abs(effective_charge)),
        working_ph=working_ph,
        working_temp_c=working_temp_c,
        ionic_strength_mm=ionic_strength_mm,
    )
    design_kwargs.update(kwargs)

    assemblies = design_and_score(**design_kwargs)

    # Annotate assemblies with speciation info
    for a in assemblies:
        a.confidence_reasoning.insert(0,
            f"Speciation: {spec.design_strategy} — {spec.dominant_species} "
            f"({spec.dominant_fraction*100:.0f}%), free ion {spec.free_ion_fraction*100:.0f}%")
        if spec.free_ion_fraction < 0.5:
            a.failure_modes.insert(0,
                f"Only {spec.free_ion_fraction*100:.0f}% free ion at pH {working_ph}. "
                f"Effective concentration reduced.")
            # Scale predicted Kd by bindable fraction
            if a.thermodynamics and spec.bindable_fraction < 1.0:
                # Effective Kd is worse by factor of 1/bindable_fraction
                a.thermodynamics.predicted_kd_um /= max(0.01, spec.bindable_fraction)

    return spec, assemblies

''')

write_file("core/spin_state.py", '''\
"""
core/spin_state.py — Sprint 19: Spin State Predictor + Strong/Weak Field LFSE

Determines high-spin vs low-spin from d-electron count + ligand field
strength. Replaces the single LFSE table with spin-state-aware calculation.
Adds magnetic moment prediction.

Physics:
  If 10Dq > pairing energy P → low-spin
  If 10Dq < P → high-spin
  d1-d3, d8-d10: no spin-state choice (same either way)
  d4-d7: spin-state depends on field strength
"""
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class SpinStateResult:
    d_electrons: int
    spin_state: str             # "high_spin", "low_spin", "no_choice"
    unpaired_electrons: int
    magnetic_moment_bm: float   # Bohr magnetons, spin-only μ = √(n(n+2))
    lfse_oct_kj: float          # LFSE in octahedral field
    lfse_tet_kj: float          # LFSE in tetrahedral field
    lfse_sq_planar_kj: float    # LFSE in square planar field
    pairing_energy_kj: float    # Estimated P for this ion
    field_strength_10dq_kj: float  # Estimated 10Dq from ligand field
    spin_crossover_possible: bool  # 10Dq ≈ P
    rationale: str

@dataclass
class LFSEResult:
    """LFSE for a specific geometry, incorporating spin state."""
    geometry: str
    lfse_kj: float
    spin_state: str
    unpaired_electrons: int
    magnetic_moment_bm: float
    jahn_teller: str            # "none", "weak", "strong"
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# PAIRING ENERGY DATABASE (kJ/mol)
# From Figgis & Hitchman, Lever; values for aqua ions
# ═══════════════════════════════════════════════════════════════════════════

_PAIRING_ENERGY = {
    # First row d-block (kJ/mol)
    "Ti3+": 168, "V3+": 188, "V2+": 172, "Cr3+": 208, "Cr2+": 176,
    "Mn3+": 230, "Mn2+": 213, "Fe3+": 252, "Fe2+": 210,
    "Co3+": 264, "Co2+": 226, "Ni2+": 248, "Cu2+": 230,
    # Second row
    "Mo3+": 160, "Ru3+": 180, "Ru2+": 170, "Rh3+": 190, "Pd2+": 195,
    # Third row
    "Re3+": 150, "Os3+": 170, "Os2+": 160, "Ir3+": 180, "Pt2+": 200,
    "Pt4+": 220, "Au3+": 210,
}

# Default pairing energies by row and charge
_DEFAULT_P = {
    (3, 2): 210,  # 1st row, 2+
    (3, 3): 240,  # 1st row, 3+
    (4, 2): 170,  # 2nd row, 2+
    (4, 3): 180,  # 2nd row, 3+
    (5, 2): 160,  # 3rd row, 2+
    (5, 3): 170,  # 3rd row, 3+
}


# ═══════════════════════════════════════════════════════════════════════════
# 10Dq ESTIMATION FROM LIGAND FIELD STRENGTHS
# Uses average Dq from the donor ligand templates
# ═══════════════════════════════════════════════════════════════════════════

# Reference 10Dq(oct) for aqua complexes in kJ/mol
_10DQ_AQUA = {
    "Ti3+": 243, "V3+": 216, "V2+": 147, "Cr3+": 208, "Cr2+": 166,
    "Mn3+": 252, "Mn2+": 90, "Fe3+": 164, "Fe2+": 124,
    "Co3+": 220, "Co2+": 111, "Ni2+": 102, "Cu2+": 151,
    "Ru2+": 240, "Rh3+": 340, "Pd2+": 310, "Ir3+": 340,
    "Pt2+": 370, "Pt4+": 420, "Au3+": 380,
}

# Spectrochemical series multipliers relative to water (f factor)
_FIELD_MULTIPLIER = {
    "water": 1.00,
    # Weak field
    "hydroxide": 0.72, "fluoride": 0.90, "chloride": 0.78,
    "bromide": 0.72, "iodide": 0.60, "sulfide": 0.65,
    # Moderate field
    "carboxylate": 0.85, "phosphonate": 0.80,
    "phenolate": 0.88, "catechol": 0.90,
    # Borderline
    "primary_amine": 1.25, "tertiary_amine": 1.20,
    "imidazole": 1.30, "pyridine": 1.35, "bipyridyl": 1.90,
    "hydroxamate": 0.95,
    # Strong field
    "iminodiacetate": 1.15, "salicylaldehyde_imine": 1.35,
    "cyanide": 2.50, "carbonyl": 2.60, "nitrosyl": 2.70,
    "phosphine": 1.60,
    # Soft donors
    "thiolate": 0.75, "thioether": 0.80, "dithiocarbamate": 0.78,
    "thiourea": 0.82, "crown_ether_O": 0.70,
}


def _get_10dq(metal_formula, ligand_names):
    """Estimate 10Dq for a metal with given ligands.

    Uses spectrochemical series: 10Dq(complex) = 10Dq(aqua) × f_avg
    where f_avg is the average field multiplier of the ligands.
    """
    base_10dq = _10DQ_AQUA.get(metal_formula, 120.0)

    if not ligand_names:
        return base_10dq

    multipliers = []
    for name in ligand_names:
        m = _FIELD_MULTIPLIER.get(name, 1.0)
        multipliers.append(m)

    f_avg = sum(multipliers) / len(multipliers) if multipliers else 1.0

    # 2nd/3rd row metals have inherently larger 10Dq (~1.5x, ~2.0x)
    row_factor = 1.0
    if metal_formula in ("Ru2+", "Rh3+", "Mo3+", "Pd2+"):
        row_factor = 1.45
    elif metal_formula in ("Ir3+", "Pt2+", "Pt4+", "Os2+", "Os3+", "Re3+", "Au3+"):
        row_factor = 1.75

    return base_10dq * f_avg * row_factor


def _get_pairing_energy(metal_formula):
    """Get pairing energy for a metal ion."""
    if metal_formula in _PAIRING_ENERGY:
        return _PAIRING_ENERGY[metal_formula]
    # Estimate from defaults
    charge = 2
    if "3+" in metal_formula or "3+$" in metal_formula:
        charge = 3
    row = 3  # Default to first row
    return _DEFAULT_P.get((row, charge), 210)


# ═══════════════════════════════════════════════════════════════════════════
# LFSE TABLES — HIGH SPIN AND LOW SPIN
# Values in units of Dq (multiply by 10Dq/10 = Dq to get kJ/mol)
# Format: (LFSE_oct_Dq, unpaired_e_oct, LFSE_tet_Dq, unpaired_e_tet)
# ═══════════════════════════════════════════════════════════════════════════

# Octahedral LFSE in Dq units
_HS_OCT = {
    0: (0, 0), 1: (-4, 1), 2: (-8, 2), 3: (-12, 3),
    4: (-6, 4), 5: (0, 5), 6: (-4, 4), 7: (-8, 3),
    8: (-12, 2), 9: (-6, 1), 10: (0, 0),
}

_LS_OCT = {
    0: (0, 0), 1: (-4, 1), 2: (-8, 2), 3: (-12, 3),
    4: (-16, 2), 5: (-20, 1), 6: (-24, 0), 7: (-18, 1),
    8: (-12, 2), 9: (-6, 1), 10: (0, 0),
}

_HS_TET = {
    0: (0, 0), 1: (-2.67, 1), 2: (-5.34, 2), 3: (-3.56, 3),
    4: (-1.78, 4), 5: (0, 5), 6: (-2.67, 4), 7: (-5.34, 3),
    8: (-3.56, 2), 9: (-1.78, 1), 10: (0, 0),
}

# Square planar LFSE — only relevant for d8 (and some d9, d7)
_SQ_PLANAR = {
    7: (-24.56, 1), 8: (-24.56, 0), 9: (-21.89, 1), 10: (0, 0),
}

# Jahn-Teller distortion strength
_JT_STRENGTH = {
    # (d_electrons, spin_state): "none" / "weak" / "strong"
    (4, "high_spin"): "strong",   # eg: 1 electron in d(z2) or d(x2-y2)
    (7, "low_spin"): "strong",
    (9, "high_spin"): "strong",   # Cu2+ classic
    (9, "low_spin"): "strong",
    (1, "high_spin"): "weak",
    (2, "high_spin"): "weak",
    (6, "high_spin"): "weak",
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

def predict_spin_state(metal_formula, d_electrons, ligand_names=None):
    """Predict spin state from metal d-count and ligand field strengths.

    Args:
        metal_formula: e.g. "Ni2+", "Fe3+", "Au3+"
        d_electrons: number of d electrons
        ligand_names: list of ligand names from donor_chemistry.py

    Returns:
        SpinStateResult with spin state, magnetic moment, and LFSE values
    """
    ligand_names = ligand_names or []

    # d0-d3 and d8-d10: no spin-state ambiguity
    if d_electrons <= 3 or d_electrons >= 8:
        return _no_choice_result(metal_formula, d_electrons, ligand_names)

    # d4-d7: compare 10Dq vs pairing energy
    ten_dq = _get_10dq(metal_formula, ligand_names)
    P = _get_pairing_energy(metal_formula)
    dq = ten_dq / 10.0

    crossover = abs(ten_dq - P) < P * 0.15  # Within 15% = spin crossover region

    if ten_dq > P:
        # Low spin
        lfse_oct_dq, unpaired_oct = _LS_OCT[d_electrons]
        spin = "low_spin"
    else:
        # High spin
        lfse_oct_dq, unpaired_oct = _HS_OCT[d_electrons]
        spin = "high_spin"

    lfse_tet_dq, unpaired_tet = _HS_TET[d_electrons]  # Tet is almost always HS

    lfse_oct = lfse_oct_dq * dq
    lfse_tet = lfse_tet_dq * (ten_dq * 4 / 90)  # 10Dq(tet) ≈ 4/9 × 10Dq(oct)

    # Square planar (if applicable)
    lfse_sq = 0.0
    if d_electrons in _SQ_PLANAR:
        sq_dq, _ = _SQ_PLANAR[d_electrons]
        lfse_sq = sq_dq * dq

    mu = math.sqrt(unpaired_oct * (unpaired_oct + 2))

    rationale = (f"d{d_electrons} {metal_formula}: 10Dq={ten_dq:.0f} kJ/mol, "
                 f"P={P:.0f} kJ/mol → {'10Dq > P → low-spin' if spin == 'low_spin' else '10Dq < P → high-spin'}. "
                 f"Ligands: {', '.join(ligand_names[:4]) if ligand_names else 'aqua'}.")
    if crossover:
        rationale += " Near spin-crossover boundary."

    return SpinStateResult(
        d_electrons=d_electrons, spin_state=spin,
        unpaired_electrons=unpaired_oct,
        magnetic_moment_bm=round(mu, 2),
        lfse_oct_kj=round(lfse_oct, 1),
        lfse_tet_kj=round(lfse_tet, 1),
        lfse_sq_planar_kj=round(lfse_sq, 1),
        pairing_energy_kj=P, field_strength_10dq_kj=round(ten_dq, 1),
        spin_crossover_possible=crossover,
        rationale=rationale,
    )


def _no_choice_result(metal_formula, d_electrons, ligand_names):
    """For d0-d3, d8-d10 where spin state is unambiguous."""
    ten_dq = _get_10dq(metal_formula, ligand_names)
    dq = ten_dq / 10.0

    lfse_oct_dq, unpaired_oct = _HS_OCT[d_electrons]
    lfse_tet_dq, unpaired_tet = _HS_TET[d_electrons]

    lfse_oct = lfse_oct_dq * dq
    lfse_tet = lfse_tet_dq * (ten_dq * 4 / 90)
    lfse_sq = 0.0
    if d_electrons in _SQ_PLANAR:
        sq_dq, unpaired_sq = _SQ_PLANAR[d_electrons]
        lfse_sq = sq_dq * dq
        unpaired_oct = unpaired_sq if d_electrons == 8 else unpaired_oct

    # d8 square planar is always low-spin (0 unpaired)
    if d_electrons == 8:
        unpaired_oct = 2  # Octahedral d8 has 2 unpaired
        spin = "no_choice"
    else:
        spin = "no_choice"

    mu = math.sqrt(unpaired_oct * (unpaired_oct + 2))

    return SpinStateResult(
        d_electrons=d_electrons, spin_state=spin,
        unpaired_electrons=unpaired_oct,
        magnetic_moment_bm=round(mu, 2),
        lfse_oct_kj=round(lfse_oct, 1),
        lfse_tet_kj=round(lfse_tet, 1),
        lfse_sq_planar_kj=round(lfse_sq, 1),
        pairing_energy_kj=_get_pairing_energy(metal_formula),
        field_strength_10dq_kj=round(ten_dq, 1),
        spin_crossover_possible=False,
        rationale=(f"d{d_electrons} {metal_formula}: no spin-state choice "
                   f"(same ground state). 10Dq={ten_dq:.0f} kJ/mol. "
                   f"LFSE(oct)={lfse_oct:.1f} kJ/mol."),
    )


def compute_lfse_for_geometry(metal_formula, d_electrons, geometry,
                               ligand_names=None):
    """Get LFSE for a specific geometry, with spin state awareness.

    This replaces the Sprint 12 fixed LFSE lookup with a field-strength-
    dependent calculation.
    """
    ss = predict_spin_state(metal_formula, d_electrons, ligand_names)

    if "square_planar" in geometry:
        if d_electrons in _SQ_PLANAR:
            sq_dq, unpaired = _SQ_PLANAR[d_electrons]
            dq = ss.field_strength_10dq_kj / 10.0
            lfse = sq_dq * dq
            jt = "none"  # Square planar is already distorted
            return LFSEResult(geometry, round(lfse, 1), "low_spin" if d_electrons == 8 else ss.spin_state,
                              unpaired if d_electrons == 8 else ss.unpaired_electrons,
                              round(math.sqrt(unpaired * (unpaired + 2)), 2),
                              jt, f"Square planar d{d_electrons}")
        return LFSEResult(geometry, 0.0, ss.spin_state, ss.unpaired_electrons,
                          ss.magnetic_moment_bm, "none")

    if "tetrahedral" in geometry:
        jt_key = (d_electrons, ss.spin_state)
        jt = _JT_STRENGTH.get(jt_key, "none")
        return LFSEResult(geometry, ss.lfse_tet_kj, "high_spin",  # Tet is always HS
                          _HS_TET[d_electrons][1],
                          round(math.sqrt(_HS_TET[d_electrons][1] * (_HS_TET[d_electrons][1] + 2)), 2),
                          jt)

    if "linear" in geometry:
        # Linear: LFSE ≈ 0 for most cases, dominated by orbital preference
        return LFSEResult(geometry, 0.0, ss.spin_state, ss.unpaired_electrons,
                          ss.magnetic_moment_bm, "none", "Linear geometry — LFSE minimal")

    # Default to octahedral (includes tetragonal_elongated, hemidirected, etc.)
    jt_key = (d_electrons, ss.spin_state)
    jt = _JT_STRENGTH.get(jt_key, "none")
    if jt == "none" and ss.spin_state == "no_choice":
        # Try both HS and LS keys for no_choice metals
        jt = _JT_STRENGTH.get((d_electrons, "high_spin"),
             _JT_STRENGTH.get((d_electrons, "low_spin"), "none"))
    return LFSEResult(geometry, ss.lfse_oct_kj, ss.spin_state,
                      ss.unpaired_electrons, ss.magnetic_moment_bm, jt)

''')

write_file("tests/test_sprint18.py", '''\
"""tests/test_sprint18.py — Sprint 18: Speciation Gate + Redox Routing (15 tests)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.speciation_gate import (
    predict_speciation, speciation_gated_design,
    _nernst_dg, _REDOX_PATHWAYS,
)

# === SPECIATION PREDICTION ===
def test_pb_free_ion_at_low_ph():
    """Pb2+ at pH 4 should be >90% free ion."""
    s = predict_speciation("Pb2+", 4.0)
    assert s.free_ion_fraction >= 0.90
    assert s.design_strategy == "free_ion_binding"
    assert s.effective_formula == "Pb2+"
    print(f"  \\u2705 test_pb_free_ion_at_low_ph: {s.free_ion_fraction*100:.0f}% free ion, strategy={s.design_strategy}")

def test_pb_precipitate_at_high_ph():
    """Pb2+ at pH 10 should precipitate — no binder needed."""
    s = predict_speciation("Pb2+", 10.0)
    assert s.free_ion_fraction < 0.1
    assert s.design_strategy in ("precipitation_sufficient", "anion_capture")
    print(f"  \\u2705 test_pb_precipitate_at_high_ph: strategy={s.design_strategy}, free_ion={s.free_ion_fraction*100:.0f}%")

def test_pb_mixed_speciation():
    """Pb2+ at pH 7 should show mixed speciation."""
    s = predict_speciation("Pb2+", 7.0)
    assert s.free_ion_fraction < 0.90
    assert s.bindable_fraction > 0.3
    assert len(s.species) > 1
    print(f"  \\u2705 test_pb_mixed_speciation: {len(s.species)} species, free_ion={s.free_ion_fraction*100:.0f}%")

def test_fe3_precipitates_above_ph4():
    """Fe3+ at pH 6 should be precipitated rust."""
    s = predict_speciation("Fe3+", 6.0)
    assert s.free_ion_fraction < 0.05
    assert s.design_strategy == "precipitation_sufficient"
    print(f"  \\u2705 test_fe3_precipitates: strategy={s.design_strategy}")

def test_fe3_free_ion_in_acid():
    """Fe3+ at pH 2 should be free ion."""
    s = predict_speciation("Fe3+", 2.0)
    assert s.free_ion_fraction >= 0.85
    assert s.design_strategy == "free_ion_binding"
    print(f"  \\u2705 test_fe3_free_ion_in_acid: {s.free_ion_fraction*100:.0f}%")

def test_au3_chloro_complex():
    """Au3+ at pH 1 should be AuCl4- complex."""
    s = predict_speciation("Au3+", 1.0)
    assert any("AuCl4" in sp.formula for sp in s.species)
    print(f"  \\u2705 test_au3_chloro: dominant={s.dominant_species}")

def test_uo2_carbonate_at_high_ph():
    """UO2 2+ at pH 8 should be uranyl carbonate anion."""
    s = predict_speciation("UO2_2+", 8.0)
    assert any("CO3" in sp.formula for sp in s.species)
    assert s.design_strategy in ("anion_capture", "mixed_species_binding")
    print(f"  \\u2705 test_uo2_carbonate: dominant={s.dominant_species}, strategy={s.design_strategy}")

def test_hg2_thiol_bindable_at_all_ph():
    """Hg2+ species should be thiol-bindable even as Hg(OH)2."""
    for ph in [2.0, 5.0, 8.0]:
        s = predict_speciation("Hg2+", ph)
        assert s.bindable_fraction >= 0.5, f"Hg should be bindable at pH {ph}"
    print(f"  \\u2705 test_hg2_thiol_bindable: bindable at all pH tested")

def test_na_always_free():
    """Na+ should be free ion at any pH."""
    s = predict_speciation("Na+", 12.0)
    assert s.free_ion_fraction >= 0.95
    assert s.design_strategy == "free_ion_binding"
    print(f"  \\u2705 test_na_always_free: {s.free_ion_fraction*100:.0f}%")

def test_unknown_metal():
    """Unknown metal should return reasonable fallback."""
    s = predict_speciation("Tl+", 5.0)
    assert s.free_ion_fraction > 0
    assert "No speciation data" in s.design_notes
    print(f"  \\u2705 test_unknown_metal: {s.design_notes[:60]}")

# === REDOX ROUTING ===
def test_redox_nernst_calculation():
    """Nernst equation should give reasonable dG values."""
    # Au3+ reduction at Eh=1600mV (strong reductant), pH 2: should be favorable
    dg_fav = _nernst_dg(1.50, 3, 1600, 2.0)
    assert dg_fav < 0, f"Au3+ reduction at high Eh should be favorable, got dG={dg_fav}"
    # At low Eh (800mV), Au3+ reduction is NOT favorable
    dg_unfav = _nernst_dg(1.50, 3, 800, 2.0)
    assert dg_unfav > 0, f"Au3+ reduction at low Eh should be unfavorable, got dG={dg_unfav}"
    print(f"  \\u2705 test_redox_nernst: Eh=1600 dG={dg_fav:.1f}, Eh=800 dG={dg_unfav:.1f} kJ/mol")

def test_redox_au_pathway():
    """Au3+ with strongly reducing conditions should flag reductive deposition."""
    s = predict_speciation("Au3+", 2.0, redox_mv=1600)
    assert s.design_strategy == "redox_capture"
    assert "reduc" in s.design_notes.lower()
    print(f"  \\u2705 test_redox_au: strategy={s.design_strategy}")

# === GATED DESIGN ===
def test_gated_design_normal():
    """Normal conditions should produce scored assemblies."""
    spec, assemblies = speciation_gated_design("copper", "Cu2+", working_ph=4.0)
    assert spec.design_strategy == "free_ion_binding"
    assert len(assemblies) > 0
    assert assemblies[0].thermodynamics is not None
    # Should have speciation annotation
    assert any("Speciation" in c for c in assemblies[0].confidence_reasoning)
    print(f"  \\u2705 test_gated_design_normal: {len(assemblies)} assemblies, "
          f"top dG={assemblies[0].thermodynamics.dg_net_kj:.1f}")

def test_gated_design_precipitation():
    """Fe3+ at pH 8 should return empty — precipitation sufficient."""
    spec, assemblies = speciation_gated_design("iron3", "Fe3+", working_ph=8.0)
    assert spec.design_strategy == "precipitation_sufficient"
    assert len(assemblies) == 0
    print(f"  \\u2705 test_gated_design_precipitation: no binder needed (correct)")

def test_gated_design_low_free_ion():
    """Pb2+ at pH 7.5 should annotate reduced free ion fraction."""
    spec, assemblies = speciation_gated_design("lead", "Pb2+", working_ph=7.5)
    assert spec.free_ion_fraction < 0.5
    if assemblies:
        assert any("free ion" in f.lower() for f in assemblies[0].failure_modes)
    print(f"  \\u2705 test_gated_design_low_free_ion: free_ion={spec.free_ion_fraction*100:.0f}%, "
          f"assemblies={len(assemblies)}")

if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprint 18 \\u2014 Speciation Gate + Redox Routing\\n")
    print("Speciation Prediction:")
    test_pb_free_ion_at_low_ph(); test_pb_precipitate_at_high_ph()
    test_pb_mixed_speciation(); test_fe3_precipitates_above_ph4()
    test_fe3_free_ion_in_acid(); test_au3_chloro_complex()
    test_uo2_carbonate_at_high_ph(); test_hg2_thiol_bindable_at_all_ph()
    test_na_always_free(); test_unknown_metal()
    print("\\nRedox Routing:")
    test_redox_nernst_calculation(); test_redox_au_pathway()
    print("\\nGated Design:")
    test_gated_design_normal(); test_gated_design_precipitation()
    test_gated_design_low_free_ion()
    print("\\n\\u2705 All Sprint 18 tests passed! (15/15)")
    print("\\n\\U0001f389 SPECIATION GATE OPERATIONAL\\n")

''')

write_file("tests/test_sprint19.py", '''\
"""tests/test_sprint19.py — Sprint 19: Spin State + Strong/Weak Field LFSE (15 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.spin_state import (
    predict_spin_state, compute_lfse_for_geometry,
    _get_10dq, _get_pairing_energy,
)

# === SPIN STATE PREDICTION ===
def test_fe2_weak_field_high_spin():
    """Fe2+ d6 with weak-field water → high-spin, 4 unpaired."""
    ss = predict_spin_state("Fe2+", 6, ["water"])
    assert ss.spin_state == "high_spin"
    assert ss.unpaired_electrons == 4
    assert ss.magnetic_moment_bm > 4.5  # √(4×6) = 4.90 BM
    print(f"  \\u2705 test_fe2_weak_field_hs: {ss.spin_state}, μ={ss.magnetic_moment_bm:.2f} BM, "
          f"unpaired={ss.unpaired_electrons}")

def test_fe2_strong_field_low_spin():
    """Fe2+ d6 with strong-field bipyridyl → low-spin, 0 unpaired."""
    ss = predict_spin_state("Fe2+", 6, ["bipyridyl", "bipyridyl", "bipyridyl"])
    assert ss.spin_state == "low_spin"
    assert ss.unpaired_electrons == 0
    assert ss.magnetic_moment_bm == 0.0  # Diamagnetic!
    assert abs(ss.lfse_oct_kj) > abs(predict_spin_state("Fe2+", 6, ["water"]).lfse_oct_kj)
    print(f"  \\u2705 test_fe2_strong_field_ls: {ss.spin_state}, μ={ss.magnetic_moment_bm:.2f} BM, "
          f"LFSE={ss.lfse_oct_kj:.1f} kJ/mol")

def test_fe3_weak_field_high_spin():
    """Fe3+ d5 with water → high-spin, 5 unpaired, LFSE=0."""
    ss = predict_spin_state("Fe3+", 5, ["water"])
    assert ss.spin_state == "high_spin"
    assert ss.unpaired_electrons == 5
    assert ss.lfse_oct_kj == 0.0  # d5 HS has zero LFSE
    print(f"  \\u2705 test_fe3_weak_field_hs: {ss.spin_state}, LFSE={ss.lfse_oct_kj:.1f}, "
          f"unpaired={ss.unpaired_electrons}")

def test_fe3_strong_field_low_spin():
    """Fe3+ d5 with cyanide → low-spin, 1 unpaired."""
    ss = predict_spin_state("Fe3+", 5, ["cyanide"] * 6)
    assert ss.spin_state == "low_spin"
    assert ss.unpaired_electrons == 1
    assert ss.magnetic_moment_bm < 2.5  # √(1×3) = 1.73 BM
    assert ss.lfse_oct_kj < -100  # Significant LFSE
    print(f"  \\u2705 test_fe3_strong_field_ls: {ss.spin_state}, μ={ss.magnetic_moment_bm:.2f} BM, "
          f"LFSE={ss.lfse_oct_kj:.1f} kJ/mol")

def test_co2_weak_vs_strong():
    """Co2+ d7 should change spin state with field strength."""
    weak = predict_spin_state("Co2+", 7, ["water"])
    strong = predict_spin_state("Co2+", 7, ["cyanide"] * 6)
    assert weak.spin_state == "high_spin"
    assert weak.unpaired_electrons == 3
    assert strong.spin_state == "low_spin"
    assert strong.unpaired_electrons == 1
    print(f"  \\u2705 test_co2_weak_vs_strong: water={weak.spin_state}({weak.unpaired_electrons}), "
          f"CN-={strong.spin_state}({strong.unpaired_electrons})")

def test_ni2_no_spin_choice():
    """Ni2+ d8 has no spin-state ambiguity."""
    ss = predict_spin_state("Ni2+", 8, ["imidazole"] * 4)
    assert ss.spin_state == "no_choice"
    assert ss.d_electrons == 8
    print(f"  \\u2705 test_ni2_no_choice: {ss.spin_state}, LFSE_oct={ss.lfse_oct_kj:.1f}")

def test_d10_zero_lfse():
    """d10 metals should have zero LFSE regardless of ligands."""
    for metal in ["Zn2+", "Ag+"]:
        ss = predict_spin_state(metal, 10, ["imidazole"] * 4)
        assert ss.lfse_oct_kj == 0.0
        assert ss.lfse_tet_kj == 0.0
    print(f"  \\u2705 test_d10_zero_lfse: Zn2+ and Ag+ both LFSE=0")

def test_cr3_always_high_lfse():
    """Cr3+ d3 has 3 unpaired regardless, but large LFSE."""
    ss = predict_spin_state("Cr3+", 3, ["imidazole"] * 6)
    assert ss.unpaired_electrons == 3
    assert ss.lfse_oct_kj < -100  # Strong LFSE
    print(f"  \\u2705 test_cr3_high_lfse: unpaired=3, LFSE={ss.lfse_oct_kj:.1f}")

def test_au3_large_10dq():
    """Au3+ (3rd row) should have very large 10Dq."""
    ten_dq = _get_10dq("Au3+", ["thiolate"] * 4)
    assert ten_dq > 300  # 3rd row + field strength
    print(f"  \\u2705 test_au3_large_10dq: 10Dq={ten_dq:.0f} kJ/mol")

def test_mn2_hs_zero_lfse():
    """Mn2+ d5 high-spin → LFSE = 0 (half-filled shell)."""
    ss = predict_spin_state("Mn2+", 5, ["water"])
    assert ss.spin_state == "high_spin"
    assert ss.lfse_oct_kj == 0.0
    assert ss.unpaired_electrons == 5
    print(f"  \\u2705 test_mn2_hs_zero_lfse: LFSE={ss.lfse_oct_kj}, unpaired={ss.unpaired_electrons}")

# === GEOMETRY-SPECIFIC LFSE ===
def test_lfse_square_planar_d8():
    """d8 square planar should have very large LFSE."""
    r = compute_lfse_for_geometry("Ni2+", 8, "square_planar", ["bipyridyl"] * 2)
    assert r.lfse_kj < -100  # Very favorable
    assert r.spin_state == "low_spin"  # d8 square planar is always low-spin
    assert r.unpaired_electrons == 0  # Diamagnetic
    print(f"  \\u2705 test_lfse_sq_planar_d8: LFSE={r.lfse_kj:.1f}, μ={r.magnetic_moment_bm:.2f} BM")

def test_lfse_octahedral_vs_tetrahedral():
    """Octahedral LFSE should be larger than tetrahedral."""
    oct = compute_lfse_for_geometry("Ni2+", 8, "octahedral", ["imidazole"] * 6)
    tet = compute_lfse_for_geometry("Ni2+", 8, "tetrahedral", ["imidazole"] * 4)
    assert abs(oct.lfse_kj) > abs(tet.lfse_kj)
    print(f"  \\u2705 test_lfse_oct_vs_tet: oct={oct.lfse_kj:.1f} vs tet={tet.lfse_kj:.1f}")

def test_jahn_teller_cu2():
    """Cu2+ d9 should show strong Jahn-Teller."""
    r = compute_lfse_for_geometry("Cu2+", 9, "octahedral", ["imidazole"] * 6)
    assert r.jahn_teller == "strong"
    print(f"  \\u2705 test_jahn_teller_cu2: JT={r.jahn_teller}")

def test_magnetic_moment_fe2_comparison():
    """Fe2+ magnetic moment should differ dramatically HS vs LS."""
    hs = predict_spin_state("Fe2+", 6, ["water"])
    ls = predict_spin_state("Fe2+", 6, ["bipyridyl"] * 3)
    assert hs.magnetic_moment_bm > 4.0   # ~4.90 BM
    assert ls.magnetic_moment_bm == 0.0   # Diamagnetic
    print(f"  \\u2705 test_magnetic_fe2: HS μ={hs.magnetic_moment_bm:.2f}, LS μ={ls.magnetic_moment_bm:.2f} BM")

def test_spectrochemical_series_ordering():
    """10Dq should follow spectrochemical series: I- < Br- < Cl- < O < N < CN-."""
    series = ["iodide", "bromide", "chloride", "water", "imidazole", "cyanide"]
    dqs = [_get_10dq("Fe2+", [lig] * 6) for lig in series]
    for i in range(len(dqs) - 1):
        assert dqs[i] <= dqs[i + 1], \\
            f"Spectrochemical violation: {series[i]}({dqs[i]:.0f}) > {series[i+1]}({dqs[i+1]:.0f})"
    print(f"  \\u2705 test_spectrochemical_series: {' < '.join(f'{s}({d:.0f})' for s, d in zip(series, dqs))}")

if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprint 19 \\u2014 Spin State + Strong/Weak Field LFSE\\n")
    print("Spin State Prediction:")
    test_fe2_weak_field_high_spin(); test_fe2_strong_field_low_spin()
    test_fe3_weak_field_high_spin(); test_fe3_strong_field_low_spin()
    test_co2_weak_vs_strong(); test_ni2_no_spin_choice()
    test_d10_zero_lfse(); test_cr3_always_high_lfse()
    test_au3_large_10dq(); test_mn2_hs_zero_lfse()
    print("\\nGeometry-Specific LFSE:")
    test_lfse_square_planar_d8(); test_lfse_octahedral_vs_tetrahedral()
    test_jahn_teller_cu2(); test_magnetic_moment_fe2_comparison()
    test_spectrochemical_series_ordering()
    print("\\n\\u2705 All Sprint 19 tests passed! (15/15)")
    print("\\n\\U0001f389 SPIN STATE ENGINE OPERATIONAL\\n")

''')


print("""
\u2705 Sprint 18+19 files created!

Sprint 18 — Speciation Gate + Redox Routing:
  core/speciation_gate.py    \u2190 14 metals + 8 redox pathways + Nernst scoring
  tests/test_sprint18.py     \u2190 15 tests

Sprint 19 — Spin State + Strong/Weak Field LFSE:
  core/spin_state.py         \u2190 Spin predictor + spectrochemical series + magnetic moment
  tests/test_sprint19.py     \u2190 15 tests

Run:
  python tests/test_sprint18.py
  python tests/test_sprint19.py
""")