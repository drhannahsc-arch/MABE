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

