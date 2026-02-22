"""
core/thermodynamics.py - Energy landscape calculations for molecular binding.

Everything here is in kJ/mol. Everything is semi-empirical: grounded in real
physical scales from coordination chemistry, not arbitrary 0-1 scores.

The fundamental equation:
    ΔG_net = ΔG_bind + ΔG_desolv + ΔG_preorg + ΔG_chelate + ΔG_electrostatic

Where:
    ΔG_bind    = intrinsic donor-atom binding energy (negative = favorable)
    ΔG_desolv  = desolvation penalty (positive = costs energy to strip water)
    ΔG_preorg  = preorganization bonus (negative = structure pays entropy cost)
    ΔG_chelate = chelate effect bonus (negative = multidentate advantage)
    ΔG_electrostatic = long-range charge-charge attraction/repulsion

Selectivity:
    ΔΔG = ΔG_net(target) - ΔG_net(competitor)
    Negative ΔΔG = target preferred. Magnitude determines selectivity factor.

Probability:
    K_eq = exp(-ΔG_net / RT)
    At 298K, RT = 2.479 kJ/mol
    ΔG = -10 kJ/mol → K_eq ~ 57 (good)
    ΔG = -20 kJ/mol → K_eq ~ 3300 (strong)
    ΔG = -40 kJ/mol → K_eq ~ 10^7 (very strong, may be hard to elute)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.problem import Problem, TargetSpecies, CompetingSpecies
from core.assembly import RecognitionChemistry, StructuralConstraint, InteriorDesign


R_GAS = 8.314e-3   # kJ/(mol·K)


@dataclass
class BindingThermodynamics:
    """Complete thermodynamic profile for a binding interaction."""
    # Individual energy contributions (kJ/mol, negative = favorable)
    dG_bind: float = 0.0          # intrinsic donor-atom binding
    dG_desolv: float = 0.0        # desolvation penalty (positive)
    dG_preorg: float = 0.0        # preorganization bonus (negative)
    dG_chelate: float = 0.0       # chelate effect (negative)
    dG_electrostatic: float = 0.0 # charge-charge interaction

    # Net
    dG_net: float = 0.0           # sum of all contributions

    # Selectivity
    ddG_vs_top_competitor: float = 0.0      # ΔΔG (negative = target preferred)
    top_competitor_identity: str = ""
    selectivity_factor: float = 1.0          # exp(-ΔΔG/RT)

    # Derived
    K_eq: float = 1.0                        # equilibrium constant
    predicted_kd_um: Optional[float] = None  # predicted dissociation constant
    temperature_k: float = 298.15

    # Breakdown for transparency
    energy_breakdown: list[str] = field(default_factory=list)

    @property
    def dG_net_computed(self) -> float:
        return self.dG_bind + self.dG_desolv + self.dG_preorg + self.dG_chelate + self.dG_electrostatic

    def probability_of_binding(self) -> float:
        """Boltzmann probability: fraction bound at equilibrium (1 mM reference)."""
        RT = R_GAS * self.temperature_k
        if RT == 0:
            return 0.0
        K = math.exp(-self.dG_net / RT) if abs(self.dG_net) < 200 else (1e30 if self.dG_net < 0 else 0.0)
        # At 1 mM reference concentration, fraction bound = K / (K + 1000)
        # (since Kd in uM and 1 mM = 1000 uM)
        kd_um = 1e6 / K if K > 0 else 1e12
        return K / (K + 1000.0)

    def summary(self) -> str:
        parts = [
            f"ΔG_net = {self.dG_net:+.1f} kJ/mol",
            f"  bind: {self.dG_bind:+.1f}  desolv: {self.dG_desolv:+.1f}  "
            f"preorg: {self.dG_preorg:+.1f}  chelate: {self.dG_chelate:+.1f}  "
            f"elec: {self.dG_electrostatic:+.1f}",
            f"K_eq = {self.K_eq:.1e}  |  Kd ~ {self.predicted_kd_um:.1f} µM" if self.predicted_kd_um else f"K_eq = {self.K_eq:.1e}",
        ]
        if self.top_competitor_identity:
            parts.append(
                f"ΔΔG vs {self.top_competitor_identity}: {self.ddG_vs_top_competitor:+.1f} kJ/mol "
                f"(selectivity: {self.selectivity_factor:.0f}x)"
            )
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# DONOR ATOM BINDING ENERGIES
# Semi-empirical: from coordination chemistry literature.
# Values are ΔG contribution per donor atom for metal-donor interaction.
# Scaled by HSAB match quality.
# ═══════════════════════════════════════════════════════════════════════════

# Base donor-metal binding energy (kJ/mol, negative = favorable)
# These are approximate ΔG contributions per donor for a typical divalent cation
DONOR_ENERGIES = {
    "S": -25.0,   # thiolate/thioether — strong for soft metals
    "N": -18.0,   # amine/imine/imidazole — moderate, versatile
    "O": -12.0,   # carboxylate/hydroxyl — ubiquitous, weaker per atom
    "P": -15.0,   # phosphonate — moderate
    "electrostatic": -8.0,  # pure charge interaction (zeolite, LDH)
}

# HSAB match multiplier: how well donor type matches metal character
# Correct match amplifies binding; mismatch penalizes
HSAB_MATCH = {
    # (donor_type, metal_hsab) → multiplier
    ("soft", "soft"): 1.8,       # thiol + Au/Hg/Ag — excellent
    ("soft", "borderline"): 1.2, # thiol + Pb/Cu/Ni — decent
    ("soft", "hard"): 0.5,       # thiol + Ca/Fe3+ — poor
    ("hard", "hard"): 1.5,       # carboxylate + Fe3+/Ca — good
    ("hard", "borderline"): 1.1, # carboxylate + Pb — ok
    ("hard", "soft"): 0.4,       # carboxylate + Au — poor
    ("borderline", "borderline"): 1.4,  # amine + Pb/Cu — good
    ("borderline", "soft"): 1.1, # amine + Au — ok
    ("borderline", "hard"): 1.0, # amine + Fe3+ — ok
}

# Electronegativity-based refinement: higher eneg metals bind harder
# ΔG_bind scales as ~(eneg_metal * eneg_donor)^0.5
# This captures the Irving-Williams series trend


def estimate_binding_energy(recognition: RecognitionChemistry,
                             target: TargetSpecies) -> tuple[float, list[str]]:
    """
    Estimate ΔG_bind from donor atoms, HSAB match, and electronegativity.

    Returns (dG_bind in kJ/mol, list of reasoning strings).
    Negative = favorable.
    """
    breakdown = []
    dG = 0.0
    metal_hsab = target.electronic.hardness_softness or "borderline"
    metal_eneg = target.electronic.electronegativity or 1.8

    donor_type = recognition.donor_type or "borderline"
    donors = recognition.donor_atoms or ["O", "N"]

    for donor in donors:
        base = DONOR_ENERGIES.get(donor, -10.0)

        # HSAB match
        match_key = (donor_type, metal_hsab)
        hsab_mult = HSAB_MATCH.get(match_key, 1.0)

        # Electronegativity scaling: stronger bonds with higher eneg metals
        eneg_scale = (metal_eneg / 2.0) ** 0.5  # normalized to ~1.0 for typical divalent

        # Charge scaling: higher charge = stronger electrostatic component
        charge = abs(target.charge) if target.charge else 2.0
        charge_scale = (charge / 2.0) ** 0.4

        atom_dG = base * hsab_mult * eneg_scale * charge_scale
        dG += atom_dG
        breakdown.append(
            f"{donor} donor: {base:.1f} × HSAB({donor_type}/{metal_hsab})={hsab_mult:.1f} "
            f"× eneg={eneg_scale:.2f} × charge={charge_scale:.2f} → {atom_dG:.1f} kJ/mol"
        )

    breakdown.append(f"Total ΔG_bind = {dG:.1f} kJ/mol ({len(donors)} donors)")
    return dG, breakdown


def estimate_desolvation_penalty(target: TargetSpecies,
                                  n_donors: int) -> tuple[float, list[str]]:
    """
    Estimate the energy cost of stripping water molecules from the target
    to make room for binder donor atoms.

    Each donor atom displaces one water from the coordination shell.
    Cost is proportional to the total dehydration energy divided by
    coordination number (energy per water molecule removed).

    Returns (dG_desolv in kJ/mol, list of reasoning strings). Positive.
    """
    breakdown = []

    dehydration_total = target.hydration.dehydration_energy_kj_mol
    cn = target.hydration.coordination_number_water

    if dehydration_total is None or cn is None or cn == 0:
        # Estimate from charge: higher charge = more tightly held water
        charge = abs(target.charge) if target.charge else 2.0
        r_ion = target.size.ionic_radius_angstrom if target.size and target.size.ionic_radius_angstrom else 1.0
        # Born model estimate: ΔG_solv ∝ z²/r
        dehydration_total = 300.0 * (charge ** 2) / r_ion
        cn = int(4 + charge * 2)  # rough estimate
        breakdown.append(f"Estimated dehydration energy from Born model: {dehydration_total:.0f} kJ/mol, CN ~ {cn}")

    # Energy per water molecule in shell
    energy_per_water = dehydration_total / cn

    # The key physics: when a donor atom replaces a coordinated water,
    # the cost is NOT the full water removal energy. It is the DIFFERENCE
    # between the water-metal interaction and the donor-metal interaction.
    # If the donor binds MORE strongly than water, the net penalty is small.
    # We model the residual penalty as ~8% of per-water energy per donor.
    # Calibrated against EDTA-Pb (log K~18, dG~-103 kJ/mol).
    fraction = 0.08
    dG_desolv = energy_per_water * n_donors * fraction

    breakdown.append(
        f"Dehydration: {dehydration_total:.0f} kJ/mol total / {cn} waters = "
        f"{energy_per_water:.1f} kJ/mol per water"
    )
    breakdown.append(
        f"{n_donors} donors displace {n_donors} waters × {fraction:.0%} penalty = "
        f"+{dG_desolv:.1f} kJ/mol"
    )

    return dG_desolv, breakdown


def estimate_chelate_effect(n_donors: int, is_macrocyclic: bool = False) -> tuple[float, list[str]]:
    """
    The chelate effect: multidentate ligands bind more strongly than
    equivalent monodentate ligands because of entropy.

    Each additional donor beyond the first adds ~5-8 kJ/mol stability
    (entropy of displacement: releasing n-1 waters costs less entropy
    than n separate binding events).

    Macrocyclic effect adds another ~10-15 kJ/mol (preorganized cavity).

    Returns (dG_chelate in kJ/mol, list of reasoning). Negative = favorable.
    """
    breakdown = []

    if n_donors <= 1:
        breakdown.append("Monodentate — no chelate effect")
        return 0.0, breakdown

    # ~6 kJ/mol per additional ring closure (empirical, well-established)
    dG = -6.0 * (n_donors - 1)
    breakdown.append(
        f"Chelate effect: {n_donors}-dentate → {n_donors - 1} ring closures × -6 kJ/mol = {dG:.1f} kJ/mol"
    )

    if is_macrocyclic:
        macro_bonus = -12.0
        dG += macro_bonus
        breakdown.append(f"Macrocyclic effect: {macro_bonus:.1f} kJ/mol (preorganized cavity)")

    return dG, breakdown


def estimate_preorganization(structure: StructuralConstraint,
                               interior: InteriorDesign) -> tuple[float, list[str]]:
    """
    Preorganization bonus: if the structure holds donors in the correct
    geometry BEFORE the target arrives, binding is more favorable because
    the entropy cost of organizing the donors is already paid.

    DNA origami: excellent preorganization (rigid scaffold)
    MOF: good (crystalline framework)
    Zeolite: excellent (rigid framework)
    MIP: excellent (templated cavity)
    Free in solution: no preorganization
    Dendrimer: poor (flexible)

    Returns (dG_preorg in kJ/mol, list of reasoning). Negative = favorable.
    """
    breakdown = []

    preorg_map = {
        "none": 0.0,
        "dna_origami_cage": -8.0,   # rigid scaffold, addressable
        "mof": -6.0,               # crystalline, rigid pores
        "zeolite": -10.0,          # angstrom-precision framework
        "mip": -12.0,              # templated to EXACT target — best preorg
        "protein_cage": -5.0,      # somewhat flexible
        "cof": -7.0,               # crystalline, rigid
        "mesoporous_silica": -4.0, # channels, but silane linkers flexible
        "silica_np": -2.0,         # surface only, flexible linkers
        "ldh": -8.0,               # layered, well-defined gallery
        "coordination_cage": -9.0, # atomically precise
        "dendrimer": -2.0,         # flexible branches
        "carbon_nanotube": -3.0,   # rigid tube, but functionalization flexible
        "graphene_oxide": -2.0,    # sheet, flexible groups
    }

    dG = preorg_map.get(structure.type, 0.0)

    # Bonus for tertiary design: multiple recognition types positioned together
    if interior.design_level == "tertiary" and interior.unique_recognition_types >= 2:
        tertiary_bonus = -3.0
        dG += tertiary_bonus
        breakdown.append(f"Tertiary pocket bonus: {tertiary_bonus:.1f} kJ/mol (mixed donors prepositioned)")

    breakdown.append(
        f"Preorganization ({structure.type}): {dG:.1f} kJ/mol"
    )

    return dG, breakdown


def estimate_electrostatic(target: TargetSpecies,
                            recognition: RecognitionChemistry,
                            structure: StructuralConstraint) -> tuple[float, list[str]]:
    """
    Long-range electrostatic attraction/repulsion.

    Charged structures (zeolite framework, LDH layers) provide electrostatic
    pre-concentration of oppositely charged targets.

    Returns (dG_elec in kJ/mol, list of reasoning).
    """
    breakdown = []
    charge = target.charge if target.charge else 0.0

    if structure.type == "zeolite" and charge > 0:
        # Framework negative charge attracts cations
        dG = -4.0 * abs(charge)
        breakdown.append(f"Zeolite framework attracts {target.formula}: {dG:.1f} kJ/mol")
        return dG, breakdown

    if structure.type == "ldh" and charge < 0:
        # Positive layers attract anions
        dG = -4.0 * abs(charge)
        breakdown.append(f"LDH layers attract {target.formula}: {dG:.1f} kJ/mol")
        return dG, breakdown

    # Charged donors can have weak electrostatic pre-attraction
    if charge != 0 and "O" in recognition.donor_atoms:
        dG = -1.5 * abs(charge)
        breakdown.append(f"Donor-charge electrostatic: {dG:.1f} kJ/mol")
        return dG, breakdown

    breakdown.append("No significant long-range electrostatic contribution")
    return 0.0, breakdown


def estimate_competitor_binding(recognition: RecognitionChemistry,
                                 competitor_identity: str,
                                 competitor_charge: float,
                                 metal_hsab: str) -> float:
    """
    Quick estimate of ΔG for a competitor binding to the same recognition.
    Used to compute ΔΔG selectivity.
    """
    donor_type = recognition.donor_type or "borderline"
    donors = recognition.donor_atoms or ["O", "N"]

    # Competitors: estimate HSAB class from common knowledge
    competitor_hsab_guess = {
        "calcium": "hard", "magnesium": "hard", "sodium": "hard",
        "potassium": "hard", "iron": "hard", "aluminum": "hard",
        "manganese": "borderline", "zinc": "borderline",
        "copper": "borderline", "nickel": "borderline",
        "cobalt": "borderline", "cadmium": "soft",
        "sulfate": "hard", "chloride": "borderline",
        "phosphate": "hard", "arsenate": "hard",
    }
    comp_hsab = competitor_hsab_guess.get(competitor_identity.lower(), "borderline")

    dG = 0.0
    for donor in donors:
        base = DONOR_ENERGIES.get(donor, -10.0)
        match_key = (donor_type, comp_hsab)
        hsab_mult = HSAB_MATCH.get(match_key, 1.0)
        charge_scale = (abs(competitor_charge) / 2.0) ** 0.4
        dG += base * hsab_mult * charge_scale

    return dG


def compute_thermodynamics(recognition: RecognitionChemistry,
                            structure: StructuralConstraint,
                            interior: InteriorDesign,
                            problem: Problem) -> BindingThermodynamics:
    """
    Full thermodynamic calculation for a binder-target interaction.
    """
    target = problem.target
    temp_k = (problem.matrix.temperature_c or 25.0) + 273.15
    RT = R_GAS * temp_k

    all_breakdown = []

    # 1. Binding energy
    dG_bind, bd = estimate_binding_energy(recognition, target)
    all_breakdown.extend(bd)

    # 2. Desolvation penalty
    n_donors = len(recognition.donor_atoms)
    dG_desolv, bd = estimate_desolvation_penalty(target, n_donors)
    all_breakdown.extend(bd)

    # 3. Chelate effect
    is_macro = "macrocycl" in recognition.structure.lower() if recognition.structure else False
    dG_chelate, bd = estimate_chelate_effect(n_donors, is_macro)
    all_breakdown.extend(bd)

    # 4. Preorganization
    dG_preorg, bd = estimate_preorganization(structure, interior)
    all_breakdown.extend(bd)

    # 5. Electrostatic
    dG_elec, bd = estimate_electrostatic(target, recognition, structure)
    all_breakdown.extend(bd)

    # Net
    dG_net = dG_bind + dG_desolv + dG_preorg + dG_chelate + dG_elec

    # Equilibrium constant
    if abs(dG_net / RT) < 500:
        K_eq = math.exp(-dG_net / RT)
    else:
        K_eq = 1e30 if dG_net < 0 else 0.0

    # Predicted Kd
    kd_um = 1e6 / K_eq if K_eq > 1e-6 else 1e12

    # Avidity correction from interior design
    if interior.avidity_factor > 1.0:
        kd_um /= interior.avidity_factor
        K_eq *= interior.avidity_factor
        dG_avidity = -RT * math.log(interior.avidity_factor)
        dG_net += dG_avidity
        all_breakdown.append(
            f"Avidity ({interior.avidity_factor:.0f}x): {dG_avidity:.1f} kJ/mol effective"
        )

    # Selectivity vs competitors
    best_ddG = 0.0
    worst_competitor = ""
    for comp in problem.matrix.competing_species:
        comp_dG_bind = estimate_competitor_binding(
            recognition, comp.identity, comp.charge,
            target.electronic.hardness_softness or "borderline"
        )
        # Competitor gets same desolv penalty roughly (simplified)
        comp_dG_net = comp_dG_bind + dG_desolv * 0.8  # competitors have similar desolv
        ddG = dG_net - comp_dG_net  # negative = target preferred
        if ddG > best_ddG or worst_competitor == "":
            best_ddG = ddG
            worst_competitor = comp.identity

    selectivity_factor = math.exp(-best_ddG / RT) if abs(best_ddG / RT) < 500 else 1e30

    all_breakdown.append(
        f"ΔΔG vs {worst_competitor}: {best_ddG:+.1f} kJ/mol → selectivity {selectivity_factor:.0f}x"
    )

    return BindingThermodynamics(
        dG_bind=round(dG_bind, 2),
        dG_desolv=round(dG_desolv, 2),
        dG_preorg=round(dG_preorg, 2),
        dG_chelate=round(dG_chelate, 2),
        dG_electrostatic=round(dG_elec, 2),
        dG_net=round(dG_net, 2),
        ddG_vs_top_competitor=round(best_ddG, 2),
        top_competitor_identity=worst_competitor,
        selectivity_factor=round(selectivity_factor, 1),
        K_eq=K_eq,
        predicted_kd_um=round(kd_um, 3) if kd_um < 1e9 else None,
        temperature_k=temp_k,
        energy_breakdown=all_breakdown,
    )
