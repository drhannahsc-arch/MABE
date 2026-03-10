"""
adapters/porphyrin_adapter.py — Planar Coordination Ring Adapter

Maps InteractionGeometrySpec to porphyrin, phthalocyanine, and corrole designs
for metal coordination pockets requiring planar N₄ geometry.

Physics basis:
    - Porphyrin core: 4 pyrrolic N in a square-planar arrangement
    - Core hole diameter: ~2.0 Å (porphyrin), ~1.9 Å (corrole), ~2.7 Å (expanded)
    - Metal selectivity from ionic radius → core hole size matching
    - meso/beta substituents tune electronics (electron-withdrawing = harder Lewis acid)
    - Axial ligand sites for 5th/6th coordination
    - Guest binding via: axial coordination, peripheral H-bonds, or face-to-face π-stacking

Also handles non-porphyrin planar macrocycles:
    - Phthalocyanines (larger cavity, 4 isoindole N)
    - Corroles (trianionic, 3 meso-C, smaller core)
    - Expanded porphyrins (hexaphyrins, octaphyrins — larger core holes)
    - Salen (N₂O₂ planar) — for non-symmetric pockets

Does NOT:
    - Run DFT on porphyrin structures
    - Use fitted parameters against porphyrin binding data
    - Design non-planar macrocycles (those go through crown ether adapter)
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# PORPHYRIN CORE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PlanarMacrocycle:
    """A planar macrocyclic core scaffold."""
    name: str
    core_type: str                # "porphyrin", "phthalocyanine", "corrole", "salen", "expanded"
    n_donors: int                 # number of coordinating atoms
    donor_atoms: tuple            # ("N","N","N","N") or ("N","N","O","O")
    core_hole_A: float            # diameter of central cavity
    charge_when_coordinated: int  # charge on deprotonated macrocycle
    symmetry: str                 # "D4h", "C2v", "D2h", etc.
    aromatic: bool
    smiles_core: str              # SMILES of the unsubstituted free base
    axial_sites: int              # number of axial coordination sites (0, 1, or 2)
    meso_positions: int           # substitutable meso positions
    beta_positions: int           # substitutable beta positions
    notes: str = ""


MACROCYCLE_LIBRARY = [
    PlanarMacrocycle(
        "porphyrin", "porphyrin", 4, ("N", "N", "N", "N"), 2.0, -2,
        "D4h", True,
        "c1cc2cc3ccc([nH]3)cc3ccc([nH]3)cc3ccc(n3)cc3ccc1[nH]3",
        2, 4, 8,
        "Standard porphyrin. 4 meso + 8 beta substitutions. Most versatile.",
    ),
    PlanarMacrocycle(
        "phthalocyanine", "phthalocyanine", 4, ("N", "N", "N", "N"), 2.7, -2,
        "D4h", True,
        "c1ccc2c(c1)[nH]c1nc3nc4nc5nc6ccc7ccccc7c6nc5ccc4c3c1c2",
        2, 0, 16,
        "Larger core hole. Better for larger metals (Pb²⁺, lanthanides). Very stable.",
    ),
    PlanarMacrocycle(
        "corrole", "corrole", 4, ("N", "N", "N", "N"), 1.9, -3,
        "C2v", True,
        "c1cc2ccc([nH]2)c2ccc([nH]2)c2cc3ccc1[nH]3",
        1, 3, 8,
        "Trianionic. Stabilizes high-valent metals (Mn(V), Fe(IV)). Smaller core.",
    ),
    PlanarMacrocycle(
        "salen", "salen", 4, ("N", "N", "O", "O"), 2.0, -2,
        "C2v", False,
        "Oc1ccc(/C=N/CC/N=C/c2ccc(O)cc2)cc1",
        2, 0, 4,
        "N₂O₂ donor. Asymmetric. Chiral variants (Jacobsen catalyst). Easy synthesis.",
    ),
    PlanarMacrocycle(
        "TAPH", "expanded", 6, ("N", "N", "N", "N", "N", "N"), 3.5, -2,
        "D6h", True,
        "",  # complex SMILES
        2, 6, 12,
        "Hexaphyrin. Very large core hole for actinides (U, Th). Rare.",
    ),
]

_MACROCYCLE_BY_NAME = {m.name: m for m in MACROCYCLE_LIBRARY}


# ═══════════════════════════════════════════════════════════════════════════
# SUBSTITUENT LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Substituent:
    """A functional group for porphyrin meso/beta positions."""
    name: str
    position_type: str            # "meso" or "beta"
    smiles_fragment: str
    electronic_effect: str        # "EDG" (electron-donating), "EWG" (electron-withdrawing), "neutral"
    provides_donor: str           # what this adds: "hb_donor", "hb_acceptor", "none", "conjugation_handle"
    solubility_effect: str        # "hydrophilic", "lipophilic", "neutral"
    steric_bulk: str              # "small", "medium", "large"
    notes: str = ""


SUBSTITUENT_LIBRARY = [
    # meso substituents
    Substituent("phenyl", "meso", "c1ccccc1", "neutral", "none", "lipophilic", "medium",
                "TPP (tetraphenylporphyrin). Most common."),
    Substituent("4-pyridyl", "meso", "c1ccncc1", "EWG", "hb_acceptor", "hydrophilic", "medium",
                "Axial coordination to metals. Water-soluble if charged."),
    Substituent("4-carboxyphenyl", "meso", "c1ccc(C(=O)O)cc1", "EWG", "hb_donor", "hydrophilic", "medium",
                "MOF linker attachment. Water-soluble. TCPP."),
    Substituent("4-sulfonatophenyl", "meso", "c1ccc(S(=O)(=O)[O-])cc1", "EWG", "none", "hydrophilic", "medium",
                "TPPS. Water-soluble. Anionic."),
    Substituent("pentafluorophenyl", "meso", "c1(F)c(F)c(F)c(F)c1F", "EWG", "none", "lipophilic", "medium",
                "TPPF₂₀. Strong EWG. Higher-valent metal stabilization."),
    Substituent("mesityl", "meso", "c1c(C)cc(C)cc1C", "EDG", "none", "lipophilic", "large",
                "Steric protection of core. Prevents mu-oxo dimerization."),
    Substituent("4-hydroxyphenyl", "meso", "c1ccc(O)cc1", "EDG", "hb_donor", "hydrophilic", "medium",
                "Catechol-like. H-bond donor."),
    Substituent("4-aminophenyl", "meso", "c1ccc(N)cc1", "EDG", "hb_donor", "hydrophilic", "medium",
                "Amine handle for conjugation."),
    Substituent("4-azidophenyl", "meso", "c1ccc(N=[N+]=[N-])cc1", "EWG", "conjugation_handle",
                "neutral", "medium", "Click chemistry handle. SPAAC-ready."),
    # beta substituents
    Substituent("Br", "beta", "Br", "EWG", "none", "neutral", "small",
                "Suzuki/Sonogashira coupling handle."),
    Substituent("NO2", "beta", "[N+](=O)[O-]", "EWG", "hb_acceptor", "neutral", "small",
                "Strong EWG. Reducible to NH₂."),
    Substituent("vinyl", "beta", "C=C", "neutral", "none", "neutral", "small",
                "Polymerizable. MIP cross-linking."),
]


# ═══════════════════════════════════════════════════════════════════════════
# METAL-MACROCYCLE COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════

# Ionic radii (Å) for square-planar or octahedral coordination
_METAL_RADII = {
    "Fe2+": 0.78, "Fe3+": 0.65, "Co2+": 0.74, "Co3+": 0.61,
    "Ni2+": 0.69, "Cu2+": 0.73, "Zn2+": 0.74, "Mn2+": 0.83,
    "Mn3+": 0.64, "Cr3+": 0.62, "Ru2+": 0.73, "Ru3+": 0.68,
    "Rh3+": 0.66, "Ir3+": 0.68, "Pd2+": 0.86, "Pt2+": 0.80,
    "Al3+": 0.53, "Ga3+": 0.62, "In3+": 0.80,
    "Pb2+": 1.19, "Cd2+": 0.95, "Hg2+": 1.02,
    "La3+": 1.03, "Gd3+": 0.94, "Lu3+": 0.86,
    "U4+": 0.89, "Th4+": 0.94,
    "Mg2+": 0.72, "Ca2+": 1.00,
}


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PorphyrinDesign:
    """A scored planar macrocycle design."""
    macrocycle: PlanarMacrocycle
    target_metal: str             # metal to bind in the core
    meso_substituents: list = field(default_factory=list)   # list[Substituent]
    beta_substituents: list = field(default_factory=list)
    axial_ligands: list = field(default_factory=list)       # list[str] for axial positions

    # Scores
    core_fit_score: float = 0.0   # metal radius vs core hole
    electronic_match: float = 0.0 # EWG/EDG for target oxidation state
    donor_match: float = 0.0      # peripheral donors for guest H-bonds
    composite_score: float = 0.0

    # Properties
    water_soluble: bool = False
    has_click_handle: bool = False
    estimated_log_Ka_metal: float = 0.0  # estimated from Irving-Williams + core fit

    # Synthesis
    synthesis_route: str = ""
    estimated_cost_usd: float = 0.0
    notes: str = ""


def design_porphyrin_for_spec(
    spec,
    target_metal: str = "",
    guest_smiles: str = "",
    require_water_soluble: bool = False,
    require_click_handle: bool = False,
    max_designs: int = 5,
) -> list:
    """Design planar macrocycle candidates for a pocket spec.

    Two modes:
    A) Metal-binding mode (target_metal set): select macrocycle that best
       coordinates the target metal. Substituents tune selectivity.
    B) Guest-binding mode (guest_smiles set, no metal): select macrocycle
       that uses metalloporphyrin axial coordination + peripheral H-bonds
       to bind a guest molecule (e.g., 6PPD-Q on Zn-porphyrin).

    Args:
        spec: InteractionGeometrySpec
        target_metal: metal to bind in core (e.g., "Pb2+")
        guest_smiles: guest to bind via axial/peripheral interactions
        require_water_soluble: filter for aqueous compatibility
        require_click_handle: must have azide/alkyne for deployment
        max_designs: max results

    Returns:
        list[PorphyrinDesign] sorted by composite score.
    """
    designs = []

    # Determine if this is metal-binding or guest-binding
    metal_mode = bool(target_metal)
    metal_radius = _METAL_RADII.get(target_metal, 0.75) if metal_mode else 0.75

    # Collect spec donor requirements
    spec_donor_roles = []
    for dp in spec.donor_positions:
        if dp.charge_state < 0:
            spec_donor_roles.append("acceptor")
        else:
            spec_donor_roles.append("donor")
    n_hb_donors_needed = spec_donor_roles.count("donor")
    n_hb_acceptors_needed = spec_donor_roles.count("acceptor")

    for macro in MACROCYCLE_LIBRARY:
        # Core fit: metal radius vs core hole
        if metal_mode:
            core_fit = _core_fit_score(metal_radius, macro.core_hole_A)
        else:
            core_fit = 0.7  # guest-binding mode uses Zn/Fe metalloporphyrin

        if core_fit < 0.1:
            continue  # terrible fit, skip

        # Select substituents
        meso_subs, beta_subs = _select_substituents(
            macro, spec, require_water_soluble, require_click_handle,
            n_hb_donors_needed, n_hb_acceptors_needed,
        )

        # Electronic match
        if metal_mode:
            e_match = _electronic_match(target_metal, meso_subs, beta_subs)
        else:
            e_match = 0.7

        # Donor match from periphery
        n_donors_provided = sum(
            1 for s in meso_subs + beta_subs
            if s.provides_donor in ("hb_donor", "hb_acceptor")
        )
        donor_match = min(1.0, n_donors_provided / max(
            n_hb_donors_needed + n_hb_acceptors_needed, 1
        ))

        # Water solubility check
        water_sol = any(s.solubility_effect == "hydrophilic" for s in meso_subs)
        if require_water_soluble and not water_sol:
            continue

        has_click = any(s.provides_donor == "conjugation_handle" for s in meso_subs + beta_subs)
        if require_click_handle and not has_click:
            continue

        # Axial ligands
        axial = []
        if macro.axial_sites > 0 and not metal_mode:
            axial = ["pyridine"]  # default axial for guest binding

        # Estimated metal log Ka (very rough Irving-Williams + core fit)
        if metal_mode:
            base_log_ka = _estimate_metal_log_ka(target_metal, macro)
        else:
            base_log_ka = 0.0

        composite = (
            core_fit * 4.0 +
            e_match * 2.0 +
            donor_match * 3.0 +
            (0.5 if water_sol else 0) +
            (0.5 if has_click else 0)
        )

        # Synthesis route
        if macro.core_type == "porphyrin":
            synth = "Lindsey condensation (pyrrole + aldehyde, BF₃·OEt₂, then DDQ oxidation)"
        elif macro.core_type == "phthalocyanine":
            synth = "Cyclotetramerization of phthalonitrile (metal salt, high-boiling solvent, 180°C)"
        elif macro.core_type == "corrole":
            synth = "Dipyrromethane + aldehyde condensation (Gross method)"
        elif macro.core_type == "salen":
            synth = "Schiff base condensation (diamine + 2× salicylaldehyde, EtOH, reflux)"
        else:
            synth = "Literature-specific synthesis"

        designs.append(PorphyrinDesign(
            macrocycle=macro,
            target_metal=target_metal or "Zn2+",
            meso_substituents=meso_subs,
            beta_substituents=beta_subs,
            axial_ligands=axial,
            core_fit_score=core_fit,
            electronic_match=e_match,
            donor_match=donor_match,
            composite_score=composite,
            water_soluble=water_sol,
            has_click_handle=has_click,
            estimated_log_Ka_metal=base_log_ka,
            synthesis_route=synth,
            estimated_cost_usd=100 if macro.core_type in ("porphyrin", "salen") else 300,
        ))

    designs.sort(key=lambda d: d.composite_score, reverse=True)
    return designs[:max_designs]


def _core_fit_score(metal_radius_A, core_hole_A):
    """Score metal-macrocycle fit from ionic radius vs core hole.

    Optimal: metal diameter ≈ core hole diameter.
    Too small: rattling, weak binding.
    Too large: out-of-plane distortion, strain.
    """
    metal_d = metal_radius_A * 2
    mismatch = abs(metal_d - core_hole_A)
    # Gaussian decay: σ = 0.3 Å
    return math.exp(-(mismatch ** 2) / (2 * 0.3 ** 2))


def _electronic_match(metal, meso_subs, beta_subs):
    """Score electronic tuning of macrocycle for target metal.

    High-valent metals (Fe³⁺, Mn³⁺) benefit from EWG.
    Low-valent metals (Cu²⁺, Zn²⁺) benefit from EDG or neutral.
    """
    n_ewg = sum(1 for s in meso_subs + beta_subs if s.electronic_effect == "EWG")
    n_edg = sum(1 for s in meso_subs + beta_subs if s.electronic_effect == "EDG")

    # Metals that prefer EWG stabilization
    high_valent = {"Fe3+", "Mn3+", "Mn5+", "Cr3+", "Co3+", "Ru3+"}
    if metal in high_valent:
        return min(1.0, 0.5 + n_ewg * 0.15)
    else:
        return min(1.0, 0.6 + n_edg * 0.1)


def _select_substituents(macro, spec, water_sol, click, n_donors, n_acceptors):
    """Select meso and beta substituents based on spec requirements."""
    meso_subs = []
    beta_subs = []

    # Priority 1: H-bond donors/acceptors for guest binding
    available_meso = macro.meso_positions
    for s in SUBSTITUENT_LIBRARY:
        if len(meso_subs) >= available_meso:
            break
        if s.position_type != "meso":
            continue
        if n_donors > 0 and s.provides_donor == "hb_donor":
            meso_subs.append(s)
            n_donors -= 1
        elif n_acceptors > 0 and s.provides_donor == "hb_acceptor":
            meso_subs.append(s)
            n_acceptors -= 1

    # Priority 2: water solubility
    if water_sol and not any(s.solubility_effect == "hydrophilic" for s in meso_subs):
        for s in SUBSTITUENT_LIBRARY:
            if s.position_type == "meso" and s.solubility_effect == "hydrophilic" \
               and len(meso_subs) < available_meso and s not in meso_subs:
                meso_subs.append(s)
                break

    # Priority 3: click handle
    if click and not any(s.provides_donor == "conjugation_handle" for s in meso_subs):
        for s in SUBSTITUENT_LIBRARY:
            if s.provides_donor == "conjugation_handle" and s.position_type == "meso" \
               and len(meso_subs) < available_meso and s not in meso_subs:
                meso_subs.append(s)
                break

    # Fill remaining meso with phenyl (default)
    phenyl = next((s for s in SUBSTITUENT_LIBRARY if s.name == "phenyl"), None)
    while len(meso_subs) < min(available_meso, 4) and phenyl:
        meso_subs.append(phenyl)

    return meso_subs, beta_subs


def _estimate_metal_log_ka(metal, macro):
    """Rough estimate of metal-macrocycle log Ka.

    Based on Irving-Williams order + core fit correction.
    Literature compilation: Smith & Martell, macrocyclic constants.
    """
    # Approximate stability order for porphyrins (log K₁)
    _PORPH_LOG_K = {
        "Mn2+": 7, "Fe2+": 9, "Fe3+": 12, "Co2+": 10, "Ni2+": 11,
        "Cu2+": 14, "Zn2+": 12, "Pd2+": 16, "Pt2+": 18,
        "Pb2+": 6, "Cd2+": 8, "Hg2+": 10, "Mg2+": 5, "Ca2+": 3,
        "Al3+": 10, "Ga3+": 12, "In3+": 10,
        "Ru2+": 14, "Ru3+": 16, "Rh3+": 15, "Ir3+": 17,
        "La3+": 4, "Gd3+": 5, "Lu3+": 7,
    }
    base = _PORPH_LOG_K.get(metal, 8)

    # Core fit adjustment
    metal_r = _METAL_RADII.get(metal, 0.75)
    fit = _core_fit_score(metal_r, macro.core_hole_A)
    adjusted = base * fit

    # Corrole: trianionic → ~2 log K units higher for +3 metals
    if macro.core_type == "corrole" and metal.endswith("3+"):
        adjusted += 2.0

    # Phthalocyanine: larger core → better for larger metals
    if macro.core_type == "phthalocyanine" and metal_r > 0.9:
        adjusted += 1.5

    return round(adjusted, 1)
