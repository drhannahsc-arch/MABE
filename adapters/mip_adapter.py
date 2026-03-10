"""
adapters/mip_adapter.py — Molecularly Imprinted Polymer adapter for MABE.

Translates an InteractionGeometrySpec (or direct guest SMILES) into:
    1. Ranked functional monomer selections (matched to pocket donors)
    2. Crosslinker recommendations
    3. Predicted imprinting factor (IF) range
    4. Synthesis protocol outline

Physics basis:
    - Monomer-template interaction: H-bond, π-π, charge-transfer
    - Each monomer's functional group is matched to the required
      pocket interaction elements from the InteractionGeometrySpec
    - Imprinting factor estimated from complementarity score

Does NOT:
    - Run DFT (that requires ORCA/Gaussian — flags as "DFT refinement recommended")
    - Use fitted parameters against MIP training data
    - Promise specific Ka values (MIP binding is inherently heterogeneous)
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONAL MONOMER LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FunctionalMonomer:
    """A polymerizable monomer for MIP synthesis."""
    name: str
    abbreviation: str
    smiles: str
    # What pocket interaction it provides:
    provides: str          # "hb_donor", "hb_acceptor", "pi_stack", "charge_transfer", "hydrophobic"
    provides_subtype: str  # "carboxylic_acid", "pyridine_N", "amine", etc.
    # Polymerization
    polymer_class: str     # "acrylic", "vinyl", "conducting", "thiol"
    # Cost / availability
    availability: str      # "commodity", "specialty", "custom"
    # Electrochemical capability
    electroactive: bool = False
    # Notes
    notes: str = ""


MONOMER_LIBRARY = [
    # ── H-bond donors (provide N-H or O-H to bind guest acceptors) ──
    FunctionalMonomer(
        "methacrylic acid", "MAA",
        "CC(=C)C(=O)O",
        "hb_donor", "carboxylic_acid", "acrylic", "commodity",
        notes="Most common MIP monomer. Strong H-bond donor via COOH.",
    ),
    FunctionalMonomer(
        "acrylamide", "AAm",
        "C=CC(=O)N",
        "hb_donor", "amide_NH", "acrylic", "commodity",
        notes="Moderate H-bond donor via NH₂. Water-compatible.",
    ),
    FunctionalMonomer(
        "allylamine", "AA",
        "C=CCN",
        "hb_donor", "amine", "acrylic", "commodity",
        notes="Strong H-bond donor. Protonation state depends on pH.",
    ),
    FunctionalMonomer(
        "itaconic acid", "IA",
        "C(=C)C(CC(=O)O)=O",
        "hb_donor", "dicarboxylic_acid", "acrylic", "commodity",
        notes="Bidentate H-bond donor. Two COOH groups.",
    ),
    FunctionalMonomer(
        "2-aminophenol", "2AP",
        "OC1=CC=CC=C1N",
        "hb_donor", "aminophenol", "vinyl", "commodity",
        notes="Dual H-bond donor (OH + NH₂). Also π-stacker.",
    ),
    FunctionalMonomer(
        "dopamine methacrylamide", "DMA",
        "CC(=C)C(=O)NCCC1=CC(O)=C(O)C=C1",
        "hb_donor", "catechol", "acrylic", "specialty",
        notes="Catechol H-bond donor. Strong with quinones.",
    ),

    # ── H-bond acceptors (provide lone pair to bind guest donors) ──
    FunctionalMonomer(
        "4-vinylpyridine", "4VP",
        "C=CC1=CC=NC=C1",
        "hb_acceptor", "pyridine_N", "vinyl", "commodity",
        notes="Pyridine N lone pair. Also π-stacker with electron-poor guests.",
    ),
    FunctionalMonomer(
        "2-vinylpyridine", "2VP",
        "C=CC1=CC=CC=N1",
        "hb_acceptor", "pyridine_N", "vinyl", "commodity",
        notes="Like 4VP but steric constraints differ.",
    ),
    FunctionalMonomer(
        "1-vinylimidazole", "VIm",
        "C=CN1C=CN=C1",
        "hb_acceptor", "imidazole_N", "vinyl", "commodity",
        notes="Imidazole N lone pair. Versatile H-bond acceptor.",
    ),

    # ── π-stacking / charge-transfer ──
    FunctionalMonomer(
        "EDOT", "EDOT",
        "C1=CSC2=C1OCCO2",
        "charge_transfer", "thiophene", "conducting", "specialty",
        electroactive=True,
        notes="3,4-ethylenedioxythiophene. Electron-rich → CT with quinones.",
    ),
    FunctionalMonomer(
        "pyrrole", "Py",
        "C1=CNC=C1",
        "charge_transfer", "pyrrole", "conducting", "commodity",
        electroactive=True,
        notes="Electron-rich. Electropolymerizable. CT with quinones.",
    ),
    FunctionalMonomer(
        "o-phenylenediamine", "oPD",
        "NC1=CC=CC=C1N",
        "charge_transfer", "diamine_aromatic", "conducting", "commodity",
        electroactive=True,
        notes="Electropolymerizable. Both π-stacker and H-bond donor.",
    ),
    FunctionalMonomer(
        "vinyl benzoic acid", "VBA",
        "C=CC1=CC=C(C(=O)O)C=C1",
        "pi_stack", "aromatic_acid", "vinyl", "commodity",
        notes="Aromatic π-stacker + COOH donor.",
    ),

    # ── Covalent capture (quinone-specific) ──
    FunctionalMonomer(
        "2-mercaptoethyl methacrylate", "MEMA",
        "CC(=C)C(=O)OCCS",
        "hb_donor", "thiol", "acrylic", "specialty",
        notes="Thiol-quinone Michael addition. Covalent capture of quinones.",
    ),
    FunctionalMonomer(
        "N-allylthiourea", "ATU",
        "C=CCNC(=S)N",
        "hb_donor", "thiourea", "acrylic", "specialty",
        notes="Thiourea H-bond donor. Also thiol-like reactivity.",
    ),

    # ── Hydrophobic ──
    FunctionalMonomer(
        "HEMA", "HEMA",
        "CC(=C)C(=O)OCCO",
        "hydrophobic", "hydroxyl_ester", "acrylic", "commodity",
        notes="2-hydroxyethyl methacrylate. Mild hydrophobic + OH.",
    ),
    FunctionalMonomer(
        "trifluoromethyl acrylamide", "TFMAA",
        "C=CC(=O)NC(F)(F)F",
        "hydrophobic", "fluorinated", "acrylic", "specialty",
        notes="Fluorophilic. Enhanced hydrophobic burial in aqueous.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# CROSSLINKER DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Crosslinker:
    name: str
    abbreviation: str
    smiles: str
    rigidity: str           # "rigid", "semi-rigid", "flexible"
    click_compatible: bool  # has azide/alkyne/DBCO for SPAAC
    notes: str = ""


CROSSLINKER_LIBRARY = [
    Crosslinker(
        "ethylene glycol dimethacrylate", "EGDMA",
        "CC(=C)C(=O)OCCOC(=O)C(=C)C",
        "semi-rigid", False,
        "Standard MIP crosslinker. ~20:1 crosslinker:monomer typical.",
    ),
    Crosslinker(
        "divinylbenzene", "DVB",
        "C=CC1=CC=C(C=C)C=C1",
        "rigid", False,
        "Rigid aromatic crosslinker. Good for small cavities.",
    ),
    Crosslinker(
        "trimethylolpropane trimethacrylate", "TRIM",
        "CC(COC(=O)C(=C)C)(COC(=O)C(=C)C)COC(=O)C(=C)C",
        "rigid", False,
        "Trifunctional. Higher crosslink density.",
    ),
    Crosslinker(
        "bis-azide-PEG-dimethacrylate", "BA-PEG-DMA",
        "CC(=C)C(=O)OCCOCCOCCOC(=O)C(=C)C",
        "flexible", True,
        "Click-compatible via azide termini. For SPAAC deployment.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# MONOMER SELECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MonomerMatch:
    """One monomer matched to a pocket interaction requirement."""
    monomer: FunctionalMonomer
    matched_feature: str      # what guest feature it complements
    match_quality: str        # "exact", "good", "partial"
    score: float              # 0-1 match score
    rationale: str = ""


@dataclass
class MIPDesign:
    """Complete MIP formulation design."""
    guest_name: str
    guest_smiles: str

    # Ranked monomer selections
    primary_monomers: list = field(default_factory=list)    # list[MonomerMatch]
    secondary_monomers: list = field(default_factory=list)  # complementary monomers

    # Formulation
    recommended_crosslinker: str = "EGDMA"
    monomer_template_ratio: str = "4:1"
    crosslinker_monomer_ratio: str = "20:1"
    porogen: str = "acetonitrile"
    initiator: str = "AIBN"
    polymerization: str = "thermal"    # "thermal", "UV", "electrochemical"

    # Predictions
    predicted_if_range: tuple = (1.0, 1.0)  # imprinting factor range
    electrochemical_sensor: bool = False
    click_deployable: bool = False

    # DFT refinement flag
    dft_recommended: bool = True
    dft_note: str = "DFT monomer-template ΔE would refine rankings"

    # Synthesis protocol summary
    synthesis_steps: list = field(default_factory=list)


def select_monomers_for_guest(
    guest_smiles: str,
    guest_name: str = "",
    pharmacophore=None,
    n_primary: int = 3,
    prefer_electroactive: bool = False,
    require_click: bool = False,
) -> MIPDesign:
    """Select optimal MIP monomers for a guest molecule.

    Args:
        guest_smiles: Guest SMILES
        guest_name: Display name
        pharmacophore: Optional pre-computed GuestPharmacophore
        n_primary: Number of top monomers to recommend
        prefer_electroactive: Prioritize electrochemical MIP (sensor mode)
        require_click: Require click-chemistry deployability

    Returns:
        MIPDesign with ranked monomers and formulation.
    """
    # Get pharmacophore if not provided
    if pharmacophore is None:
        from core.small_molecule_target import analyze_guest
        pharmacophore = analyze_guest(guest_smiles, name=guest_name)

    design = MIPDesign(
        guest_name=guest_name or guest_smiles,
        guest_smiles=guest_smiles,
    )

    # ── Match monomers to guest features ──
    matches = []
    for monomer in MONOMER_LIBRARY:
        score, quality, rationale, matched_feat = _score_monomer_guest(
            monomer, pharmacophore
        )
        if score > 0:
            matches.append(MonomerMatch(
                monomer=monomer,
                matched_feature=matched_feat,
                match_quality=quality,
                score=score,
                rationale=rationale,
            ))

    # Apply preference weighting
    if prefer_electroactive:
        for m in matches:
            if m.monomer.electroactive:
                m.score *= 1.5

    # Sort by score
    matches.sort(key=lambda m: m.score, reverse=True)

    # Assign primary (top N diverse) and secondary
    design.primary_monomers = _select_diverse(matches, n_primary)
    remaining = [m for m in matches if m not in design.primary_monomers]
    design.secondary_monomers = remaining[:3]

    # ── Crosslinker selection ──
    if require_click:
        design.recommended_crosslinker = "BA-PEG-DMA"
        design.click_deployable = True
    elif pharmacophore.volume_A3 < 200:
        design.recommended_crosslinker = "DVB"  # rigid for small cavities
    else:
        design.recommended_crosslinker = "EGDMA"  # standard

    # ── Electrochemical assessment ──
    electroactive_primary = any(
        m.monomer.electroactive for m in design.primary_monomers
    )
    if electroactive_primary:
        design.electrochemical_sensor = True
        design.polymerization = "electrochemical"

    # ── Imprinting factor estimate ──
    # Based on total complementarity score
    total_score = sum(m.score for m in design.primary_monomers)
    if total_score > 2.5:
        design.predicted_if_range = (3.0, 8.0)
    elif total_score > 1.5:
        design.predicted_if_range = (2.0, 5.0)
    elif total_score > 0.5:
        design.predicted_if_range = (1.5, 3.0)
    else:
        design.predicted_if_range = (1.0, 2.0)

    # ── Porogen ──
    if pharmacophore.logP > 3.0:
        design.porogen = "toluene"
    else:
        design.porogen = "acetonitrile"

    # ── Synthesis protocol ──
    design.synthesis_steps = _generate_synthesis_protocol(design, pharmacophore)

    return design


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL: SCORING
# ═══════════════════════════════════════════════════════════════════════════

def _score_monomer_guest(monomer, pharmacophore):
    """Score how well a monomer complements the guest's pharmacophore.

    Returns (score, quality, rationale, matched_feature).
    """
    score = 0.0
    quality = "partial"
    rationale = ""
    matched = ""

    # Check if monomer's provided interaction matches a guest need
    # Guest acceptors need pocket donors → monomer hb_donor
    # Guest donors need pocket acceptors → monomer hb_acceptor
    # Guest aromatics need π-stacker → monomer pi_stack / charge_transfer
    # Guest hydrophobics need pocket hydrophobic → monomer hydrophobic

    provides = monomer.provides

    if provides == "hb_donor":
        # This monomer donates H-bonds → matches guest H-bond acceptors
        n_acc = pharmacophore.n_hb_acceptors
        if n_acc > 0:
            score = 0.8
            quality = "good"
            rationale = f"{monomer.abbreviation} COOH/NH donates H to {n_acc} guest acceptor(s)"
            matched = "guest_hb_acceptor"

            # Quinone-specific boost for thiol monomers (Michael addition)
            if monomer.provides_subtype == "thiol":
                has_quinone = any(
                    f.subtype == "carbonyl_O" for f in pharmacophore.features
                    if f.feature_type == "hb_acceptor"
                )
                if has_quinone:
                    score = 1.0
                    quality = "exact"
                    rationale = f"{monomer.abbreviation} thiol → Michael addition to quinone C=O"
                    matched = "quinone_covalent"

            # Catechol monomers boost with quinone guests
            if monomer.provides_subtype == "catechol":
                has_quinone = any(
                    f.subtype == "carbonyl_O" for f in pharmacophore.features
                    if f.feature_type == "hb_acceptor"
                )
                if has_quinone:
                    score = 0.95
                    quality = "exact"
                    rationale = f"{monomer.abbreviation} catechol H-bonds to quinone carbonyls"

    elif provides == "hb_acceptor":
        n_don = pharmacophore.n_hb_donors
        if n_don > 0:
            score = 0.75
            quality = "good"
            rationale = f"{monomer.abbreviation} lone pair accepts H from {n_don} guest donor(s)"
            matched = "guest_hb_donor"

    elif provides in ("pi_stack", "charge_transfer"):
        n_arom = pharmacophore.n_aromatic_rings
        if n_arom > 0:
            score = 0.7
            quality = "good"
            rationale = f"{monomer.abbreviation} π-interaction with {n_arom} guest aromatic(s)"
            matched = "guest_aromatic"

            # Charge-transfer boost for electron-poor guests (quinones)
            if provides == "charge_transfer":
                has_quinone_acc = any(
                    f.subtype == "carbonyl_O" for f in pharmacophore.features
                    if f.feature_type == "hb_acceptor"
                )
                if has_quinone_acc:
                    score = 0.9
                    quality = "exact"
                    rationale = f"{monomer.abbreviation} charge-transfer with electron-poor quinone"
                    matched = "quinone_CT"

    elif provides == "hydrophobic":
        if pharmacophore.logP > 2.0 or pharmacophore.n_hydrophobic_centers > 0:
            score = 0.5
            quality = "partial"
            rationale = f"{monomer.abbreviation} hydrophobic contact with nonpolar guest region"
            matched = "guest_hydrophobic"

    return score, quality, rationale, matched


def _select_diverse(matches, n):
    """Select top N monomers with diversity (different provides types)."""
    selected = []
    seen_types = set()
    # First pass: one per type
    for m in matches:
        if m.monomer.provides not in seen_types and len(selected) < n:
            selected.append(m)
            seen_types.add(m.monomer.provides)
    # Fill remaining from best scores
    for m in matches:
        if m not in selected and len(selected) < n:
            selected.append(m)
    return selected


def _generate_synthesis_protocol(design, pharmacophore):
    """Generate a basic MIP synthesis protocol."""
    steps = []

    template_mg = max(10, round(pharmacophore.mw * 0.1))
    steps.append(
        f"1. Dissolve {template_mg} mg template ({design.guest_name}) "
        f"in 5 mL {design.porogen}"
    )

    monomer_names = ", ".join(m.monomer.abbreviation for m in design.primary_monomers)
    steps.append(
        f"2. Add functional monomers ({monomer_names}) at "
        f"{design.monomer_template_ratio} monomer:template molar ratio"
    )

    steps.append(
        f"3. Pre-polymerization complexation: stir 2 h at RT to allow "
        f"monomer-template self-assembly"
    )

    steps.append(
        f"4. Add crosslinker ({design.recommended_crosslinker}) at "
        f"{design.crosslinker_monomer_ratio} crosslinker:monomer ratio"
    )

    if design.polymerization == "thermal":
        steps.append(
            f"5. Add initiator ({design.initiator}, 1 mol% of monomers). "
            f"Purge N₂. Heat 60°C, 24 h"
        )
    elif design.polymerization == "UV":
        steps.append(
            f"5. Add photoinitiator (Irgacure 184, 1 mol%). "
            f"UV irradiate 365 nm, 4 h at RT"
        )
    elif design.polymerization == "electrochemical":
        steps.append(
            f"5. Electropolymerize on electrode: cyclic voltammetry "
            f"0 → +1.0V → 0V, 20 cycles, 50 mV/s"
        )

    steps.append(
        "6. Template removal: Soxhlet extraction with methanol/acetic acid "
        "(9:1) for 48 h, or sonicate in methanol 3×30 min"
    )

    steps.append(
        "7. Verify removal: UV-vis of wash solvent until template peak absent"
    )

    if design.click_deployable:
        steps.append(
            "8. Click-handle: surface azide groups from crosslinker available "
            "for SPAAC conjugation to DBCO-functionalized beads/surfaces"
        )

    return steps
