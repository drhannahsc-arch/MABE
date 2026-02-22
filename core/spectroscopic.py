"""
core/spectroscopic.py — Sprint 26: Spectroscopic Prediction + Photoresponsive Design

Predicts how to DETECT binding (absorption shift, fluorescence quench,
color change) and enables light-triggered release mechanisms.

Physics:
  d-d transitions: ΔE = 10Dq (from spin_state module)
  Charge transfer: LMCT/MLCT energies from donor/acceptor orbital energies
  Beer-Lambert: A = εcl
  Photoswitch: azobenzene trans→cis isomerization energy
"""
from dataclasses import dataclass, field
import math


@dataclass
class SpectroscopicPrediction:
    """Predicted spectroscopic properties of a metal-binder complex."""
    # d-d transitions
    dd_transition_nm: float         # Wavelength of d-d band
    dd_transition_ev: float         # Energy in eV
    dd_extinction: float            # ε in M⁻¹cm⁻¹ (d-d are weak)
    predicted_color: str            # Perceived color of complex
    # Charge transfer
    ct_transition_nm: float         # LMCT or MLCT band
    ct_type: str                    # "LMCT", "MLCT", "none"
    ct_extinction: float            # ε (CT bands are intense)
    # Detection strategy
    detection_method: str           # "colorimetric", "fluorescence", "SPR", "UV-Vis"
    detection_signal: str           # What changes upon binding
    sensitivity_estimate: str       # "nM", "µM", "mM"
    # Photoresponsive
    photoresponsive: bool
    photoswitch_type: str           # "azobenzene", "diarylethene", "spiropyran", "none"
    trigger_wavelength_nm: float    # Wavelength for photoswitching
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# WAVELENGTH ↔ COLOR
# ═══════════════════════════════════════════════════════════════════════════

def _wavelength_to_color(nm):
    """Convert absorption wavelength to perceived color (complementary)."""
    # Absorbed → perceived complementary
    if nm < 380: return "colorless"
    elif nm < 430: return "yellow"       # Absorbs violet → yellow
    elif nm < 490: return "orange"       # Absorbs blue → orange
    elif nm < 570: return "red"          # Absorbs green → red
    elif nm < 590: return "purple"       # Absorbs yellow → purple
    elif nm < 610: return "blue"         # Absorbs orange → blue
    elif nm < 780: return "green"        # Absorbs red → green/blue-green
    elif nm < 900: return "blue"         # Absorbs deep red/NIR tail → blue tinge
    else: return "colorless"


def _ev_to_nm(ev):
    """Convert eV to nm."""
    if ev <= 0: return 0.0
    return 1239.8 / ev


def _nm_to_ev(nm):
    """Convert nm to eV."""
    if nm <= 0: return 0.0
    return 1239.8 / nm


# ═══════════════════════════════════════════════════════════════════════════
# d-d TRANSITION PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

# Typical extinction coefficients for d-d bands (M⁻¹cm⁻¹)
_DD_EPSILON = {
    "octahedral": 10.0,     # Laporte-forbidden
    "tetrahedral": 100.0,   # Slightly relaxed (no center of symmetry)
    "square_planar": 50.0,
    "other": 20.0,
}

# d-electron count → number of d-d transitions and approximate position
# relative to 10Dq (in units of Dq)
_DD_TRANSITIONS = {
    # d_electrons: [(transition_name, energy_in_Dq, relative_intensity)]
    # For color prediction, we pick the band most likely in visible range
    1: [("²T₂g→²Eg", 10.0, 1.0)],
    2: [("³T₁g→³T₂g", 8.0, 1.0), ("³T₁g→³T₁g(P)", 18.0, 0.5)],
    3: [("⁴A₂g→⁴T₁g(F)", 18.0, 0.7), ("⁴A₂g→⁴T₂g", 10.0, 1.0)],
    4: [("⁵Eg→⁵T₂g", 10.0, 1.0)],
    5: [],  # d5 HS: all spin-forbidden, very weak
    6: [("⁵T₂g→⁵Eg", 10.0, 1.0)],
    7: [("⁴T₁g→⁴T₂g", 8.0, 1.0), ("⁴T₁g→⁴A₂g", 18.0, 0.3)],
    8: [("³A₂g→³T₁g(F)", 18.0, 0.8), ("³A₂g→³T₂g", 10.0, 1.0)],
    9: [("²Eg→²T₂g", 10.0, 1.0)],  # Cu2+ JT-split but centroid at ~10Dq
    10: [],  # d10: no d-d transitions
    0: [],   # d0: no d-d transitions
}


def predict_dd_transition(d_electrons, ten_dq_kj, geometry="octahedral"):
    """Predict d-d transition wavelength and color."""
    if d_electrons in (0, 10) or d_electrons == 5:
        # No visible d-d transition (d0, d10, or d5 HS spin-forbidden)
        return 0.0, 0.0, 0.0, "colorless"

    transitions = _DD_TRANSITIONS.get(d_electrons, [])
    if not transitions:
        return 0.0, 0.0, 0.0, "colorless"

    # Primary (most intense) transition
    name, energy_dq, rel_int = transitions[0]
    dq_kj = ten_dq_kj / 10.0
    energy_kj = energy_dq * dq_kj

    energy_ev = energy_kj / _EV_TO_KJ if energy_kj > 0 else 0.0
    wavelength = _ev_to_nm(energy_ev)
    epsilon = _DD_EPSILON.get(geometry, 20.0) * rel_int

    color = _wavelength_to_color(wavelength)

    return round(wavelength, 0), round(energy_ev, 3), round(epsilon, 1), color


_EV_TO_KJ = 96.485


# ═══════════════════════════════════════════════════════════════════════════
# CHARGE TRANSFER PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

# LMCT energies depend on donor ionization energy and metal reduction potential
# Higher E0(metal) + lower IE(donor) = lower energy LMCT
_DONOR_IE_EV = {
    "O": 9.0,     # Carboxylate/hydroxide
    "N": 8.0,     # Amine/imine
    "S": 6.5,     # Thiolate — low IE, strong LMCT
    "P": 7.0,     # Phosphine
    "Cl": 8.5,
    "Br": 7.5,
    "I": 6.5,
}

_METAL_EA_EV = {
    # Approximate electron affinity / reduction tendency
    "Fe3+": 4.5, "Cr3+": 3.0, "Mn3+": 4.0, "Co3+": 5.0,
    "Cu2+": 3.5, "Ni2+": 2.5, "Au3+": 5.5, "Pt2+": 4.0,
    "Ru3+": 3.8, "Os3+": 3.5, "Pd2+": 3.8,
    "Fe2+": 2.0, "Co2+": 2.2, "Mn2+": 1.5, "Zn2+": 1.0,
}


def predict_ct_transition(metal_formula, donor_atoms, d_electrons=0):
    """Predict charge transfer bands (LMCT or MLCT)."""
    metal_ea = _METAL_EA_EV.get(metal_formula, 2.5)

    # Average donor IE
    ies = [_DONOR_IE_EV.get(da, 8.0) for da in donor_atoms]
    avg_ie = sum(ies) / len(ies) if ies else 8.0
    min_ie = min(ies) if ies else 8.0

    # LMCT energy: ~ IE(donor) - EA(metal)
    lmct_ev = min_ie - metal_ea
    lmct_nm = _ev_to_nm(lmct_ev) if lmct_ev > 0.5 else 0.0

    # MLCT: relevant for low-oxidation-state metals with π-acceptor ligands
    has_pi_acceptor = any(da in ("N", "P") for da in donor_atoms)
    if d_electrons >= 6 and has_pi_acceptor and metal_ea < 3.0:
        mlct_ev = metal_ea + 1.0  # Rough: HOMO(metal) → π*(ligand)
        mlct_nm = _ev_to_nm(mlct_ev)
        ct_type = "MLCT"
        ct_nm = mlct_nm
    elif lmct_nm > 200:
        ct_type = "LMCT"
        ct_nm = lmct_nm
    else:
        return 0.0, "none", 0.0

    # CT bands are intense: ε = 1000-50000 M⁻¹cm⁻¹
    epsilon = 5000.0 if ct_type == "LMCT" else 10000.0
    if "S" in donor_atoms:
        epsilon *= 2.0  # S→M CT very intense

    return round(ct_nm, 0), ct_type, round(epsilon, 0)


# ═══════════════════════════════════════════════════════════════════════════
# DETECTION STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

def recommend_detection(dd_nm, dd_eps, ct_nm, ct_eps, ct_type,
                         metal_formula, donor_atoms):
    """Recommend optimal detection strategy for binding confirmation."""
    strategies = []

    # Colorimetric: if color change is visible (ε > 50 in visible range)
    if dd_nm > 380 and dd_nm < 780 and dd_eps > 20:
        strategies.append(("colorimetric", f"d-d color change at {dd_nm:.0f} nm",
                          "µM"))

    # CT-based UV-Vis: intense bands
    if ct_nm > 250 and ct_eps > 1000:
        region = "visible" if ct_nm > 380 else "UV"
        sens = "nM" if ct_eps > 5000 else "µM"
        strategies.append(("UV-Vis", f"{ct_type} band at {ct_nm:.0f} nm ({region})",
                          sens))

    # Fluorescence quench: paramagnetic metals quench fluorophores
    if metal_formula in ("Cu2+", "Fe3+", "Co2+", "Ni2+", "Mn2+"):
        strategies.append(("fluorescence_quench",
                          "Paramagnetic quenching of appended fluorophore",
                          "nM"))

    # SPR: universal for any binding event
    strategies.append(("SPR", "Surface plasmon resonance (universal)", "nM"))

    # ICP-MS: direct metal quantification (always available)
    strategies.append(("ICP-MS", "Direct metal quantification", "ppt"))

    # Best strategy
    if strategies:
        best = strategies[0]
        return best[0], best[1], best[2]
    return "SPR", "Surface plasmon resonance", "nM"


# ═══════════════════════════════════════════════════════════════════════════
# PHOTORESPONSIVE RELEASE
# ═══════════════════════════════════════════════════════════════════════════

_PHOTOSWITCHES = {
    "azobenzene": {
        "forward_nm": 365,   # trans → cis (UV)
        "reverse_nm": 450,   # cis → trans (blue) or thermal
        "geometry_change_nm": 0.35,  # End-to-end distance change
        "fatigue_cycles": 1e5,
        "thermal_half_life_h": 12,  # cis → trans dark reversion
    },
    "diarylethene": {
        "forward_nm": 313,   # Open → closed (UV)
        "reverse_nm": 530,   # Closed → open (green)
        "geometry_change_nm": 0.15,
        "fatigue_cycles": 1e4,
        "thermal_half_life_h": 1e6,  # Thermally stable both forms
    },
    "spiropyran": {
        "forward_nm": 365,   # Spiropyran → merocyanine (UV)
        "reverse_nm": 530,   # Merocyanine → spiropyran (green) or thermal
        "geometry_change_nm": 0.20,
        "fatigue_cycles": 1e3,
        "thermal_half_life_h": 1,  # Fast dark reversion
    },
}


def assess_photoresponsive(scaffold_type, site_spacing_nm=0.0):
    """Assess whether photoresponsive release is feasible.

    Photorelease requires geometry change larger than binding pocket
    tolerance. Azobenzene: 0.35 nm change — good for DNA origami,
    MIP cavities. Diarylethene: 0.15 nm — more subtle.
    """
    if scaffold_type in ("dna_origami_icosahedron", "dna_origami_tetrahedron"):
        # DNA origami can incorporate azobenzene in staples
        return True, "azobenzene", 365, \
            "Azobenzene-modified staples: 365 nm triggers cage opening/closing"
    elif scaffold_type == "MIP":
        # MIP with photoresponsive monomer
        return True, "spiropyran", 365, \
            "Spiropyran monomer: UV triggers polarity change → cavity distortion → release"
    elif scaffold_type in ("MOF_UiO66", "MOF_MIL101"):
        # Azobenzene linkers in MOF
        return True, "azobenzene", 365, \
            "Azobenzene-dicarboxylate linkers: photoisomerization changes pore access"
    elif scaffold_type == "COF":
        return True, "diarylethene", 313, \
            "Diarylethene crosslinks: ring-closing changes pore geometry"
    else:
        return False, "none", 0, "No established photoswitch strategy for this scaffold"


# ═══════════════════════════════════════════════════════════════════════════
# FULL SPECTROSCOPIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def predict_spectroscopy(metal_formula, donor_atoms, d_electrons,
                          ten_dq_kj=120.0, geometry="octahedral",
                          scaffold_type="free"):
    """Full spectroscopic prediction for a metal-binder-scaffold assembly."""
    dd_nm, dd_ev, dd_eps, color = predict_dd_transition(
        d_electrons, ten_dq_kj, geometry)

    ct_nm, ct_type, ct_eps = predict_ct_transition(
        metal_formula, donor_atoms, d_electrons)

    det_method, det_signal, sensitivity = recommend_detection(
        dd_nm, dd_eps, ct_nm, ct_eps, ct_type, metal_formula, donor_atoms)

    photo, photo_type, photo_nm, photo_notes = assess_photoresponsive(
        scaffold_type)

    notes_parts = []
    if color != "colorless":
        notes_parts.append(f"Complex is {color} (absorbs at {dd_nm:.0f} nm)")
    if ct_type != "none":
        notes_parts.append(f"{ct_type} band at {ct_nm:.0f} nm (ε={ct_eps:.0f})")
    if photo:
        notes_parts.append(f"Photorelease: {photo_type} at {photo_nm} nm")

    return SpectroscopicPrediction(
        dd_transition_nm=dd_nm, dd_transition_ev=dd_ev,
        dd_extinction=dd_eps, predicted_color=color,
        ct_transition_nm=ct_nm, ct_type=ct_type, ct_extinction=ct_eps,
        detection_method=det_method, detection_signal=det_signal,
        sensitivity_estimate=sensitivity,
        photoresponsive=photo, photoswitch_type=photo_type,
        trigger_wavelength_nm=photo_nm,
        notes="; ".join(notes_parts),
    )

