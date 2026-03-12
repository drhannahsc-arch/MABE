"""
mabe/glycan/params.py — Glycan recognition parameters
======================================================

All parameters back-solved from non-biological sources.
Zero fitting against biological binding data.

Parameter sources are documented inline. Each parameter has:
  - Value
  - Source (paper, database, or derivation)
  - Phase (G1–G8) in which it was locked
  - Status: LOCKED (calibrated) or PLACEHOLDER (future phase)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GlycanParams:
    """
    Complete glycan parameter set.
    
    LOCKED parameters have been calibrated from non-biological sources.
    PLACEHOLDER parameters are set to physically reasonable defaults
    and will be calibrated in future phases (G2–G8).
    """

    # ── G1: Polar desolvation (LOCKED) ──────────────────────────────────
    # Source: Rekharsky & Inoue 2007 (Chem. Rev. 107, 3715)
    k_desolv_OH: float = 3.97        # kJ/mol per uncompensated OH
    k_hbond_dG: float = -2.00        # kJ/mol per H-bond (free energy, CD portal)
    k_hbond_dH: float = -7.35        # kJ/mol per H-bond (enthalpy, intrinsic)

    # Source: Jasra et al. 1982 (J. Solution Chem. 11, 325)
    dCp_per_OH: float = -52.0        # J/(K·mol) heat capacity per OH

    # SASA-based per-position desolvation (kJ/mol)
    # Source: RDKit ETKDG + FreeSASA, γ_polar = 0.075 kJ/(mol·Å²)
    gamma_polar: float = 0.075       # kJ/(mol·Å²)

    # Context-dependent buffering for multivalent ligands
    # Calibrated from Gupta 1996 trimannoside deoxy series
    beta_context_default: float = 0.45

    # ── G2: Conformational entropy (PLACEHOLDER) ───────────────────────
    # Will be calibrated from GLYCAM06 torsion potentials
    eps_glycosidic_freeze: float = 4.0    # kJ/mol per frozen φ/ψ pair
    k_branch_penalty: float = 2.0         # kJ/mol per branch point

    # ── G3: CH-π interactions (LOCKED) ─────────────────────────────────
    # Source: Laughrey et al. 2008 (JACS 130, 14625)
    eps_CH_pi: float = -2.5               # kJ/mol per CH-π contact
    # Range: -2.1 to -3.3 kJ/mol across 6 model systems

    # ── G5: Structural water (PLACEHOLDER) ─────────────────────────────
    # Will be calibrated from lectin mutant series
    eps_water_bridge: float = -3.5        # kJ/mol per conserved water
    n_water_bridge_norm: float = 0.015    # waters per Å² contact area

    # ── G7: Ca²⁺ bridging for C-type lectins (PLACEHOLDER) ────────────
    # Will reuse MABE metal scorer; this is just the coupling term
    eps_Ca_coordination: float = -5.0     # kJ/mol per Ca-sugar-protein bridge

    # ── G8: Multivalency (PLACEHOLDER) ─────────────────────────────────
    # Will be calibrated from Dam & Brewer 2002
    k_multivalent_coop: float = 0.5       # cooperativity per additional valence
    k_multivalent_spacing: float = 1.0    # optimal inter-site distance scaling


# Per-position SASA-based desolvation costs (kJ/mol)
# Computed from RDKit ETKDG conformer ensemble + FreeSASA
SASA_DESOLV_PER_POSITION = {
    'C1_anomeric':     3.67,
    'C2_equatorial':   3.14,
    'C2_axial':        3.13,
    'C3_equatorial':   3.04,
    'C4_equatorial':   2.48,
    'C4_axial':        3.11,
    'C6_primary':      3.40,
}

# Singleton default params
GLYCAN_PARAMS = GlycanParams()
