"""
knowledge/entropy_data.py - Enthalpy/entropy decomposition parameters.

Convention: ΔG = ΔH - TΔS
  ΔH < 0 = exothermic (favorable)
  ΔS > 0 = entropy increase (favorable)
"""

ENTHALPY_FRACTIONS = {
    "dG_bind":          0.85,   # bond formation: mostly enthalpic
    "dG_desolv":        0.60,   # mixed: lost bonds (ΔH) + released water (ΔS)
    "dG_chelate":       0.10,   # chelate effect IS entropy
    "dG_preorg":        0.15,   # preorg = pre-paid conformational entropy
    "dG_electrostatic": 0.95,   # Coulomb = direct energy
    "dG_protonation":   0.70,   # mixed
    "dG_lfse":          1.00,   # crystal field = electronic energy
    "dG_activity":      0.50,   # mixed
    "dG_repulsion":     1.00,   # Pauli/Born = energy barriers
}

# ΔS per donor coordination (J/(mol·K))
DONOR_ENTROPY = {
    "S": -15.0,  "N": -20.0,  "O": +5.0,
    "P": -10.0,  "electrostatic": +8.0,
}

WATER_RELEASE_ENTROPY = 25.0   # J/(mol·K) per water displaced
ROTATABLE_BOND_ENTROPY_LOSS = -6.0  # J/(mol·K) per bond

DONOR_HEAT_CAPACITY = {
    "S": -30.0,  "N": -10.0,  "O": +5.0,
    "P": -20.0,  "electrostatic": 0.0,
}
