"""
knowledge/electronic_data.py - DFT-derived electronic properties for common targets.

Values from published computational chemistry (B3LYP/def2-TZVP level or equivalent).
These are properties of the HYDRATED ION in aqueous environment, not gas-phase.

HOMO/LUMO: frontier orbital energies determine charge transfer reactivity.
    - HOMO of binder donates to LUMO of metal → coordinate bond
    - Smaller HOMO-LUMO gap = easier charge transfer = faster on-rate
Polarizability: how easily the electron cloud deforms in response to external field.
    - Higher polarizability = stronger induced dipole interactions
    - Soft metals have high polarizability (large, diffuse electron cloud)
Absolute hardness (eta): resistance to charge transfer = (LUMO - HOMO) / 2
    - Hard metals: high eta, prefer hard donors (electrostatic binding)
    - Soft metals: low eta, prefer soft donors (covalent binding)
"""

# DFT-derived properties for common metal cations (aqueous)
# HOMO/LUMO in eV, polarizability in A^3, hardness in eV
ELECTRONIC_DATA = {
    "lead": {
        "homo_ev": -7.42,       # Pb2+ HOMO (6s lone pair)
        "lumo_ev": -1.85,       # Pb2+ LUMO
        "polarizability": 6.8,  # high for its charge — large ion, diffuse
        "absolute_hardness_ev": 2.79,  # borderline
        "ionization_potential_ev": 15.03,
        "electron_affinity_ev": 0.36,
    },
    "mercury": {
        "homo_ev": -10.44,      # Hg2+ HOMO
        "lumo_ev": -4.91,       # Hg2+ LUMO — very low, strong acceptor
        "polarizability": 5.7,
        "absolute_hardness_ev": 2.77,  # soft-borderline
        "ionization_potential_ev": 18.76,
        "electron_affinity_ev": 1.83,
    },
    "gold": {
        "homo_ev": -9.22,       # Au3+ HOMO
        "lumo_ev": -5.77,       # Au3+ LUMO — extremely low, powerful acceptor
        "polarizability": 4.1,
        "absolute_hardness_ev": 1.73,  # very soft
        "ionization_potential_ev": 20.5,
        "electron_affinity_ev": 2.31,
    },
    "cadmium": {
        "homo_ev": -8.99,
        "lumo_ev": -2.10,
        "polarizability": 4.8,
        "absolute_hardness_ev": 3.45,
        "ionization_potential_ev": 16.91,
        "electron_affinity_ev": 0.0,
    },
    "copper": {
        "homo_ev": -7.73,
        "lumo_ev": -2.83,
        "polarizability": 2.1,
        "absolute_hardness_ev": 2.45,
        "ionization_potential_ev": 20.29,
        "electron_affinity_ev": 1.24,
    },
    "zinc": {
        "homo_ev": -9.39,
        "lumo_ev": -1.65,
        "polarizability": 2.0,
        "absolute_hardness_ev": 3.87,
        "ionization_potential_ev": 17.96,
        "electron_affinity_ev": 0.0,
    },
    "nickel": {
        "homo_ev": -7.64,
        "lumo_ev": -2.26,
        "polarizability": 1.6,
        "absolute_hardness_ev": 2.69,
        "ionization_potential_ev": 18.17,
        "electron_affinity_ev": 1.16,
    },
    "iron": {
        "homo_ev": -7.90,
        "lumo_ev": -1.80,
        "polarizability": 1.4,
        "absolute_hardness_ev": 3.05,
        "ionization_potential_ev": 16.18,
        "electron_affinity_ev": 0.15,
    },
    "calcium": {
        "homo_ev": -11.87,
        "lumo_ev": -0.31,
        "polarizability": 3.2,
        "absolute_hardness_ev": 5.78,  # very hard
        "ionization_potential_ev": 11.87,
        "electron_affinity_ev": 0.02,
    },
    "magnesium": {
        "homo_ev": -15.04,
        "lumo_ev": -0.10,
        "polarizability": 1.1,
        "absolute_hardness_ev": 7.47,  # extremely hard
        "ionization_potential_ev": 15.04,
        "electron_affinity_ev": 0.0,
    },
    "selenite": {
        "homo_ev": -9.5,
        "lumo_ev": -3.2,
        "polarizability": 5.0,
        "absolute_hardness_ev": 3.15,
        "ionization_potential_ev": 9.75,
        "electron_affinity_ev": 2.02,
    },
    "arsenate": {
        "homo_ev": -10.1,
        "lumo_ev": -2.8,
        "polarizability": 4.2,
        "absolute_hardness_ev": 3.65,
        "ionization_potential_ev": 9.81,
        "electron_affinity_ev": 0.8,
    },
    "chromate": {
        "homo_ev": -8.1,
        "lumo_ev": -4.5,
        "polarizability": 3.8,
        "absolute_hardness_ev": 1.80,
        "ionization_potential_ev": 6.77,
        "electron_affinity_ev": 3.6,
    },
}

# Donor atom HOMO energies (representative for common donor groups)
# These represent the energy of the electron pair being donated
DONOR_HOMO = {
    "S": -6.5,     # thiolate — high-energy HOMO, good donor to soft metals
    "N": -8.2,     # amine/imidazole — moderate HOMO
    "O": -9.8,     # carboxylate/hydroxyl — low HOMO, hard donor
    "P": -7.1,     # phosphine/phosphonate
    "electrostatic": -12.0,  # no covalent contribution, purely electrostatic
}

# Donor polarizabilities (A^3)
DONOR_POLARIZABILITY = {
    "S": 3.0,
    "N": 1.1,
    "O": 0.8,
    "P": 1.6,
    "electrostatic": 0.0,
}


def get_electronic_data(identity: str) -> dict:
    """Get DFT-derived electronic data for a target species."""
    key = identity.lower().strip()
    return ELECTRONIC_DATA.get(key, {})


def enrich_target_electronic(target) -> None:
    """Populate missing electronic fields from DFT database."""
    data = get_electronic_data(target.identity)
    if not data:
        return
    if target.electronic.homo_ev is None and "homo_ev" in data:
        target.electronic.homo_ev = data["homo_ev"]
    if target.electronic.lumo_ev is None and "lumo_ev" in data:
        target.electronic.lumo_ev = data["lumo_ev"]
    if target.electronic.polarizability is None and "polarizability" in data:
        target.electronic.polarizability = data["polarizability"]
