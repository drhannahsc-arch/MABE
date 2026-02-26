"""
Crown Ether & Cryptand Knowledge Base.

Properties from Izatt critical compilations, Pedersen original data,
and Buschmann/Schollmeyer thermodynamic reviews.

Size-match selectivity from Hancock & Martell (Chem. Rev. 1989).
Macrocyclic/cryptate effects from Haymore et al. (Inorg. Chem. 1982).

Data sources:
    - Izatt RM et al. (Chem. Rev. 1985, 1991, 1995) — stability constant compilations
    - Pedersen CJ (J. Am. Chem. Soc. 1967) — original crown ether work
    - Buschmann HJ, Schollmeyer E (J. Incl. Phenom. 2000) — thermodynamics
    - Hancock RD, Martell AE (Chem. Rev. 1989) — chelate ring size selectivity
    - Lehn JM (Angew. Chem. 1988) — cryptand design principles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class CrownEtherHost:
    """Physical properties of one crown ether or cryptand."""

    name: str
    host_class: str                    # "crown_ether", "aza_crown", "thia_crown", "cryptand"
    common_name: str                   # e.g., "18-crown-6", "[2.2.2]cryptand"

    # ── Ring/cage geometry ──
    n_donors: int
    donor_types: list[str]             # ["O","O","O","O","O","O"] for 18-crown-6
    cavity_radius_A: float             # from crystallography
    ring_atoms: int                    # total atoms in ring backbone
    is_3d_cage: bool                   # True for cryptands

    # ── Physicochemical ──
    mw: float
    water_soluble: bool
    organic_soluble: bool
    smiles: str

    # ── Selectivity data (Izatt compilations) ──
    best_match_ion: str                # ion with highest log K
    best_match_ionic_radius_A: float
    best_match_logK: float             # in water, 25°C
    selectivity_profile: dict[str, float]  # ion → log K

    # ── Macrocyclic / cryptate effect ──
    macrocyclic_stabilization_kJ_mol: float

    # ── Production ──
    commercial: bool
    cost_per_gram_usd: float
    common_suppliers: list[str]
    synthesis_route: str               # "Pedersen", "template", "high-dilution"

    # ── Fields with defaults (must come last) ──
    cryptate_stabilization_kJ_mol: float = 0.0  # only for cryptands
    lariat_arm_possible: bool = True
    n_substitution_possible: bool = False  # N-donor variants
    benzo_fusion_possible: bool = True


@dataclass(frozen=True)
class CationTarget:
    """Target cation properties for size-match calculation."""

    symbol: str
    charge: int
    ionic_radius_A: float              # Shannon effective ionic radius (CN=6)
    ionic_radius_cn4_A: float          # CN=4 where available
    hsab_class: str                    # "hard", "borderline", "soft"
    hydration_dG_kJ_mol: float         # Marcus 1991 / Kepp 2019
    preferred_donors: list[str]        # HSAB-matched donor types


# ─────────────────────────────────────────────
# Cation database (Shannon radii, Marcus hydration)
# ─────────────────────────────────────────────

CATION_DB: dict[str, CationTarget] = {
    "Li+": CationTarget("Li+", 1, 0.76, 0.59, "hard", -475.0, ["O", "N"]),
    "Na+": CationTarget("Na+", 1, 1.02, 0.99, "hard", -365.0, ["O", "N"]),
    "K+": CationTarget("K+", 1, 1.38, 1.37, "hard", -295.0, ["O"]),
    "Rb+": CationTarget("Rb+", 1, 1.52, 1.52, "hard", -275.0, ["O"]),
    "Cs+": CationTarget("Cs+", 1, 1.67, 1.67, "hard", -250.0, ["O"]),
    "Mg2+": CationTarget("Mg2+", 2, 0.72, 0.57, "hard", -1830.0, ["O", "N"]),
    "Ca2+": CationTarget("Ca2+", 2, 1.00, 1.00, "hard", -1505.0, ["O"]),
    "Sr2+": CationTarget("Sr2+", 2, 1.18, 1.18, "hard", -1380.0, ["O"]),
    "Ba2+": CationTarget("Ba2+", 2, 1.35, 1.35, "hard", -1250.0, ["O"]),
    "Cu2+": CationTarget("Cu2+", 2, 0.73, 0.57, "borderline", -2010.0, ["N", "S", "O"]),
    "Zn2+": CationTarget("Zn2+", 2, 0.74, 0.60, "borderline", -1955.0, ["N", "S", "O"]),
    "Ni2+": CationTarget("Ni2+", 2, 0.69, 0.55, "borderline", -1980.0, ["N", "S"]),
    "Co2+": CationTarget("Co2+", 2, 0.75, 0.58, "borderline", -1915.0, ["N", "O"]),
    "Pb2+": CationTarget("Pb2+", 2, 1.19, 0.98, "borderline", -1425.0, ["S", "N", "O"]),
    "Cd2+": CationTarget("Cd2+", 2, 0.95, 0.78, "soft", -1755.0, ["S", "N"]),
    "Hg2+": CationTarget("Hg2+", 2, 1.02, 0.96, "soft", -1760.0, ["S"]),
    "Ag+": CationTarget("Ag+", 1, 1.15, 1.00, "soft", -430.0, ["S", "N"]),
    "Tl+": CationTarget("Tl+", 1, 1.50, 1.50, "soft", -300.0, ["S", "N"]),
    "NH4+": CationTarget("NH4+", 1, 1.48, 1.48, "hard", -285.0, ["O"]),
    "La3+": CationTarget("La3+", 3, 1.03, 1.03, "hard", -3145.0, ["O"]),
    "Eu3+": CationTarget("Eu3+", 3, 0.95, 0.95, "hard", -3360.0, ["O"]),
    "UO2_2+": CationTarget("UO2_2+", 2, 0.73, 0.73, "hard", -1600.0, ["O"]),
}


# ─────────────────────────────────────────────
# Crown ether hosts
# ─────────────────────────────────────────────

CROWN_12C4 = CrownEtherHost(
    name="12-crown-4",
    host_class="crown_ether",
    common_name="12-crown-4",
    n_donors=4,
    donor_types=["O", "O", "O", "O"],
    cavity_radius_A=0.60,
    ring_atoms=12,
    is_3d_cage=False,
    mw=176.21,
    water_soluble=True,
    organic_soluble=True,
    smiles="C1COCCOCCOCCO1",
    best_match_ion="Li+",
    best_match_ionic_radius_A=0.76,
    best_match_logK=1.7,
    selectivity_profile={
        "Li+": 1.7, "Na+": 1.3, "K+": 0.8,
    },
    macrocyclic_stabilization_kJ_mol=8.0,
    commercial=True,
    cost_per_gram_usd=5.00,
    common_suppliers=["Sigma-Aldrich", "TCI"],
    synthesis_route="Pedersen (Williamson ether)",
)

CROWN_15C5 = CrownEtherHost(
    name="15-crown-5",
    host_class="crown_ether",
    common_name="15-crown-5",
    n_donors=5,
    donor_types=["O", "O", "O", "O", "O"],
    cavity_radius_A=0.86,
    ring_atoms=15,
    is_3d_cage=False,
    mw=220.26,
    water_soluble=True,
    organic_soluble=True,
    smiles="C1COCCOCCOCCOCCO1",
    best_match_ion="Na+",
    best_match_ionic_radius_A=1.02,
    best_match_logK=3.24,
    selectivity_profile={
        "Li+": 1.0, "Na+": 3.24, "K+": 2.27, "Rb+": 1.56,
        "Cs+": 1.17, "Ca2+": 2.30, "Sr2+": 2.80,
    },
    macrocyclic_stabilization_kJ_mol=10.0,
    commercial=True,
    cost_per_gram_usd=3.00,
    common_suppliers=["Sigma-Aldrich", "TCI", "Alfa Aesar"],
    synthesis_route="Pedersen (Williamson ether)",
)

CROWN_18C6 = CrownEtherHost(
    name="18-crown-6",
    host_class="crown_ether",
    common_name="18-crown-6",
    n_donors=6,
    donor_types=["O", "O", "O", "O", "O", "O"],
    cavity_radius_A=1.34,
    ring_atoms=18,
    is_3d_cage=False,
    mw=264.32,
    water_soluble=True,
    organic_soluble=True,
    smiles="C1COCCOCCOCCOCCOCCO1",
    best_match_ion="K+",
    best_match_ionic_radius_A=1.38,
    best_match_logK=6.10,
    selectivity_profile={
        "Li+": 0.8, "Na+": 4.35, "K+": 6.10, "Rb+": 5.35,
        "Cs+": 4.79, "Ca2+": 3.90, "Sr2+": 5.30, "Ba2+": 5.76,
        "Pb2+": 4.27, "NH4+": 4.27, "Tl+": 5.87, "Ag+": 4.76,
    },
    macrocyclic_stabilization_kJ_mol=12.0,
    commercial=True,
    cost_per_gram_usd=1.50,
    common_suppliers=["Sigma-Aldrich", "TCI", "Alfa Aesar", "Merck"],
    synthesis_route="Pedersen (Williamson ether)",
)

CROWN_21C7 = CrownEtherHost(
    name="21-crown-7",
    host_class="crown_ether",
    common_name="21-crown-7",
    n_donors=7,
    donor_types=["O", "O", "O", "O", "O", "O", "O"],
    cavity_radius_A=1.70,
    ring_atoms=21,
    is_3d_cage=False,
    mw=308.37,
    water_soluble=True,
    organic_soluble=True,
    smiles="C1COCCOCCOCCOCCOCCOCCO1",
    best_match_ion="Cs+",
    best_match_ionic_radius_A=1.67,
    best_match_logK=4.50,
    selectivity_profile={
        "Na+": 2.10, "K+": 4.30, "Rb+": 4.40,
        "Cs+": 4.50, "Ba2+": 4.90,
    },
    macrocyclic_stabilization_kJ_mol=10.0,
    commercial=True,
    cost_per_gram_usd=8.00,
    common_suppliers=["Sigma-Aldrich", "TCI"],
    synthesis_route="Pedersen (Williamson ether)",
)

# ── Aza-crowns (N-donor substitution) ──

DIAZA_18C6 = CrownEtherHost(
    name="diaza-18-crown-6",
    host_class="aza_crown",
    common_name="1,10-diaza-18-crown-6",
    n_donors=6,
    donor_types=["O", "O", "N", "O", "O", "N"],
    cavity_radius_A=1.30,
    ring_atoms=18,
    is_3d_cage=False,
    mw=262.35,
    water_soluble=True,
    organic_soluble=True,
    smiles="C1COCCN(CCOCCOCCN1)CCO",
    best_match_ion="Cu2+",
    best_match_ionic_radius_A=0.73,
    best_match_logK=7.9,
    selectivity_profile={
        "Cu2+": 7.9, "Ni2+": 5.4, "Zn2+": 5.1,
        "Co2+": 3.8, "Pb2+": 6.2, "K+": 3.9,
    },
    macrocyclic_stabilization_kJ_mol=15.0,  # N-donors enhance macrocyclic effect
    commercial=True,
    cost_per_gram_usd=12.00,
    common_suppliers=["Sigma-Aldrich", "TCI"],
    synthesis_route="Richman-Atkins (tosyl route)",
    n_substitution_possible=True,
)

# ── Thia-crown (S-donor for soft metals) ──

DITHIA_18C6 = CrownEtherHost(
    name="dithia-18-crown-6",
    host_class="thia_crown",
    common_name="1,10-dithia-18-crown-6",
    n_donors=6,
    donor_types=["O", "O", "S", "O", "O", "S"],
    cavity_radius_A=1.35,
    ring_atoms=18,
    is_3d_cage=False,
    mw=296.43,
    water_soluble=False,
    organic_soluble=True,
    smiles="C1COCCSCCOCCSCCO1",
    best_match_ion="Ag+",
    best_match_ionic_radius_A=1.15,
    best_match_logK=7.2,
    selectivity_profile={
        "Ag+": 7.2, "Hg2+": 6.8, "Pb2+": 5.5,
        "Cu2+": 4.9, "K+": 3.1,
    },
    macrocyclic_stabilization_kJ_mol=12.0,
    commercial=True,
    cost_per_gram_usd=25.00,
    common_suppliers=["Sigma-Aldrich", "TCI"],
    synthesis_route="Pedersen (modified, thiol precursor)",
)

# ── Cryptands (3D cage) ──

CRYPTAND_211 = CrownEtherHost(
    name="[2.1.1]cryptand",
    host_class="cryptand",
    common_name="[2.1.1]cryptand",
    n_donors=5,
    donor_types=["O", "O", "O", "N", "N"],
    cavity_radius_A=0.80,
    ring_atoms=17,
    is_3d_cage=True,
    mw=247.33,
    water_soluble=True,
    organic_soluble=True,
    smiles="C1COCCN2CCOCCN(CCO1)CC2",
    best_match_ion="Li+",
    best_match_ionic_radius_A=0.76,
    best_match_logK=5.50,
    selectivity_profile={
        "Li+": 5.50, "Na+": 3.20, "K+": 1.50,
    },
    macrocyclic_stabilization_kJ_mol=10.0,
    cryptate_stabilization_kJ_mol=12.0,
    commercial=True,
    cost_per_gram_usd=50.00,
    common_suppliers=["Sigma-Aldrich"],
    synthesis_route="Lehn (high-dilution)",
)

CRYPTAND_221 = CrownEtherHost(
    name="[2.2.1]cryptand",
    host_class="cryptand",
    common_name="[2.2.1]cryptand",
    n_donors=6,
    donor_types=["O", "O", "O", "O", "N", "N"],
    cavity_radius_A=1.10,
    ring_atoms=20,
    is_3d_cage=True,
    mw=291.39,
    water_soluble=True,
    organic_soluble=True,
    smiles="C1COCCOCCN2CCOCCN(CCO1)CCO2",
    best_match_ion="Na+",
    best_match_ionic_radius_A=1.02,
    best_match_logK=9.70,
    selectivity_profile={
        "Li+": 5.40, "Na+": 9.70, "K+": 5.30,
        "Ca2+": 6.95, "Sr2+": 7.35,
    },
    macrocyclic_stabilization_kJ_mol=12.0,
    cryptate_stabilization_kJ_mol=15.0,
    commercial=True,
    cost_per_gram_usd=80.00,
    common_suppliers=["Sigma-Aldrich"],
    synthesis_route="Lehn (high-dilution)",
)

CRYPTAND_222 = CrownEtherHost(
    name="[2.2.2]cryptand",
    host_class="cryptand",
    common_name="[2.2.2]cryptand (Kryptofix 222)",
    n_donors=8,
    donor_types=["O", "O", "O", "O", "O", "O", "N", "N"],
    cavity_radius_A=1.40,
    ring_atoms=24,
    is_3d_cage=True,
    mw=376.50,
    water_soluble=True,
    organic_soluble=True,
    smiles="C1COCCOCCN2CCOCCOCCN(CCOCC1)CCO2",
    best_match_ion="K+",
    best_match_ionic_radius_A=1.38,
    best_match_logK=10.40,
    selectivity_profile={
        "Li+": 2.50, "Na+": 7.21, "K+": 10.40, "Rb+": 8.80,
        "Cs+": 4.40, "Ca2+": 8.10, "Sr2+": 10.80, "Ba2+": 11.60,
        "Pb2+": 12.40, "Ag+": 9.60, "La3+": 5.80,
    },
    macrocyclic_stabilization_kJ_mol=15.0,
    cryptate_stabilization_kJ_mol=20.0,
    commercial=True,
    cost_per_gram_usd=30.00,
    common_suppliers=["Sigma-Aldrich", "Merck", "TCI"],
    synthesis_route="Lehn (high-dilution), commercial as Kryptofix 222",
)


# ─────────────────────────────────────────────
# Indexed registries
# ─────────────────────────────────────────────

ALL_CROWN_HOSTS: dict[str, CrownEtherHost] = {
    "12-crown-4": CROWN_12C4,
    "15-crown-5": CROWN_15C5,
    "18-crown-6": CROWN_18C6,
    "21-crown-7": CROWN_21C7,
    "diaza-18-crown-6": DIAZA_18C6,
    "dithia-18-crown-6": DITHIA_18C6,
    "[2.1.1]cryptand": CRYPTAND_211,
    "[2.2.1]cryptand": CRYPTAND_221,
    "[2.2.2]cryptand": CRYPTAND_222,
}

CROWNS_ONLY: list[CrownEtherHost] = [
    CROWN_12C4, CROWN_15C5, CROWN_18C6, CROWN_21C7,
]

AZA_CROWNS: list[CrownEtherHost] = [DIAZA_18C6]

THIA_CROWNS: list[CrownEtherHost] = [DITHIA_18C6]

CRYPTANDS: list[CrownEtherHost] = [CRYPTAND_211, CRYPTAND_221, CRYPTAND_222]

ALL_HOSTS_LIST: list[CrownEtherHost] = list(ALL_CROWN_HOSTS.values())


# ─────────────────────────────────────────────
# Physics functions
# ─────────────────────────────────────────────

# Size-match Gaussian parameters
SIZE_MATCH_SIGMA_A = 0.20  # Å — selectivity width (tighter = more selective)
SIZE_MATCH_K = 5.0         # kJ/mol — max stabilization at perfect match


def size_match_score(ion_radius_A: float, cavity_radius_A: float) -> float:
    """
    Gaussian size-match selectivity.

    Returns 0.0–1.0: 1.0 = perfect r_ion == r_cavity match.
    From Hancock & Martell cavity size-match model.
    """
    return math.exp(
        -0.5 * ((ion_radius_A - cavity_radius_A) / SIZE_MATCH_SIGMA_A) ** 2
    )


def size_match_dG(ion_radius_A: float, cavity_radius_A: float) -> float:
    """
    Free energy contribution from size-match.

    Returns negative (stabilizing) when well-matched.
    """
    return -SIZE_MATCH_K * size_match_score(ion_radius_A, cavity_radius_A)


def hsab_donor_score(ion_hsab: str, donor_types: list[str]) -> float:
    """
    HSAB compatibility score for donor set.

    Returns 0.0–1.0: how well the donor types match the ion's preference.
    """
    preference_map = {
        "hard": {"O": 1.0, "N": 0.7, "S": 0.2},
        "borderline": {"O": 0.5, "N": 1.0, "S": 0.7},
        "soft": {"O": 0.2, "N": 0.5, "S": 1.0},
    }
    prefs = preference_map.get(ion_hsab, preference_map["borderline"])
    if not donor_types:
        return 0.0
    scores = [prefs.get(d, 0.3) for d in donor_types]
    return sum(scores) / len(scores)


def select_best_crown(
    target_ion: str,
    require_3d: bool = False,
    require_water_soluble: bool = True,
) -> list[tuple[CrownEtherHost, float, float, str]]:
    """
    Rank all crown/cryptand hosts for a target cation.

    Returns list of (host, size_match_score, predicted_logK, rationale)
    sorted by combined score.
    """
    if target_ion not in CATION_DB:
        return []

    cation = CATION_DB[target_ion]
    results = []

    candidates = ALL_HOSTS_LIST
    if require_3d:
        candidates = [h for h in candidates if h.is_3d_cage]
    if require_water_soluble:
        candidates = [h for h in candidates if h.water_soluble]

    for host in candidates:
        sm = size_match_score(cation.ionic_radius_A, host.cavity_radius_A)
        hsab = hsab_donor_score(cation.hsab_class, host.donor_types)

        # Combined score: size-match dominates, HSAB secondary
        combined = sm * 0.7 + hsab * 0.3

        # Use known logK if available, else estimate
        known_logK = host.selectivity_profile.get(target_ion)
        if known_logK is not None:
            predicted_logK = known_logK
        else:
            # Estimate from size-match + macrocyclic effect
            base_logK = sm * 6.0  # rough: perfect match ≈ 6 log K
            macro_bonus = host.macrocyclic_stabilization_kJ_mol / 5.71  # convert to log K
            crypto_bonus = host.cryptate_stabilization_kJ_mol / 5.71
            charge_factor = cation.charge * 0.8  # divalent binds stronger
            predicted_logK = base_logK + macro_bonus + crypto_bonus + (charge_factor - 0.8)

        # Rationale
        parts = [f"size-match={sm:.2f}"]
        parts.append(f"HSAB={hsab:.2f}")
        if known_logK is not None:
            parts.append(f"known logK={known_logK:.1f}")
        else:
            parts.append(f"estimated logK={predicted_logK:.1f}")
        if host.is_3d_cage:
            parts.append("3D encapsulation (cryptate effect)")
        if sm > 0.8:
            parts.append("excellent size match")
        elif sm < 0.3:
            parts.append("poor size match")

        results.append((host, combined, predicted_logK, "; ".join(parts)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
