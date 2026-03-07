"""
core/conformer_3d.py — 3D Conformer Pipeline for Binding Geometry Assessment

Converts 2D SMILES into 3D binding geometry predictions:

  Step 1: SMILES → 3D conformer ensemble (ETKDG + MMFF)
  Step 2: Identify donor atoms in 3D (map auto_descriptor subtypes to xyz)
  Step 3: Multi-conformer sampling (find best-fit to binding geometry)
  Step 4: Kabsch alignment of donor positions to ideal pocket
  Step 5: Strain energy (E_binding_conf − E_global_min)
  Step 6: Preorganization score (how close is free-ligand min-E to binding)

This is the bridge between 2D molecular design (de_novo_generator) and
3D reality. A molecule with perfect 2D descriptors (right donors, right
count, right HSAB) may still be a poor binder if its 3D shape can't
reproduce the ideal pocket geometry.

Uses RDKit only: ETKDG v3 for conformer generation, MMFF94 for energetics.
No external MD, no fitted parameters.

References:
  Riniker & Landrum, JCIM 55:2562 (2015) — ETKDG
  Halgren, JACS 14:490 (1996) — MMFF94
  Kabsch, Acta Cryst. A32:922 (1976) — optimal rotation
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DonorAtom3D:
    """A donor atom located in a 3D conformer."""
    atom_idx: int                # RDKit atom index
    element: str                 # "N", "O", "S", "P"
    donor_subtype: str           # "N_amine", "O_carboxylate", etc.
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class ConformerResult:
    """Result for one conformer."""
    conf_id: int
    energy_kcal: float           # MMFF energy
    donor_positions: list = field(default_factory=list)  # list[DonorAtom3D]
    rmsd_to_ideal: float = 999.0 # after Kabsch alignment
    aligned_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class Binding3DResult:
    """Full 3D binding geometry assessment."""
    smiles: str
    target: str
    n_conformers_generated: int = 0
    n_conformers_valid: int = 0

    # Best binding conformer
    best_conf_id: int = -1
    best_rmsd_A: float = 999.0       # RMSD of donors to ideal pocket (Å)
    best_energy_kcal: float = 0.0

    # Global minimum conformer
    min_energy_conf_id: int = -1
    min_energy_kcal: float = 0.0

    # Strain
    strain_energy_kcal: float = 0.0  # E(binding) - E(min)
    strain_energy_kJ: float = 0.0    # × 4.184

    # Preorganization
    preorganization_rmsd: float = 999.0  # RMSD of min-E conformer donors to ideal
    preorganization_score: float = 0.0   # 0-1, 1 = perfectly preorganized

    # Deviation from ideal
    per_donor_deviation_A: list = field(default_factory=list)
    mean_deviation_A: float = 0.0
    max_deviation_A: float = 0.0

    # Geometry quality
    fidelity_score: float = 0.0      # 0-1, combined geometry match
    geometry_accessible: bool = True  # can metal reach the binding site?

    # Donor info
    n_donors_found: int = 0
    n_donors_expected: int = 0
    donor_subtypes_found: list = field(default_factory=list)

    # All conformers (for analysis)
    conformers: list = field(default_factory=list)  # list[ConformerResult]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: 3D CONFORMER GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_conformers(smiles, n_conformers=50, random_seed=42,
                        prune_thresh=0.5, optimize=True):
    """Generate 3D conformer ensemble from SMILES.

    Args:
        smiles:         Input SMILES
        n_conformers:   Number of conformers to attempt
        random_seed:    Reproducibility seed
        prune_thresh:   RMSD threshold for pruning similar conformers (Å)
        optimize:       Run MMFF optimization on each conformer

    Returns:
        (mol_with_conformers, energies_kcal) or (None, []) on failure
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []

    mol = Chem.AddHs(mol)

    # ETKDG v3 conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = prune_thresh
    params.numThreads = 1

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(cids) == 0:
        # Fallback: try with less strict parameters
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)

    if len(cids) == 0:
        return None, []

    # MMFF optimization + energy
    energies = []
    if optimize:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            # MMFF can't parameterize — use UFF fallback
            for cid in cids:
                try:
                    AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                    energies.append(ff.CalcEnergy() if ff else 999.0)
                except Exception:
                    energies.append(999.0)
        else:
            for cid in cids:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500,
                                                  mmffVariant="MMFF94")
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                    energies.append(ff.CalcEnergy() if ff else 999.0)
                except Exception:
                    energies.append(999.0)
    else:
        energies = [0.0] * len(cids)

    return mol, energies


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: DONOR ATOM LOCATOR IN 3D
# ═══════════════════════════════════════════════════════════════════════════

# SMARTS patterns for donor atom identification
_DONOR_SMARTS = {
    "N_amine":         "[NX3;!$(NC=O);!$(NC=S);!$(N=*)]",   # any sp3 amine (1°/2°/3°)
    "N_pyridine":      "[nX2]",                              # aromatic N
    "N_imidazole":     "[nX3H1]",                            # imidazole NH
    "N_imine":         "[NX2]=*",                            # imine
    "N_amide":         "[NX3;$(NC=O)]",                      # amide N
    "O_carboxylate":   "[OX1;$(O=C)]",                       # C=O oxygen (carboxylate/acid)
    "O_hydroxyl":      "[OX2H;!$(OC=O)]",                   # hydroxyl (not COOH)
    "O_carbonyl":      "[OX1;$(O=C);!$(OC(=O)[O,N])]",      # pure carbonyl
    "O_phenolate":     "[OX2H;$(Oc)]",                       # phenol OH
    "O_ether":         "[OX2;!H;$(C-O-C)]",                  # ether
    "O_phosphonate":   "[OX1;$(OP)]",                        # phosphonate O
    "S_thiolate":      "[SX2H]",                             # thiol
    "S_thioether":     "[#16X2;!H;$(C~S~C)]",               # thioether
    "S_dithiocarbamate":"[SX1;$(SC(=S)N)]",                  # DTC S
    "P_phosphine":     "[PX3]",                              # phosphine
}

# Compiled SMARTS
_COMPILED_SMARTS = {}


def _get_smarts(subtype):
    if subtype not in _COMPILED_SMARTS:
        smarts_str = _DONOR_SMARTS.get(subtype)
        if smarts_str:
            _COMPILED_SMARTS[subtype] = Chem.MolFromSmarts(smarts_str)
        else:
            _COMPILED_SMARTS[subtype] = None
    return _COMPILED_SMARTS[subtype]


def find_donors_3d(mol, conf_id, donor_subtypes=None):
    """Locate donor atoms in a 3D conformer.

    If donor_subtypes is provided, searches for those specific types.
    Otherwise, finds all potential donors.

    Args:
        mol:            RDKit mol with conformers
        conf_id:        Conformer ID
        donor_subtypes: list of expected donor subtypes (optional)

    Returns:
        list[DonorAtom3D]
    """
    conf = mol.GetConformer(conf_id)
    # Work on molecule without explicit H for SMARTS matching
    mol_noH = Chem.RemoveHs(mol)
    conf_noH = mol_noH.GetConformer(0) if mol_noH.GetNumConformers() > 0 else None

    # Map: noH atom idx → H atom idx (for position lookup)
    # Actually, we need positions from the full mol
    # Get atom mapping
    h_to_noH = {}
    noH_to_h = {}
    j = 0
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() != 1:
            h_to_noH[i] = j
            noH_to_h[j] = i
            j += 1

    donors = []
    found_indices = set()

    search_types = donor_subtypes if donor_subtypes else list(_DONOR_SMARTS.keys())

    for subtype in search_types:
        pattern = _get_smarts(subtype)
        if pattern is None:
            continue

        matches = mol_noH.GetSubstructMatches(pattern)
        for match in matches:
            noH_idx = match[0]
            h_idx = noH_to_h.get(noH_idx)
            if h_idx is None or h_idx in found_indices:
                continue
            found_indices.add(h_idx)

            pos = conf.GetAtomPosition(h_idx)
            element = mol.GetAtomWithIdx(h_idx).GetSymbol()

            donors.append(DonorAtom3D(
                atom_idx=h_idx,
                element=element,
                donor_subtype=subtype,
                position=np.array([pos.x, pos.y, pos.z]),
            ))

    return donors


def find_all_donors(mol, conf_id):
    """Find all potential donor atoms without type constraint."""
    return find_donors_3d(mol, conf_id, donor_subtypes=None)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 + 4: KABSCH ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def kabsch_rmsd(P, Q):
    """Compute RMSD after optimal Kabsch rotation+translation.

    P, Q: Nx3 arrays of corresponding point sets.
    Returns (rmsd, rotation_matrix, translation).
    """
    assert P.shape == Q.shape
    n = P.shape[0]
    if n == 0:
        return 999.0, np.eye(3), np.zeros(3)

    # Center both
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q

    # Covariance matrix
    H = P_c.T @ Q_c

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and compute RMSD
    P_aligned = (P_c @ R.T) + centroid_Q
    diff = P_aligned - Q
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd, R, centroid_Q - centroid_P @ R.T


def align_donors_to_ideal(donor_positions, ideal_positions):
    """Align actual donor positions to ideal pocket positions.

    Handles donor count mismatch by finding the best subset match.

    Args:
        donor_positions: Nx3 array of actual donor xyz
        ideal_positions: Mx3 array of ideal donor xyz

    Returns:
        (rmsd, aligned_positions, assignment)
        assignment: list of (actual_idx, ideal_idx) pairs
    """
    n_actual = len(donor_positions)
    n_ideal = len(ideal_positions)

    if n_actual == 0 or n_ideal == 0:
        return 999.0, donor_positions, []

    P = np.array(donor_positions)
    Q = np.array(ideal_positions)

    if n_actual == n_ideal:
        # Try all permutations for small N, greedy for large N
        if n_actual <= 6:
            return _exhaustive_alignment(P, Q)
        else:
            return _greedy_alignment(P, Q)
    elif n_actual > n_ideal:
        # More actual than ideal: find best subset of actual
        return _subset_alignment(P, Q, n_ideal)
    else:
        # Fewer actual than ideal: align what we have
        return _subset_alignment(Q, P, n_actual, reverse=True)


def _exhaustive_alignment(P, Q):
    """Try all permutations of P against Q, return best RMSD."""
    from itertools import permutations
    n = P.shape[0]
    best_rmsd = 999.0
    best_perm = list(range(n))

    for perm in permutations(range(n)):
        P_perm = P[list(perm)]
        rmsd, R, t = kabsch_rmsd(P_perm, Q)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = list(perm)

    P_best = P[best_perm]
    rmsd, R, t = kabsch_rmsd(P_best, Q)
    aligned = (P_best - P_best.mean(axis=0)) @ R.T + Q.mean(axis=0)
    assignment = list(zip(best_perm, range(n)))
    return rmsd, aligned, assignment


def _greedy_alignment(P, Q):
    """Greedy nearest-neighbor assignment then Kabsch."""
    n = min(P.shape[0], Q.shape[0])
    used_P = set()
    used_Q = set()
    assignment = []

    for _ in range(n):
        best_dist = float("inf")
        best_ij = (-1, -1)
        for i in range(P.shape[0]):
            if i in used_P:
                continue
            for j in range(Q.shape[0]):
                if j in used_Q:
                    continue
                d = np.linalg.norm(P[i] - Q[j])
                if d < best_dist:
                    best_dist = d
                    best_ij = (i, j)
        if best_ij[0] >= 0:
            used_P.add(best_ij[0])
            used_Q.add(best_ij[1])
            assignment.append(best_ij)

    if not assignment:
        return 999.0, P, []

    P_sub = np.array([P[i] for i, _ in assignment])
    Q_sub = np.array([Q[j] for _, j in assignment])
    rmsd, R, t = kabsch_rmsd(P_sub, Q_sub)
    aligned = (P_sub - P_sub.mean(axis=0)) @ R.T + Q_sub.mean(axis=0)
    return rmsd, aligned, assignment


def _subset_alignment(P_large, P_small, n_match, reverse=False):
    """Find best n_match points from P_large to match P_small."""
    from itertools import combinations
    n_large = P_large.shape[0]

    if n_large <= 8:
        best_rmsd = 999.0
        best_subset = None
        for subset in combinations(range(n_large), n_match):
            P_sub = P_large[list(subset)]
            rmsd, _, _ = kabsch_rmsd(P_sub, P_small)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_subset = list(subset)
    else:
        # Greedy for large sets
        best_rmsd, _, assignment = _greedy_alignment(P_large, P_small)
        best_subset = [a[0] for a in assignment]

    if best_subset is None:
        return 999.0, P_large, []

    P_sub = P_large[best_subset]
    rmsd, R, t = kabsch_rmsd(P_sub, P_small)
    aligned = (P_sub - P_sub.mean(axis=0)) @ R.T + P_small.mean(axis=0)

    if reverse:
        assignment = [(j, i) for i, j in enumerate(best_subset)]
    else:
        assignment = [(i, j) for j, i in enumerate(best_subset)]
    return rmsd, aligned, assignment


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: STRAIN ENERGY
# ═══════════════════════════════════════════════════════════════════════════

def conformer_strain(energy_binding_kcal, energy_min_kcal):
    """Strain energy = E(binding conformer) - E(global minimum).

    Returns (strain_kcal, strain_kJ).
    """
    strain = max(0.0, energy_binding_kcal - energy_min_kcal)
    return strain, strain * 4.184


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: PREORGANIZATION SCORE
# ═══════════════════════════════════════════════════════════════════════════

def preorganization_score(preorg_rmsd, tolerance_A=0.5):
    """Score how preorganized the free ligand is for binding.

    preorg_rmsd: RMSD between min-E conformer donors and ideal pocket.
    tolerance_A: characteristic decay length.

    Returns 0-1 (1 = perfectly preorganized, e.g. macrocycle).
    """
    return math.exp(-preorg_rmsd / tolerance_A)


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATED 3D SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_3d(
    smiles: str,
    target: str,
    donor_subtypes: list,
    ideal_positions: np.ndarray = None,
    geometry: str = "auto",
    n_conformers: int = 50,
    random_seed: int = 42,
) -> Binding3DResult:
    """Full 3D binding geometry assessment.

    Args:
        smiles:          Ligand SMILES
        target:          Metal ion (e.g. "Cu2+")
        donor_subtypes:  Expected donor subtypes
        ideal_positions: Nx3 ideal donor positions (from ideal_pocket).
                         If None, computed from target + donor_subtypes.
        geometry:        Coordination geometry
        n_conformers:    Number of conformers to sample
        random_seed:     Reproducibility seed

    Returns:
        Binding3DResult with full 3D assessment
    """
    result = Binding3DResult(
        smiles=smiles,
        target=target,
        n_donors_expected=len(donor_subtypes),
    )

    # Compute ideal positions if not provided
    if ideal_positions is None:
        from core.ideal_pocket import compute_ideal_pocket
        pocket = compute_ideal_pocket(target, donor_subtypes, geometry)
        ideal_positions = np.array([d.position_A for d in pocket.donors])

    # Step 1: Generate conformers
    mol, energies = generate_conformers(smiles, n_conformers, random_seed)
    if mol is None or len(energies) == 0:
        return result

    result.n_conformers_generated = len(energies)

    # Find global minimum
    min_E_idx = int(np.argmin(energies))
    result.min_energy_conf_id = min_E_idx
    result.min_energy_kcal = energies[min_E_idx]

    # Step 2-4: For each conformer, find donors and align
    conformer_results = []

    for cid in range(len(energies)):
        # Find donors in this conformer
        donors = find_donors_3d(mol, cid, donor_subtypes)
        if len(donors) == 0:
            continue

        donor_pos = np.array([d.position for d in donors])

        # Align to ideal pocket
        rmsd, aligned, assignment = align_donors_to_ideal(donor_pos, ideal_positions)

        cr = ConformerResult(
            conf_id=cid,
            energy_kcal=energies[cid],
            donor_positions=donors,
            rmsd_to_ideal=rmsd,
            aligned_positions=aligned,
        )
        conformer_results.append(cr)

    result.n_conformers_valid = len(conformer_results)
    result.conformers = conformer_results

    if not conformer_results:
        return result

    # Find best binding conformer (lowest RMSD to ideal)
    best_cr = min(conformer_results, key=lambda c: c.rmsd_to_ideal)
    result.best_conf_id = best_cr.conf_id
    result.best_rmsd_A = round(best_cr.rmsd_to_ideal, 3)
    result.best_energy_kcal = best_cr.energy_kcal

    # Step 5: Strain energy
    strain_kcal, strain_kJ = conformer_strain(best_cr.energy_kcal,
                                               result.min_energy_kcal)
    result.strain_energy_kcal = round(strain_kcal, 2)
    result.strain_energy_kJ = round(strain_kJ, 2)

    # Step 6: Preorganization
    # Find RMSD of min-E conformer to ideal
    min_E_cr = next((c for c in conformer_results
                     if c.conf_id == min_E_idx), None)
    if min_E_cr:
        result.preorganization_rmsd = round(min_E_cr.rmsd_to_ideal, 3)
        result.preorganization_score = round(
            preorganization_score(min_E_cr.rmsd_to_ideal), 3)
    else:
        result.preorganization_rmsd = 999.0
        result.preorganization_score = 0.0

    # Per-donor deviations (from best conformer alignment)
    if len(best_cr.aligned_positions) > 0 and len(ideal_positions) > 0:
        n = min(len(best_cr.aligned_positions), len(ideal_positions))
        devs = [np.linalg.norm(best_cr.aligned_positions[i] - ideal_positions[i])
                for i in range(n)]
        result.per_donor_deviation_A = [round(d, 3) for d in devs]
        result.mean_deviation_A = round(np.mean(devs), 3)
        result.max_deviation_A = round(max(devs), 3)

    # Donor info
    result.n_donors_found = len(best_cr.donor_positions)
    result.donor_subtypes_found = [d.donor_subtype for d in best_cr.donor_positions]

    # Fidelity score: combines geometry match + strain penalty
    geo_fidelity = math.exp(-result.best_rmsd_A / 0.5)  # decay at 0.5 Å
    strain_penalty = math.exp(-strain_kJ / 20.0)  # decay at 20 kJ/mol
    donor_match = min(1.0, result.n_donors_found / max(result.n_donors_expected, 1))
    result.fidelity_score = round(geo_fidelity * strain_penalty * donor_match, 3)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BATCH SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_3d_batch(smiles_list, target, donor_subtypes, geometry="auto",
                   n_conformers=30, random_seed=42):
    """Score a batch of SMILES for 3D binding geometry.

    Returns list of Binding3DResult, sorted by fidelity (best first).
    """
    ideal = None
    # Compute ideal once
    from core.ideal_pocket import compute_ideal_pocket
    pocket = compute_ideal_pocket(target, donor_subtypes, geometry)
    ideal = np.array([d.position_A for d in pocket.donors])

    results = []
    for smi in smiles_list:
        try:
            r = score_3d(smi, target, donor_subtypes,
                         ideal_positions=ideal,
                         n_conformers=n_conformers,
                         random_seed=random_seed)
            results.append(r)
        except Exception:
            results.append(Binding3DResult(smiles=smi, target=target))

    results.sort(key=lambda r: r.fidelity_score, reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_3d_result(r):
    """Pretty-print a 3D binding assessment."""
    print()
    print(f"  MABE 3D Binding Assessment — {r.target}")
    print(f"  SMILES: {r.smiles[:60]}")
    print(f"  Conformers: {r.n_conformers_generated} generated, "
          f"{r.n_conformers_valid} valid")
    print(f"  Donors: {r.n_donors_found}/{r.n_donors_expected} found "
          f"({', '.join(r.donor_subtypes_found)})")
    print()
    print(f"  ── Geometry ──")
    print(f"  Best RMSD to ideal:  {r.best_rmsd_A:.3f} Å")
    print(f"  Mean donor deviation: {r.mean_deviation_A:.3f} Å")
    print(f"  Max donor deviation:  {r.max_deviation_A:.3f} Å")
    if r.per_donor_deviation_A:
        print(f"  Per-donor: {r.per_donor_deviation_A}")
    print()
    print(f"  ── Energetics ──")
    print(f"  Min-E conformer:     {r.min_energy_kcal:.1f} kcal/mol")
    print(f"  Binding conformer:   {r.best_energy_kcal:.1f} kcal/mol")
    print(f"  Strain energy:       {r.strain_energy_kcal:.1f} kcal/mol "
          f"({r.strain_energy_kJ:.1f} kJ/mol)")
    print()
    print(f"  ── Preorganization ──")
    print(f"  Free-ligand RMSD to ideal: {r.preorganization_rmsd:.3f} Å")
    print(f"  Preorganization score:     {r.preorganization_score:.3f}")
    print()
    print(f"  ── Combined ──")
    print(f"  Fidelity score:  {r.fidelity_score:.3f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 70)
    print("MABE 3D Conformer Pipeline — Self-Test")
    print("=" * 70)

    # Test 1: Ethylenediamine (en) — simple bidentate
    print("\n--- Ethylenediamine (en) for Cu2+ ---")
    r1 = score_3d("NCCN", "Cu2+", ["N_amine", "N_amine"],
                   geometry="square_planar", n_conformers=30)
    print_3d_result(r1)

    # Test 2: EDTA — hexadentate, should be well-preorganized
    print("--- EDTA for Cu2+ ---")
    r2 = score_3d("OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
                   "Cu2+", ["N_amine", "N_amine", "O_carboxylate",
                            "O_carboxylate", "O_carboxylate", "O_carboxylate"],
                   geometry="octahedral", n_conformers=30)
    print_3d_result(r2)

    # Test 3: Cyclam — macrocyclic, should be highly preorganized
    print("--- Cyclam (1,4,8,11-tetraazacyclotetradecane) for Ni2+ ---")
    r3 = score_3d("C1CNCCNCCNCCNC1",
                   "Ni2+", ["N_amine"] * 4,
                   geometry="square_planar", n_conformers=30)
    print_3d_result(r3)

    # Test 4: Simple monodentate (ammonia analog) — trivial
    print("--- Methylamine for Cu2+ (monodentate) ---")
    r4 = score_3d("CN", "Cu2+", ["N_amine"],
                   geometry="linear", n_conformers=10)
    print_3d_result(r4)
