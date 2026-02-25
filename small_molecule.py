"""
realization_ranker/geometric_fidelity/small_molecule.py

Score how well a small molecule scaffold can reproduce the required
donor geometry from an InteractionGeometrySpec.

Physics basis:
  - MMFF94 force field conformer generation (RDKit)
  - RMSD between required donor positions and achievable positions
  - Gaussian decay: fidelity = exp(-RMSD² / (2σ²))

For chelators, crown ethers, porphyrins: same physics, different
candidate libraries. The scoring function is universal.
"""

import math
from typing import Optional
import numpy as np

from ..epistemic import EpistemicScore, EpistemicBasis

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def score_geometric_fidelity_small_molecule(
    required_donor_positions: np.ndarray,
    candidate_smiles: Optional[str] = None,
    candidate_conformer_coords: Optional[np.ndarray] = None,
    donor_atom_indices: Optional[list[int]] = None,
    tolerance_angstrom: float = 0.3,
    n_conformers: int = 50,
) -> EpistemicScore:
    """
    Score geometric fidelity of a small molecule realization.

    Parameters
    ----------
    required_donor_positions : np.ndarray
        Shape (n_donors, 3) — required donor atom positions in Angstroms
        from Layer 2 InteractionGeometrySpec.
    candidate_smiles : str, optional
        SMILES of candidate molecule. If provided, conformers are generated.
    candidate_conformer_coords : np.ndarray, optional
        Pre-computed conformer donor coordinates, shape (n_conf, n_donors, 3).
        Use this if conformers were already generated upstream.
    donor_atom_indices : list[int], optional
        Atom indices in the molecule corresponding to donor atoms.
        Required if candidate_smiles is provided.
    tolerance_angstrom : float
        σ in the Gaussian decay. Default 0.3 Å for coordination chemistry,
        use 0.5 Å for host-guest.
    n_conformers : int
        Number of conformers to generate for RMSD sampling.

    Returns
    -------
    EpistemicScore with fidelity value and physics provenance.
    """

    if candidate_conformer_coords is not None:
        # Pre-computed: find best RMSD across conformers
        best_rmsd = _best_rmsd_from_coords(
            required_donor_positions, candidate_conformer_coords
        )
    elif candidate_smiles is not None and HAS_RDKIT:
        best_rmsd = _best_rmsd_from_smiles(
            required_donor_positions, candidate_smiles,
            donor_atom_indices, n_conformers
        )
    else:
        # No structure available — return heuristic based on donor count
        n_donors = len(required_donor_positions)
        # Small molecules with few donors can usually achieve required geometry
        heuristic_value = max(0.3, 1.0 - 0.1 * n_donors)
        return EpistemicScore(
            value=heuristic_value,
            basis=EpistemicBasis.HEURISTIC_ESTIMATE,
            note=(
                "No candidate structure provided. Heuristic based on donor count. "
                "Best guess, more data required"
            ),
            uncertainty=0.3,
        )

    # Gaussian decay: tight tolerance → sharp penalty for misfit
    fidelity = math.exp(-(best_rmsd ** 2) / (2 * tolerance_angstrom ** 2))

    return EpistemicScore(
        value=fidelity,
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation="fidelity = exp(-RMSD² / (2σ²)), σ={:.2f} Å".format(tolerance_angstrom),
        data_source="RDKit MMFF94 conformer ensemble" if candidate_smiles else "pre-computed coordinates",
        uncertainty=0.05,  # Conformer sampling uncertainty
    )


def _best_rmsd_from_coords(
    required: np.ndarray,
    conformer_donors: np.ndarray,
) -> float:
    """Find minimum RMSD across pre-computed conformers."""
    best = float("inf")
    for conf_donors in conformer_donors:
        rmsd = _kabsch_rmsd(required, conf_donors)
        if rmsd < best:
            best = rmsd
    return best


def _best_rmsd_from_smiles(
    required: np.ndarray,
    smiles: str,
    donor_indices: list[int],
    n_conformers: int,
) -> float:
    """Generate conformers from SMILES, extract donor positions, find best RMSD."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("inf")

    mol = Chem.AddHs(mol)

    # Embed conformers
    params = AllChem.ETKDGv3()
    params.numThreads = 0  # Use all available
    params.randomSeed = 42
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)

    if len(conf_ids) == 0:
        return float("inf")

    # Optimize with MMFF94
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)

    best = float("inf")
    for conf_id in conf_ids:
        conf = mol.GetConformer(conf_id)
        # Extract donor atom positions
        donor_pos = np.array([
            list(conf.GetAtomPosition(idx)) for idx in donor_indices
        ])
        rmsd = _kabsch_rmsd(required, donor_pos)
        if rmsd < best:
            best = rmsd

    return best


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Optimal RMSD after rigid-body alignment (Kabsch algorithm).

    This finds the rotation + translation that minimizes RMSD between
    two point sets, giving the true geometric similarity regardless of
    orientation.
    """
    if P.shape != Q.shape:
        return float("inf")

    n = P.shape[0]
    if n == 0:
        return 0.0

    # Center both point sets
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    # Covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and compute RMSD
    P_rotated = (R @ P_centered.T).T
    diff = P_rotated - Q_centered
    rmsd = math.sqrt((diff ** 2).sum() / n)

    return rmsd


def score_class_geometric_fidelity(
    realization_type: str,
    required_cavity_nm: float,
    required_donor_count: int,
    required_donor_types: set[str],
) -> EpistemicScore:
    """
    Class-level geometric fidelity estimate when no specific candidate is known.

    Uses physics-based heuristics:
    - How close is the required cavity to the material's natural cavity range?
    - Can the material present the required number of donors simultaneously?
    """
    from .material_geometry import (
        NATURAL_CAVITY_RANGE_NM,
        MAX_SIMULTANEOUS_DONORS,
    )

    # Cavity size match: Gaussian centered on natural range midpoint
    cav_range = NATURAL_CAVITY_RANGE_NM.get(realization_type, (0.1, 1.0))
    mid = (cav_range[0] + cav_range[1]) / 2
    span = (cav_range[1] - cav_range[0]) / 2
    if span <= 0:
        span = 0.1
    cav_score = math.exp(-((required_cavity_nm - mid) ** 2) / (2 * span ** 2))

    # Donor count feasibility
    max_donors = MAX_SIMULTANEOUS_DONORS.get(realization_type, 4)
    if required_donor_count <= max_donors:
        donor_score = 1.0
    else:
        # Penalty for exceeding natural coordination capacity
        donor_score = max(0.1, max_donors / required_donor_count)

    fidelity = 0.6 * cav_score + 0.4 * donor_score

    return EpistemicScore(
        value=min(1.0, fidelity),
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation=(
            "fidelity = 0.6 × exp(-(d_cav - d_natural)²/(2σ²)) "
            "+ 0.4 × min(1, max_donors/n_required)"
        ),
        uncertainty=0.15,
        note="Class-level estimate; improves with specific candidate structure",
    )