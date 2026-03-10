"""
core/small_molecule_target.py — Guest Pharmacophore → Pocket Geometry

Takes a small-molecule SMILES and produces an InteractionGeometrySpec
describing the COMPLEMENTARY binding pocket: where the pocket must place
H-bond donors to match guest acceptors, acceptors to match guest donors,
aromatic walls to match guest aromatics, and hydrophobic surfaces to
bury guest nonpolar area.

Physics basis:
    - RDKit pharmacophore perception for donor/acceptor/aromatic/hydrophobic
    - 3D conformer embedding for spatial coordinates
    - Complementarity inversion: guest donor → pocket acceptor, etc.
    - Cavity sizing from guest volume + 55% packing rule (Rebek)

Does NOT:
    - Use any fitted parameters against target-domain data
    - Assume a specific scaffold or material system
    - Collapse Layer 2 into Layer 3 (output is realization-agnostic)
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski
    from rdkit.Chem import rdMolTransforms
    _RDKIT = True
except ImportError:
    _RDKIT = False

from realization.models import (
    InteractionGeometrySpec,
    DonorPosition,
    CavityShape,
    CavityDimensions,
    HydrophobicSurface,
    HBondSpec,
    Solvent,
    ApplicationContext,
    ScaleClass,
    ExclusionSpec,
)


# ═══════════════════════════════════════════════════════════════════════════
# PHARMACOPHORE FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PharmacophoreFeature:
    """One pharmacophore feature on the guest molecule."""
    feature_type: str          # "hb_donor", "hb_acceptor", "aromatic", "hydrophobic"
    atom_indices: list = field(default_factory=list)
    position_A: tuple = (0.0, 0.0, 0.0)  # 3D coords in Angstrom
    strength: str = "medium"   # "strong", "medium", "weak"
    subtype: str = ""          # e.g. "quinone_carbonyl", "amine_NH", "phenyl"


@dataclass
class GuestPharmacophore:
    """Complete pharmacophore description of a guest molecule."""
    smiles: str
    name: str = ""
    features: list = field(default_factory=list)  # list[PharmacophoreFeature]

    # Aggregate counts
    n_hb_donors: int = 0
    n_hb_acceptors: int = 0
    n_aromatic_rings: int = 0
    n_hydrophobic_centers: int = 0

    # Guest geometry
    volume_A3: float = 0.0
    max_dimension_A: float = 0.0   # longest axis
    min_dimension_A: float = 0.0   # shortest axis
    aspect_ratio: float = 1.0

    # Computed properties
    mw: float = 0.0
    logP: float = 0.0
    tpsa: float = 0.0
    sasa_nonpolar_A2: float = 0.0
    sasa_polar_A2: float = 0.0
    rotatable_bonds: int = 0

    @property
    def is_too_large_for_cavity(self) -> bool:
        """Guests >600 Å³ are unlikely to fit in simple host cavities."""
        return self.volume_A3 > 600.0

    @property
    def dominant_interaction(self) -> str:
        """Rough guide to which physics term dominates binding."""
        if self.logP > 3.0 and self.n_hb_donors + self.n_hb_acceptors < 3:
            return "hydrophobic"
        if self.n_hb_donors + self.n_hb_acceptors >= 4:
            return "h_bond"
        if self.n_aromatic_rings >= 2:
            return "pi_stacking"
        return "mixed"


def analyze_guest(smiles: str, name: str = "") -> GuestPharmacophore:
    """Extract pharmacophore features from guest SMILES.

    Returns GuestPharmacophore with 3D feature positions.
    Requires RDKit.
    """
    if not _RDKIT:
        raise RuntimeError("RDKit required for pharmacophore analysis")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol_h = Chem.AddHs(mol)
    # Embed 3D — try multiple seeds for best conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol_h, params)
    if cid < 0:
        # Fallback: less strict embedding
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol_h, params)
    if cid >= 0:
        AllChem.MMFFOptimizeMolecule(mol_h, confId=cid)

    conf = mol_h.GetConformer(0) if mol_h.GetNumConformers() > 0 else None
    pharma = GuestPharmacophore(smiles=smiles, name=name or smiles)

    # ── Molecular properties ──
    pharma.mw = Descriptors.ExactMolWt(mol)
    pharma.logP = Descriptors.MolLogP(mol)
    pharma.tpsa = Descriptors.TPSA(mol)
    pharma.rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    # Volume from Van der Waals
    if conf is not None:
        try:
            pharma.volume_A3 = AllChem.ComputeMolVolume(mol_h, confId=0)
        except Exception:
            pharma.volume_A3 = pharma.mw * 0.98  # rough fallback

    # ── H-bond donors (NH, OH on guest → need acceptors in pocket) ──
    hb_donors = _find_hb_donors(mol_h, conf)
    pharma.features.extend(hb_donors)
    pharma.n_hb_donors = len(hb_donors)

    # ── H-bond acceptors (C=O, N:, on guest → need donors in pocket) ──
    hb_acceptors = _find_hb_acceptors(mol_h, conf)
    pharma.features.extend(hb_acceptors)
    pharma.n_hb_acceptors = len(hb_acceptors)

    # ── Aromatic rings ──
    aromatics = _find_aromatic_rings(mol_h, conf)
    pharma.features.extend(aromatics)
    pharma.n_aromatic_rings = len(aromatics)

    # ── Hydrophobic centers ──
    hydrophobics = _find_hydrophobic_centers(mol_h, conf)
    pharma.features.extend(hydrophobics)
    pharma.n_hydrophobic_centers = len(hydrophobics)

    # ── Guest shape ──
    if conf is not None:
        pharma.max_dimension_A, pharma.min_dimension_A = _guest_dimensions(mol_h, conf)
        pharma.aspect_ratio = (
            pharma.max_dimension_A / pharma.min_dimension_A
            if pharma.min_dimension_A > 0.1 else 1.0
        )

    # ── SASA ──
    try:
        from core.guest_compute import compute_guest_properties
        props = compute_guest_properties(smiles)
        pharma.sasa_nonpolar_A2 = props.get("guest_sasa_nonpolar_A2", 0.0)
        pharma.sasa_polar_A2 = props.get("guest_sasa_polar_A2", 0.0)
    except Exception:
        pass

    return pharma


# ═══════════════════════════════════════════════════════════════════════════
# PHARMACOPHORE → INTERACTION GEOMETRY SPEC (pocket inversion)
# ═══════════════════════════════════════════════════════════════════════════

def guest_to_pocket_spec(
    pharma: GuestPharmacophore,
    application: str = "research",
    scale: str = "µmol",
    solvent: str = "aqueous",
    pH: float = 7.0,
    exclude_species: list = None,
) -> InteractionGeometrySpec:
    """Convert guest pharmacophore to complementary pocket geometry.

    This is the Layer 1 → Layer 2 bridge for small-molecule guests.

    The pocket spec inverts the guest's features:
        guest H-bond donor  → pocket H-bond acceptor (DonorPosition with O/N)
        guest H-bond acceptor → pocket H-bond donor (DonorPosition with N-H/O-H)
        guest aromatic ring → pocket aromatic wall (HydrophobicSurface with π character)
        guest hydrophobic → pocket hydrophobic wall (HydrophobicSurface)

    Cavity sizing follows Rebek's 55% rule: V_cavity ≈ V_guest / 0.55
    (optimal packing coefficient for inclusion complexes).
    """

    # ── Cavity dimensions ──
    # Rebek rule: optimal packing ~0.55 for organic guests in cavities
    optimal_packing = 0.55
    ideal_cavity_volume = pharma.volume_A3 / optimal_packing if pharma.volume_A3 > 0 else 300.0

    # Aperture: guest must enter. Use min dimension + 1 Å clearance.
    aperture = pharma.min_dimension_A + 1.0 if pharma.min_dimension_A > 0 else 6.0

    # Depth: from aspect ratio
    depth = pharma.max_dimension_A * 0.8 if pharma.max_dimension_A > 0 else 8.0

    max_diam = (ideal_cavity_volume * 6 / math.pi) ** (1/3) * 2  # sphere equiv
    if pharma.aspect_ratio > 1.5:
        cavity_shape = CavityShape.CHANNEL
    elif pharma.aspect_ratio > 1.2:
        cavity_shape = CavityShape.CONE
    else:
        cavity_shape = CavityShape.SPHERE

    cavity_dims = CavityDimensions(
        volume_A3=ideal_cavity_volume,
        aperture_A=aperture,
        depth_A=depth,
        max_internal_diameter_A=max_diam,
        aspect_ratio=pharma.aspect_ratio,
    )

    # ── Donor positions (COMPLEMENTARY to guest features) ──
    donor_positions = []
    for feat in pharma.features:
        if feat.feature_type == "hb_donor":
            # Guest donates H → pocket needs acceptor atom (O or N with lone pair)
            # Position: along the H-bond vector, ~2.8 Å from guest donor
            pocket_pos = _offset_position(feat.position_A, offset_A=2.8)
            donor_positions.append(DonorPosition(
                atom_type="O",
                coordination_role="terminal",
                position_vector_A=pocket_pos,
                tolerance_A=0.5,
                required_hybridization="sp2" if feat.subtype in ("amine_NH",) else "any",
                charge_state=-0.3,
            ))

        elif feat.feature_type == "hb_acceptor":
            # Guest accepts H → pocket needs donor atom (N-H or O-H)
            pocket_pos = _offset_position(feat.position_A, offset_A=2.8)
            donor_positions.append(DonorPosition(
                atom_type="N",
                coordination_role="terminal",
                position_vector_A=pocket_pos,
                tolerance_A=0.5,
                required_hybridization="any",
                charge_state=0.2,
            ))

    # ── Hydrophobic surfaces ──
    hydrophobic_surfaces = []
    for feat in pharma.features:
        if feat.feature_type == "aromatic":
            # Guest aromatic → pocket aromatic wall for π-stacking
            # Optimal π-π distance: 3.5 Å face-to-face
            wall_center = _offset_position(feat.position_A, offset_A=3.5)
            # Approximate ring area: ~24 Å² for 6-membered ring
            hydrophobic_surfaces.append(HydrophobicSurface(
                center_A=wall_center,
                area_A2=24.0,
                normal_vector=(0.0, 0.0, 1.0),  # placeholder normal
            ))

        elif feat.feature_type == "hydrophobic":
            # Guest hydrophobic center → pocket hydrophobic wall
            wall_center = _offset_position(feat.position_A, offset_A=3.8)
            hydrophobic_surfaces.append(HydrophobicSurface(
                center_A=wall_center,
                area_A2=max(12.0, len(feat.atom_indices) * 6.0),
                normal_vector=(0.0, 1.0, 0.0),  # placeholder
            ))

    # ── H-bond network spec ──
    pocket_donor_positions = [
        dp.position_vector_A for dp in donor_positions
        if dp.charge_state > 0  # pocket donors (complement guest acceptors)
    ]
    pocket_acceptor_positions = [
        dp.position_vector_A for dp in donor_positions
        if dp.charge_state < 0  # pocket acceptors (complement guest donors)
    ]
    h_bond_network = HBondSpec(
        donors=pocket_donor_positions,
        acceptors=pocket_acceptor_positions,
        required_geometry="any",
    ) if pocket_donor_positions or pocket_acceptor_positions else None

    # ── Exclusion specs ──
    exclusions = []
    if exclude_species:
        for species in exclude_species:
            exclusions.append(ExclusionSpec(
                species=species,
                max_allowed_affinity_kJ_mol=-10.0,
                exclusion_mechanism="geometry",
            ))

    # ── Rigidity ──
    # More rotatable bonds → more preorganization needed
    if pharma.rotatable_bonds <= 2:
        rigidity = "semi-rigid"
        rmsd_budget = 0.5
    elif pharma.rotatable_bonds <= 5:
        rigidity = "semi-rigid"
        rmsd_budget = 1.0
    else:
        rigidity = "semi-flexible"
        rmsd_budget = 1.5

    # Conformational penalty: ~2.5 kJ/mol per frozen rotor (MABE convention)
    conf_penalty_budget = pharma.rotatable_bonds * 2.5

    # ── Application / Scale ──
    app_map = {
        "research": ApplicationContext.RESEARCH,
        "remediation": ApplicationContext.REMEDIATION,
        "diagnostic": ApplicationContext.DIAGNOSTIC,
        "separation": ApplicationContext.SEPARATION,
    }
    scale_map = {
        "nmol": ScaleClass.NMOL, "µmol": ScaleClass.UMOL,
        "mmol": ScaleClass.MMOL, "mol": ScaleClass.MOL,
    }

    spec = InteractionGeometrySpec(
        cavity_shape=cavity_shape,
        cavity_dimensions=cavity_dims,
        donor_positions=donor_positions,
        hydrophobic_surfaces=hydrophobic_surfaces,
        h_bond_network=h_bond_network,
        rigidity_requirement=rigidity,
        max_backbone_rmsd_A=rmsd_budget,
        conformational_penalty_budget_kJ_mol=conf_penalty_budget,
        pocket_scale_nm=pharma.max_dimension_A / 10.0 if pharma.max_dimension_A > 0 else 0.8,
        multivalency=1,
        must_exclude=exclusions,
        pH_range=(pH - 1.0, pH + 1.0),
        solvent=Solvent.AQUEOUS if solvent == "aqueous" else Solvent.ORGANIC,
        ionic_strength_M=0.1,
        target_application=app_map.get(application, ApplicationContext.RESEARCH),
        required_scale=scale_map.get(scale, ScaleClass.UMOL),
    )

    return spec


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: SMILES → HOST SCREENING via unified_scorer_v2
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HostScreenResult:
    """Scoring result for one host-guest pair."""
    host_key: str
    host_display: str
    log_Ka_pred: float
    dg_total_kJ: float
    dg_hydrophobic: float = 0.0
    dg_cavity_dehydration: float = 0.0
    dg_hbond: float = 0.0
    dg_pi: float = 0.0
    dg_size_mismatch: float = 0.0
    dg_conf_entropy: float = 0.0
    dg_shape: float = 0.0
    packing_coefficient: float = 0.0
    feasibility_note: str = ""


def screen_hosts(
    smiles: str,
    hosts: list = None,
    name: str = "",
) -> list:
    """Score a guest SMILES against all available host cavities.

    Args:
        smiles: Guest SMILES
        hosts: List of host keys (default: all in HOST_DB)
        name: Display name for guest

    Returns:
        List[HostScreenResult] sorted by log_Ka descending.
    """
    from core.auto_descriptor import from_smiles
    from core.unified_scorer_v2 import predict
    from knowledge.hg_dataset import HOST_DB

    if hosts is None:
        hosts = list(HOST_DB.keys())

    results = []
    for h in hosts:
        try:
            uc = from_smiles(smiles, host=h)
            uc.name = f"{name or smiles}:{h}"
            pred = predict(uc)

            # Size feasibility check
            note = ""
            if uc.packing_coefficient > 1.0:
                note = f"guest too large (packing={uc.packing_coefficient:.2f})"
            elif uc.packing_coefficient > 0 and uc.packing_coefficient < 0.3:
                note = f"guest too small (packing={uc.packing_coefficient:.2f})"

            results.append(HostScreenResult(
                host_key=h,
                host_display=HOST_DB[h].get("full_name", h),
                log_Ka_pred=pred.log_Ka_pred,
                dg_total_kJ=pred.dg_total_kj,
                dg_hydrophobic=pred.dg_hydrophobic,
                dg_cavity_dehydration=pred.dg_cavity_dehydration,
                dg_hbond=pred.dg_hbond,
                dg_pi=pred.dg_pi,
                dg_size_mismatch=pred.dg_size_mismatch,
                dg_conf_entropy=pred.dg_conf_entropy,
                dg_shape=pred.dg_shape,
                packing_coefficient=uc.packing_coefficient,
                feasibility_note=note,
            ))
        except Exception as e:
            results.append(HostScreenResult(
                host_key=h,
                host_display=h,
                log_Ka_pred=float("-inf"),
                dg_total_kJ=0.0,
                feasibility_note=f"error: {e}",
            ))

    results.sort(key=lambda r: r.log_Ka_pred, reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL: FEATURE FINDERS
# ═══════════════════════════════════════════════════════════════════════════

def _find_hb_donors(mol_h, conf):
    """Find H-bond donor groups on the molecule."""
    features = []
    # SMARTS for H-bond donors: N-H, O-H
    patterns = [
        ("[NX3;H1,H2]", "amine_NH"),
        ("[nH]", "aromatic_NH"),
        ("[OH]", "hydroxyl_OH"),
        ("[NH]C(=O)", "amide_NH"),
    ]
    for smarts, subtype in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol_h.GetSubstructMatches(patt)
        for match in matches:
            pos = _get_atom_position(conf, match[0]) if conf else (0, 0, 0)
            features.append(PharmacophoreFeature(
                feature_type="hb_donor",
                atom_indices=list(match),
                position_A=pos,
                strength="strong" if subtype in ("amide_NH",) else "medium",
                subtype=subtype,
            ))
    return features


def _find_hb_acceptors(mol_h, conf):
    """Find H-bond acceptor groups on the molecule."""
    features = []
    patterns = [
        ("[C]=[O]", "carbonyl_O"),
        ("[#7;!H1;!H2;!H3]", "lone_pair_N"),  # N with lone pair (not NH)
        ("[OX2H0]", "ether_O"),
        ("[SX2H0]", "thioether_S"),
    ]
    for smarts, subtype in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol_h.GetSubstructMatches(patt)
        for match in matches:
            # For C=O, the acceptor is the O (last atom in match)
            acceptor_idx = match[-1] if subtype == "carbonyl_O" else match[0]
            pos = _get_atom_position(conf, acceptor_idx) if conf else (0, 0, 0)
            features.append(PharmacophoreFeature(
                feature_type="hb_acceptor",
                atom_indices=[acceptor_idx],
                position_A=pos,
                strength="strong" if subtype == "carbonyl_O" else "medium",
                subtype=subtype,
            ))
    return features


def _find_aromatic_rings(mol_h, conf):
    """Find aromatic ring systems."""
    features = []
    ring_info = mol_h.GetRingInfo()
    for ring in ring_info.AtomRings():
        # Check if all atoms in ring are aromatic
        if all(mol_h.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            pos = _ring_centroid(conf, ring) if conf else (0, 0, 0)
            features.append(PharmacophoreFeature(
                feature_type="aromatic",
                atom_indices=list(ring),
                position_A=pos,
                strength="strong" if len(ring) == 6 else "medium",
                subtype=f"aromatic_{len(ring)}ring",
            ))
    return features


def _find_hydrophobic_centers(mol_h, conf):
    """Find hydrophobic (alkyl/lipophilic) regions."""
    features = []
    # Identify contiguous carbon-only chains ≥3 atoms
    patt = Chem.MolFromSmarts("[CX4,CX3]~[CX4,CX3]~[CX4,CX3]")
    if patt is None:
        return features
    matches = mol_h.GetSubstructMatches(patt)
    # Deduplicate overlapping matches
    seen = set()
    for match in matches:
        key = tuple(sorted(match))
        if key in seen:
            continue
        seen.add(key)
        # Only include if all atoms are C (not already counted as aromatic)
        atoms = [mol_h.GetAtomWithIdx(i) for i in match]
        if all(not a.GetIsAromatic() for a in atoms):
            pos = _centroid(conf, list(match)) if conf else (0, 0, 0)
            features.append(PharmacophoreFeature(
                feature_type="hydrophobic",
                atom_indices=list(match),
                position_A=pos,
                strength="medium",
                subtype="alkyl_chain",
            ))
    return features


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_atom_position(conf, idx):
    """Get 3D position of atom from conformer."""
    if conf is None:
        return (0.0, 0.0, 0.0)
    pos = conf.GetAtomPosition(idx)
    return (pos.x, pos.y, pos.z)


def _ring_centroid(conf, ring_indices):
    """Compute centroid of a ring from conformer."""
    if conf is None:
        return (0.0, 0.0, 0.0)
    xs, ys, zs = [], [], []
    for idx in ring_indices:
        pos = conf.GetAtomPosition(idx)
        xs.append(pos.x)
        ys.append(pos.y)
        zs.append(pos.z)
    n = len(ring_indices)
    return (sum(xs)/n, sum(ys)/n, sum(zs)/n)


def _centroid(conf, indices):
    """Compute centroid of atom group."""
    if conf is None or not indices:
        return (0.0, 0.0, 0.0)
    xs, ys, zs = [], [], []
    for idx in indices:
        pos = conf.GetAtomPosition(idx)
        xs.append(pos.x)
        ys.append(pos.y)
        zs.append(pos.z)
    n = len(indices)
    return (sum(xs)/n, sum(ys)/n, sum(zs)/n)


def _guest_dimensions(mol_h, conf):
    """Estimate max and min molecular dimensions from conformer."""
    positions = []
    for i in range(mol_h.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append((pos.x, pos.y, pos.z))
    if len(positions) < 2:
        return 5.0, 5.0

    # Max distance between any two atoms
    max_dist = 0.0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d > max_dist:
                max_dist = d

    # Add VdW radii (~1.7 Å per side)
    max_dim = max_dist + 3.4

    # Min dimension: approximate from volume and max dimension
    # V ≈ π/6 * max * mid * min → min ≈ 6V / (π * max * mid)
    # Approximate mid ≈ sqrt(max * min), so min ≈ sqrt(6V / (π * max))
    if max_dim > 0.1:
        vol = AllChem.ComputeMolVolume(mol_h, confId=0) if conf else 200.0
        min_dim = max(3.0, math.sqrt(6 * vol / (math.pi * max_dim)))
    else:
        min_dim = 5.0

    return max_dim, min_dim


def _offset_position(pos, offset_A=2.8):
    """Offset a position radially outward from origin by offset_A.

    Used to place pocket interaction elements complementary to guest features.
    Direction: radially outward from molecule centroid (assumed at origin).
    """
    x, y, z = pos
    r = math.sqrt(x*x + y*y + z*z)
    if r < 0.01:
        # At origin — offset along z
        return (0.0, 0.0, offset_A)
    scale = (r + offset_A) / r
    return (x * scale, y * scale, z * scale)
