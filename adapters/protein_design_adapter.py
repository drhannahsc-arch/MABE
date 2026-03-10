"""
adapters/protein_design_adapter.py — De Novo Protein Binder Design Adapter

Translates InteractionGeometrySpec into protein design tool inputs:
    RFdiffusion3:  hotspot residues + contig specification
    ProteinMPNN:   backbone PDB → sequence design
    BindCraft:     end-to-end target → binder (spec mode)
    AlphaFold2:    validation spec

Pipeline:
    1. Pocket spec donors → amino acid hotspot residues
       (H-bond acceptor C=O → Asp/Glu; donor NH → Lys/Arg/His; aromatic → Trp/Phe/Tyr)
    2. Donor positions → 3D coordinates for RFdiffusion motif scaffolding
    3. Hydrophobic surfaces → Leu/Ile/Val packing residues
    4. Cavity volume → scaffold topology (helical bundle, β-barrel, TIM barrel)
    5. Generate RFdiffusion contig spec + ProteinMPNN config
    6. Score designed pocket using protein-ligand physics

Physics basis (from Lu et al. Science 2024):
    - First shell: residues that directly contact guest (donor matching)
    - Second shell: residues that position first shell (structural support)
    - Two-shell design validated at <5 nM for PARPi binders
    - Charged residues biased toward surface (Born solvation penalty)

Does NOT:
    - Run RFdiffusion, ProteinMPNN, AlphaFold2, or BindCraft
    - Generate actual protein sequences (that's the tool's job)
    - Require GPU or external API access
    - Fit parameters against protein binding data
"""

import math
import json
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# AMINO ACID → DONOR MAPPING
# ═══════════════════════════════════════════════════════════════════════════
# Maps pocket interaction requirements to amino acid side chains.
# Each donor position in InteractionGeometrySpec maps to residues that
# provide the complementary interaction.

@dataclass(frozen=True)
class HotspotResidue:
    """An amino acid placed at a specific position to create a binding pocket."""
    residue_3: str               # "ASP", "HIS", "TRP", etc.
    residue_1: str               # "D", "H", "W"
    role: str                    # "hb_acceptor", "hb_donor", "aromatic", "hydrophobic", "charged"
    donor_type: str              # what the side chain provides
    position_A: tuple            # (x, y, z) from pocket spec
    shell: int                   # 1 = direct contact, 2 = structural support
    chi_preference: str          # "gauche+", "trans", "any" — preferred rotamer
    notes: str = ""


# Donor type → ranked amino acid choices
# Pocket needs an acceptor (to receive guest H-bond donor) → use Asp/Glu (C=O)
# Pocket needs a donor (to donate to guest H-bond acceptor) → use Lys/Arg/His (NH)
_DONOR_TO_RESIDUES = {
    # Pocket spec says "acceptor" → guest has NH, pocket provides C=O/COO⁻
    "hb_acceptor": [
        ("ASP", "D", "carboxylate_acceptor", "Oδ accepts guest NH/OH. Strong. Short side chain."),
        ("GLU", "E", "carboxylate_acceptor", "Oε accepts guest NH/OH. Longer reach than Asp."),
        ("ASN", "N", "amide_acceptor", "Oδ accepts guest NH. Neutral. Good for non-ionic interactions."),
        ("SER", "S", "hydroxyl_acceptor", "Oγ accepts guest NH. Weak but compact."),
        ("THR", "T", "hydroxyl_acceptor", "Oγ accepts guest NH. Branched, less flexible."),
    ],
    # Pocket spec says "donor" → guest has C=O, pocket provides NH
    "hb_donor": [
        ("ARG", "R", "guanidinium_donor", "Nη donates to guest C=O. Bidentate. Strongest H-bond donor."),
        ("LYS", "K", "amine_donor", "Nζ donates to guest C=O. Long flexible side chain."),
        ("HIS", "H", "imidazole_donor", "Nε/Nδ donates to guest C=O. pH-switchable."),
        ("ASN", "N", "amide_donor", "Nδ donates to guest C=O. Dual donor/acceptor."),
        ("GLN", "Q", "amide_donor", "Nε donates to guest C=O. Longer than Asn."),
        ("TRP", "W", "indole_donor", "Nε donates to guest C=O. Also provides aromatic stacking."),
    ],
    # Aromatic stacking with guest
    "aromatic_wall": [
        ("TRP", "W", "indole_pi", "Largest aromatic side chain. Best for π-stacking + CH-π."),
        ("PHE", "F", "phenyl_pi", "Phenyl ring. Good π-stacking. Hydrophobic."),
        ("TYR", "Y", "phenol_pi", "Phenyl + OH. π-stacking + H-bond capability."),
    ],
    # Hydrophobic packing
    "hydrophobic_wall": [
        ("LEU", "L", "alkyl", "Isobutyl. Best hydrophobic packing residue."),
        ("ILE", "I", "alkyl", "Sec-butyl. Branched. Good cavity wall."),
        ("VAL", "V", "alkyl", "Isopropyl. Small hydrophobic. Tight packing."),
        ("MET", "M", "thioether", "Flexible. Good for adaptive binding."),
        ("ALA", "A", "methyl", "Smallest hydrophobic. Fine-tuning only."),
    ],
    # Metal coordination (if pocket spec has metal requirement)
    "metal_coord": [
        ("HIS", "H", "imidazole_N", "Nδ/Nε coordinates metal. Most common in metalloenzymes."),
        ("CYS", "C", "thiolate_S", "Sγ coordinates soft metals (Zn, Cu, Fe). Strong."),
        ("ASP", "D", "carboxylate_O", "Oδ coordinates hard metals (Fe³⁺, Mn²⁺). Bidentate possible."),
        ("GLU", "E", "carboxylate_O", "Oε coordinates hard metals. Longer reach."),
    ],
    # Charge-transfer (for electron-poor guests like quinones)
    "charge_transfer": [
        ("TRP", "W", "indole_ct", "Electron-rich indole → charge-transfer with quinone."),
        ("TYR", "Y", "phenol_ct", "Electron-rich phenol → moderate CT."),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD TOPOLOGY SELECTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScaffoldTopology:
    """A protein fold topology for hosting a designed pocket."""
    name: str
    description: str
    pocket_volume_range_A3: tuple  # (min, max) cavity volume this fold can host
    n_helices: int                 # 0 for β-structures
    n_strands: int                 # 0 for all-α
    typical_size_aa: int           # approximate total residues
    preorganization: float         # 0-1, higher = more rigid pocket
    designability: float           # 0-1, how well RFdiffusion handles this fold
    notes: str = ""


SCAFFOLD_TOPOLOGIES = [
    ScaffoldTopology(
        "4-helix_bundle", "Parametric 4-helix bundle (Lu et al. 2024)",
        (100, 800), 4, 0, 120, 0.85, 0.95,
        "Best validated. Lu et al. PARPi binder. RFdiffusion excels at helical bundles.",
    ),
    ScaffoldTopology(
        "3-helix_bundle", "Compact 3-helix bundle",
        (50, 400), 3, 0, 80, 0.80, 0.90,
        "Smaller pocket. Good for small-molecule guests.",
    ),
    ScaffoldTopology(
        "TIM_barrel", "TIM barrel (8 αβ units)",
        (200, 2000), 8, 8, 250, 0.90, 0.70,
        "Large central cavity. Natural enzyme fold. Harder to design de novo.",
    ),
    ScaffoldTopology(
        "beta_barrel", "β-barrel (lipocalin-like)",
        (300, 1500), 0, 10, 170, 0.75, 0.65,
        "Calyx-shaped pocket. Natural small-molecule carrier fold.",
    ),
    ScaffoldTopology(
        "NTF2_like", "NTF2-like cone (Baker lab favorite)",
        (100, 600), 2, 5, 130, 0.80, 0.85,
        "Mixed α/β. Good pocket geometry. Well-validated in design.",
    ),
    ScaffoldTopology(
        "beta_propeller", "β-propeller (lectin-like)",
        (150, 1000), 0, 20, 300, 0.85, 0.50,
        "Flat binding surface. Good for glycan/sugar recognition.",
    ),
    ScaffoldTopology(
        "miniprotein", "Miniprotein/knot (~40-60 aa)",
        (30, 200), 1, 2, 50, 0.70, 0.90,
        "Smallest designable fold. Cystine-stabilized. Good for constrained pockets.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# RFDIFFUSION SPEC GENERATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RFdiffusionSpec:
    """Input specification for RFdiffusion motif scaffolding."""
    # Hotspot residues (the pocket)
    hotspot_residues: list = field(default_factory=list)  # list[HotspotResidue]

    # Contig specification
    contig: str = ""              # e.g., "10-40/A163-181/10-40" for scaffolding
    scaffold_topology: str = ""

    # RFdiffusion config
    n_designs: int = 100
    noise_scale: float = 1.0
    partial_T: int = 0            # 0 = full diffusion, >0 = partial from template

    # Pocket constraints
    pocket_center_A: tuple = (0, 0, 0)
    pocket_radius_A: float = 8.0

    # Output
    output_dir: str = "rfd_output"

    def to_json(self):
        return json.dumps({
            "contig": self.contig,
            "scaffold_topology": self.scaffold_topology,
            "n_designs": self.n_designs,
            "noise_scale": self.noise_scale,
            "partial_T": self.partial_T,
            "pocket_center": list(self.pocket_center_A),
            "pocket_radius": self.pocket_radius_A,
            "hotspot_residues": [
                {
                    "residue": hr.residue_3,
                    "position": list(hr.position_A),
                    "role": hr.role,
                    "shell": hr.shell,
                }
                for hr in self.hotspot_residues
            ],
        }, indent=2)

    def to_command(self):
        """Generate RFdiffusion command line."""
        # Hotspot string for --hotspot_res
        hotspot_str = ",".join(
            f"A{i+1}" for i, _ in enumerate(self.hotspot_residues)
        )
        return (
            f"python run_inference.py "
            f"'contigmap.contigs=[{self.contig}]' "
            f"'ppi.hotspot_res=[{hotspot_str}]' "
            f"inference.num_designs={self.n_designs} "
            f"denoiser.noise_scale_ca={self.noise_scale} "
            f"inference.output_prefix={self.output_dir}/design"
        )


@dataclass
class ProteinMPNNSpec:
    """Input specification for ProteinMPNN sequence design."""
    pdb_path: str = ""            # from RFdiffusion output
    fixed_positions: list = field(default_factory=list)  # hotspot residues to keep
    n_sequences: int = 8
    temperature: float = 0.1      # lower = more conservative
    model: str = "v_48_020"       # ProteinMPNN model version

    def to_json(self):
        return json.dumps({
            "pdb_path": self.pdb_path,
            "fixed_positions": self.fixed_positions,
            "n_sequences": self.n_sequences,
            "temperature": self.temperature,
            "model": self.model,
        }, indent=2)


@dataclass
class BindCraftSpec:
    """Input specification for BindCraft end-to-end design."""
    target_pdb: str = ""          # target structure (if available)
    target_smiles: str = ""       # for small-molecule targets
    hotspot_residues: list = field(default_factory=list)
    binder_length: int = 80       # target binder size in residues
    n_designs: int = 100
    use_mpnn: bool = True
    use_af2: bool = True

    def to_json(self):
        return json.dumps({
            "target_pdb": self.target_pdb,
            "target_smiles": self.target_smiles,
            "hotspot_residues": [
                {"residue": hr.residue_3, "position": list(hr.position_A)}
                for hr in self.hotspot_residues
            ],
            "binder_length": self.binder_length,
            "n_designs": self.n_designs,
            "use_mpnn": self.use_mpnn,
            "use_af2": self.use_af2,
        }, indent=2)


@dataclass
class AF2ValidationSpec:
    """Spec for AlphaFold2 fold validation of designed sequences."""
    sequences: list = field(default_factory=list)  # list of amino acid sequences
    expected_rmsd_threshold_A: float = 2.0
    n_recycles: int = 3

    def to_json(self):
        return json.dumps({
            "sequences": self.sequences,
            "expected_rmsd_threshold": self.expected_rmsd_threshold_A,
            "n_recycles": self.n_recycles,
        }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# POCKET SPEC → PROTEIN DESIGN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProteinDesign:
    """A complete protein binder design specification."""
    guest_name: str
    guest_smiles: str

    # Hotspot residues
    hotspot_residues: list = field(default_factory=list)
    n_first_shell: int = 0
    n_second_shell: int = 0

    # Scaffold
    scaffold: object = None       # ScaffoldTopology
    total_residues: int = 0

    # Tool specs (all in spec mode)
    rfdiffusion_spec: object = None
    proteinmpnn_spec: object = None
    bindcraft_spec: object = None
    af2_spec: object = None

    # Scoring
    estimated_log_Ka: float = 0.0
    predicted_kd_nM: float = 0.0
    confidence: str = ""           # "high", "medium", "low"
    design_success_rate: float = 0.0  # estimated % of designs that would work

    # Practical
    expression_system: str = ""    # "E. coli BL21(DE3)", "insect cell", etc.
    estimated_cost_usd: float = 0.0
    estimated_time_weeks: int = 0
    notes: str = ""


def design_protein_for_guest(
    spec,
    guest_smiles: str,
    guest_name: str = "",
    guest_volume_A3: float = 0.0,
    guest_charge: int = 0,
    max_hotspots: int = 12,
) -> ProteinDesign:
    """Design a protein binder for a guest from InteractionGeometrySpec.

    Args:
        spec: InteractionGeometrySpec from guest_to_pocket_spec()
        guest_smiles: target guest SMILES
        guest_name: display name
        guest_volume_A3: guest molecular volume
        guest_charge: guest formal charge
        max_hotspots: max hotspot residues (computational cost scales with this)

    Returns:
        ProteinDesign with tool specs ready for execution.
    """
    result = ProteinDesign(
        guest_name=guest_name or guest_smiles[:40],
        guest_smiles=guest_smiles,
    )

    # Step 1: Map pocket donors to hotspot amino acids
    hotspots = _map_donors_to_residues(spec, max_hotspots)
    result.hotspot_residues = hotspots
    result.n_first_shell = sum(1 for h in hotspots if h.shell == 1)
    result.n_second_shell = sum(1 for h in hotspots if h.shell == 2)

    if not hotspots:
        result.confidence = "infeasible"
        result.notes = "No donor positions in spec"
        return result

    # Step 2: Select scaffold topology
    scaffold = _select_scaffold(guest_volume_A3, len(hotspots))
    result.scaffold = scaffold
    result.total_residues = scaffold.typical_size_aa

    # Step 3: Generate RFdiffusion spec
    result.rfdiffusion_spec = _build_rfdiffusion_spec(hotspots, scaffold)

    # Step 4: Generate ProteinMPNN spec
    result.proteinmpnn_spec = ProteinMPNNSpec(
        pdb_path=f"{result.rfdiffusion_spec.output_dir}/design_0.pdb",
        fixed_positions=list(range(1, len(hotspots) + 1)),
        n_sequences=8,
    )

    # Step 5: Generate BindCraft spec (alternative pathway)
    result.bindcraft_spec = BindCraftSpec(
        target_smiles=guest_smiles,
        hotspot_residues=hotspots,
        binder_length=scaffold.typical_size_aa,
    )

    # Step 6: AF2 validation spec
    result.af2_spec = AF2ValidationSpec(
        expected_rmsd_threshold_A=2.0,
    )

    # Step 7: Score the designed pocket
    _score_protein_design(result, spec, guest_volume_A3, guest_charge)

    return result


def _map_donors_to_residues(spec, max_hotspots):
    """Map InteractionGeometrySpec donors + surfaces to amino acid hotspots."""
    hotspots = []

    # First shell: direct contact residues (from spec donors)
    for i, dp in enumerate(spec.donor_positions):
        if len(hotspots) >= max_hotspots:
            break

        if dp.charge_state < 0:
            role = "hb_acceptor"
        elif dp.charge_state > 0:
            role = "hb_donor"
        else:
            role = "hb_donor"  # default to donor for neutral

        candidates = _DONOR_TO_RESIDUES.get(role, [])
        if not candidates:
            continue

        # Pick top candidate
        res3, res1, donor_type, notes = candidates[0]

        hotspots.append(HotspotResidue(
            residue_3=res3,
            residue_1=res1,
            role=role,
            donor_type=donor_type,
            position_A=dp.position_vector_A,
            shell=1,
            chi_preference="gauche+",
            notes=notes,
        ))

    # Aromatic surfaces → Trp/Phe first-shell
    for i, hs in enumerate(spec.hydrophobic_surfaces[:4]):
        if len(hotspots) >= max_hotspots:
            break
        if hs.area_A2 >= 20.0:
            role = "aromatic_wall"
        else:
            role = "hydrophobic_wall"

        candidates = _DONOR_TO_RESIDUES.get(role, [])
        if not candidates:
            continue
        res3, res1, donor_type, notes = candidates[0]

        hotspots.append(HotspotResidue(
            residue_3=res3,
            residue_1=res1,
            role=role,
            donor_type=donor_type,
            position_A=hs.center_A,
            shell=1,
            chi_preference="any",
            notes=notes,
        ))

    # Second shell: structural support (every first-shell residue gets
    # one Leu/Ile behind it for packing)
    n_first = len(hotspots)
    for i in range(n_first):
        if len(hotspots) >= max_hotspots:
            break
        # Position: 4 Å behind the first-shell residue (away from pocket center)
        fs = hotspots[i]
        pos = fs.position_A
        r = math.sqrt(sum(c**2 for c in pos))
        if r > 0.01:
            # Push outward by 4 Å
            scale = (r + 4.0) / r
            back_pos = tuple(c * scale for c in pos)
        else:
            back_pos = (4.0, 0, 0)

        hotspots.append(HotspotResidue(
            residue_3="LEU",
            residue_1="L",
            role="hydrophobic_wall",
            donor_type="alkyl",
            position_A=back_pos,
            shell=2,
            chi_preference="trans",
            notes="Second-shell packing behind first-shell contact.",
        ))

    return hotspots


def _select_scaffold(guest_volume_A3, n_hotspots):
    """Select best scaffold topology for pocket size and complexity."""
    best = None
    best_score = -1

    for topo in SCAFFOLD_TOPOLOGIES:
        vmin, vmax = topo.pocket_volume_range_A3
        # Volume fit
        if guest_volume_A3 < vmin:
            vol_score = 0.5 * guest_volume_A3 / vmin
        elif guest_volume_A3 > vmax:
            vol_score = 0.5 * vmax / guest_volume_A3
        else:
            vol_score = 1.0

        # Hotspot capacity (need ~n_hotspots positions in the pocket)
        if n_hotspots <= topo.typical_size_aa * 0.15:  # <15% of residues are hotspots
            capacity_score = 1.0
        else:
            capacity_score = 0.5

        score = (
            vol_score * 3.0
            + topo.designability * 4.0
            + topo.preorganization * 2.0
            + capacity_score * 1.0
        )

        if score > best_score:
            best_score = score
            best = topo

    return best or SCAFFOLD_TOPOLOGIES[0]


def _build_rfdiffusion_spec(hotspots, scaffold):
    """Build RFdiffusion contig specification from hotspots and scaffold."""
    n_res = scaffold.typical_size_aa
    n_hotspot = len([h for h in hotspots if h.shell == 1])

    # Contig: flexible linker / hotspot block / flexible linker
    # e.g., "10-40/0 A1-A{n_hotspot}/10-40"
    flank = max(10, (n_res - n_hotspot) // 2)

    contig = f"{flank}-{flank+30}/0 A1-A{n_hotspot}/{flank}-{flank+30}"

    # Pocket center (average of first-shell positions)
    first_shell = [h for h in hotspots if h.shell == 1]
    if first_shell:
        cx = sum(h.position_A[0] for h in first_shell) / len(first_shell)
        cy = sum(h.position_A[1] for h in first_shell) / len(first_shell)
        cz = sum(h.position_A[2] for h in first_shell) / len(first_shell)
        center = (cx, cy, cz)
    else:
        center = (0, 0, 0)

    return RFdiffusionSpec(
        hotspot_residues=hotspots,
        contig=contig,
        scaffold_topology=scaffold.name,
        n_designs=100,
        pocket_center_A=center,
        pocket_radius_A=max(8.0, math.sqrt(sum(c**2 for c in center)) + 5),
    )


def _score_protein_design(result, spec, guest_volume_A3, guest_charge):
    """Estimate binding affinity for the designed protein pocket.

    Uses existing MABE PL physics parameters + hotspot quality assessment.
    """
    n_first = result.n_first_shell
    n_total = len(result.hotspot_residues)

    if n_first == 0:
        result.estimated_log_Ka = 0.0
        result.confidence = "infeasible"
        return

    # H-bond contribution: each first-shell H-bond residue contributes
    # ~1.5 log Ka units (literature: tight H-bonds in proteins ≈ 1-2 log Ka)
    n_hb = sum(1 for h in result.hotspot_residues
               if h.shell == 1 and h.role in ("hb_donor", "hb_acceptor"))
    dg_hbond = n_hb * 1.5  # log Ka units

    # Hydrophobic packing: each aromatic/hydrophobic first-shell residue
    # contributes ~0.8 log Ka (from buried SASA arguments)
    n_hydrophobic = sum(1 for h in result.hotspot_residues
                        if h.shell == 1 and h.role in ("aromatic_wall", "hydrophobic_wall"))
    dg_hydrophobic = n_hydrophobic * 0.8

    # Preorganization bonus: rigid scaffold = less entropy penalty
    preorg = result.scaffold.preorganization if result.scaffold else 0.5
    dg_preorg = preorg * 1.5  # up to 1.5 log Ka bonus

    # Size penalty: very small or very large guests are harder
    if guest_volume_A3 > 0:
        if guest_volume_A3 < 150:
            size_penalty = 0.5  # small guests have fewer contacts
        elif guest_volume_A3 > 500:
            size_penalty = 0.3  # large guests harder to fully encapsulate
        else:
            size_penalty = 0.0
    else:
        size_penalty = 0.0

    # Estimated log Ka
    log_ka = dg_hbond + dg_hydrophobic + dg_preorg - size_penalty
    result.estimated_log_Ka = round(log_ka, 1)
    result.predicted_kd_nM = round(10 ** (9 - log_ka), 1)  # nM = 10^9 / Ka

    # Design success rate (from literature)
    # Baker lab: ~19% for protein binders, ~10% for small-molecule pockets
    # BindCraft: 10-100% depending on target
    designability = result.scaffold.designability if result.scaffold else 0.5
    result.design_success_rate = round(designability * 0.30, 2)  # ~15-30%

    # Confidence
    if n_first >= 4 and result.scaffold and result.scaffold.designability > 0.8:
        result.confidence = "medium"
    elif n_first >= 2:
        result.confidence = "low"
    else:
        result.confidence = "speculative"

    # Expression
    result.expression_system = "E. coli BL21(DE3), IPTG induction, 18°C overnight"
    result.estimated_cost_usd = 300.0  # gene synthesis + expression + purification
    result.estimated_time_weeks = 4    # gene to purified protein
