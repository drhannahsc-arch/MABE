"""
knowledge/structural_library.py - Structural scaffolds from physics up.

The question is not "what biology uses" but "what atomic properties enable
the structural and electronic behaviors we need?"

Backbone atoms ranked by bond stability in aqueous/harsh conditions:
    Si-O: 452 kJ/mol (silicates, zeolites, mesoporous silica)
    B-O:  536 kJ/mol (borates, borosilicates)
    C-C:  346 kJ/mol (organic frameworks, polymers)
    C-N:  305 kJ/mol (peptides, proteins, DNA)
    M-O:  variable   (MOFs, POMs, coordination cages)

Biology chose carbon because it was available, not because it was optimal.
"""

from core.assembly import StructuralConstraint, SelectivityFilter, ReleaseMechanism


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL OPTIONS — organized by backbone chemistry, not by biology
# ═══════════════════════════════════════════════════════════════════════════

STRUCTURAL_OPTIONS = [

    # ── NO STRUCTURE (free in solution) ──────────────────────────
    StructuralConstraint(
        name="Free in solution (no constraint)",
        type="none",
        geometry="free",
        max_interior_sites=1,
        ph_stable_range=(0.0, 14.0),
        temp_stable_c=(0, 100),
        cost_per_unit="included in recognition chemistry cost",
        synthesis_complexity="trivial",
        notes="No structural constraint. Simplest. Fastest to deploy.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # SILICON-OXYGEN BACKBONE
    # Si-O bond: 452 kJ/mol. Strongest common covalent framework.
    # Natural nanoporous architectures. pH/temp invincible.
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="Mesoporous silica MCM-41 (2-3nm pores)",
        type="mesoporous_silica",
        geometry="hexagonal_channels",
        interior_volume_nm3=None,
        pore_size_nm=2.5,
        max_interior_sites=50,
        recognition_spacing_nm=2.5,
        ph_stable_range=(1.0, 12.0),
        temp_stable_c=(0, 500),
        cost_per_unit="$5/g material",
        synthesis_complexity="standard",
        notes="Si-O backbone. Hexagonal array of uniform 2-3nm channels. "
              "1000+ m2/g surface area. Functionalize interior with organosilanes "
              "(amine, thiol, carboxylate, phosphonate). Survives pH 1-12 and 500C. "
              "Industrial scale synthesis (tons/year). Interior walls are pure silica = "
              "no leaching, no degradation products.",
    ),
    StructuralConstraint(
        name="Mesoporous silica SBA-15 (6-10nm pores)",
        type="mesoporous_silica",
        geometry="hexagonal_channels",
        pore_size_nm=8.0,
        max_interior_sites=30,
        recognition_spacing_nm=4.0,
        ph_stable_range=(0.5, 13.0),
        temp_stable_c=(0, 600),
        cost_per_unit="$8/g material",
        synthesis_complexity="standard",
        notes="Si-O backbone. Thicker walls than MCM-41 = more stable. "
              "6-10nm pores accommodate larger recognition elements (peptides, aptamers). "
              "Microporous wall interconnects aid diffusion. "
              "The workhorse of mesoporous silica for applications.",
    ),
    StructuralConstraint(
        name="Zeolite Y (FAU topology, 0.74nm pores)",
        type="zeolite",
        geometry="sodalite_cages",
        interior_volume_nm3=0.8,
        pore_size_nm=0.74,
        max_interior_sites=4,
        recognition_spacing_nm=1.0,
        ph_stable_range=(2.0, 12.0),
        temp_stable_c=(0, 700),
        cost_per_unit="$2/g (commodity)",
        synthesis_complexity="trivial",
        notes="Si-O-Al backbone. Angstrom-precision pores — molecular sieving at atomic scale. "
              "Natural ion exchanger. Framework charge from Al substitution creates "
              "cation binding sites WITHOUT added recognition chemistry. "
              "The zeolite IS the binder for many metal cations. "
              "Used industrially for decades (water softening, catalysis). Dirt cheap.",
    ),
    StructuralConstraint(
        name="Zeolite ZSM-5 (MFI topology, 0.55nm pores)",
        type="zeolite",
        geometry="channel_intersection",
        interior_volume_nm3=0.3,
        pore_size_nm=0.55,
        max_interior_sites=2,
        recognition_spacing_nm=0.8,
        ph_stable_range=(1.0, 13.0),
        temp_stable_c=(0, 800),
        cost_per_unit="$3/g (commodity)",
        synthesis_complexity="trivial",
        notes="Si-O backbone, high silica. Intersecting 10-ring channels. "
              "Ultra-tight pores = extreme size selectivity. "
              "Only small cations (Li+, Na+, K+) fit in channels. "
              "Most thermally stable zeolite. Acidic mine drainage compatible.",
    ),
    StructuralConstraint(
        name="Silica nanoparticle (20-200nm, functionalized surface)",
        type="silica_np",
        geometry="sphere",
        pore_size_nm=None,
        max_interior_sites=0,
        recognition_spacing_nm=3.0,
        ph_stable_range=(2.0, 11.0),
        temp_stable_c=(0, 400),
        cost_per_unit="$3/g",
        synthesis_complexity="standard",
        notes="Si-O core. Not a cage — a high surface area particle. "
              "Functionalize surface with organosilanes for recognition attachment. "
              "~200 m2/g. Colloidal stability. Well-characterized. "
              "The default nanoparticle platform for a reason.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # METAL-OXYGEN / METAL-ORGANIC FRAMEWORKS
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="MOF UiO-66 (Zr-based, 0.6nm pores)",
        type="mof",
        geometry="octahedral",
        interior_volume_nm3=1.2,
        pore_size_nm=0.6,
        max_interior_sites=1,
        ph_stable_range=(1.0, 10.0),
        temp_stable_c=(0, 300),
        cost_per_unit="$5/g material",
        synthesis_complexity="standard",
        notes="Zr6O4(OH)4 nodes + terephthalate linkers. "
              "Exceptional chemical stability (survives pH 1 HCl). "
              "Sub-nm pores = molecular sieving. Post-synthetic modification: "
              "amine, thiol, or click-chemistry on linker. 1200 m2/g.",
    ),
    StructuralConstraint(
        name="MOF MIL-101(Cr) (large pore, 1.2-1.6nm)",
        type="mof",
        geometry="mesoporous",
        interior_volume_nm3=20.0,
        pore_size_nm=1.6,
        max_interior_sites=6,
        recognition_spacing_nm=1.5,
        ph_stable_range=(0.0, 12.0),
        temp_stable_c=(0, 275),
        cost_per_unit="$10/g",
        synthesis_complexity="standard",
        notes="Cr3+ nodes + terephthalate. One of the most chemically stable MOFs. "
              "Giant pores (1.2nm + 1.6nm cages). 4000 m2/g — highest surface area "
              "of common MOFs. Space for larger recognition elements inside pores. "
              "Demonstrated for heavy metal adsorption.",
    ),
    StructuralConstraint(
        name="COF (covalent organic framework, 2-4nm pores)",
        type="cof",
        geometry="hexagonal_channels",
        pore_size_nm=3.0,
        max_interior_sites=20,
        recognition_spacing_nm=2.0,
        ph_stable_range=(1.0, 13.0),
        temp_stable_c=(0, 400),
        cost_per_unit="$20/g",
        synthesis_complexity="complex",
        notes="All-covalent bonds (C-C, C-N, B-O). No metal nodes = no leaching risk. "
              "More chemically stable than MOFs. Crystalline with defined pore geometry. "
              "Functionalize through monomer design. "
              "Emerging platform — fewer off-shelf options but superior stability.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # MOLECULARLY IMPRINTED POLYMERS
    # The structure IS the recognition. No separate binder needed.
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="Molecularly imprinted polymer (MIP)",
        type="mip",
        geometry="templated_cavity",
        interior_volume_nm3=0.5,
        pore_size_nm=0.5,
        max_interior_sites=1,
        recognition_spacing_nm=None,
        ph_stable_range=(0.0, 14.0),
        temp_stable_c=(0, 150),
        cost_per_unit="$10/g first batch, $2/g at scale",
        synthesis_complexity="standard",
        notes="THE STRUCTURE IS THE BINDER. Template target ion with functional monomers "
              "(methacrylic acid, vinylpyridine, allylthiourea). Cross-link (EGDMA). "
              "Remove template. Cavity left behind has shape + charge + donor atom "
              "complementarity for target. No separate recognition chemistry needed. "
              "Extreme pH stability (all covalent). Cheap at scale. "
              "Selectivity factors of 10-1000x demonstrated for metal ions. "
              "Limitation: batch variation in cavity quality. Cannot inspect individual cavities.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # CARBON STRUCTURES
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="Carbon nanotube (single-wall, 1-2nm diameter)",
        type="carbon_nanotube",
        geometry="tube",
        interior_volume_nm3=None,
        pore_size_nm=1.5,
        max_interior_sites=10,
        recognition_spacing_nm=1.0,
        ph_stable_range=(1.0, 13.0),
        temp_stable_c=(0, 400),
        cost_per_unit="$50/g (purified)",
        synthesis_complexity="complex",
        notes="sp2 carbon tube. Conductive = enables electrochemical detection. "
              "Interior functionalization possible but challenging. "
              "Primary value: ELECTRODE SUBSTRATE for electrochemical sensors. "
              "High aspect ratio = large surface area.",
    ),
    StructuralConstraint(
        name="Graphene oxide sheet (2D, functionalized)",
        type="graphene_oxide",
        geometry="sheet",
        pore_size_nm=None,
        max_interior_sites=100,
        recognition_spacing_nm=2.0,
        ph_stable_range=(2.0, 10.0),
        temp_stable_c=(0, 200),
        cost_per_unit="$15/g",
        synthesis_complexity="standard",
        notes="2D carbon sheet with -OH, -COOH, epoxide groups. "
              "Inherent metal binding via oxygen functional groups. "
              "Layer stacking creates tunable interlayer spacing. "
              "Conductive (reduced form) for electrochemical applications.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # DNA/RNA SCAFFOLDS
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="DNA origami icosahedron (40nm, DX wireframe)",
        type="dna_origami_cage",
        geometry="icosahedron",
        interior_volume_nm3=33500.0,
        pore_size_nm=8.0,
        max_interior_sites=12,
        recognition_spacing_nm=6.0,
        ph_stable_range=(5.0, 9.0),
        temp_stable_c=(4, 50),
        cost_per_unit="$2000 first (staple set), $50/assembly",
        synthesis_complexity="complex",
        notes="C-N-P backbone. Most programmable self-assembly known. "
              "Addressable at single-staple resolution. Interior decoration via overhang extension. "
              "LIMITATION: pH 5-9 only (depurination), nuclease-sensitive, Mg2+-dependent. "
              "ADVANTAGE: absolute control over recognition element placement.",
    ),
    StructuralConstraint(
        name="DNA origami tetrahedron (20nm)",
        type="dna_origami_cage",
        geometry="tetrahedron",
        interior_volume_nm3=940.0,
        pore_size_nm=12.0,
        max_interior_sites=4,
        recognition_spacing_nm=8.0,
        ph_stable_range=(5.5, 8.5),
        temp_stable_c=(4, 45),
        cost_per_unit="$500 first, $20/assembly",
        synthesis_complexity="standard",
        notes="Simplest DNA cage. Large pores = less steric selectivity but faster diffusion. "
              "Proven renal clearance (therapeutic applications). "
              "Good for rapid prototyping of interior designs.",
    ),
    StructuralConstraint(
        name="DNA origami 6HB cage (30nm, tight pores)",
        type="dna_origami_cage",
        geometry="icosahedron",
        interior_volume_nm3=14100.0,
        pore_size_nm=3.0,
        max_interior_sites=8,
        recognition_spacing_nm=5.0,
        ph_stable_range=(5.5, 8.5),
        temp_stable_c=(4, 45),
        cost_per_unit="$3000 first, $80/assembly",
        synthesis_complexity="complex",
        notes="6-helix bundle edges. Tightest DNA origami pores (~3nm). "
              "Best steric selectivity among DNA scaffolds.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # PROTEIN CAGES
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="Ferritin cage (24-subunit, 8nm cavity)",
        type="protein_cage",
        geometry="icosahedron",
        interior_volume_nm3=340.0,
        pore_size_nm=0.4,
        max_interior_sites=24,
        recognition_spacing_nm=2.0,
        ph_stable_range=(4.0, 10.0),
        temp_stable_c=(4, 70),
        cost_per_unit="$500 expression, $5/mg",
        synthesis_complexity="complex",
        notes="Natural iron storage cage. 0.4nm pores = ions pass, proteins excluded. "
              "Interior engineerable by mutagenesis. "
              "pH-driven disassembly (pH<4) for release. Biocompatible.",
    ),
    StructuralConstraint(
        name="Encapsulin cage (60-mer, 24nm cavity)",
        type="protein_cage",
        geometry="icosahedron",
        interior_volume_nm3=7200.0,
        pore_size_nm=3.0,
        max_interior_sites=60,
        recognition_spacing_nm=3.0,
        ph_stable_range=(5.0, 9.0),
        temp_stable_c=(4, 80),
        cost_per_unit="$800 expression, $8/mg",
        synthesis_complexity="complex",
        notes="Bacterial nanocompartment. Larger cavity than ferritin. "
              "Cargo-loading peptide tags for interior targeting. "
              "Thermostable (some variants to 80C). Growing toolkit.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # COORDINATION CAGES
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="Pd/Pt coordination cage (Fujita-type, M12L24)",
        type="coordination_cage",
        geometry="cuboctahedron",
        interior_volume_nm3=4.0,
        pore_size_nm=1.0,
        max_interior_sites=1,
        recognition_spacing_nm=None,
        ph_stable_range=(3.0, 10.0),
        temp_stable_c=(4, 80),
        cost_per_unit="$200/mg (precious metal)",
        synthesis_complexity="complex",
        notes="Pd2+ or Pt2+ corners + bent dipyridyl linkers. "
              "Quantitative self-assembly in water. Atomically precise cavity. "
              "Guest binding by hydrophobic effect + shape complementarity. "
              "Small cavity = very selective. Limited to small molecule guests.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # POLYMER / DENDRIMER
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="PAMAM dendrimer G4 (amine surface)",
        type="dendrimer",
        geometry="sphere",
        interior_volume_nm3=22.0,
        pore_size_nm=1.5,
        max_interior_sites=16,
        recognition_spacing_nm=1.0,
        ph_stable_range=(2.0, 10.0),
        temp_stable_c=(0, 80),
        cost_per_unit="$100/g",
        synthesis_complexity="standard",
        notes="Generation 4 PAMAM. 64 amine surface groups. "
              "Interior cavities encapsulate metals. Well-characterized.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # INORGANIC LAYERED MATERIALS
    # ═══════════════════════════════════════════════════════════════

    StructuralConstraint(
        name="Layered double hydroxide (Mg-Al LDH)",
        type="ldh",
        geometry="layered",
        pore_size_nm=0.7,
        max_interior_sites=20,
        recognition_spacing_nm=0.3,
        ph_stable_range=(4.0, 12.0),
        temp_stable_c=(0, 300),
        cost_per_unit="$3/g",
        synthesis_complexity="standard",
        notes="Positively charged brucite-like layers with exchangeable interlayer anions. "
              "NATURAL ANION BINDER — the structure IS the recognition for oxyanions "
              "(arsenate, chromate, selenite, phosphate). "
              "Ion exchange capacity 2-5 meq/g. Cheap, scalable, nontoxic. "
              "The LDH IS the binder for anionic targets.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY FILTER GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_selectivity_filter(structure: StructuralConstraint,
                                 target_radius_nm: float,
                                 competitor_sizes: list = None) -> SelectivityFilter:
    if structure.type == "none":
        return SelectivityFilter(
            name="Chemical selectivity only",
            mechanism="none",
            description="No steric filtering. Selectivity from recognition chemistry only.",
        )

    if structure.type == "mip":
        return SelectivityFilter(
            name="Templated cavity (shape + charge + donor complementarity)",
            mechanism="template_imprint",
            description=(
                "Cavity molded around target template. Shape complementarity "
                "at sub-angstrom level. Charge distribution matches target. "
                "Donor atoms positioned by template geometry. "
                "Combined shape+charge+donor selectivity — the polymer IS the binder."
            ),
            selectivity_enhancement="Selectivity factor 10-1000x demonstrated for metal ion MIPs",
        )

    if structure.type == "ldh":
        return SelectivityFilter(
            name="Charge-selective interlayer gallery",
            mechanism="charge_gating",
            description=(
                "Positively charged layers exclude cations. "
                "Only anions (and neutral species) enter interlayer gallery. "
                "Size selectivity from gallery height (~0.7nm)."
            ),
            selectivity_enhancement="Intrinsic anion selectivity — cations electrostatically excluded",
        )

    if structure.type == "zeolite":
        return SelectivityFilter(
            name=f"Zeolite molecular sieve ({structure.pore_size_nm} nm)",
            mechanism="molecular_sieve",
            description=(
                f"Angstrom-precision pore window ({structure.pore_size_nm} nm). "
                f"Only species smaller than pore enter framework. "
                f"Framework charge provides additional electrostatic selectivity. "
                f"Combined size + charge discrimination at atomic scale."
            ),
            selectivity_enhancement=f"Molecular sieving at {structure.pore_size_nm} nm — atomic-scale size discrimination",
        )

    if structure.pore_size_nm and structure.pore_size_nm > 0:
        return SelectivityFilter(
            name=f"Pore exclusion ({structure.pore_size_nm} nm)",
            mechanism="pore_exclusion",
            description=(
                f"Pore diameter {structure.pore_size_nm} nm. "
                f"Species larger than pore are physically excluded. "
                f"Combined with interior recognition for dual selectivity."
            ),
            selectivity_enhancement=f"Steric exclusion at {structure.pore_size_nm} nm",
        )

    return SelectivityFilter(
        name="Structural selectivity",
        mechanism="size_exclusion",
        description="Constrained geometry limits access to binding sites.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# RELEASE MECHANISMS
# ═══════════════════════════════════════════════════════════════════════════

RELEASE_OPTIONS = [
    ReleaseMechanism(
        name="pH shift release",
        trigger="ph_shift",
        description="Lower pH protonates donor atoms, releasing metal.",
        reversible=True, cycles=50,
        trigger_conditions="pH shift to 2.0-3.0 with dilute HCl",
        release_efficiency=">90% in 5 min",
    ),
    ReleaseMechanism(
        name="Competitor displacement (EDTA wash)",
        trigger="competitor",
        description="Strong universal chelator strips target from binder.",
        reversible=True, cycles=100,
        trigger_conditions="10 mM EDTA, pH 7.0, 10 min",
        release_efficiency=">95%",
    ),
    ReleaseMechanism(
        name="Competitor displacement (imidazole)",
        trigger="competitor",
        description="Imidazole displaces His-tag coordination. Standard IMAC elution.",
        reversible=True, cycles=100,
        trigger_conditions="250 mM imidazole, pH 7.5",
        release_efficiency=">95%",
    ),
    ReleaseMechanism(
        name="Toehold strand displacement",
        trigger="toehold_strand",
        description="DNA strand hybridizes to toehold, unfolding binder. Programmable, isothermal.",
        reversible=True, cycles=20,
        trigger_conditions="10x excess trigger strand, RT, 30 min",
        release_efficiency="80-95%",
    ),
    ReleaseMechanism(
        name="UV photocleavage",
        trigger="uv_light",
        description="UV cleaves photolabile linker. Spatial control possible.",
        reversible=False, cycles=1,
        trigger_conditions="365 nm UV, 5-15 min",
        release_efficiency=">90%",
    ),
    ReleaseMechanism(
        name="Thermal release",
        trigger="thermal",
        description="Heat denatures structure, releasing target.",
        reversible=True, cycles=10,
        trigger_conditions="65-95 C, 5-15 min",
        release_efficiency=">90%",
    ),
    ReleaseMechanism(
        name="Electrochemical release",
        trigger="electrochemical",
        description="Applied voltage changes oxidation state, reducing binding affinity.",
        reversible=True, cycles=200,
        trigger_conditions="-0.5V to -1.0V vs Ag/AgCl, 1-5 min",
        release_efficiency="70-95%",
        notes="Requires conductive substrate (electrode, CNT, graphene).",
    ),
    ReleaseMechanism(
        name="Ion exchange displacement",
        trigger="ion_exchange",
        description="Concentrated salt solution displaces captured ions from framework sites.",
        reversible=True, cycles=500,
        trigger_conditions="1M NaCl or NaNO3, 30 min",
        release_efficiency="60-90%",
        notes="For zeolites, LDHs, ion exchange resins. Simple, cheap, scalable.",
    ),
    ReleaseMechanism(
        name="Solvent wash (MIP regeneration)",
        trigger="solvent_wash",
        description="Acid/base wash + solvent removes template ion from MIP cavities.",
        reversible=True, cycles=100,
        trigger_conditions="0.1M HCl or thiourea/HCl for heavy metals",
        release_efficiency="80-95%",
        notes="MIP-specific. Cavities retain shape after washing. Highly reusable.",
    ),
    ReleaseMechanism(
        name="No active release (permanent capture)",
        trigger="none",
        description="Target permanently captured. For removal, not recovery.",
        reversible=False, cycles=1,
    ),
]


def get_compatible_releases(recognition_type: str, structure_type: str,
                             outcome_wants_release: bool) -> list[ReleaseMechanism]:
    """Return release mechanisms compatible with recognition + structure combo."""
    compatible = []
    for release in RELEASE_OPTIONS:
        if release.trigger == "toehold_strand" and recognition_type not in ("dnazyme", "aptamer", "motif"):
            continue
        if release.trigger == "competitor" and "imidazole" in release.name and recognition_type != "peptide":
            continue
        if release.trigger == "electrochemical" and structure_type not in ("none", "carbon_nanotube", "graphene_oxide"):
            continue
        if release.trigger == "ion_exchange" and structure_type not in ("zeolite", "ldh"):
            continue
        if release.trigger == "solvent_wash" and structure_type != "mip":
            continue
        if release.trigger == "none" and outcome_wants_release:
            continue
        compatible.append(release)
    return compatible
