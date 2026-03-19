"""
glycan/realization_ranker.py -- Fabrication-ready specs for immune cell pulldown.

Two capture strategies:
  Strategy A: Sugar on scaffold -> binds lectin on cell surface
  Strategy B: Synthetic lectin-mimic on scaffold -> binds glycan on cell surface

For each cell type, ranks all viable (strategy, scaffold, binder) combinations
by predicted affinity, cost, complexity, and time-to-batch.

Scaffold options:
  - Dendrimer (PAMAM G4-G6)
  - Glycopolymer (RAFT/ROMP)
  - Gold nanoparticle (AuNP / Fe3O4@Au)
  - DNA origami exterior (ATHENA-compatible)
  - Liposome (glycolipid / SPION-loaded)
  - Streptavidin-bead direct (commercial kit)
  - MIP (molecularly imprinted polymer)
  - Boronic acid polymer/surface
  - Designed mini-lectin (RFdiffusion)
  - Aptamer (SELEX-derived)

Binder types for Strategy B:
  - Davis-type synthetic lectin (anthracene cage)
  - Boronic acid array (reversible covalent diol capture)
  - Designed mini-protein (RFdiffusion/BindCraft)
  - Anti-glycan aptamer (DNA/RNA)
  - MIP cavity (template-imprinted)
  - Plant lectin (ConA, WGA, PNA — cheap, off-the-shelf)

Usage:
    from glycan.realization_ranker import rank_all_options
    options = rank_all_options("macrophage_m2", bead_diameter_nm=50)
    for opt in options[:5]:
        print(opt.summary)
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ── Cell surface glycan profiles ────────────────────────────────────────
# What glycans each cell type displays on its surface.
# These are the TARGETS for Strategy B (synthetic lectin -> cell glycan).
# Sources: glycomics literature (Cummings 2009, Varki 2017 Essentials of Glycobiology).

@dataclass
class GlycanProfile:
    glycan: str                    # dominant glycan class
    abundance: str                 # "high" | "medium" | "low"
    linkage: str                   # e.g., "alpha2-6", "beta1-4"
    terminal_sugar: str            # the outermost sugar accessible to binders
    note: str
    source: str

CELL_GLYCAN_PROFILES: dict[str, list[GlycanProfile]] = {
    "t_cell": [
        GlycanProfile("sialyl-LacNAc", "high", "alpha2-6", "Neu5Ac",
                       "T cells: high alpha2-6 sialylation on N-glycans",
                       "Baum 2002 Immunity 16:273"),
        GlycanProfile("poly-LacNAc", "medium", "beta1-4", "Gal",
                       "Extended LacNAc chains on activated T cells",
                       "Demetriou 2001 Nature 409:733"),
        GlycanProfile("core-2 O-glycans", "medium", "beta1-6", "GlcNAc",
                       "Selectin ligand scaffold",
                       "Ellies 2002 Immunity 17:153"),
    ],
    "t_cell_activated": [
        GlycanProfile("sialyl-LacNAc", "high", "alpha2-6", "Neu5Ac",
                       "Maintained after activation",
                       "Baum 2002 Immunity 16:273"),
        GlycanProfile("poly-LacNAc", "high", "beta1-4", "Gal",
                       "Upregulated on activation; Galectin-3 ligand",
                       "Demetriou 2001 Nature 409:733"),
    ],
    "macrophage_m2": [
        GlycanProfile("high-mannose N-glycans", "high", "alpha1-2", "Man",
                       "M2 macrophages display high-mannose glycans",
                       "van Vliet 2005 Immunobiology 210:185"),
        GlycanProfile("sialyl-Lewis-x", "medium", "alpha2-3", "Neu5Ac",
                       "Selectin ligand",
                       "McEver 2015 Annu Rev Pathol 10:425"),
    ],
    "macrophage_m1": [
        GlycanProfile("complex N-glycans", "high", "beta1-4", "Gal",
                       "Fully processed N-glycans with terminal Gal/Neu5Ac",
                       "Baum 2002 Immunity 16:273"),
        GlycanProfile("sialyl-LacNAc", "high", "alpha2-6", "Neu5Ac",
                       "High sialylation on M1",
                       "Crocker 2007 Nat Rev Immunol 7:255"),
    ],
    "dendritic_cell": [
        GlycanProfile("high-mannose N-glycans", "high", "alpha1-2", "Man",
                       "DC surface rich in high-mannose for pathogen mimicry",
                       "van Vliet 2005 Immunobiology 210:185"),
        GlycanProfile("Lewis-x/y", "medium", "alpha1-3", "Fuc",
                       "Fucosylated glycans on DC",
                       "Appelmelk 2003 J Immunol 170:1635"),
    ],
    "nk_cell": [
        GlycanProfile("complex N-glycans", "high", "beta1-4", "Gal",
                       "Standard complex N-glycan profile",
                       "Varki 2017 Essentials of Glycobiology Ch.41"),
        GlycanProfile("sialyl-LacNAc", "high", "alpha2-6", "Neu5Ac",
                       "High sialylation",
                       "Crocker 2007 Nat Rev Immunol 7:255"),
    ],
    "b_cell": [
        GlycanProfile("sialyl-LacNAc", "high", "alpha2-6", "Neu5Ac",
                       "B cells: high alpha2-6 Sia; CD22 cis ligand",
                       "Collins 2004 J Immunol 172:5543"),
        GlycanProfile("poly-LacNAc", "medium", "beta1-4", "Gal",
                       "Galectin ligands",
                       "Demetriou 2001 Nature 409:733"),
    ],
    "neutrophil": [
        GlycanProfile("sialyl-Lewis-x", "high", "alpha2-3", "Neu5Ac",
                       "Primary selectin ligand; essential for rolling",
                       "McEver 2015 Annu Rev Pathol 10:425"),
        GlycanProfile("poly-LacNAc", "high", "beta1-4", "GlcNAc",
                       "Long LacNAc chains on neutrophil PSGL-1",
                       "Ellies 2002 Immunity 17:153"),
    ],
    "hepatocyte": [
        GlycanProfile("complex N-glycans", "high", "beta1-4", "Gal",
                       "Hepatocyte N-glycans terminate in Gal (ASGPR removes Sia)",
                       "Stockert 1980 JBC 255:3830"),
    ],
    "tumor_epithelial": [
        GlycanProfile("truncated O-glycans (Tn/STn)", "high", "alpha", "GalNAc",
                       "Tumor-associated carbohydrate antigens; truncated O-glycans",
                       "Pinho 2015 Nat Rev Cancer 15:540"),
        GlycanProfile("sialyl-Lewis-x/a", "high", "alpha2-3", "Neu5Ac",
                       "Overexpressed on metastatic tumors",
                       "Pinho 2015 Nat Rev Cancer 15:540"),
        GlycanProfile("poly-LacNAc (branched)", "high", "beta1-6", "Gal",
                       "beta1-6 branched N-glycans (MGAT5); Galectin lattice ligand",
                       "Demetriou 2001 Nature 409:733"),
    ],
}

# Aliases (reuse from pulldown_selector)
_GLYCAN_ALIASES = {
    "t cell": "t_cell", "t-cell": "t_cell", "macrophage": "macrophage_m2",
    "m2 macrophage": "macrophage_m2", "m1 macrophage": "macrophage_m1",
    "dc": "dendritic_cell", "dendritic": "dendritic_cell",
    "nk": "nk_cell", "nk cell": "nk_cell", "b cell": "b_cell",
    "b-cell": "b_cell", "hepatocyte": "hepatocyte", "liver": "hepatocyte",
    "tumor": "tumor_epithelial", "cancer": "tumor_epithelial",
}

def _resolve_glycan_cell_type(cell_type: str) -> str:
    key = cell_type.strip().lower().replace("_", " ")
    if key in _GLYCAN_ALIASES:
        return _GLYCAN_ALIASES[key]
    canonical = cell_type.strip().lower().replace(" ", "_")
    if canonical in CELL_GLYCAN_PROFILES:
        return canonical
    raise ValueError(f"Unknown cell type: '{cell_type}'. Available: {sorted(CELL_GLYCAN_PROFILES.keys())}")


# ── Scaffold database ──────────────────────────────────────────────────

@dataclass
class ScaffoldType:
    name: str
    display_name: str
    diameter_nm: float            # typical scaffold diameter
    max_loading: int              # max binder sites
    typical_loading: int          # realistic loading after conjugation
    cost_per_batch_usd: float     # reagent cost for one batch (enough for ~10-100 tests)
    cost_per_test_usd: float      # estimated cost per 10^6 cells sorted
    complexity: str               # "trivial" | "standard" | "advanced" | "expert"
    time_to_first_batch: str
    magnetic_strategy: str        # how to make it magnetic
    strategies: list[str]         # which capture strategies it supports: ["A", "B", or both]
    athena_compatible: bool
    rfd_compatible: bool
    notes: str
    source: str

SCAFFOLDS: dict[str, ScaffoldType] = {
    "streptavidin_bead": ScaffoldType(
        name="streptavidin_bead",
        display_name="Streptavidin-magnetic bead (direct)",
        diameter_nm=1000, max_loading=4, typical_loading=4,
        cost_per_batch_usd=15, cost_per_test_usd=1.5,
        complexity="trivial", time_to_first_batch="1 hour",
        magnetic_strategy="direct (bead is magnetic)",
        strategies=["A", "B"], athena_compatible=False, rfd_compatible=False,
        notes="Lowest effort. Biotin-PEG-sugar or biotin-lectin onto commercial SA bead. 4 sites only.",
        source="Thermo Fisher Dynabeads catalog",
    ),
    "dendrimer_g4": ScaffoldType(
        name="dendrimer_g4",
        display_name="PAMAM G4 glycodendrimer",
        diameter_nm=4.5, max_loading=64, typical_loading=40,
        cost_per_batch_usd=80, cost_per_test_usd=0.50,
        complexity="standard", time_to_first_batch="3 days",
        magnetic_strategy="biotin terminus -> SA-magnetic bead",
        strategies=["A", "B"], athena_compatible=False, rfd_compatible=False,
        notes="Sigma PAMAM G4-NH2 + sugar-NHS coupling. 60% conjugation typical. Strong multivalent effect.",
        source="Chabre 2010 Chem Soc Rev 39:1538",
    ),
    "dendrimer_g6": ScaffoldType(
        name="dendrimer_g6",
        display_name="PAMAM G6 glycodendrimer",
        diameter_nm=6.7, max_loading=256, typical_loading=150,
        cost_per_batch_usd=200, cost_per_test_usd=1.0,
        complexity="standard", time_to_first_batch="3 days",
        magnetic_strategy="biotin terminus -> SA-magnetic bead",
        strategies=["A", "B"], athena_compatible=False, rfd_compatible=False,
        notes="Higher valency than G4 but larger excluded volume. Best for high-density receptors.",
        source="Chabre 2010 Chem Soc Rev 39:1538",
    ),
    "glycopolymer": ScaffoldType(
        name="glycopolymer",
        display_name="RAFT glycopolymer (pendant sugars)",
        diameter_nm=10, max_loading=200, typical_loading=100,
        cost_per_batch_usd=60, cost_per_test_usd=0.30,
        complexity="advanced", time_to_first_batch="1 week",
        magnetic_strategy="biotin chain-end -> SA-magnetic bead",
        strategies=["A"], athena_compatible=False, rfd_compatible=False,
        notes="Sugar-acrylate RAFT polymerization. DP 50-200. Cheapest per-sugar display. Requires polymer chemistry expertise.",
        source="Ladmiral 2004 Eur Polym J 40:431; Spain 2007 Polym Chem",
    ),
    "aunp_50": ScaffoldType(
        name="aunp_50",
        display_name="Gold nanoparticle (50 nm) + thiol-PEG-sugar SAM",
        diameter_nm=50, max_loading=5000, typical_loading=2000,
        cost_per_batch_usd=150, cost_per_test_usd=1.0,
        complexity="standard", time_to_first_batch="2 days",
        magnetic_strategy="Fe3O4@Au core-shell or biotin-SA bridge",
        strategies=["A", "B"], athena_compatible=False, rfd_compatible=False,
        notes="Thiol-PEG-sugar:thiol-PEG-OH mixed SAM. Sugar density tunable by feed ratio. Colorimetric readout possible.",
        source="Marradi 2013 Chem Soc Rev 42:4728",
    ),
    "liposome": ScaffoldType(
        name="liposome",
        display_name="Glycoliposome (sugar-PEG-DSPE in lipid bilayer)",
        diameter_nm=100, max_loading=50000, typical_loading=10000,
        cost_per_batch_usd=40, cost_per_test_usd=0.20,
        complexity="standard", time_to_first_batch="2 days",
        magnetic_strategy="SPION encapsulation or magnetoliposome",
        strategies=["A"], athena_compatible=False, rfd_compatible=False,
        notes="Highest sugar density and lowest cost per sugar. Stability concern for long-term storage. Extrusion required.",
        source="Ravoo 2001 JACS 123:10245; Sandstrom 2004 Langmuir",
    ),
    "dna_origami": ScaffoldType(
        name="dna_origami",
        display_name="DNA origami cage (ATHENA) + sugar-click staples",
        diameter_nm=30, max_loading=200, typical_loading=80,
        cost_per_batch_usd=800, cost_per_test_usd=8.0,
        complexity="expert", time_to_first_batch="2 weeks",
        magnetic_strategy="biotin staples -> SA-magnetic bead",
        strategies=["A", "B"], athena_compatible=True, rfd_compatible=False,
        notes="Highest geometric precision. Each sugar position controlled to ~2 nm. Cost dominated by modified staples ($10-20 each).",
        source="Douglas 2009 Nature 459:414; Veneziano 2016 Science 352:1534",
    ),
    "boronic_acid_polymer": ScaffoldType(
        name="boronic_acid_polymer",
        display_name="Boronic acid polymer/hydrogel beads",
        diameter_nm=1000, max_loading=100000, typical_loading=50000,
        cost_per_batch_usd=30, cost_per_test_usd=0.10,
        complexity="standard", time_to_first_batch="3 days",
        magnetic_strategy="Fe3O4 core + boronic acid shell",
        strategies=["B"], athena_compatible=False, rfd_compatible=False,
        notes="Reversible covalent diol capture. Broad glycan binding. Release with fructose or acid. Not sugar-specific but cheap.",
        source="Li 2018 Chem Soc Rev 47:2279; Nishiyama 2006 Chem Commun",
    ),
    "mip_beads": ScaffoldType(
        name="mip_beads",
        display_name="Molecularly imprinted polymer (MIP) beads",
        diameter_nm=500, max_loading=1000, typical_loading=500,
        cost_per_batch_usd=50, cost_per_test_usd=0.30,
        complexity="advanced", time_to_first_batch="1 week",
        magnetic_strategy="Fe3O4 core + MIP shell",
        strategies=["B"], athena_compatible=False, rfd_compatible=False,
        notes="Template imprinting with sugar as template. Moderate selectivity. Cheap once template batch made. Reusable.",
        source="Haupt 2012 Chem Rev 112:4598; Yin 2016 Angew Chem",
    ),
    "plant_lectin_bead": ScaffoldType(
        name="plant_lectin_bead",
        display_name="Plant lectin on magnetic bead (ConA/WGA/PNA)",
        diameter_nm=1000, max_loading=10000, typical_loading=5000,
        cost_per_batch_usd=25, cost_per_test_usd=0.50,
        complexity="trivial", time_to_first_batch="2 hours",
        magnetic_strategy="direct (lectin-coated magnetic bead, commercial)",
        strategies=["B"], athena_compatible=False, rfd_compatible=False,
        notes="Off-the-shelf. ConA-agarose/magnetic beads available from Sigma, Vector Labs. Cheapest Strategy B option.",
        source="Sigma-Aldrich lectin-agarose catalog; Vector Labs",
    ),
    "designed_miniprotein": ScaffoldType(
        name="designed_miniprotein",
        display_name="Designed mini-lectin (RFdiffusion + ProteinMPNN)",
        diameter_nm=3, max_loading=1, typical_loading=1,
        cost_per_batch_usd=3000, cost_per_test_usd=30.0,
        complexity="expert", time_to_first_batch="2 months",
        magnetic_strategy="biotin tag -> SA-magnetic bead",
        strategies=["B"], athena_compatible=False, rfd_compatible=True,
        notes="Design protein with exact pharmacophore. Highest potential affinity (nM). Highest cost/risk. Gene synthesis + E. coli expression.",
        source="Polizzi 2024 Science 383:1098 (COMBS); Watson 2023 Nature (RFdiffusion)",
    ),
    "aptamer_bead": ScaffoldType(
        name="aptamer_bead",
        display_name="Anti-glycan DNA aptamer on magnetic bead",
        diameter_nm=1000, max_loading=10000, typical_loading=5000,
        cost_per_batch_usd=200, cost_per_test_usd=2.0,
        complexity="advanced", time_to_first_batch="1 month (SELEX)",
        magnetic_strategy="biotin-aptamer -> SA-magnetic bead",
        strategies=["B"], athena_compatible=False, rfd_compatible=False,
        notes="Requires SELEX against target glycan. Once selected, cheap to produce (IDT oligo synthesis). Few anti-glycan aptamers published.",
        source="Kawakami 2012 Analyst 137:4539; Ferreira 2008 Anal Chem",
    ),
}


# ── Strategy B binder types ─────────────────────────────────────────────

@dataclass
class BinderType:
    name: str
    display_name: str
    target_glycans: list[str]     # which terminal sugars it binds
    selectivity: str              # "broad" | "class-specific" | "sugar-specific"
    mechanism: str                # binding mechanism
    estimated_Kd_uM: float       # typical Kd range in uM
    notes: str
    source: str

STRATEGY_B_BINDERS: dict[str, BinderType] = {
    "ConA_lectin": BinderType(
        name="ConA_lectin",
        display_name="Concanavalin A (plant lectin)",
        target_glycans=["Man", "Glc", "GlcNAc"],
        selectivity="class-specific",
        mechanism="CRD binding to Man/Glc via C3/C4/C6 pharmacophore",
        estimated_Kd_uM=130,  # Ka ~7600 M-1 for αMeMan
        notes="Cheapest lectin. Tetravalent. Requires Mn2+/Ca2+. $20/10mg from Sigma.",
        source="Chervenak 1995 Biochemistry 34:5685",
    ),
    "WGA_lectin": BinderType(
        name="WGA_lectin",
        display_name="Wheat Germ Agglutinin (WGA)",
        target_glycans=["GlcNAc", "Neu5Ac"],
        selectivity="class-specific",
        mechanism="Multi-subsite binding to GlcNAc/Sia",
        estimated_Kd_uM=2500,  # Ka ~400 M-1 monovalent
        notes="Binds GlcNAc and sialic acid. Divalent. No metal required. $25/10mg.",
        source="Bains 1992 Biochemistry 31:12624",
    ),
    "PNA_lectin": BinderType(
        name="PNA_lectin",
        display_name="Peanut Agglutinin (PNA)",
        target_glycans=["Gal", "GalNAc"],
        selectivity="class-specific",
        mechanism="CRD binding to Gal via Trp CH-pi + HB",
        estimated_Kd_uM=53,  # Ka ~18900 M-1 for Gal
        notes="Gal-specific. Useful for tumor Tn antigen detection (after neuraminidase). $20/10mg.",
        source="Swaminathan 1998",
    ),
    "SNA_lectin": BinderType(
        name="SNA_lectin",
        display_name="Sambucus nigra Agglutinin (SNA/EBL)",
        target_glycans=["Neu5Ac"],
        selectivity="sugar-specific",
        mechanism="alpha2-6 sialic acid specific",
        estimated_Kd_uM=100,
        notes="Highly specific for alpha2-6 Sia. Key for B cell/T cell sialylation. $40/10mg.",
        source="Shibuya 1987 JBC 262:1596",
    ),
    "MAL_lectin": BinderType(
        name="MAL_lectin",
        display_name="Maackia amurensis Lectin (MAL-II)",
        target_glycans=["Neu5Ac"],
        selectivity="sugar-specific",
        mechanism="alpha2-3 sialic acid specific",
        estimated_Kd_uM=200,
        notes="Complementary to SNA: alpha2-3 Sia. For neutrophil sLex capture. $40/10mg.",
        source="Wang 1988 JBC 263:4576",
    ),
    "boronic_acid": BinderType(
        name="boronic_acid",
        display_name="Phenylboronic acid (diol capture)",
        target_glycans=["Man", "Glc", "Gal", "Neu5Ac", "GlcNAc", "GalNAc", "Fuc"],
        selectivity="broad",
        mechanism="Reversible covalent boronate ester with cis-diols",
        estimated_Kd_uM=500,
        notes="Binds ALL sugars (any cis-diol). Not selective. Good for total glycoprotein capture. Release with fructose.",
        source="Nishiyama 2006 Chem Commun; Li 2018 Chem Soc Rev 47:2279",
    ),
    "davis_receptor": BinderType(
        name="davis_receptor",
        display_name="Davis synthetic lectin (GluHUT cage)",
        target_glycans=["Glc"],
        selectivity="sugar-specific",
        mechanism="Urea HB array + anthracene CH-pi; all-equatorial selective",
        estimated_Kd_uM=54,  # Ka ~18600 M-1
        notes="Highest selectivity synthetic receptor. Glc > Gal ~100x. Not commercially available; requires synthesis (5 steps).",
        source="Tromans 2019 Nat Chem 11:52",
    ),
    "mip_cavity": BinderType(
        name="mip_cavity",
        display_name="Molecularly imprinted polymer cavity",
        target_glycans=["Man", "Glc", "Gal", "Neu5Ac", "GlcNAc"],
        selectivity="class-specific",
        mechanism="Template-shaped cavity with functional monomer contacts",
        estimated_Kd_uM=100,
        notes="Selectivity depends on template sugar. Moderate. Reusable. Batch-variable.",
        source="Haupt 2012 Chem Rev 112:4598",
    ),
}


# ── Fabrication spec output ─────────────────────────────────────────────

@dataclass
class FabricationSpec:
    # Identity
    strategy: str                  # "A" (sugar->lectin) or "B" (synth_lectin->glycan)
    scaffold: str                  # scaffold name key
    scaffold_display: str
    binder: str                    # sugar name (A) or binder type key (B)
    binder_display: str
    target_on_cell: str            # what the binder engages on the cell

    # Performance
    estimated_Kd_nM: float         # per-site Kd in nM
    n_sites: int                   # binder loading
    effective_valency: int
    multivalent_enhancement_log10: float
    estimated_Kd_effective_nM: float  # after multivalent enhancement

    # Practical
    cost_per_batch_usd: float
    cost_per_test_usd: float
    complexity: str
    time_to_first_batch: str
    magnetic_strategy: str

    # Compatibility
    athena_compatible: bool
    rfd_compatible: bool

    # Ranking
    composite_score: float
    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [
            f"[{self.strategy}] {self.scaffold_display}",
            f"binder={self.binder_display}",
            f"Kd_eff={self.estimated_Kd_effective_nM:.0f}nM",
            f"valency={self.effective_valency}",
            f"${self.cost_per_test_usd:.2f}/test",
            f"[{self.complexity}]",
        ]
        return " | ".join(parts)


# ── Core ranking engine ─────────────────────────────────────────────────

def rank_all_options(
    cell_type: str,
    bead_diameter_nm: float = 50.0,
    receptor_density_per_um2: float = 1000.0,
) -> list[FabricationSpec]:
    """Rank all (strategy, scaffold, binder) combinations for a cell type.

    Returns list sorted by composite_score (descending).
    """
    canonical = _resolve_glycan_cell_type(cell_type)
    results = []

    # Strategy A: sugar on scaffold -> binds lectin on cell
    results.extend(_strategy_a_options(canonical, bead_diameter_nm, receptor_density_per_um2))

    # Strategy B: synthetic lectin on scaffold -> binds glycan on cell
    results.extend(_strategy_b_options(canonical, bead_diameter_nm, receptor_density_per_um2))

    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results


def _strategy_a_options(
    cell_type: str,
    bead_diameter_nm: float,
    receptor_density_per_um2: float,
) -> list[FabricationSpec]:
    """Generate Strategy A options: sugar on scaffold -> lectin on cell."""
    results = []

    # Import pulldown_selector to get the scored sugar recommendations
    try:
        from glycan.pulldown_selector import recommend_pulldown
        recs = recommend_pulldown(cell_type, bead_diameter_nm=bead_diameter_nm)
    except (ImportError, ValueError):
        recs = []

    if not recs:
        return results

    # Take top 3 sugar recommendations
    top_recs = recs[:3]

    for rec in top_recs:
        if rec.dG_pred is None:
            continue
        Kd_uM = _dG_to_Kd_uM(rec.dG_pred)

        for scaffold_key, scaffold in SCAFFOLDS.items():
            if "A" not in scaffold.strategies:
                continue

            spec = _build_spec_a(rec, scaffold, Kd_uM, receptor_density_per_um2)
            results.append(spec)

    return results


def _strategy_b_options(
    cell_type: str,
    bead_diameter_nm: float,
    receptor_density_per_um2: float,
) -> list[FabricationSpec]:
    """Generate Strategy B options: synthetic lectin on scaffold -> glycan on cell."""
    results = []

    if cell_type not in CELL_GLYCAN_PROFILES:
        return results

    glycan_profiles = CELL_GLYCAN_PROFILES[cell_type]

    for profile in glycan_profiles:
        terminal = profile.terminal_sugar
        # Find binders that target this glycan
        matching_binders = [
            b for b in STRATEGY_B_BINDERS.values()
            if terminal in b.target_glycans
        ]

        for binder in matching_binders:
            for scaffold_key, scaffold in SCAFFOLDS.items():
                if "B" not in scaffold.strategies:
                    continue

                spec = _build_spec_b(
                    profile, binder, scaffold, receptor_density_per_um2, cell_type
                )
                results.append(spec)

    return results


def _build_spec_a(rec, scaffold, Kd_uM, receptor_density_per_um2) -> FabricationSpec:
    """Build a FabricationSpec for Strategy A."""
    Kd_nM = Kd_uM * 1000
    n_sites = scaffold.typical_loading
    valency = _effective_valency(scaffold, receptor_density_per_um2)
    enh = _multivalent_enhancement(valency)
    Kd_eff = max(0.01, Kd_nM / (10 ** enh))

    composite = _composite_score(Kd_eff, scaffold.cost_per_test_usd, scaffold.complexity, enh,
                                 selectivity="sugar-specific")

    target_str = f"{rec.lectin} on cell" if hasattr(rec, 'lectin') else "lectin on cell"

    return FabricationSpec(
        strategy="A",
        scaffold=scaffold.name,
        scaffold_display=scaffold.display_name,
        binder=rec.sugar,
        binder_display=f"{rec.sugar} @ {rec.position}",
        target_on_cell=target_str,
        estimated_Kd_nM=round(Kd_nM, 1),
        n_sites=n_sites,
        effective_valency=valency,
        multivalent_enhancement_log10=enh,
        estimated_Kd_effective_nM=round(Kd_eff, 2),
        cost_per_batch_usd=scaffold.cost_per_batch_usd,
        cost_per_test_usd=scaffold.cost_per_test_usd,
        complexity=scaffold.complexity,
        time_to_first_batch=scaffold.time_to_first_batch,
        magnetic_strategy=scaffold.magnetic_strategy,
        athena_compatible=scaffold.athena_compatible,
        rfd_compatible=scaffold.rfd_compatible,
        composite_score=composite,
        notes=[scaffold.notes],
    )


def _build_spec_b(profile, binder, scaffold, receptor_density_per_um2, cell_type) -> FabricationSpec:
    """Build a FabricationSpec for Strategy B."""
    Kd_nM = binder.estimated_Kd_uM * 1000
    n_sites = scaffold.typical_loading
    valency = _effective_valency(scaffold, receptor_density_per_um2)
    enh = _multivalent_enhancement(valency)
    Kd_eff = max(0.01, Kd_nM / (10 ** enh))

    composite = _composite_score(Kd_eff, scaffold.cost_per_test_usd, scaffold.complexity, enh,
                                 selectivity=binder.selectivity)

    return FabricationSpec(
        strategy="B",
        scaffold=scaffold.name,
        scaffold_display=scaffold.display_name,
        binder=binder.name,
        binder_display=binder.display_name,
        target_on_cell=f"{profile.terminal_sugar} ({profile.glycan}) on {cell_type}",
        estimated_Kd_nM=round(Kd_nM, 1),
        n_sites=n_sites,
        effective_valency=valency,
        multivalent_enhancement_log10=enh,
        estimated_Kd_effective_nM=round(Kd_eff, 2),
        cost_per_batch_usd=scaffold.cost_per_batch_usd,
        cost_per_test_usd=scaffold.cost_per_test_usd,
        complexity=scaffold.complexity,
        time_to_first_batch=scaffold.time_to_first_batch,
        magnetic_strategy=scaffold.magnetic_strategy,
        athena_compatible=scaffold.athena_compatible,
        rfd_compatible=scaffold.rfd_compatible,
        composite_score=composite,
        notes=[scaffold.notes, binder.notes, f"Glycan abundance: {profile.abundance}"],
    )


# ── Physics helpers ─────────────────────────────────────────────────────

def _dG_to_Kd_uM(dG_kJ: float) -> float:
    """Convert ΔG (kJ/mol) to Kd (μM). ΔG = RT ln(Kd)."""
    RT = 2.479  # kJ/mol at 298K
    Kd_M = math.exp(dG_kJ / RT)  # dG is negative -> Kd < 1
    return Kd_M * 1e6  # M -> μM


def _effective_valency(scaffold: ScaffoldType, receptor_density_per_um2: float) -> int:
    """Effective valency: limited by sugar loading or receptor density under scaffold."""
    # Contact area from scaffold diameter
    R_nm = scaffold.diameter_nm / 2
    linker_reach_nm = 20.0  # shorter than bead model (scaffold is smaller)
    if scaffold.diameter_nm < 2 * linker_reach_nm:
        contact_area_nm2 = 2 * math.pi * R_nm ** 2  # hemisphere
    else:
        contact_area_nm2 = 2 * math.pi * R_nm * linker_reach_nm

    contact_area_um2 = contact_area_nm2 / 1e6
    n_receptors = max(1, int(contact_area_um2 * receptor_density_per_um2))
    return min(scaffold.typical_loading, n_receptors)


def _multivalent_enhancement(valency: int) -> float:
    """Log10 enhancement from multivalent binding. Capped at 6."""
    if valency <= 1:
        return 0.0
    enh = 0.5 * (valency - 1)
    return min(enh, 6.0)


def _composite_score(Kd_eff_nM: float, cost_per_test: float, complexity: str,
                     enh_log10: float, selectivity: str = "sugar-specific") -> float:
    """Composite score: higher = better pulldown candidate.

    Balances affinity, cost, practical complexity, and selectivity.
    Broad-spectrum binders (boronic acid) are penalized because they
    capture all glycoproteins, not just the target cell type.
    """
    # Affinity: log scale, 1 nM = 9, 1 uM = 3, 1 mM = 0
    if Kd_eff_nM > 0:
        affinity_score = max(0.1, 9.0 - math.log10(max(0.01, Kd_eff_nM)))
    else:
        affinity_score = 9.0

    # Cost: antibody baseline ~$2/test. Lower = better.
    cost_score = max(0.1, 2.0 / max(0.01, cost_per_test))

    # Complexity penalty
    complexity_weight = {
        "trivial": 1.5, "standard": 1.0, "advanced": 0.6, "expert": 0.3
    }.get(complexity, 0.5)

    # Selectivity weight: specific binders valued over broad-spectrum
    selectivity_weight = {
        "sugar-specific": 1.0,
        "class-specific": 0.8,
        "broad": 0.3,
    }.get(selectivity, 0.5)

    return round(affinity_score * cost_score * complexity_weight * selectivity_weight, 2)


# ── Convenience ─────────────────────────────────────────────────────────

def list_scaffolds() -> list[str]:
    return sorted(SCAFFOLDS.keys())

def list_binders() -> list[str]:
    return sorted(STRATEGY_B_BINDERS.keys())

def list_glycan_cell_types() -> list[str]:
    return sorted(CELL_GLYCAN_PROFILES.keys())
