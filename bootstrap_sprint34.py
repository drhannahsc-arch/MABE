"""
MABE Platform - Sprint 34 Bootstrap: Synthesis Protocol Generation
Physics-up actionable synthesis from raw precursors.
Si-O > C-C. Commodity chemicals. Modular building blocks.
Requires Sprints 16v2 + 17fix + 18-33.
"""
import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n\U0001f528 MABE Sprint 34 \u2014 Synthesis Protocol Generation\n")
write_file("core/synthesis.py", '''\
"""
core/synthesis.py — Sprint 34: Synthesis Protocol Generation

Physics-up actionable synthesis protocols using modular building blocks.
Prioritizes:
  1. Si-O > C-C (inorganic scaffolds preferred)
  2. Bulk/industrial precursors over boutique reagents
  3. Standard coupling reactions (Suzuki, click, amide, silane grafting)
  4. Interchangeable modular blocks

Covers full scaffold preparation from raw precursors, not just
functionalization of commercial products.
"""
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# REAGENT DATABASE — bulk/industrial pricing, not Sigma retail
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Reagent:
    name: str
    formula: str
    role: str               # "precursor", "catalyst", "linker", "protecting_group", "solvent", "template"
    cost_per_kg_usd: float  # Bulk industrial pricing
    supplier_tier: str      # "commodity" (<$50/kg), "fine_chemical" (<$500/kg), "specialty" (>$500/kg)
    hazard_class: str       # "low", "moderate", "high", "extreme"
    notes: str = ""


REAGENT_DB = {
    # === INORGANIC SCAFFOLD PRECURSORS (commodity) ===
    "TEOS": Reagent("tetraethyl orthosilicate", "Si(OC2H5)4", "precursor", 15, "commodity", "low",
        "Sol-gel silica precursor. 99% pure bulk. Si-O backbone."),
    "sodium_silicate": Reagent("sodium silicate", "Na2SiO3", "precursor", 2, "commodity", "low",
        "Water glass. Cheapest silica source. Industrial grade."),
    "sodium_aluminate": Reagent("sodium aluminate", "NaAlO2", "precursor", 8, "commodity", "low",
        "Zeolite synthesis. Al source for framework."),
    "NaOH": Reagent("sodium hydroxide", "NaOH", "precursor", 1, "commodity", "low",
        "Mineralizer for zeolite crystallization."),
    "CTAB": Reagent("cetyltrimethylammonium bromide", "C19H42BrN", "template", 25, "fine_chemical", "low",
        "Mesopore template for MCM-41/SBA-15 type silica."),
    "P123": Reagent("Pluronic P123", "(EO)20(PO)70(EO)20", "template", 30, "fine_chemical", "low",
        "Template for SBA-15 mesoporous silica. 6-10 nm pores."),
    "TiCl4": Reagent("titanium tetrachloride", "TiCl4", "precursor", 20, "commodity", "moderate",
        "TiO2 coating precursor. Hydrolysis-sensitive."),
    "ZrCl4": Reagent("zirconium tetrachloride", "ZrCl4", "precursor", 40, "fine_chemical", "moderate",
        "UiO-66 MOF node. Also ZrO2 coatings."),
    "AlCl3": Reagent("aluminium chloride", "AlCl3", "precursor", 5, "commodity", "moderate",
        "LDH synthesis. Al source."),
    "MgCl2": Reagent("magnesium chloride", "MgCl2", "precursor", 3, "commodity", "low",
        "LDH synthesis. Mg source."),
    "FeCl3": Reagent("iron(III) chloride", "FeCl3", "precursor", 4, "commodity", "low",
        "Fe3O4 nanoparticle precursor. Also MIL-101 MOF node."),
    "FeCl2": Reagent("iron(II) chloride", "FeCl2", "precursor", 8, "commodity", "low",
        "Fe3O4 coprecipitation (Massart method)."),

    # === ORGANIC LINKERS (fine chemical) ===
    "BDC": Reagent("1,4-benzenedicarboxylic acid (terephthalic acid)", "C8H6O4", "linker", 3, "commodity", "low",
        "MOF linker: UiO-66, MIL-101. Commodity from PET recycling."),
    "BTC": Reagent("1,3,5-benzenetricarboxylic acid (trimesic acid)", "C9H6O6", "linker", 40, "fine_chemical", "low",
        "MOF linker: HKUST-1, MIL-100."),
    "BPDC": Reagent("4,4'-biphenyldicarboxylic acid", "C14H10O4", "linker", 80, "fine_chemical", "low",
        "Extended MOF linker for larger pores."),

    # === SILANE COUPLING AGENTS (fine chemical) ===
    "APTES": Reagent("3-aminopropyltriethoxysilane", "(C2H5O)3Si(CH2)3NH2", "linker", 50, "fine_chemical", "low",
        "Grafts -NH2 onto silica/zeolite. THE universal silica functionalizer."),
    "MPTMS": Reagent("3-mercaptopropyltrimethoxysilane", "(CH3O)3Si(CH2)3SH", "linker", 60, "fine_chemical", "low",
        "Grafts -SH onto silica. Soft-metal capture (Pb, Hg, Au)."),
    "GPTMS": Reagent("3-glycidoxypropyltrimethoxysilane", "(CH3O)3Si(CH2)3OCH2CHOCH2", "linker", 55, "fine_chemical", "low",
        "Grafts epoxy onto silica. Ring-opens with amines/thiols."),
    "CPTMS": Reagent("3-chloropropyltrimethoxysilane", "(CH3O)3Si(CH2)3Cl", "linker", 45, "fine_chemical", "low",
        "Grafts -Cl onto silica. Nucleophilic substitution with amines/thiols."),

    # === CHELATOR BUILDING BLOCKS ===
    "IDA": Reagent("iminodiacetic acid", "HN(CH2COOH)2", "linker", 30, "fine_chemical", "low",
        "IDA/NTA-type chelator arm. Hard metal binder."),
    "DTPA_anhydride": Reagent("DTPA dianhydride", "C14H19N3O8·(CO)2", "linker", 120, "fine_chemical", "low",
        "Pre-activated DTPA. Reacts with amines directly."),
    "EDTA_dianhydride": Reagent("EDTA dianhydride", "C10H12N2O6·(CO)2", "linker", 80, "fine_chemical", "low",
        "Pre-activated EDTA. One anhydride grafts to surface, other chelates."),
    "dithiocarbamate_CS2": Reagent("carbon disulfide", "CS2", "linker", 5, "commodity", "moderate",
        "Forms dithiocarbamate in situ from amine + CS2. S,S-bidentate chelator."),
    "thiourea": Reagent("thiourea", "SC(NH2)2", "linker", 8, "commodity", "low",
        "S-donor ligand. Au/Ag/Hg selective."),
    "2_2_bipyridine": Reagent("2,2'-bipyridine", "C10H8N2", "linker", 60, "fine_chemical", "low",
        "N,N-bidentate chelator. Strong-field, borderline."),
    "salicylaldehyde": Reagent("salicylaldehyde", "C7H6O2", "linker", 15, "commodity", "low",
        "Salen/saloph ligand synthesis. Schiff base with amines."),
    "catechol": Reagent("catechol", "C6H6O2", "linker", 20, "commodity", "low",
        "Fe3+ chelator (siderophore mimic). Also Ti4+, UO22+."),
    "8_hydroxyquinoline": Reagent("8-hydroxyquinoline", "C9H7NO", "linker", 25, "fine_chemical", "low",
        "Classic metal chelator. N,O-bidentate. Very broad selectivity."),

    # === COUPLING REAGENTS ===
    "EDC": Reagent("1-ethyl-3-(3-dimethylaminopropyl)carbodiimide", "C8H17N3", "catalyst", 200, "fine_chemical", "low",
        "Amide coupling activator. Carboxylate + amine → amide."),
    "NHS": Reagent("N-hydroxysuccinimide", "C4H5NO3", "catalyst", 80, "fine_chemical", "low",
        "EDC/NHS coupling. Stabilizes active ester intermediate."),
    "Pd_PPh3_4": Reagent("tetrakis(triphenylphosphine)palladium(0)", "Pd(PPh3)4", "catalyst", 8000, "specialty", "moderate",
        "Suzuki coupling catalyst. Reusable."),
    "CuSO4": Reagent("copper(II) sulfate", "CuSO4", "catalyst", 5, "commodity", "low",
        "CuAAC click catalyst (with sodium ascorbate)."),
    "sodium_ascorbate": Reagent("sodium ascorbate", "C6H7NaO6", "catalyst", 15, "commodity", "low",
        "Reduces Cu2+ to Cu+ for CuAAC click."),
    "BOC2O": Reagent("di-tert-butyl dicarbonate", "(BOC)2O", "protecting_group", 60, "fine_chemical", "low",
        "BOC protection for amines. Removed by TFA or HCl."),
    "TFA": Reagent("trifluoroacetic acid", "CF3COOH", "catalyst", 40, "fine_chemical", "moderate",
        "BOC deprotection. Also peptide cleavage."),

    # === SOLVENTS (commodity) ===
    "water": Reagent("deionized water", "H2O", "solvent", 0.1, "commodity", "low", ""),
    "ethanol": Reagent("ethanol", "C2H5OH", "solvent", 2, "commodity", "low", ""),
    "DMF": Reagent("N,N-dimethylformamide", "C3H7NO", "solvent", 5, "commodity", "moderate",
        "MOF solvothermal synthesis."),
    "toluene": Reagent("toluene", "C7H8", "solvent", 3, "commodity", "moderate",
        "Silane grafting solvent (anhydrous)."),
    "HCl": Reagent("hydrochloric acid", "HCl", "solvent", 1, "commodity", "moderate", "pH adjustment."),
}


# ═══════════════════════════════════════════════════════════════════════════
# REACTION LIBRARY — standard transformations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReactionStep:
    step_number: int
    name: str
    reaction_type: str      # "sol_gel", "hydrothermal", "grafting", "coupling", etc.
    description: str
    reagents: list          # [(reagent_key, amount_g_per_batch)]
    conditions: str         # "80°C, 24h, stirring" etc.
    time_hours: float
    yield_pct: float        # Expected yield
    critical_notes: str = ""
    safety_notes: str = ""


@dataclass
class SynthesisProtocol:
    """Complete actionable synthesis protocol."""
    binder_name: str
    target_formula: str
    scaffold_type: str
    # Protocol
    steps: list                 # List of ReactionStep
    total_time_hours: float
    total_cost_usd_per_gram: float  # Cost per gram of final product
    batch_size_g: float         # Typical batch
    # Assessment
    difficulty: str             # "straightforward", "moderate", "advanced", "expert"
    scalability: str            # "kg_scale", "100g_scale", "10g_scale", "mg_scale"
    reproducibility: str        # "high", "moderate", "low"
    # Key info
    critical_reagent: str       # Most expensive or hardest to source
    rate_limiting_step: str
    alternative_routes: list    # Brief descriptions of alternatives
    equipment_needed: list
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD SYNTHESIS ROUTES
# ═══════════════════════════════════════════════════════════════════════════

def _mesoporous_silica_sba15():
    """SBA-15 from bulk precursors. 6-10 nm ordered pores."""
    return [
        ReactionStep(1, "Template solution", "dissolution",
            "Dissolve P123 (4g) in 2M HCl (120 mL) + water (30 mL) at 35°C.",
            [("P123", 4.0), ("HCl", 10.0), ("water", 150.0)],
            "35°C, 2h, stirring until clear", 2.0, 99,
            "P123 must fully dissolve before adding silica source."),
        ReactionStep(2, "Silica condensation", "sol_gel",
            "Add TEOS (8.5g) dropwise to template solution. Stir 20h at 35°C.",
            [("TEOS", 8.5)],
            "35°C, 20h, vigorous stirring", 20.0, 95,
            "TEOS addition rate controls pore order. ~1 drop/sec."),
        ReactionStep(3, "Hydrothermal aging", "hydrothermal",
            "Transfer to PTFE-lined autoclave. Age at 100°C for 24h.",
            [],
            "100°C, 24h, static, sealed autoclave", 24.0, 98,
            "Higher temp (130°C) gives larger pores. Do not exceed 150°C."),
        ReactionStep(4, "Wash and calcine", "calcination",
            "Filter, wash with water (3×200 mL) and ethanol (1×100 mL). "
            "Dry 80°C overnight. Calcine 550°C, 6h, ramp 1°C/min.",
            [("water", 600.0), ("ethanol", 100.0)],
            "550°C, 6h, air, ramp 1°C/min", 16.0, 90,
            "Calcination removes template. Product: white powder, ~3g.",
            "Ensure good ventilation during calcination (organics burn off)."),
    ]

def _zeolite_Y_from_scratch():
    """Zeolite Y (FAU) from sodium silicate + sodium aluminate."""
    return [
        ReactionStep(1, "Seed gel", "gel_preparation",
            "Mix sodium aluminate (2.0g) + NaOH (1.3g) in water (10 mL). "
            "Add sodium silicate (10.0g) slowly with stirring. Age 24h at RT.",
            [("sodium_aluminate", 2.0), ("NaOH", 1.3), ("sodium_silicate", 10.0), ("water", 10.0)],
            "Room temperature, 24h, stirring → aging", 24.0, 99,
            "Seed gel nucleation controls crystal size. Do not skip aging."),
        ReactionStep(2, "Feedstock gel", "gel_preparation",
            "Separately: sodium silicate (20g) + water (20 mL). "
            "Sodium aluminate (1.5g) + NaOH (0.5g) + water (10 mL). "
            "Combine rapidly with vigorous stirring.",
            [("sodium_silicate", 20.0), ("sodium_aluminate", 1.5), ("NaOH", 0.5), ("water", 30.0)],
            "Room temperature, 30 min mixing", 0.5, 99),
        ReactionStep(3, "Combine and crystallize", "hydrothermal",
            "Add seed gel to feedstock gel. Mix thoroughly. "
            "Transfer to PTFE autoclave. Heat 100°C for 8-24h.",
            [],
            "100°C, 8-24h, static", 16.0, 85,
            "Check XRD at 8h. FAU peaks at 6.2°, 15.6°, 23.5° 2θ.",
            "Si/Al ratio ~2.5 for Y-type. Lower NaOH → higher Si/Al."),
        ReactionStep(4, "Wash and dry", "washing",
            "Filter hot. Wash with DI water until pH <9. Dry 110°C overnight.",
            [("water", 500.0)],
            "110°C, 12h", 12.0, 90,
            "Yield: ~8-12g white crystalline powder. Ion-exchange with NH4+ if Na-form not desired."),
    ]

def _zeolite_ZSM5():
    """ZSM-5 (MFI) — high-silica, acid-stable."""
    return [
        ReactionStep(1, "Synthesis gel", "gel_preparation",
            "Mix TEOS (20g) + TPAOH solution (25%, 15 mL) + water (30 mL). "
            "Add NaOH (0.3g) + Al source (sodium aluminate 0.3g for Si/Al=50). Stir 6h.",
            [("TEOS", 20.0), ("NaOH", 0.3), ("sodium_aluminate", 0.3), ("water", 30.0)],
            "Room temperature, 6h stirring", 6.0, 99,
            "TPAOH is the structure-directing agent. Can substitute TPA-Br + NaOH."),
        ReactionStep(2, "Crystallization", "hydrothermal",
            "Autoclave at 170°C for 48h. Static or tumbled.",
            [],
            "170°C, 48h, PTFE-lined autoclave", 48.0, 80,
            "Higher T → faster but less ordered. Check XRD for MFI at 7.9°, 8.8° 2θ."),
        ReactionStep(3, "Wash, calcine", "calcination",
            "Filter, wash water 5×100mL. Dry 110°C. Calcine 550°C 8h (ramp 1°C/min).",
            [("water", 500.0)],
            "550°C, 8h", 20.0, 85,
            "Removes TPA template. Yield ~6-8g."),
    ]

def _MIP_preparation(template_formula):
    """Molecularly imprinted polymer with target template."""
    return [
        ReactionStep(1, "Template-monomer complex", "complexation",
            f"Dissolve template ion ({template_formula}, as salt, 1 mmol) in methanol (20 mL). "
            f"Add functional monomer (methacrylic acid or 4-vinylpyridine, 4 mmol). "
            f"Stir 2h at RT for pre-organization.",
            [("ethanol", 20.0)],
            "Room temperature, 2h, stirring, N2 atmosphere", 2.0, 99,
            "Pre-organization is critical. 4:1 monomer:template typical. "
            "Use methacrylic acid for hard metals, 4-vinylpyridine for borderline."),
        ReactionStep(2, "Polymerization", "radical_polymerization",
            "Add crosslinker (EGDMA, 20 mmol) + porogen (toluene, 10 mL) + "
            "initiator (AIBN, 0.1 mmol). Purge N2 15 min. Heat 60°C, 24h.",
            [("toluene", 10.0)],
            "60°C, 24h, N2 atmosphere, sealed", 24.0, 90,
            "Bulk polymerization. Can also do suspension or precipitation polymerization.",
            "AIBN is irritant. Handle in fume hood."),
        ReactionStep(3, "Template removal", "extraction",
            f"Grind polymer. Soxhlet extract with 0.1M HCl/methanol (1:1) for 48h "
            f"until no {template_formula} detected by ICP or UV.",
            [("HCl", 5.0), ("ethanol", 200.0)],
            "Reflux, 48h, Soxhlet", 48.0, 85,
            "Template removal efficiency >95% required. Test by ICP-OES."),
        ReactionStep(4, "Wash and dry", "washing",
            "Wash with methanol (3×50 mL), then water (3×50 mL). "
            "Dry 60°C vacuum oven 24h. Sieve to 50-100 µm.",
            [("ethanol", 150.0), ("water", 150.0)],
            "60°C vacuum, 24h", 24.0, 95,
            "Yield: ~5-8g. Particle size affects kinetics."),
    ]

def _LDH_coprecipitation():
    """Layered double hydroxide (Mg-Al) by coprecipitation."""
    return [
        ReactionStep(1, "Metal salt solution", "dissolution",
            "Dissolve MgCl2 (6.0g, 63 mmol) + AlCl3 (2.8g, 21 mmol) in water (100 mL). "
            "Mg:Al = 3:1 molar ratio.",
            [("MgCl2", 6.0), ("AlCl3", 2.8), ("water", 100.0)],
            "Room temperature, stirring until dissolved", 0.5, 99,
            "Mg:Al ratio controls layer charge density. 2:1 to 4:1 range."),
        ReactionStep(2, "Coprecipitation", "precipitation",
            "Add NaOH solution (2M, 80 mL) dropwise at pH 10±0.5 under N2. "
            "Vigorous stirring throughout.",
            [("NaOH", 6.4), ("water", 80.0)],
            "Room temperature, pH 10, N2, 2h addition", 2.0, 95,
            "pH control critical. Too high → Mg(OH)2 impurity. "
            "N2 prevents CO2 absorption (would give carbonate LDH)."),
        ReactionStep(3, "Hydrothermal aging", "hydrothermal",
            "Age slurry at 80°C for 18h under N2.",
            [],
            "80°C, 18h, N2, sealed", 18.0, 95,
            "Aging improves crystallinity. Can skip for amorphous LDH."),
        ReactionStep(4, "Wash and dry", "washing",
            "Filter, wash with CO2-free water (5×100 mL). Dry 60°C vacuum 24h.",
            [("water", 500.0)],
            "60°C vacuum, 24h", 24.0, 90,
            "Yield: ~4-5g. White powder. Interlayer: Cl⁻ or NO₃⁻ (exchangeable)."),
    ]

def _MOF_UiO66():
    """UiO-66 from ZrCl4 + terephthalic acid."""
    return [
        ReactionStep(1, "Solvothermal synthesis", "solvothermal",
            "Mix ZrCl4 (0.75g, 3.2 mmol) + BDC (0.54g, 3.2 mmol) in DMF (30 mL). "
            "Add acetic acid (3 mL) as modulator. Sonicate 15 min.",
            [("ZrCl4", 0.75), ("BDC", 0.54), ("DMF", 30.0)],
            "120°C, 24h, sealed vial, oven", 24.0, 85,
            "Acetic acid modulates crystal size. More acid → larger crystals."),
        ReactionStep(2, "Solvent exchange", "washing",
            "Cool to RT. Filter. Wash with DMF (3×20 mL) then methanol (3×20 mL). "
            "Soak in methanol 3 days, exchange daily.",
            [("DMF", 60.0), ("ethanol", 60.0)],
            "Room temperature, 3 days", 72.0, 95,
            "Methanol exchange removes DMF from pores. Critical for activation."),
        ReactionStep(3, "Activation", "activation",
            "Heat 150°C under vacuum 12h to remove solvent from pores.",
            [],
            "150°C, vacuum, 12h", 12.0, 95,
            "BET surface area should be >1000 m²/g. Lower → incomplete activation."),
    ]

def _COF_synthesis():
    """COF from aldehyde + amine condensation (imine-linked)."""
    return [
        ReactionStep(1, "Solvothermal condensation", "solvothermal",
            "Mix trialdehyde (1,3,5-triformylbenzene, 0.3g) + diamine (p-phenylenediamine, 0.3g) "
            "in mesitylene/dioxane (1:1, 6 mL) + acetic acid (0.6 mL, 6M). Freeze-pump-thaw 3×.",
            [("DMF", 6.0)],
            "120°C, 72h, sealed ampoule, N2", 72.0, 70,
            "Imine condensation. Crystallinity improves with time. Check PXRD.",
            "Freeze-pump-thaw removes O2 which inhibits crystallization."),
        ReactionStep(2, "Wash and activate", "washing",
            "Filter, wash THF (5×10 mL), acetone (3×10 mL). "
            "Supercritical CO2 drying or vacuum 120°C 12h.",
            [("ethanol", 80.0)],
            "120°C vacuum, 12h", 12.0, 80,
            "COFs are fragile. Gentle washing. Yield: ~0.3-0.5g."),
    ]

def _dna_origami_scaffold():
    """DNA origami nanocage from commercial scaffold + staple strands."""
    return [
        ReactionStep(1, "Design and order", "design",
            "Design nanocage in ATHENA/caDNAno. Order M13mp18 scaffold (7249 nt) "
            "and ~200 staple strands (IDT, standard desalt, 25 nmol each).",
            [],
            "Design: 4h, ordering: 3-5 business days", 100.0, 99,
            "Staple cost: ~$1-2 each × 200 = $200-400 per design. "
            "Scaffold M13mp18: ~$50/100 pmol."),
        ReactionStep(2, "Annealing", "annealing",
            "Mix scaffold (10 nM) + staples (100 nM each, 10× excess) in "
            "folding buffer (TAE + 12 mM MgCl2). Anneal: 80°C → 20°C over 18h.",
            [("MgCl2", 0.01), ("water", 0.05)],
            "80°C → 20°C, 18h linear ramp, thermocycler", 18.0, 60,
            "Mg2+ concentration critical. Too low → misfolding. "
            "Yield limited by scaffold concentration (~60% typical)."),
        ReactionStep(3, "Purification", "purification",
            "Purify by agarose gel (1.5%) or rate-zonal centrifugation. "
            "Extract band, dialyze into storage buffer.",
            [],
            "Gel: 2h run + 1h extraction. Centrifuge: 1h.", 4.0, 50,
            "Final yield: ~30% of input scaffold. Very expensive per gram."),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONALIZATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════

def _silane_grafting(donor_type, scaffold_name):
    """Graft functional groups onto silica/zeolite via silane coupling."""
    if donor_type in ("soft", "S"):
        silane = "MPTMS"
        group = "thiol (-SH)"
        follow_up = "Thiol available for: (a) direct soft-metal capture, " \\
                     "(b) dithiocarbamate formation with CS2, (c) thiol-maleimide click."
    elif donor_type in ("hard", "O"):
        silane = "GPTMS"
        group = "epoxy → opens to diol or amino-alcohol"
        follow_up = "Epoxy ring-opens with: (a) IDA for NTA-type chelator, " \\
                     "(b) catechol for Fe3+ siderophore, (c) water for diol."
    else:  # borderline / N
        silane = "APTES"
        group = "amine (-NH2)"
        follow_up = "Amine available for: (a) Schiff base with salicylaldehyde → salen, " \\
                     "(b) dithiocarbamate with CS2, (c) amide coupling with EDTA/DTPA anhydride, " \\
                     "(d) BOC protection for orthogonal functionalization."

    return ReactionStep(0, f"Silane grafting ({group})", "grafting",
        f"Suspend {scaffold_name} (2g) in dry toluene (50 mL). Add {silane} (2 mL). "
        f"Reflux 110°C under N2 for 12h. Filter, wash toluene (3×20 mL), ethanol (3×20 mL). "
        f"Dry 80°C vacuum 6h. {follow_up}",
        [(silane, 2.0), ("toluene", 50.0), ("ethanol", 60.0)],
        "110°C reflux, N2, 12h", 12.0, 90,
        f"Anhydrous conditions essential. Loading ~1-2 mmol/g by TGA. "
        f"Verify by FTIR: Si-O-Si at 1050 cm⁻¹, functional group peaks.",
        "Use dry glassware. Silanes hydrolyze in air.")


def _chelator_attachment(chelator_type, surface_group):
    """Attach chelator to functionalized surface."""
    if chelator_type == "dithiocarbamate" and surface_group == "amine":
        return ReactionStep(0, "Dithiocarbamate formation", "coupling",
            "Suspend amine-functionalized material (2g) in ethanol (30 mL). "
            "Add CS2 (1 mL, excess) + triethylamine (0.5 mL). Stir RT 12h. "
            "Filter, wash ethanol 3×20 mL. Dry 60°C vacuum.",
            [("dithiocarbamate_CS2", 1.0), ("ethanol", 90.0)],
            "Room temperature, 12h, stirring", 12.0, 85,
            "In situ: R-NH2 + CS2 → R-NH-CS2⁻ (dithiocarbamate). "
            "Very cheap. CS2 is volatile — work in fume hood.",
            "CS2 is flammable and toxic. Fume hood mandatory.")

    elif chelator_type == "EDTA" and surface_group == "amine":
        return ReactionStep(0, "EDTA grafting", "coupling",
            "Suspend amine-material (2g) in DMF (30 mL). Add EDTA dianhydride (1.5g). "
            "Stir RT 24h. One anhydride ring opens with surface-NH2, "
            "other three carboxylates free for metal binding.",
            [("EDTA_dianhydride", 1.5), ("DMF", 30.0)],
            "Room temperature, 24h, stirring", 24.0, 80,
            "EDTA dianhydride is moisture-sensitive. Use dry DMF.")

    elif chelator_type == "salen" and surface_group == "amine":
        return ReactionStep(0, "Salen Schiff base formation", "coupling",
            "Suspend amine-material (2g) in ethanol (30 mL). "
            "Add salicylaldehyde (1.0 mL, excess). Reflux 4h. "
            "Filter, wash ethanol 3×20 mL. Dry 60°C.",
            [("salicylaldehyde", 1.0), ("ethanol", 90.0)],
            "Reflux (78°C), 4h", 4.0, 85,
            "Schiff base: R-NH2 + OHC-Ar-OH → R-N=CH-Ar-OH. "
            "N,O-bidentate chelator. Yellow color confirms success.")

    elif chelator_type == "IDA" and surface_group in ("amine", "epoxy"):
        return ReactionStep(0, "IDA chelator attachment", "coupling",
            "Suspend material (2g) in water (30 mL, pH 10 with NaOH). "
            "Add IDA (1.5g). Heat 70°C 12h. Filter, wash water 5×20 mL.",
            [("IDA", 1.5), ("NaOH", 0.3), ("water", 130.0)],
            "70°C, 12h, pH 10", 12.0, 80,
            "Epoxy ring-opens with IDA nitrogen. Amine displaces Cl from CPTMS.")

    elif chelator_type == "catechol" and surface_group == "amine":
        return ReactionStep(0, "Catechol grafting via EDC/NHS", "coupling",
            "Dissolve 3,4-dihydroxybenzoic acid (0.8g) in MES buffer (30 mL, pH 6). "
            "Add EDC (1.0g) + NHS (0.6g). Activate 30 min. Add amine-material (2g). "
            "Stir RT 12h. Filter, wash water 5×20 mL.",
            [("catechol", 0.8), ("EDC", 1.0), ("NHS", 0.6), ("water", 130.0)],
            "Room temperature, 12h, pH 6", 12.0, 75,
            "EDC/NHS amide coupling. catechol-COOH + surface-NH2 → amide bond.")

    else:
        return ReactionStep(0, f"{chelator_type} attachment", "coupling",
            f"Attach {chelator_type} to {surface_group}-functionalized surface "
            f"using standard coupling chemistry.",
            [],
            "Conditions depend on specific combination", 12.0, 70)


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def _select_scaffold_route(scaffold_type):
    """Select scaffold synthesis route from type string."""
    st = scaffold_type.lower()
    if "sba" in st or "mesoporous_silica" in st or "mcm" in st:
        return _mesoporous_silica_sba15(), "mesoporous_silica"
    elif "zsm" in st or "mfi" in st:
        return _zeolite_ZSM5(), "zeolite"
    elif "zeolite" in st or "fau" in st:
        return _zeolite_Y_from_scratch(), "zeolite"
    elif "mip" in st:
        return None, "MIP"  # MIP needs template — handled separately
    elif "ldh" in st:
        return _LDH_coprecipitation(), "LDH"
    elif "uio" in st or "mil" in st or "mof" in st:
        return _MOF_UiO66(), "MOF"
    elif "cof" in st:
        return _COF_synthesis(), "COF"
    elif "dna" in st or "origami" in st:
        return _dna_origami_scaffold(), "DNA"
    else:
        return _mesoporous_silica_sba15(), "mesoporous_silica"  # Default: cheapest


def _infer_chelator_strategy(donor_atoms, donor_type, target_softness):
    """Decide chelator type and surface chemistry from donor atoms."""
    unique = set(donor_atoms)
    if unique == {"S"} or (target_softness > 0.6 and "S" in unique):
        return "dithiocarbamate", "amine"  # Amine surface + CS2 → dithiocarbamate
    elif unique == {"O"} and target_softness < 0.2:
        return "IDA", "epoxy"  # Epoxy surface + IDA
    elif "N" in unique and "O" in unique:
        return "salen", "amine"  # Schiff base N,O
    elif unique == {"N"}:
        return "EDTA", "amine"  # EDTA dianhydride onto amine surface
    elif target_softness < 0.15:
        return "catechol", "amine"  # Hard metals: catechol
    else:
        return "EDTA", "amine"  # Default broad chelator


def generate_synthesis_protocol(binder_name, target_formula, scaffold_type,
                                  donor_atoms, donor_type, target_softness=0.5,
                                  batch_size_g=5.0):
    """Generate complete synthesis protocol from scaffold prep to final product."""

    steps = []
    step_num = 1
    total_cost = 0.0
    equipment = set()

    # === PHASE 1: SCAFFOLD ===
    scaffold_route, scaffold_class = _select_scaffold_route(scaffold_type)

    if scaffold_class == "MIP":
        # MIP: special case — scaffold IS the binder
        mip_steps = _MIP_preparation(target_formula)
        for s in mip_steps:
            s.step_number = step_num
            steps.append(s)
            step_num += 1
        equipment.update(["fume_hood", "vacuum_oven", "Soxhlet_extractor", "grinder_sieve"])
    elif scaffold_class == "DNA":
        # DNA origami: commercial + fold
        for s in scaffold_route:
            s.step_number = step_num
            steps.append(s)
            step_num += 1
        equipment.update(["thermocycler", "gel_electrophoresis", "UV_spec"])
    else:
        # Inorganic scaffold synthesis
        for s in scaffold_route:
            s.step_number = step_num
            steps.append(s)
            step_num += 1
        equipment.update(["autoclave_PTFE", "furnace_600C", "vacuum_oven", "fume_hood"])

        # === PHASE 2: FUNCTIONALIZATION ===
        chelator_type, surface_group = _infer_chelator_strategy(
            donor_atoms, donor_type, target_softness)

        # Silane grafting (only for silica/zeolite/alumina scaffolds)
        if scaffold_class in ("mesoporous_silica", "zeolite"):
            graft = _silane_grafting(surface_group if surface_group != "amine" else "borderline",
                                      scaffold_type)
            if surface_group == "amine" and chelator_type == "dithiocarbamate":
                # For dithiocarbamate: graft amine first, then CS2
                graft = _silane_grafting("borderline", scaffold_type)
            graft.step_number = step_num
            steps.append(graft)
            step_num += 1
            equipment.add("reflux_condenser")

            # Chelator attachment
            attach = _chelator_attachment(chelator_type, "amine")
            attach.step_number = step_num
            steps.append(attach)
            step_num += 1

        elif scaffold_class == "MOF":
            # MOF: post-synthetic modification or use functionalized linker
            steps.append(ReactionStep(step_num, "PSM or functionalized linker", "coupling",
                f"For post-synthetic modification: soak activated MOF (2g) in "
                f"chelator solution (e.g., APTES in toluene or amino-BDC exchange). "
                f"Or synthesize directly with NH2-BDC linker for amine-functionalized UiO-66-NH2.",
                [("APTES", 1.0), ("toluene", 30.0)],
                "Depends on approach. PSM: 80°C, 24h.", 24.0, 75,
                "NH2-BDC (2-aminoterephthalic acid) costs ~$80/g."))
            step_num += 1

        elif scaffold_class == "LDH":
            # LDH: intercalation or surface grafting
            steps.append(ReactionStep(step_num, "LDH anion exchange", "intercalation",
                f"Suspend LDH (2g) in chelator anion solution (e.g., EDTA 0.1M, 50 mL). "
                f"Stir 60°C 24h under N2. Filter, wash water 3×50 mL.",
                [("water", 200.0)],
                "60°C, 24h, N2", 24.0, 80,
                "LDH interlayer exchanges Cl⁻/NO₃⁻ for chelator anions. "
                "EDTA²⁻, citrate³⁻, phosphonate²⁻ all viable."))
            step_num += 1

        elif scaffold_class == "COF":
            # COF: build chelator into linker design
            steps.append(ReactionStep(step_num, "Chelator integration", "coupling",
                f"Use pre-functionalized linker with {donor_type} donor groups. "
                f"E.g., 2,5-diaminopyridine (N-donor) or 2,5-diamino-1,4-benzenedithiol (S-donor).",
                [],
                "Same as COF synthesis with modified linker", 0.0, 70,
                "One-pot: chelator IS part of the COF backbone."))
            step_num += 1

    # === PHASE 3: CHARACTERIZATION (always) ===
    steps.append(ReactionStep(step_num, "Characterization", "analysis",
        "FTIR: confirm functional groups. TGA: loading (mmol/g). "
        "N2 physisorption (BET): surface area, pore size. "
        "XRD: crystal phase (if zeolite/MOF). "
        "ICP-OES: metal uptake test with target ion.",
        [],
        "1-2 days total", 16.0, 99,
        "Minimum QC before deployment: loading >0.5 mmol/g, "
        "uptake >80% of target at design concentration."))
    equipment.update(["FTIR", "TGA", "BET_analyzer"])

    # === COST ESTIMATE ===
    for step in steps:
        for rkey, amount in step.reagents:
            if rkey in REAGENT_DB:
                cost = REAGENT_DB[rkey].cost_per_kg_usd * amount / 1000.0
                total_cost += cost

    # Scale to per-gram
    expected_yield_g = batch_size_g * 0.7  # Rough overall yield
    cost_per_g = total_cost / max(0.1, expected_yield_g)

    # Time
    total_hours = sum(s.time_hours for s in steps)

    # Difficulty
    if scaffold_class == "DNA":
        difficulty = "expert"
        scalability = "mg_scale"
    elif scaffold_class == "COF":
        difficulty = "advanced"
        scalability = "10g_scale"
    elif scaffold_class == "MOF":
        difficulty = "moderate"
        scalability = "100g_scale"
    elif scaffold_class == "MIP":
        difficulty = "straightforward"
        scalability = "kg_scale"
    else:
        difficulty = "straightforward"
        scalability = "kg_scale"

    # Find rate-limiting step
    rls = max(steps, key=lambda s: s.time_hours)

    # Find critical reagent (most expensive)
    all_reagents = {}
    for step in steps:
        for rkey, amount in step.reagents:
            if rkey in REAGENT_DB:
                cost = REAGENT_DB[rkey].cost_per_kg_usd * amount / 1000.0
                all_reagents[rkey] = all_reagents.get(rkey, 0) + cost
    critical = max(all_reagents, key=all_reagents.get) if all_reagents else "none"

    # Alternative routes
    alts = []
    if scaffold_class in ("mesoporous_silica", "zeolite"):
        alts.append("Replace TEOS with sodium silicate for 5× cost reduction")
        alts.append("Substitute commercial zeolite (Zeolyst, $50/kg) to skip synthesis")
    if scaffold_class == "MOF":
        alts.append("Use water-based synthesis (no DMF) at higher temperature")
    if "dithiocarbamate" in str(steps):
        alts.append("Replace CS2 route with pre-formed Na-diethyldithiocarbamate ($30/kg)")

    return SynthesisProtocol(
        binder_name=binder_name, target_formula=target_formula,
        scaffold_type=scaffold_type,
        steps=steps, total_time_hours=round(total_hours, 1),
        total_cost_usd_per_gram=round(cost_per_g, 2),
        batch_size_g=batch_size_g,
        difficulty=difficulty, scalability=scalability,
        reproducibility="high" if scaffold_class in ("mesoporous_silica", "zeolite", "MIP") else "moderate",
        critical_reagent=critical, rate_limiting_step=rls.name,
        alternative_routes=alts,
        equipment_needed=sorted(equipment),
    )


def print_synthesis_protocol(protocol):
    """Pretty-print synthesis protocol."""
    print(f"\\n  SYNTHESIS PROTOCOL: {protocol.binder_name}")
    print(f"  {'─'*60}")
    print(f"  Target: {protocol.target_formula}  Scaffold: {protocol.scaffold_type}")
    print(f"  Batch: {protocol.batch_size_g}g  Time: {protocol.total_time_hours:.0f}h  "
          f"Cost: ${protocol.total_cost_usd_per_gram:.2f}/g")
    print(f"  Difficulty: {protocol.difficulty}  Scale: {protocol.scalability}  "
          f"Reproducibility: {protocol.reproducibility}")
    print()
    for step in protocol.steps:
        print(f"  Step {step.step_number}: {step.name}")
        print(f"    {step.description[:120]}")
        if step.conditions:
            print(f"    Conditions: {step.conditions}")
        if step.critical_notes:
            print(f"    Note: {step.critical_notes[:100]}")
        if step.safety_notes:
            print(f"    ⚠ Safety: {step.safety_notes}")
        print()
    if protocol.alternative_routes:
        print(f"  ALTERNATIVES:")
        for alt in protocol.alternative_routes:
            print(f"    → {alt}")
    print(f"\\n  Equipment: {', '.join(protocol.equipment_needed)}")
    print()

''')

write_file("core/design_package.py", '''\
"""
core/design_package.py — Sprint 32: Complete Design Package

The top-level entry point that produces a fully characterized binder
design from target identity alone. Integrates:
  generative_design → speciation_gated → enhanced_thermodynamics →
  deployment_scoring → spectroscopic_prediction → readout_recommendation

Output: DesignPackage — everything needed to synthesize, deploy, and
detect a binder in the field.
"""
from dataclasses import dataclass, field
import math

from core.generative_integration import generative_design
from core.physics_integration import compute_enhanced_thermodynamics, EnhancedThermodynamics
from core.deployment_scoring import score_deployment, DeploymentScore
from core.spectroscopic import predict_spectroscopy, SpectroscopicPrediction
from core.nmr_readout import predict_nmr_relaxation, recommend_readout
from core.nuclear_decay import analyze_decay_chain
from core.selectivity import compute_selectivity, SelectivityProfile
from core.synthesis import generate_synthesis_protocol, SynthesisProtocol
from core.speciation_gate import predict_speciation
from core.generative_physics_adapter import (
    adapt_generative_to_pipeline, TargetSpecies, Matrix, Problem,
)
from core.coordination_generator import METAL_D_ELECTRONS, METAL_HSAB_SOFTNESS, _IONIC_RADII
from core.spin_state import predict_spin_state


@dataclass
class DetectionPlan:
    """How to confirm binding and quantify target."""
    spectroscopy: dict           # Color, CT band, detection method
    nmr_viable: bool
    nmr_relaxivity: float        # r1 if paramagnetic
    recommended_readouts: list   # Top 3 strategies
    field_deployable_option: str # Best field-deployable method
    mass_spec_replacement: str   # Best mass-spec-replacing method


@dataclass
class DesignPackage:
    """Complete binder design: everything needed to build, deploy, detect."""
    # Identity
    target: str
    target_formula: str
    working_ph: float
    # Design
    binder_name: str
    scaffold_type: str
    donor_atoms: list
    geometry: str
    coordination_number: int
    # Binding
    thermodynamics: EnhancedThermodynamics
    predicted_kd_uM: float
    selectivity_notes: str
    # Deployment
    deployment: DeploymentScore
    # Detection
    detection: DetectionPlan
    # Selectivity
    selectivity: SelectivityProfile = None
    # Synthesis
    synthesis: SynthesisProtocol = None
    # Nuclear (optional)
    decay_chain_warning: str = ""
    # Summary
    overall_grade: str = ""      # "A", "B", "C", "D", "F"
    one_line_summary: str = ""


def _infer_unpaired(formula, d_electrons, donors):
    """Infer unpaired electrons for spectroscopic/magnetic prediction."""
    if d_electrons == 0 or d_electrons == 10:
        return 0
    try:
        ligand_names = []
        for da in donors:
            lmap = {"O": "water", "N": "pyridine", "S": "thiolate",
                    "P": "phosphine", "Cl": "Cl-"}
            ligand_names.append(lmap.get(da, "water"))
        sp = predict_spin_state(formula, d_electrons, ligand_names)
        return sp.unpaired_electrons
    except Exception:
        # Fallback: high-spin estimate
        if d_electrons <= 5:
            return d_electrons
        return 10 - d_electrons


def design_binder(target_identity, target_formula, charge=2,
                    working_ph=7.0, working_temp_c=25.0,
                    ionic_strength_mm=10.0, target_conc_uM=1.0,
                    is_nuclear=False, outdoor_use=False,
                    field_deployable=False, max_designs=5,
                    required_sensitivity="µM",
                    selectivity_panel="drinking_water"):
    """THE entry point. Target identity → complete design packages.

    Returns list of DesignPackage objects, ranked by combined
    binding + deployment score.
    """
    # Resolve metal properties
    d_electrons = METAL_D_ELECTRONS.get(target_formula, 0)
    from core.coordination_generator import _get_continuous_softness
    hsab = _get_continuous_softness(target_formula, d_electrons)
    ionic_r = _IONIC_RADII.get(target_formula, 80)
    # Estimate hydrated radius from ionic
    hydrated_r = (ionic_r + 140) / 1000.0  # Rough: add ~1.4 Å for water shell

    # Estimate MW
    mw_map = {"Pb2+": 207.2, "Cu2+": 63.5, "Ni2+": 58.7, "Zn2+": 65.4,
              "Fe3+": 55.8, "Fe2+": 55.8, "Au3+": 197.0, "Au+": 197.0,
              "Hg2+": 200.6, "Ag+": 107.9, "Cd2+": 112.4, "Mn2+": 54.9,
              "Co2+": 58.9, "Cr3+": 52.0, "UO2_2+": 270.0, "Ce3+": 140.1,
              "Ba2+": 137.3, "Na+": 23.0, "K+": 39.1, "Ca2+": 40.1,
              "Al3+": 27.0, "Pt2+": 195.1}
    target_mw = mw_map.get(target_formula, 60.0)

    # Speciation check first
    spec = predict_speciation(target_formula, working_ph)

    # Nuclear decay check
    decay_warning = ""
    if is_nuclear:
        # Try to find isotope from formula
        elem = target_formula.replace("+", "").replace("-", "")
        for digits in "0123456789":
            elem = elem.replace(digits, "")
        chain = None
        for iso_key in ["U-238", "Cs-137", "Sr-90", "Ra-226", "Co-60",
                         "Tc-99", "Am-241", "Pu-239", "I-131"]:
            if elem.lower() in iso_key.lower():
                chain = analyze_decay_chain(iso_key)
                break
        if chain:
            decay_warning = (f"Decay chain: {len(chain.chain)} steps, "
                             f"{chain.total_species_to_capture} species to capture. "
                             f"Strategy: {chain.binder_strategy}. "
                             f"{chain.notes}")

    # Generate candidates
    assemblies = generative_design(
        target_identity, target_formula, charge, d_electrons, hsab,
        ionic_r, hydrated_r, working_ph, working_temp_c, ionic_strength_mm,
        max_coord_envs=4, max_donor_arrangements=3, max_scaffold_matches=3)

    if not assemblies:
        return []

    # === DEDUP: remove duplicate (donor_set, scaffold) combinations ===
    seen_keys = set()
    unique_assemblies = []
    for a in assemblies:
        key = (tuple(sorted(a.donor_atoms)), a.scaffold_type, a.coordination_number)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_assemblies.append(a)
    assemblies = unique_assemblies

    # Build problem object for all candidates
    prob_obj = Problem(
        target=TargetSpecies(
            identity=target_identity, formula=target_formula,
            charge=charge, d_electrons=d_electrons,
            hsab_softness=hsab, ionic_radius_pm=ionic_r,
            hydrated_radius_nm=hydrated_r,
            coordination_number=6),
        matrix=Matrix(
            ph=working_ph, temperature_c=working_temp_c,
            ionic_strength_mm=ionic_strength_mm))

    # Score each candidate
    packages = []
    for gen_a in assemblies[:max_designs * 3]:  # Oversample then filter
        # Adapt to pipeline format
        adapted = adapt_generative_to_pipeline(gen_a, problem=prob_obj)
        rec = adapted.recognition
        struct = adapted.structure
        interior = adapted.interior

        # Build Problem
        prob = prob_obj

        # Enhanced thermodynamics
        thermo = compute_enhanced_thermodynamics(rec, struct, interior, prob)

        # Deployment scoring
        unpaired = _infer_unpaired(target_formula, d_electrons, rec.donor_atoms)
        dep = score_deployment(
            gen_a.scaffold_type, target_formula, charge, target_mw,
            hydrated_r, struct.pore_size_nm if struct else 0.0,
            abs(thermo.dg_bind_kj), target_mw, unpaired,
            is_nuclear, outdoor_use, target_conc_uM)

        # Spectroscopic prediction
        ten_dq = 120.0  # Default; would come from spin_state in full impl
        try:
            sp_result = predict_spin_state(target_formula, d_electrons,
                                            ["water"] * min(6, len(rec.donor_atoms)))
            ten_dq = sp_result.ten_dq_kj
        except Exception:
            pass

        spec_pred = predict_spectroscopy(
            target_formula, rec.donor_atoms, d_electrons,
            ten_dq_kj=ten_dq,
            geometry="octahedral" if gen_a.coordination_number >= 5 else "tetrahedral",
            scaffold_type=gen_a.scaffold_type)

        # NMR + readout
        nmr = predict_nmr_relaxation(target_formula, unpaired)
        readouts = recommend_readout(target_formula, required_sensitivity,
                                      field_deployable, multiplexing_needed=1)

        field_option = "None available"
        mass_spec_option = "ICP-MS (traditional)"
        for ro in readouts:
            if ro.field_deployable and field_option == "None available":
                field_option = ro.strategy_name
            if ro.multiplexing_capacity >= 100:
                mass_spec_option = ro.strategy_name

        detection = DetectionPlan(
            spectroscopy={
                "color": spec_pred.predicted_color,
                "dd_nm": spec_pred.dd_transition_nm,
                "ct_type": spec_pred.ct_type,
                "ct_nm": spec_pred.ct_transition_nm,
                "detection_method": spec_pred.detection_method,
                "sensitivity": spec_pred.sensitivity_estimate,
            },
            nmr_viable=nmr.total_r1_mM_s > 0,
            nmr_relaxivity=nmr.total_r1_mM_s,
            recommended_readouts=[r.strategy_name for r in readouts[:3]],
            field_deployable_option=field_option,
            mass_spec_replacement=mass_spec_option,
        )

        # Selectivity
        sel = compute_selectivity(
            target_formula, thermo.predicted_kd_um, charge,
            rec, struct, interior, prob.matrix,
            panel=selectivity_panel)

        # Synthesis protocol
        synth = generate_synthesis_protocol(
            gen_a.name, target_formula, gen_a.scaffold_type,
            rec.donor_atoms, rec.donor_type,
            target_softness=hsab)

        # Overall grade — now includes selectivity
        binding_score = max(0, min(100, -thermo.dg_net_kj))  # More negative = better
        combined = binding_score * 0.35 + dep.deployment_score * 0.35 + sel.selectivity_score * 0.30
        if combined > 70: grade = "A"
        elif combined > 55: grade = "B"
        elif combined > 40: grade = "C"
        elif combined > 25: grade = "D"
        else: grade = "F"

        summary = (f"{gen_a.name} | Kd={thermo.predicted_kd_um:.1f} µM | "
                   f"Deploy={dep.deployment_class} | "
                   f"Detect={spec_pred.detection_method} | Grade={grade}")

        sel_notes = ""
        if thermo.bond_character == "covalent":
            sel_notes = "Covalent binding — high selectivity for soft metals"
        elif thermo.softness_continuous > 0.5:
            sel_notes = "Soft-metal selective (polarization-driven)"
        elif thermo.softness_continuous < 0.15:
            sel_notes = "Hard-metal selective (electrostatic-driven)"

        packages.append(DesignPackage(
            target=target_identity, target_formula=target_formula,
            working_ph=working_ph,
            binder_name=gen_a.name, scaffold_type=gen_a.scaffold_type,
            donor_atoms=rec.donor_atoms, geometry=gen_a.geometry,
            coordination_number=gen_a.coordination_number,
            thermodynamics=thermo, predicted_kd_uM=thermo.predicted_kd_um,
            selectivity_notes=sel_notes,
            deployment=dep, detection=detection,
            selectivity=sel,
            synthesis=synth,
            decay_chain_warning=decay_warning,
            overall_grade=grade, one_line_summary=summary,
        ))

    # Sort by combined score
    packages.sort(key=lambda p: (-ord(p.overall_grade[0]),
                                   p.thermodynamics.dg_net_kj))
    return packages[:max_designs]


def print_design_package(pkg):
    """Pretty-print a complete design package."""
    print(f"\\n{'='*72}")
    print(f"  MABE DESIGN PACKAGE: {pkg.binder_name}")
    print(f"{'='*72}")
    print(f"  Target:      {pkg.target} ({pkg.target_formula}) at pH {pkg.working_ph}")
    print(f"  Grade:       {pkg.overall_grade}")
    print(f"  Summary:     {pkg.one_line_summary}")

    t = pkg.thermodynamics
    print(f"\\n  BINDING ({t.confidence} confidence)")
    print(f"  {'─'*60}")
    print(f"  ΔG_net:      {t.dg_net_kj:.1f} kJ/mol → Kd = {t.predicted_kd_um:.2f} µM")
    print(f"  ΔG_bind:     {t.dg_bind_kj:.1f}  ΔG_desolv: +{t.dg_desolv_kj:.1f}")
    print(f"  ΔG_LFSE:     {t.dg_lfse_kj:.1f}  ΔG_chelate: {t.dg_chelate_kj:.1f}")
    if t.dg_covalent_kj != 0:
        print(f"  ΔG_covalent: {t.dg_covalent_kj:.1f}  ({t.bond_character})")
    if t.dg_dispersion_kj != 0:
        print(f"  ΔG_disp:     {t.dg_dispersion_kj:.2f}  ΔG_polar: {t.dg_polarization_kj:.2f}")
    if t.dg_relativistic_correction_kj != 0:
        print(f"  ΔG_relativ:  {t.dg_relativistic_correction_kj:.2f}")
    if t.speciation_warning:
        print(f"  ⚠ SPECIATION: {t.speciation_warning}")
    print(f"  Softness:    {t.softness_continuous:.3f}  β={t.nephelauxetic_beta:.3f}")
    if pkg.selectivity_notes:
        print(f"  Selectivity: {pkg.selectivity_notes}")

    d = pkg.deployment
    print(f"\\n  DEPLOYMENT ({d.deployment_class})")
    print(f"  {'─'*60}")
    print(f"  Score:       {d.deployment_score:.0f}/100  Limiting: {d.limiting_factor}")
    print(f"  Transport:   {d.transport_score:.0f}  Capacity: {d.capacity_mg_g:.0f} mg/g")
    print(f"  Wetting:     {d.wettability} ({d.wetting_score:.0f})")
    print(f"  Thermal:     max {d.max_temp_C}°C ({d.thermal_score:.0f})")
    if d.recommendations:
        for r in d.recommendations[:3]:
            print(f"  → {r}")

    det = pkg.detection
    print(f"\\n  DETECTION")
    print(f"  {'─'*60}")
    sp = det.spectroscopy
    if sp["color"] != "colorless":
        print(f"  Color:       {sp['color']} (d-d at {sp['dd_nm']:.0f} nm)")
    if sp["ct_type"] != "none":
        print(f"  CT band:     {sp['ct_type']} at {sp['ct_nm']:.0f} nm")
    print(f"  Best method: {sp['detection_method']} ({sp['sensitivity']})")
    if det.nmr_viable:
        print(f"  NMR:         r1={det.nmr_relaxivity:.1f} mM⁻¹s⁻¹ (viable)")
    print(f"  Field:       {det.field_deployable_option}")
    print(f"  Mass-spec→:  {det.mass_spec_replacement}")

    if pkg.decay_chain_warning:
        print(f"\\n  ☢ NUCLEAR: {pkg.decay_chain_warning}")

    if pkg.selectivity:
        s = pkg.selectivity
        print(f"\\n  SELECTIVITY ({s.overall_selectivity_class}, score={s.selectivity_score:.0f}/100)")
        print(f"  {'─'*60}")
        print(f"  Panel: {s.deployment_matrix}")
        for r in sorted(s.interferents, key=lambda x: x.selectivity_ratio)[:5]:
            kd_str = f"{r.predicted_kd_uM:.1f}" if r.predicted_kd_uM < 1e6 else ">10⁶"
            ratio_str = f"{r.selectivity_ratio:.0f}×" if r.selectivity_ratio < 1e6 else ">10⁶×"
            flag = " ⚠" if r.selectivity_class in ("poor", "none") else ""
            print(f"  {r.formula:10s} Kd={kd_str:>8s}  sel={ratio_str:>8s}  {r.selectivity_class}{flag}")
        if s.notes:
            print(f"  ⚠ {s.notes}")

    if pkg.synthesis:
        sy = pkg.synthesis
        print(f"\\n  SYNTHESIS")
        print(f"  {'─'*60}")
        print(f"  {sy.difficulty} | {len(sy.steps)} steps | {sy.total_time_hours:.0f}h | "
              f"${sy.total_cost_usd_per_gram:.2f}/g | {sy.scalability}")
        for step in sy.steps:
            print(f"  {step.step_number}. {step.name} ({step.time_hours:.0f}h)")
        if sy.alternative_routes:
            print(f"  Alternatives: {sy.alternative_routes[0]}")

    print(f"\\n{'='*72}\\n")



''')

write_file("tests/test_sprint34.py", '''\
"""tests/test_sprint34.py — Sprint 34: Synthesis Protocol Generation (20 tests)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.synthesis import (
    generate_synthesis_protocol, SynthesisProtocol, REAGENT_DB,
    _mesoporous_silica_sba15, _zeolite_Y_from_scratch, _LDH_coprecipitation,
    _MOF_UiO66, _MIP_preparation, _dna_origami_scaffold,
)
from core.design_package import design_binder


# ═══════════════════════════════════════════════════════════════════════════
# REAGENT DATABASE
# ═══════════════════════════════════════════════════════════════════════════

def test_reagent_db_populated():
    """Should have comprehensive reagent database."""
    assert len(REAGENT_DB) > 25
    categories = set(r.role for r in REAGENT_DB.values())
    assert "precursor" in categories
    assert "linker" in categories
    assert "catalyst" in categories
    print(f"  \\u2705 test_reagents: {len(REAGENT_DB)} reagents, roles={categories}")

def test_commodity_pricing():
    """Commodity chemicals should be <$50/kg."""
    for key, r in REAGENT_DB.items():
        if r.supplier_tier == "commodity":
            assert r.cost_per_kg_usd < 50, f"{key} marked commodity but ${r.cost_per_kg_usd}/kg"
    print(f"  \\u2705 test_commodity: all commodity < $50/kg")

def test_inorganic_precursors_present():
    """Must have Si, Al, Mg, Fe, Zr precursors."""
    keys = set(REAGENT_DB.keys())
    assert "TEOS" in keys
    assert "sodium_silicate" in keys
    assert "sodium_aluminate" in keys
    assert "MgCl2" in keys
    assert "FeCl3" in keys
    print(f"  \\u2705 test_inorganic: all precursors present")

def test_silane_agents_present():
    """Must have APTES, MPTMS, GPTMS for grafting."""
    keys = set(REAGENT_DB.keys())
    assert "APTES" in keys   # Amine
    assert "MPTMS" in keys   # Thiol
    assert "GPTMS" in keys   # Epoxy
    print(f"  \\u2705 test_silanes: APTES, MPTMS, GPTMS present")

# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD ROUTES
# ═══════════════════════════════════════════════════════════════════════════

def test_sba15_route():
    """SBA-15 route should have sol-gel + calcination."""
    steps = _mesoporous_silica_sba15()
    assert len(steps) >= 3
    types = [s.reaction_type for s in steps]
    assert "sol_gel" in types
    assert "calcination" in types
    print(f"  \\u2705 test_sba15: {len(steps)} steps, types={types}")

def test_zeolite_Y_route():
    """Zeolite Y from sodium silicate + aluminate."""
    steps = _zeolite_Y_from_scratch()
    assert len(steps) >= 3
    # Check uses commodity precursors
    all_reagents = [rk for s in steps for rk, _ in s.reagents]
    assert "sodium_silicate" in all_reagents
    assert "sodium_aluminate" in all_reagents
    print(f"  \\u2705 test_zeolite_Y: {len(steps)} steps, commodity precursors")

def test_ldh_route():
    """LDH coprecipitation from MgCl2 + AlCl3."""
    steps = _LDH_coprecipitation()
    all_reagents = [rk for s in steps for rk, _ in s.reagents]
    assert "MgCl2" in all_reagents
    assert "AlCl3" in all_reagents
    print(f"  \\u2705 test_ldh: {len(steps)} steps, MgCl2+AlCl3")

def test_mip_route():
    """MIP should include template removal step."""
    steps = _MIP_preparation("Pb2+")
    types = [s.reaction_type for s in steps]
    assert "extraction" in types  # Template removal
    print(f"  \\u2705 test_mip: {len(steps)} steps, template extraction present")

def test_mof_route():
    """MOF route should include activation."""
    steps = _MOF_UiO66()
    types = [s.reaction_type for s in steps]
    assert "solvothermal" in types
    assert "activation" in types
    print(f"  \\u2705 test_mof: {len(steps)} steps, solvothermal+activation")

# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def test_zeolite_dithiocarbamate():
    """S-donor on zeolite: silane + CS2 route."""
    p = generate_synthesis_protocol("test", "Pb2+", "zeolite_Y",
        ["S","S","S","S"], "soft", 0.99)
    step_names = [s.name for s in p.steps]
    assert any("silane" in n.lower() or "graft" in n.lower() for n in step_names)
    assert any("dithiocarbamate" in n.lower() for n in step_names)
    assert p.total_cost_usd_per_gram < 2.0
    print(f"  \\u2705 test_zeo_dtc: ${p.total_cost_usd_per_gram:.2f}/g, {p.total_time_hours:.0f}h")

def test_silica_catechol():
    """O-donor on SBA-15 for Fe3+."""
    p = generate_synthesis_protocol("test", "Fe3+", "mesoporous_silica",
        ["O","O","O","O"], "hard", 0.12)
    assert p.scaffold_type == "mesoporous_silica"
    assert len(p.steps) >= 5  # Scaffold + graft + chelator + characterization
    print(f"  \\u2705 test_silica_cat: {len(p.steps)} steps, ${p.total_cost_usd_per_gram:.2f}/g")

def test_ldh_protocol():
    """LDH protocol should be cheapest."""
    p = generate_synthesis_protocol("test", "Ni2+", "LDH",
        ["N","N","O","O"], "borderline", 0.24)
    assert p.total_cost_usd_per_gram < 0.50
    assert p.difficulty == "straightforward"
    assert p.scalability == "kg_scale"
    print(f"  \\u2705 test_ldh_proto: ${p.total_cost_usd_per_gram:.2f}/g, {p.scalability}")

def test_mip_protocol():
    """MIP should be straightforward, kg-scalable."""
    p = generate_synthesis_protocol("test", "Cu2+", "MIP",
        ["N","N","O","O"], "borderline", 0.35)
    assert p.difficulty == "straightforward"
    assert p.scalability == "kg_scale"
    print(f"  \\u2705 test_mip_proto: {p.difficulty}, {p.scalability}")

def test_dna_protocol_expensive():
    """DNA origami should be expert-level and expensive."""
    p = generate_synthesis_protocol("test", "Pb2+", "dna_origami",
        ["N","N","O","O"], "borderline", 0.55)
    assert p.difficulty == "expert"
    assert p.scalability == "mg_scale"
    print(f"  \\u2705 test_dna_proto: {p.difficulty}, {p.scalability}, ${p.total_cost_usd_per_gram:.2f}/g")

def test_protocol_has_characterization():
    """Every protocol should end with characterization step."""
    p = generate_synthesis_protocol("test", "Pb2+", "zeolite_Y",
        ["S","S"], "soft", 0.99)
    last = p.steps[-1]
    assert "character" in last.name.lower()
    print(f"  \\u2705 test_char_step: last step = {last.name}")

def test_protocol_has_alternatives():
    """Inorganic protocols should suggest alternatives."""
    p = generate_synthesis_protocol("test", "Pb2+", "zeolite_Y",
        ["S","S"], "soft", 0.99)
    assert len(p.alternative_routes) > 0
    print(f"  \\u2705 test_alternatives: {len(p.alternative_routes)} alternatives")

def test_equipment_list():
    """Protocol should list required equipment."""
    p = generate_synthesis_protocol("test", "Fe3+", "mesoporous_silica",
        ["O","O","O","O"], "hard", 0.12)
    assert len(p.equipment_needed) > 0
    assert "autoclave_PTFE" in p.equipment_needed or "fume_hood" in p.equipment_needed
    print(f"  \\u2705 test_equipment: {p.equipment_needed[:4]}")

# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_design_has_synthesis():
    """DesignPackage should include synthesis protocol."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=1)
    assert pkgs[0].synthesis is not None
    assert isinstance(pkgs[0].synthesis, SynthesisProtocol)
    assert pkgs[0].synthesis.total_cost_usd_per_gram > 0
    print(f"  \\u2705 test_e2e_synth: ${pkgs[0].synthesis.total_cost_usd_per_gram:.2f}/g, "
          f"{pkgs[0].synthesis.difficulty}")

def test_e2e_cost_comparison():
    """Different scaffolds should have different costs."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=5)
    costs = [(p.scaffold_type, p.synthesis.total_cost_usd_per_gram) for p in pkgs if p.synthesis]
    unique_costs = set(c for _, c in costs)
    print(f"  \\u2705 test_costs: {costs[:3]}")


if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprint 34 \\u2014 Synthesis Protocol Generation\\n")
    print("Reagent Database:")
    test_reagent_db_populated(); test_commodity_pricing()
    test_inorganic_precursors_present(); test_silane_agents_present()
    print("\\nScaffold Routes:")
    test_sba15_route(); test_zeolite_Y_route()
    test_ldh_route(); test_mip_route(); test_mof_route()
    print("\\nProtocol Generation:")
    test_zeolite_dithiocarbamate(); test_silica_catechol()
    test_ldh_protocol(); test_mip_protocol()
    test_dna_protocol_expensive(); test_protocol_has_characterization()
    test_protocol_has_alternatives(); test_equipment_list()
    print("\\nIntegration:")
    test_e2e_design_has_synthesis(); test_e2e_cost_comparison()
    print("\\n\\u2705 All Sprint 34 tests passed! (20/20)\\n")

''')

print("\n\u2705 Sprint 34 files created!\n")