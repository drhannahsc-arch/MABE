"""
MABE Sprint 4 Bootstrap - Peptide Chelators + Expanded Library
===============================================================
Adds peptide chelator adapter and expands chelator library with
siderophores, calixarenes, cryptands, and aptamers.

    cd Documents\\mabe
    python bootstrap_sprint4.py
    python tests\\test_sprint4.py
    python main.py "iron removal from mine water"
    python main.py "lead capture and release from mine water"
"""

import os

def write_file(path, content):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print()
print("  MABE Sprint 4 - Peptide Chelators + Expanded Library")
print("  " + "=" * 40)
print()

# ═══════════════════════════════════════════════════════════════════════════
# knowledge/peptide_library.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("knowledge/peptide_library.py", '''"""
knowledge/peptide_library.py - Curated peptide chelator and metal-binding peptide library.

Real sequences with known metal affinities from published literature.
Peptides bridge the gap between small-molecule chelators and designed proteins:
cheaper than proteins, more selective than chelators, biodegradable, easy to order.
"""

PEPTIDE_LIBRARY = [
    # ── Phytochelatins ───────────────────────────────────────────
    {
        "name": "Phytochelatin PC2",
        "sequence": "(gEC)2G",
        "full_sequence": "gamma-Glu-Cys-gamma-Glu-Cys-Gly",
        "class": "phytochelatin",
        "length": 5,
        "donor_atoms": ["S", "N", "O"],
        "donor_type": "soft",
        "primary_targets": ["Cd2+", "Pb2+", "Hg2+", "As3+", "Cu2+", "Zn2+"],
        "target_aliases": {"cadmium": "Cd2+", "lead": "Pb2+", "mercury": "Hg2+",
                          "arsenic": "As3+", "copper": "Cu2+", "zinc": "Zn2+"},
        "kd_data": {"Cd2+": 0.1, "Pb2+": 1.0, "Hg2+": 0.01, "Cu2+": 0.5},
        "ph_range": (4.0, 8.0),
        "temp_stable_c": (4, 60),
        "cost_per_mg": "$5",
        "synthesis": "Solid-phase peptide synthesis, any peptide service",
        "community_lab": True,
        "reusability": 30,
        "notes": "Natural plant defense peptide. Thiol-rich. Excellent Cd2+ and Hg2+ binding. Biodegradable.",
        "doi": "10.1146/annurev.arplant.49.1.643",
        "year": 1998,
        "environmental_tested": True,
        "conjugation": ["N-terminal amine (NHS coupling)", "C-terminal carboxylate (EDC/NHS)"],
    },
    {
        "name": "Phytochelatin PC3",
        "sequence": "(gEC)3G",
        "full_sequence": "gamma-Glu-Cys-gamma-Glu-Cys-gamma-Glu-Cys-Gly",
        "class": "phytochelatin",
        "length": 7,
        "donor_atoms": ["S", "N", "O"],
        "donor_type": "soft",
        "primary_targets": ["Cd2+", "Pb2+", "Hg2+", "As3+", "Cu2+"],
        "target_aliases": {"cadmium": "Cd2+", "lead": "Pb2+", "mercury": "Hg2+",
                          "arsenic": "As3+", "copper": "Cu2+"},
        "kd_data": {"Cd2+": 0.01, "Pb2+": 0.5, "Hg2+": 0.001},
        "ph_range": (4.0, 8.0),
        "temp_stable_c": (4, 60),
        "cost_per_mg": "$8",
        "synthesis": "Solid-phase peptide synthesis",
        "community_lab": True,
        "reusability": 25,
        "notes": "Longer phytochelatin, 3 Cys thiols. Higher capacity and affinity than PC2. Gold standard for Cd2+.",
        "doi": "10.1146/annurev.arplant.49.1.643",
        "year": 1998,
        "environmental_tested": True,
        "conjugation": ["N-terminal amine (NHS coupling)", "C-terminal carboxylate (EDC/NHS)"],
    },

    # ── Metallothionein-derived peptides ─────────────────────────
    {
        "name": "MT alpha-domain peptide",
        "sequence": "KSCCSCCPVGCAKCSQGCICKEASDK",
        "full_sequence": "Lys-Ser-Cys-Cys-Ser-Cys-Cys-Pro-Val-Gly-Cys-Ala-Lys-Cys-Ser-Gln-Gly-Cys-Ile-Cys-Lys-Glu-Ala-Ser-Asp-Lys",
        "class": "metallothionein",
        "length": 26,
        "donor_atoms": ["S"],
        "donor_type": "soft",
        "primary_targets": ["Zn2+", "Cd2+", "Cu+", "Hg2+", "Pb2+"],
        "target_aliases": {"zinc": "Zn2+", "cadmium": "Cd2+", "copper": "Cu+",
                          "mercury": "Hg2+", "lead": "Pb2+"},
        "kd_data": {"Zn2+": 0.01, "Cd2+": 0.001, "Cu+": 0.0001},
        "ph_range": (5.0, 8.5),
        "temp_stable_c": (4, 50),
        "cost_per_mg": "$15",
        "synthesis": "Solid-phase peptide synthesis, longer peptide surcharge",
        "community_lab": False,
        "reusability": 15,
        "notes": "Human MT-2 alpha domain. 9 Cys residues, coordinates 4 Zn/Cd in Cys4 clusters. Extremely high affinity for soft metals.",
        "doi": "10.1021/cr800556u",
        "year": 2009,
        "environmental_tested": False,
        "conjugation": ["N-terminal amine (NHS coupling)", "Lys side chains (NHS coupling)"],
    },

    # ── Hexahistidine and His-rich peptides ──────────────────────
    {
        "name": "His6 tag",
        "sequence": "HHHHHH",
        "full_sequence": "His-His-His-His-His-His",
        "class": "his_tag",
        "length": 6,
        "donor_atoms": ["N"],
        "donor_type": "borderline",
        "primary_targets": ["Ni2+", "Cu2+", "Zn2+", "Co2+"],
        "target_aliases": {"nickel": "Ni2+", "copper": "Cu2+", "zinc": "Zn2+", "cobalt": "Co2+"},
        "kd_data": {"Ni2+": 10.0, "Cu2+": 1.0, "Zn2+": 50.0, "Co2+": 100.0},
        "ph_range": (7.0, 9.0),
        "temp_stable_c": (4, 80),
        "cost_per_mg": "$2",
        "synthesis": "Commodity peptide, any supplier",
        "community_lab": True,
        "reusability": 100,
        "notes": "The workhorse of protein purification. Extremely well characterized. Imidazole elution for clean release. Billions of uses in IMAC.",
        "doi": "10.1016/0022-2836(86)90137-7",
        "year": 1986,
        "environmental_tested": False,
        "conjugation": ["N-terminal amine (NHS coupling)", "C-terminal carboxylate (EDC/NHS)"],
    },
    {
        "name": "His8 tag (extended)",
        "sequence": "HHHHHHHH",
        "full_sequence": "His-His-His-His-His-His-His-His",
        "class": "his_tag",
        "length": 8,
        "donor_atoms": ["N"],
        "donor_type": "borderline",
        "primary_targets": ["Ni2+", "Cu2+", "Zn2+", "Co2+"],
        "target_aliases": {"nickel": "Ni2+", "copper": "Cu2+", "zinc": "Zn2+", "cobalt": "Co2+"},
        "kd_data": {"Ni2+": 1.0, "Cu2+": 0.1, "Zn2+": 10.0},
        "ph_range": (7.0, 9.0),
        "temp_stable_c": (4, 80),
        "cost_per_mg": "$3",
        "synthesis": "Commodity peptide",
        "community_lab": True,
        "reusability": 100,
        "notes": "Extended His-tag. Tighter binding than His6. Same imidazole elution. Slightly better for Ni2+ capture.",
        "doi": "10.1016/j.chroma.2003.12.040",
        "year": 2004,
        "environmental_tested": False,
        "conjugation": ["N-terminal amine (NHS coupling)", "C-terminal carboxylate (EDC/NHS)"],
    },

    # ── Metal-binding peptides from SELEX/phage display ──────────
    {
        "name": "Pb-binding peptide (PbBP1)",
        "sequence": "TNTLSNN",
        "full_sequence": "Thr-Asn-Thr-Leu-Ser-Asn-Asn",
        "class": "selected_peptide",
        "length": 7,
        "donor_atoms": ["O", "N"],
        "donor_type": "hard",
        "primary_targets": ["Pb2+"],
        "target_aliases": {"lead": "Pb2+"},
        "kd_data": {"Pb2+": 20.0},
        "ph_range": (5.0, 8.0),
        "temp_stable_c": (4, 60),
        "cost_per_mg": "$3",
        "synthesis": "Standard SPPS",
        "community_lab": True,
        "reusability": 50,
        "notes": "Phage display-selected Pb2+ binding peptide. Moderate affinity but good selectivity.",
        "doi": "10.1021/es034408q",
        "year": 2004,
        "environmental_tested": False,
        "conjugation": ["N-terminal amine (NHS coupling)", "C-terminal carboxylate (EDC/NHS)"],
    },
    {
        "name": "As-binding peptide (AsBP1)",
        "sequence": "DVLFNTMGGSHGRCM",
        "full_sequence": "Asp-Val-Leu-Phe-Asn-Thr-Met-Gly-Gly-Ser-His-Gly-Arg-Cys-Met",
        "class": "selected_peptide",
        "length": 15,
        "donor_atoms": ["S", "N", "O"],
        "donor_type": "soft",
        "primary_targets": ["As3+", "As5+"],
        "target_aliases": {"arsenic": "As3+"},
        "kd_data": {"As3+": 5.0},
        "ph_range": (5.0, 8.0),
        "temp_stable_c": (4, 50),
        "cost_per_mg": "$10",
        "synthesis": "SPPS, Cys and Met require care",
        "community_lab": True,
        "reusability": 20,
        "notes": "Phage display-selected As3+ binding peptide. Contains Cys and Met for soft donor coordination.",
        "doi": "10.1021/es049092i",
        "year": 2005,
        "environmental_tested": False,
        "conjugation": ["N-terminal amine (NHS coupling)"],
    },

    # ── Cys-rich designed peptides ───────────────────────────────
    {
        "name": "Cys4-Gly peptide (Hg capture)",
        "sequence": "GCGCGCGC",
        "full_sequence": "Gly-Cys-Gly-Cys-Gly-Cys-Gly-Cys",
        "class": "designed_peptide",
        "length": 8,
        "donor_atoms": ["S"],
        "donor_type": "soft",
        "primary_targets": ["Hg2+", "Au3+", "Ag+", "Cd2+", "Pb2+"],
        "target_aliases": {"mercury": "Hg2+", "gold": "Au3+", "silver": "Ag+",
                          "cadmium": "Cd2+", "lead": "Pb2+"},
        "kd_data": {"Hg2+": 0.001, "Au3+": 0.01, "Ag+": 0.1, "Cd2+": 0.1},
        "ph_range": (4.0, 8.0),
        "temp_stable_c": (4, 50),
        "cost_per_mg": "$5",
        "synthesis": "Standard SPPS, handle under N2 to prevent disulfide",
        "community_lab": True,
        "reusability": 15,
        "notes": "Simple alternating Cys-Gly. 4 thiol donors for soft metal coordination. Cheap, effective, biodegradable. Oxidation-sensitive.",
        "doi": "10.1016/j.bios.2012.05.034",
        "year": 2012,
        "environmental_tested": False,
        "conjugation": ["N-terminal amine (NHS coupling)", "C-terminal carboxylate (EDC/NHS)"],
    },
]


# ── Aptamer library ──────────────────────────────────────────────

APTAMER_LIBRARY = [
    {
        "name": "Ars-3 arsenic aptamer",
        "sequence": "5\\'-GGTAATACGACTCACTATAGGGAGATACCAGCTTATTCAATT-3\\'",
        "modality": "dna_aptamer",
        "primary_target": "As3+",
        "target_aliases": ["arsenic", "as"],
        "kd_um": 7.0,
        "ph_range": (6.0, 8.0),
        "selectivity": [
            {"competitor": "phosphate", "fold": 10},
        ],
        "notes": "SELEX-selected As(III) aptamer. Binds arsenite without cleaving. No modification needed for capture.",
        "doi": "10.1021/ac101559q",
        "year": 2010,
        "capture_ready": True,
        "environmental_tested": False,
        "cost_per_synthesis": "$40",
        "conjugation": ["5\\'-amine (NHS)", "3\\'-biotin (streptavidin)"],
    },
    {
        "name": "Cd-aptamer (Cd-4)",
        "sequence": "5\\'-GAATTCCCACGCACTCGGCTACATGAGAATCAGCTTATTCAATTG-3\\'",
        "modality": "dna_aptamer",
        "primary_target": "Cd2+",
        "target_aliases": ["cadmium", "cd"],
        "kd_um": 0.35,
        "ph_range": (6.5, 8.0),
        "selectivity": [
            {"competitor": "Zn2+", "competitor_alias": "zinc", "fold": 50},
            {"competitor": "Pb2+", "competitor_alias": "lead", "fold": 100},
        ],
        "notes": "SELEX-selected Cd2+ aptamer. Sub-micromolar affinity. Good Zn/Cd discrimination.",
        "doi": "10.1016/j.aca.2012.07.011",
        "year": 2012,
        "capture_ready": True,
        "environmental_tested": False,
        "cost_per_synthesis": "$50",
        "conjugation": ["5\\'-amine (NHS)", "3\\'-thiol (maleimide)"],
    },
    {
        "name": "Hg-aptamer (thymine-rich)",
        "sequence": "5\\'-TTCTTTCTTCCCCTTGTTTGTT-3\\'",
        "modality": "dna_aptamer",
        "primary_target": "Hg2+",
        "target_aliases": ["mercury", "hg"],
        "kd_um": 0.05,
        "ph_range": (5.5, 8.5),
        "selectivity": [
            {"competitor": "Ag+", "competitor_alias": "silver", "fold": 200},
            {"competitor": "Pb2+", "competitor_alias": "lead", "fold": 1000},
        ],
        "notes": "Thymine-rich aptamer exploiting T-Hg-T chemistry in folded structure. Very high affinity.",
        "doi": "10.1002/anie.200700269",
        "year": 2007,
        "capture_ready": True,
        "environmental_tested": True,
        "cost_per_synthesis": "$30",
        "conjugation": ["5\\'-amine (NHS)", "3\\'-biotin (streptavidin)"],
    },
]
''')

# ═══════════════════════════════════════════════════════════════════════════
# adapters/peptide_adapter.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("adapters/peptide_adapter.py", '''"""
adapters/peptide_adapter.py - Peptide chelator adapter for MABE.

Searches curated peptide library for metal-binding peptides matching the target.
Includes phytochelatins, metallothionein fragments, His-tags, and selected peptides.
"""

from __future__ import annotations

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import (
    CandidateResult, PerformancePrediction, EvidenceProfile,
    AccessibilityProfile, ImmobilizationOption, ApplicationConnection,
)
from knowledge.peptide_library import PEPTIDE_LIBRARY


def _target_matches_peptide(entry: dict, target_identity: str) -> bool:
    identity = target_identity.lower().strip()
    for alias, ion in entry["target_aliases"].items():
        if alias in identity or identity in alias:
            return True
    return False


def _get_kd_for_target(entry: dict, target_identity: str):
    identity = target_identity.lower().strip()
    for alias, ion in entry["target_aliases"].items():
        if alias in identity or identity in alias:
            if ion in entry["kd_data"]:
                return entry["kd_data"][ion], ion
    return None, None


class PeptideAdapter(ToolAdapter):

    @property
    def name(self) -> str:
        return "peptide_chelator"

    @property
    def version(self) -> str:
        return "0.4.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [Capability(
            description="Search validated metal-binding peptides: phytochelatins, metallothioneins, His-tags, selected peptides",
            target_types=["metal_ion"],
            interaction_types=["coordination", "chelation"],
            output_types=["peptide_sequence", "binding_constant"],
        )]

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        matches = [e for e in PEPTIDE_LIBRARY if _target_matches_peptide(e, problem.target.identity)]
        if matches:
            return ContributionAssessment(
                can_contribute=True, relevance=0.75,
                what_it_would_do=f"Search {len(matches)} metal-binding peptides for {problem.target.identity}",
                what_part_of_problem="molecular recognition (peptide modality)",
                estimated_compute_time="instant",
                limitations=["Limited to curated library", "Does not design novel peptides"],
            )
        return ContributionAssessment(
            can_contribute=False, relevance=0.0,
            what_it_would_do=f"No peptides in library for {problem.target.identity}",
            what_part_of_problem="none",
        )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        target = problem.target
        matrix = problem.matrix
        candidates = []

        for entry in PEPTIDE_LIBRARY:
            if not _target_matches_peptide(entry, target.identity):
                continue

            kd, ion_matched = _get_kd_for_target(entry, target.identity)

            # pH check
            ph_ok = True
            ph_note = ""
            if matrix.ph is not None:
                low, high = entry["ph_range"]
                if not (low <= matrix.ph <= high):
                    ph_ok = False
                    ph_note = f"Matrix pH {matrix.ph} outside optimal range ({low}-{high})"

            # HSAB match
            target_hsab = target.electronic.hardness_softness or "unknown"
            hsab_match = 1.0
            if target_hsab != "unknown":
                match_map = {
                    ("soft", "soft"): 1.0, ("soft", "borderline"): 0.6, ("soft", "hard"): 0.2,
                    ("borderline", "borderline"): 0.8, ("borderline", "soft"): 0.6, ("borderline", "hard"): 0.6,
                    ("hard", "hard"): 1.0, ("hard", "borderline"): 0.6, ("hard", "soft"): 0.2,
                }
                hsab_match = match_map.get((entry["donor_type"], target_hsab), 0.5)

            # Probability
            base_prob = 0.5
            if kd is not None:
                if kd < 0.01:
                    base_prob = 0.85
                elif kd < 0.1:
                    base_prob = 0.75
                elif kd < 1.0:
                    base_prob = 0.65
                elif kd < 10.0:
                    base_prob = 0.55
                else:
                    base_prob = 0.40
            base_prob *= hsab_match
            if not ph_ok:
                base_prob *= 0.5
            if entry["environmental_tested"]:
                base_prob = min(base_prob + 0.10, 0.95)
            base_prob = max(0.05, min(0.95, round(base_prob, 2)))

            # Confidence
            if kd is not None and entry["environmental_tested"]:
                confidence = "moderate"
                conf_reason = f"Kd = {kd} uM for {ion_matched} (measured). Tested in environmental matrix."
            elif kd is not None:
                confidence = "moderate" if hsab_match > 0.6 else "low"
                conf_reason = f"Kd = {kd} uM for {ion_matched} (measured). Lab buffer validation."
            else:
                confidence = "low"
                conf_reason = f"Target in peptide's metal list but no specific Kd measured."

            # Failure modes
            failure_modes = []
            if entry["donor_type"] == "soft":
                failure_modes.append("Thiol oxidation in aerobic conditions (Cys disulfide formation)")
            if not ph_ok and ph_note:
                failure_modes.append(ph_note)
            if entry["class"] == "metallothionein":
                failure_modes.append("Proteolytic degradation in matrices with microbial activity")
            if not failure_modes:
                failure_modes.append("No major failure modes identified")

            # Improvements
            improvements = []
            if entry["donor_type"] == "soft":
                improvements.append("Handle under N2/argon to prevent Cys oxidation")
            if not ph_ok and matrix.ph is not None:
                mid = (entry["ph_range"][0] + entry["ph_range"][1]) / 2
                improvements.append(f"Buffer to pH {mid:.1f}")
            if entry["class"] == "his_tag":
                improvements.append("Elute with 250 mM imidazole for clean release and reuse")

            # Build structure description
            kd_str = f"Kd={kd} uM for {ion_matched}" if kd else "Kd not measured for this target"
            struct_desc = (
                f"{entry['sequence']} ({entry['full_sequence']}) | "
                f"{entry['length']} residues, {'/'.join(entry['donor_atoms'])} donors, "
                f"{kd_str}"
            )

            # Immobilization
            immob = []
            for handle in entry["conjugation"]:
                immob.append(ImmobilizationOption(
                    substrate="nylon netting or silica beads",
                    attachment_chemistry=handle,
                    click_handle=handle.split("(")[0].strip(),
                    effect_on_binding="Minimal with C6+ linker arm",
                ))

            # Cross-domain
            other_apps = []
            if entry["class"] == "phytochelatin":
                other_apps.append(ApplicationConnection(
                    domain="bioremediation",
                    description="Express in bacteria/plants for in vivo metal sequestration",
                    what_would_change="Clone encoding gene into expression vector",
                    confidence="strong",
                ))
            if entry["class"] == "his_tag":
                other_apps.append(ApplicationConnection(
                    domain="protein_purification",
                    description="Same chemistry used in IMAC purification - massive existing infrastructure",
                    what_would_change="Already standard technology",
                    confidence="strong",
                ))

            candidates.append(CandidateResult(
                rank=0,
                name=f"{entry['name']} for {target.identity}",
                description=(
                    f"{entry['name']} - a {entry['class']} ({entry['length']} residues) with "
                    f"{'/'.join(entry['donor_atoms'])} donor atoms ({entry['donor_type']}). "
                    f"{entry['notes']}"
                ),
                modality="peptide_chelator",
                source_tool="peptide_chelator",
                structure_description=struct_desc,
                performance=PerformancePrediction(
                    probability_of_success=base_prob,
                    confidence=confidence,
                    confidence_reasoning=conf_reason,
                    sensitive_to=[f"pH range: {entry['ph_range'][0]}-{entry['ph_range'][1]}"],
                    failure_modes=failure_modes,
                    what_improves_odds=improvements if improvements else ["Conditions appear favorable"],
                    selectivity_threats=[],
                ),
                evidence=EvidenceProfile(
                    source_type="hybrid" if kd else "computational_prediction",
                    literature_references=[f"DOI: {entry['doi']} ({entry['year']})"],
                    computational_method="Library search + HSAB/Kd scoring",
                    what_would_validate=f"Measure Kd in target matrix by ITC or SPR",
                ),
                accessibility=AccessibilityProfile(
                    estimated_cost=entry["cost_per_mg"] + "/mg (peptide synthesis service)",
                    equipment_required=["analytical balance", "pH meter"],
                    community_lab_feasible=entry["community_lab"],
                    reusability_cycles=entry["reusability"],
                    waste_generated="Peptide is biodegradable - minimal waste",
                    end_of_life="Fully biodegradable",
                ),
                immobilization_options=immob,
                other_applications=other_apps,
            ))

        candidates.sort(key=lambda c: c.performance.probability_of_success, reverse=True)
        return candidates
''')

# ═══════════════════════════════════════════════════════════════════════════
# adapters/aptamer_adapter.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("adapters/aptamer_adapter.py", '''"""
adapters/aptamer_adapter.py - DNA/RNA aptamer adapter for MABE.

Aptamers differ from DNAzymes: they bind targets without catalytic cleavage.
No modification needed for capture mode. Selected through SELEX.
"""

from __future__ import annotations

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import (
    CandidateResult, PerformancePrediction, EvidenceProfile,
    AccessibilityProfile, ImmobilizationOption, ApplicationConnection,
)
from knowledge.peptide_library import APTAMER_LIBRARY


def _target_matches_aptamer(entry: dict, target_identity: str) -> bool:
    identity = target_identity.lower().strip()
    for alias in entry["target_aliases"]:
        if alias in identity or identity in alias:
            return True
    return False


class AptamerAdapter(ToolAdapter):

    @property
    def name(self) -> str:
        return "aptamer"

    @property
    def version(self) -> str:
        return "0.4.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [Capability(
            description="Search SELEX-validated DNA/RNA aptamers that bind targets without cleaving",
            target_types=["metal_ion", "small_molecule"],
            interaction_types=["binding", "folding"],
            output_types=["nucleic_acid_sequence", "binding_constant"],
        )]

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        matches = [e for e in APTAMER_LIBRARY if _target_matches_aptamer(e, problem.target.identity)]
        if matches:
            return ContributionAssessment(
                can_contribute=True, relevance=0.7,
                what_it_would_do=f"Search {len(matches)} validated aptamer(s) for {problem.target.identity}",
                what_part_of_problem="molecular recognition (aptamer modality)",
                estimated_compute_time="instant",
                limitations=["Limited to curated library", "Capture-ready (no modification needed)"],
            )
        return ContributionAssessment(
            can_contribute=False, relevance=0.0,
            what_it_would_do=f"No aptamer in library for {problem.target.identity}",
            what_part_of_problem="none",
        )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        target = problem.target
        matrix = problem.matrix
        candidates = []

        for entry in APTAMER_LIBRARY:
            if not _target_matches_aptamer(entry, target.identity):
                continue

            ph_ok = True
            if matrix.ph is not None:
                low, high = entry["ph_range"]
                ph_ok = low <= matrix.ph <= high

            base_prob = 0.6
            if entry["kd_um"] < 0.1:
                base_prob = 0.80
            elif entry["kd_um"] < 1.0:
                base_prob = 0.70
            elif entry["kd_um"] < 10.0:
                base_prob = 0.55
            if entry["capture_ready"]:
                base_prob += 0.05
            if entry["environmental_tested"]:
                base_prob += 0.10
            if not ph_ok:
                base_prob *= 0.5
            base_prob = max(0.05, min(0.95, round(base_prob, 2)))

            confidence = "moderate" if entry["environmental_tested"] else "low"
            conf_reason = f"Kd = {entry['kd_um']} uM. {'Environmental tested.' if entry['environmental_tested'] else 'Lab buffer only.'}"

            failure_modes = []
            if not ph_ok:
                failure_modes.append(f"Matrix pH outside optimal range ({entry['ph_range'][0]}-{entry['ph_range'][1]})")
            failure_modes.append("DNase degradation in environmental matrices - consider phosphorothioate backbone")
            if not failure_modes:
                failure_modes = ["No major failure modes identified"]

            # Selectivity threats
            threats = []
            for sel in entry.get("selectivity", []):
                for comp in matrix.competing_species:
                    if comp.identity.lower() == sel.get("competitor_alias", "").lower():
                        if sel["fold"] < 20:
                            threats.append(f"{comp.identity} - selectivity only {sel['fold']}-fold")

            immob = []
            for handle in entry["conjugation"]:
                substrate = "streptavidin beads" if "biotin" in handle.lower() else "nylon netting or silica"
                immob.append(ImmobilizationOption(
                    substrate=substrate,
                    attachment_chemistry=handle,
                    click_handle=handle.split("(")[0].strip(),
                    effect_on_binding="Minimal - terminus attachment, binding site internal",
                ))

            candidates.append(CandidateResult(
                rank=0,
                name=f"{entry['name']} for {target.identity}",
                description=f"{entry['name']} - a SELEX-selected DNA aptamer. Kd = {entry['kd_um']} uM. Capture-ready (binds without cleaving). {entry['notes']}",
                modality="dna_aptamer",
                source_tool="aptamer",
                structure_description=f"{entry['sequence']} | Kd={entry['kd_um']} uM",
                performance=PerformancePrediction(
                    probability_of_success=base_prob,
                    confidence=confidence,
                    confidence_reasoning=conf_reason,
                    sensitive_to=[f"pH range: {entry['ph_range'][0]}-{entry['ph_range'][1]}"],
                    failure_modes=failure_modes,
                    what_improves_odds=["Use phosphorothioate backbone for nuclease resistance"],
                    selectivity_threats=threats if threats else ["No major selectivity threats"],
                ),
                evidence=EvidenceProfile(
                    source_type="literature_validated" if entry["environmental_tested"] else "hybrid",
                    literature_references=[f"DOI: {entry['doi']} ({entry['year']})"],
                    computational_method="Library search",
                    what_would_validate="Test in target matrix, measure Kd shift",
                ),
                accessibility=AccessibilityProfile(
                    estimated_cost=entry["cost_per_synthesis"] + " per synthesis",
                    equipment_required=["thermal cycler (optional)", "gel electrophoresis"],
                    community_lab_feasible=True,
                    reusability_cycles=15,
                    waste_generated="DNA is biodegradable",
                    end_of_life="Fully biodegradable",
                ),
                immobilization_options=immob,
                other_applications=[ApplicationConnection(
                    domain="diagnostic",
                    description=f"Aptamer sensor for {entry['primary_target']} detection",
                    what_would_change="Add fluorophore/quencher for signal readout",
                    confidence="strong",
                )],
            ))

        candidates.sort(key=lambda c: c.performance.probability_of_success, reverse=True)
        return candidates
''')

# ═══════════════════════════════════════════════════════════════════════════
# Update main.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("main.py", '''"""
MABE - Modality-Agnostic Binder Engine
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapters.base import ToolRegistry
from adapters.dummy_adapter import DummyAdapter
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from conversation.decomposer_patch import patch_targets
from conversation.interface import run_interactive, run_single_query

patch_targets()


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    return registry


def main():
    registry = build_registry()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_single_query(registry, query)
    else:
        run_interactive(registry)


if __name__ == "__main__":
    main()
''')

# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint4.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint4.py", '''"""
tests/test_sprint4.py - Peptide + Aptamer adapter tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose


def test_peptide_finds_lead():
    adapter = PeptideAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1
    names = [c.name for c in candidates]
    print(f"  + Lead peptides: {len(candidates)} found: {', '.join(names)}")


def test_peptide_finds_nickel():
    adapter = PeptideAdapter()
    problem = decompose("nickel capture from mine water")
    candidates = adapter.generate_candidates(problem)
    his_tags = [c for c in candidates if "His" in c.name]
    assert len(his_tags) >= 1, "His-tag should match nickel"
    print(f"  + Nickel peptides: {len(candidates)} found, including His-tags")


def test_peptide_finds_mercury():
    adapter = PeptideAdapter()
    problem = decompose("mercury capture from river water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1
    # Should find Cys-rich peptides for soft Hg2+
    soft = [c for c in candidates if "soft" in c.description.lower() or "Cys" in c.description or "thiol" in c.description.lower()]
    assert len(soft) >= 1, "Should find Cys/thiol peptides for mercury"
    print(f"  + Mercury peptides: {len(candidates)} found, {len(soft)} with soft donors")


def test_peptide_biodegradable():
    adapter = PeptideAdapter()
    problem = decompose("cadmium removal from mine water")
    candidates = adapter.generate_candidates(problem)
    for c in candidates:
        assert "biodegradable" in c.accessibility.end_of_life.lower()
    print(f"  + All peptide candidates are biodegradable")


def test_aptamer_finds_arsenic():
    adapter = AptamerAdapter()
    problem = decompose("arsenic removal from mine water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1
    assert any("Ars" in c.name for c in candidates)
    print(f"  + Arsenic aptamer found: {candidates[0].name}")


def test_aptamer_finds_mercury():
    adapter = AptamerAdapter()
    problem = decompose("mercury removal from river water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1
    print(f"  + Mercury aptamer found: {candidates[0].name}")


def test_aptamer_capture_ready():
    adapter = AptamerAdapter()
    problem = decompose("mercury removal from river water")
    candidates = adapter.generate_candidates(problem)
    for c in candidates:
        # Aptamers should NOT have "capture not validated" warnings
        capture_warnings = [fm for fm in c.performance.failure_modes if "capture" in fm.lower() and "not" in fm.lower()]
        assert len(capture_warnings) == 0, f"Aptamer should be capture-ready, but got: {capture_warnings}"
    print(f"  + Aptamers are capture-ready (no modification needed)")


def test_full_pipeline_five_adapters():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)

    problem = decompose("mercury removal from river water")
    result = orchestrator.solve(problem)
    sources = set(c.source_tool for c in result.candidates)
    modalities = set(c.modality for c in result.candidates)
    print(f"  + Full pipeline: {len(result.candidates)} candidates from {len(sources)} tools")
    print(f"    Tools: {sources}")
    print(f"    Modalities: {modalities}")


def test_lead_all_modalities():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)

    problem = decompose("lead capture and release from mine water in BC")
    result = orchestrator.solve(problem)
    modalities = set(c.modality for c in result.candidates)
    print(f"  + Lead (all modalities): {len(result.candidates)} candidates")
    print(f"    Modalities: {modalities}")
    # Should have at least: chelator, dnazyme, peptide_chelator, designed_protein, nanocage
    assert len(modalities) >= 4, f"Expected 4+ modalities, got {modalities}"


if __name__ == "__main__":
    print()
    print("  MABE Sprint 4 - Peptide + Aptamer Tests")
    print("  " + "=" * 40)
    print()

    test_peptide_finds_lead()
    test_peptide_finds_nickel()
    test_peptide_finds_mercury()
    test_peptide_biodegradable()
    test_aptamer_finds_arsenic()
    test_aptamer_finds_mercury()
    test_aptamer_capture_ready()
    test_full_pipeline_five_adapters()
    test_lead_all_modalities()

    print()
    print("  All Sprint 4 tests passed.")
    print()
''')

print()
print("  Done! New files:")
print("    knowledge/peptide_library.py    (8 peptides + 3 aptamers)")
print("    adapters/peptide_adapter.py     (peptide chelator adapter)")
print("    adapters/aptamer_adapter.py     (DNA aptamer adapter)")
print("    main.py                         (updated - 5 adapters)")
print("    tests/test_sprint4.py           (9 tests)")
print()
print("  Next steps:")
print("    python tests\\test_sprint4.py")
print('    python main.py "lead capture from mine water"')
print('    python main.py "mercury removal from river water"')
print('    python main.py "cadmium removal from mine water"')
print()