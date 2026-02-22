"""
MABE Sprint 5b Bootstrap - Universal Lab Tool Connections
==========================================================
The big picture: selective binder + scaffold + controlled release
is the bottleneck for most lab procedures. MABE designs all three.

    cd Documents\\mabe
    python bootstrap_sprint5b.py
    python tests\\test_sprint5b.py
    python main.py "lead capture from mine water"
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
print("  MABE Sprint 5b - Universal Lab Tool Connections")
print("  " + "=" * 40)
print()

# ═══════════════════════════════════════════════════════════════════════════
# core/connections.py — EXPANDED: sees the full picture
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/connections.py", '''"""
core/connections.py - Cross-domain connection engine.

THE BIG PICTURE:

    Selective binder + scaffold + controlled release
    = the universal bottleneck in molecular science.

MABE designs selective binders. That same binder is the recognition
element for EVERY procedure that requires selective capture:

    DIAGNOSTICS: replace mass spec with $5 test
    PULL-DOWNS: immunoprecipitation without antibodies
    COLUMN CAPTURE: affinity chromatography with designed selectivity
    CELL SORTING: target cell capture without Fc activation
    RESEARCH TOOLS: replace $500/sample ICP-MS with $5 assay
    MONITORING: community-operated sensor networks
    THERAPEUTICS: chelation therapy with designed molecules
    MANUFACTURING: selective extraction for industrial processes

The key insight about antibody replacement:
- Traditional IP/pull-down uses antibodies (expensive, Fc activation, batch variation)
- DNA/peptide binders have NO Fc region — no immune activation
- Can be immobilized on beads, columns, plates, netting — same infrastructure
- Elution by pH, imidazole, EDTA, heat — controlled release
- Synthetic = no batch variation, unlimited supply, defined chemistry

Connection rules are grounded in the physics of what the binder CAN DO,
not speculation about what it MIGHT do.
"""

from __future__ import annotations
from core.candidate import CandidateResult, ApplicationConnection


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC connections
# ═══════════════════════════════════════════════════════════════════════════

def _diagnostic_from_capture(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    connections = []
    modality = candidate.modality.lower()
    prob = candidate.performance.probability_of_success
    if prob < 0.2:
        return connections

    if modality in ("dnazyme", "dna_aptamer", "dna_motif"):
        connections.append(ApplicationConnection(
            domain="field_diagnostic",
            description=(
                f"Lateral flow strip for {target_name} detection. "
                f"Gold nanoparticle reporter + capture line. "
                f"Visual yes/no in 15 min, no equipment. "
                f"$2-5/test vs $50-500 for ICP-MS."
            ),
            what_would_change="Biotin one end, AuNP-complementary strand other end. Standard lateral flow manufacturing.",
            confidence="strong" if prob > 0.4 else "plausible",
        ))
        connections.append(ApplicationConnection(
            domain="electrochemical_sensor",
            description=(
                f"Screen-printed electrode sensor for {target_name}. "
                f"Quantitative, reusable, field-deployable with $200 handheld potentiostat. "
                f"10-100x more sensitive than lateral flow."
            ),
            what_would_change="Thiolate DNA on gold electrode. Redox reporter (methylene blue). Square wave voltammetry readout.",
            confidence="strong" if prob > 0.4 else "plausible",
        ))
        if modality == "dnazyme":
            connections.append(ApplicationConnection(
                domain="fluorescent_sensor",
                description=(
                    f"Fluorescent turn-on sensor. Native DNAzyme function with FAM/dabcyl pair. "
                    f"Cleavage on metal binding generates fluorescence. Sub-nM detection. "
                    f"Microplate reader or field fluorimeter."
                ),
                what_would_change="Use unmodified sequence. Add FAM 5\\', dabcyl 3\\'. This IS the native function.",
                confidence="strong",
            ))

    if modality == "peptide_chelator":
        connections.append(ApplicationConnection(
            domain="field_diagnostic",
            description=(
                f"Colorimetric test strip for {target_name}. "
                f"Peptide + indicator dye displacement. "
                f"Color change visible by eye. $1-3/test."
            ),
            what_would_change="Conjugate peptide to cellulose. Pre-load weak indicator. Phone camera for quantification.",
            confidence="plausible",
        ))

    if modality == "chelator":
        connections.append(ApplicationConnection(
            domain="field_diagnostic",
            description=(
                f"Indicator displacement assay for {target_name}. "
                f"Target displaces colored indicator from chelator. "
                f"Test tube chemistry. <$1/test."
            ),
            what_would_change="Select indicator with lower Kf than chelator. Calibrate. Phone camera readout.",
            confidence="strong",
        ))

    return connections


# ═══════════════════════════════════════════════════════════════════════════
# PULL-DOWN / IMMUNOPRECIPITATION replacement
# ═══════════════════════════════════════════════════════════════════════════

def _pulldown_replacement(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    THE BIG ONE: Any selective binder on beads = pull-down reagent.
    Replaces antibody-based IP for metals, and the SAME LOGIC applies to
    protein/cell targets when MABE expands beyond metals.

    Key advantages over antibody IP:
    - No Fc region = no Fc receptor activation = no immune artifacts
    - Synthetic = no batch variation
    - Defined elution conditions = cleaner release
    - Cheap = democratizes access
    """
    connections = []
    prob = candidate.performance.probability_of_success
    modality = candidate.modality.lower()

    if prob < 0.25:
        return connections

    # Bead-based pull-down
    has_conjugation = len(candidate.immobilization_options) > 0
    if has_conjugation:
        connections.append(ApplicationConnection(
            domain="affinity_pulldown",
            description=(
                f"Bead-based affinity pull-down for {target_name}. "
                f"Immobilize binder on magnetic or agarose beads. "
                f"Add sample, wash, elute. Same workflow as immunoprecipitation "
                f"but with synthetic binder: no batch variation, no Fc artifacts, "
                f"defined elution chemistry, unlimited supply. "
                f"Works with existing magnetic racks and spin columns."
            ),
            what_would_change=(
                "Conjugate to NHS-activated magnetic beads (Dynabeads, $50/mg) or "
                "NHS-agarose ($20/mL resin). Standard affinity purification protocol. "
                "Elute by pH shift, imidazole, EDTA, or competitor displacement."
            ),
            confidence="strong" if prob > 0.5 else "plausible",
        ))

    # Column chromatography
    if has_conjugation and prob > 0.3:
        connections.append(ApplicationConnection(
            domain="affinity_column",
            description=(
                f"Affinity chromatography column for {target_name}. "
                f"Pack binder-conjugated resin into column for continuous-flow capture. "
                f"Scalable from analytical (uL) to preparative (L). "
                f"Replaces ion exchange with designed selectivity. "
                f"Same FPLC/HPLC infrastructure, better specificity."
            ),
            what_would_change=(
                "Conjugate to activated sepharose or silica resin. "
                "Pack column, characterize binding/elution profile. "
                "Optimize flow rate and elution conditions for target matrix."
            ),
            confidence="plausible",
        ))

    return connections


# ═══════════════════════════════════════════════════════════════════════════
# CELL SORTING without Fc activation
# ═══════════════════════════════════════════════════════════════════════════

def _cell_sorting_connection(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    When MABE expands to protein targets, this becomes critical.
    For now, plant on the concept: binders without Fc = cell interaction without activation.
    Even for metal targets, metal-loaded cells can be targeted.
    """
    connections = []
    modality = candidate.modality.lower()
    prob = candidate.performance.probability_of_success

    if prob < 0.4:
        return connections

    if modality in ("dna_aptamer", "peptide_chelator", "dnazyme"):
        connections.append(ApplicationConnection(
            domain="cell_capture",
            description=(
                f"Cell capture/sorting without immune activation. "
                f"Binder on surface captures cells that have accumulated {target_name}. "
                f"DNA/peptide binders have NO Fc region - no FcR cross-linking, "
                f"no complement activation, no ADCC. Cells remain viable and unactivated. "
                f"Critical for: B cell isolation, stem cell purification, "
                f"circulating tumor cell capture, metal-exposed cell enrichment."
            ),
            what_would_change=(
                "Immobilize binder on plate/column/magnetic beads. "
                "Flow cells over surface. Wash. Release by gentle elution. "
                "Validate cell viability and activation markers post-capture."
            ),
            confidence="speculative",
        ))

    return connections


# ═══════════════════════════════════════════════════════════════════════════
# RESEARCH TOOLS - mass spec replacement
# ═══════════════════════════════════════════════════════════════════════════

def _research_tool_from_capture(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    connections = []
    prob = candidate.performance.probability_of_success
    if prob < 0.3:
        return connections

    connections.append(ApplicationConnection(
        domain="research_tool",
        description=(
            f"Low-cost {target_name} quantification replacing ICP-MS. "
            f"$1-10/sample vs $50-500/sample. Enables: high-throughput screening, "
            f"field studies, resource-limited settings, continuous monitoring, "
            f"dose-response studies, process optimization. "
            f"Removes the analytical bottleneck from experimental workflows."
        ),
        what_would_change=(
            "Validate response curve against ICP-MS standards. "
            "Package as kit: binder + readout + calibration standards. "
            "Cross-validate with 20+ real samples."
        ),
        confidence="plausible" if prob > 0.5 else "speculative",
    ))

    return connections


# ═══════════════════════════════════════════════════════════════════════════
# DNA TAG ENCODING - multiplexed panels
# ═══════════════════════════════════════════════════════════════════════════

def _dna_tag_encoding(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    connections = []
    modality = candidate.modality.lower()
    prob = candidate.performance.probability_of_success

    if prob < 0.3:
        return connections

    if modality in ("dnazyme", "dna_aptamer", "dna_motif"):
        connections.append(ApplicationConnection(
            domain="multiplexed_panel",
            description=(
                f"DNA-barcoded multiplexed detection panel. "
                f"Each binder carries unique DNA barcode. Pool binders for multiple targets. "
                f"One sample, many analytes. Read by Nanopore ($2/sample) or hybridization array. "
                f"Replaces multi-element ICP-MS panel. "
                f"Same principle as CITE-seq but for environmental/industrial analytes."
            ),
            what_would_change="Extend with 20-nt barcode + universal primer site. PCR amplify after capture. Sequence.",
            confidence="plausible",
        ))
    elif modality == "peptide_chelator":
        connections.append(ApplicationConnection(
            domain="multiplexed_panel",
            description=(
                f"Click-chemistry DNA tag for multiplexed {target_name} panel. "
                f"Attach unique DNA barcode to peptide via azide-DBCO click. "
                f"After capture, read barcode. Multi-analyte from single sample."
            ),
            what_would_change="Azide-modified Lys on peptide. Click to DBCO-DNA barcode. Standard bioconjugation.",
            confidence="plausible",
        ))

    return connections


# ═══════════════════════════════════════════════════════════════════════════
# MONITORING NETWORKS
# ═══════════════════════════════════════════════════════════════════════════

def _monitoring_network(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    connections = []
    prob = candidate.performance.probability_of_success
    modality = candidate.modality.lower()
    if prob < 0.4 or modality not in ("dnazyme", "dna_aptamer", "dna_motif", "chelator"):
        return connections

    connections.append(ApplicationConnection(
        domain="monitoring_network",
        description=(
            f"Distributed {target_name} monitoring network. "
            f"Binder-based sensors at multiple waterway points. "
            f"Phone-camera readout, community-operated. "
            f"$5-20/test point vs $200-1000/lab visit."
        ),
        what_would_change="Waterproof housing + sample inlet. Phone app for color reading + GPS. 15 min protocol.",
        confidence="plausible",
    ))
    return connections


# ═══════════════════════════════════════════════════════════════════════════
# THERAPEUTIC
# ═══════════════════════════════════════════════════════════════════════════

def _therapeutic_from_capture(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    connections = []
    prob = candidate.performance.probability_of_success
    modality = candidate.modality.lower()
    if prob < 0.4 or modality not in ("chelator", "peptide_chelator"):
        return connections

    if "fda" in candidate.description.lower():
        connections.append(ApplicationConnection(
            domain="therapeutic",
            description=f"FDA-approved. Direct path to therapeutic {target_name} chelation.",
            what_would_change="Existing clinical protocols. Dosing optimization.",
            confidence="strong",
        ))
    elif modality == "peptide_chelator":
        connections.append(ApplicationConnection(
            domain="therapeutic",
            description=(
                f"Peptide chelator for therapeutic {target_name} removal. "
                f"Biodegradable, predictable PK. No Fc-mediated immune effects."
            ),
            what_would_change="PEG for renal clearance. Toxicity studies. Long regulatory path.",
            confidence="speculative",
        ))
    return connections


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE PREP - replacing expensive lab bottlenecks
# ═══════════════════════════════════════════════════════════════════════════

def _sample_prep_connection(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    Sample prep is the hidden bottleneck. Every analytical workflow starts with
    selective extraction/concentration. A good binder on beads IS sample prep.
    """
    connections = []
    prob = candidate.performance.probability_of_success
    if prob < 0.3:
        return connections

    has_immob = len(candidate.immobilization_options) > 0
    if not has_immob:
        return connections

    connections.append(ApplicationConnection(
        domain="sample_preparation",
        description=(
            f"Selective pre-concentration of {target_name} from complex matrices. "
            f"Binder on magnetic beads: add to sample, capture, magnet pull-down, "
            f"wash away matrix, elute pure {target_name}. "
            f"Replaces: solid-phase extraction cartridges, liquid-liquid extraction, "
            f"precipitation steps. Works directly in environmental water, blood, "
            f"culture media, industrial process streams. "
            f"Makes downstream analysis (any method) faster, cheaper, more sensitive."
        ),
        what_would_change=(
            "Conjugate to magnetic beads. Optimize loading capacity and wash steps. "
            "Validate recovery rate and enrichment factor in target matrix."
        ),
        confidence="plausible" if prob > 0.5 else "speculative",
    ))

    return connections


# ═══════════════════════════════════════════════════════════════════════════
# PROCESS ANALYTICAL - inline QC
# ═══════════════════════════════════════════════════════════════════════════

def _process_analytical(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """Industrial process monitoring and quality control."""
    connections = []
    prob = candidate.performance.probability_of_success
    modality = candidate.modality.lower()
    if prob < 0.4:
        return connections

    if modality in ("dnazyme", "dna_aptamer", "chelator"):
        connections.append(ApplicationConnection(
            domain="process_analytical",
            description=(
                f"Inline process analytical for {target_name} in manufacturing. "
                f"Continuous monitoring of {target_name} levels in process streams. "
                f"Real-time QC without pulling samples for lab analysis. "
                f"Reduces batch failures, speeds release testing."
            ),
            what_would_change=(
                "Integrate sensor into flow cell. Calibrate for process matrix. "
                "Connect to SCADA/process control for automated response."
            ),
            confidence="plausible",
        ))
    return connections


# ═══════════════════════════════════════════════════════════════════════════
# Main engine
# ═══════════════════════════════════════════════════════════════════════════

ALL_CONNECTION_RULES = [
    _diagnostic_from_capture,
    _pulldown_replacement,
    _cell_sorting_connection,
    _research_tool_from_capture,
    _dna_tag_encoding,
    _monitoring_network,
    _therapeutic_from_capture,
    _sample_prep_connection,
    _process_analytical,
]


def discover_connections(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """Run all connection rules. Returns new connections not already present."""
    existing = {(a.domain, a.description[:40]) for a in candidate.other_applications}
    new_connections = []

    for rule in ALL_CONNECTION_RULES:
        for conn in rule(candidate, target_name):
            key = (conn.domain, conn.description[:40])
            if key not in existing:
                new_connections.append(conn)
                existing.add(key)

    return new_connections
''')

# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint5b.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint5b.py", '''"""
tests/test_sprint5b.py - Universal lab tool connection tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose


def _build():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available(): registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    return registry


def test_pulldown_connection():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_pulldown = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "affinity_pulldown":
                has_pulldown = True
                assert "Fc" in a.description or "batch variation" in a.description
                break
    assert has_pulldown, "Should discover affinity pulldown application"
    print("  + Affinity pull-down connection found (replaces antibody IP)")


def test_column_capture_connection():
    o = Orchestrator(_build())
    r = o.solve(decompose("nickel capture from mine water"))
    has_column = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "affinity_column":
                has_column = True
                break
    assert has_column, "Should discover affinity column application"
    print("  + Affinity chromatography column connection found")


def test_cell_sorting_no_fc():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_cell = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "cell_capture":
                has_cell = True
                assert "Fc" in a.description
                assert "activation" in a.description.lower()
                break
    assert has_cell, "Should discover cell capture without Fc activation"
    print("  + Cell capture connection found (no Fc activation)")


def test_sample_prep_connection():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_prep = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "sample_preparation":
                has_prep = True
                assert "pre-concentration" in a.description.lower() or "magnetic" in a.description.lower()
                break
    assert has_prep, "Should discover sample prep application"
    print("  + Sample preparation connection found (replaces SPE)")


def test_process_analytical():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_pac = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "process_analytical":
                has_pac = True
                break
    assert has_pac
    print("  + Process analytical (inline QC) connection found")


def test_no_fc_in_dna_peptide():
    """DNA and peptide binders should explicitly note no Fc artifacts."""
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    fc_mentions = 0
    for c in r.candidates:
        if c.modality in ("dnazyme", "dna_aptamer", "peptide_chelator", "dna_motif"):
            for a in c.other_applications:
                if "Fc" in a.description or "no Fc" in a.description:
                    fc_mentions += 1
    assert fc_mentions >= 1, "DNA/peptide candidates should mention Fc-free advantage"
    print(f"  + {fc_mentions} connections explicitly mention Fc-free advantage")


def test_total_connection_domains():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    all_domains = set()
    for c in r.candidates:
        for a in c.other_applications:
            all_domains.add(a.domain)
    print(f"  + All connection domains discovered: {sorted(all_domains)}")
    expected = {"field_diagnostic", "affinity_pulldown", "research_tool", "sample_preparation"}
    missing = expected - all_domains
    assert not missing, f"Missing expected domains: {missing}"


def test_mass_spec_replacement_mentioned():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_mass_spec = False
    for c in r.candidates:
        for a in c.other_applications:
            if "ICP-MS" in a.description or "mass spec" in a.description.lower():
                has_mass_spec = True
                break
    assert has_mass_spec
    print("  + ICP-MS/mass spec replacement explicitly mentioned")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 5b - Universal Lab Tool Tests")
    print("  " + "=" * 40)
    print()

    test_pulldown_connection()
    test_column_capture_connection()
    test_cell_sorting_no_fc()
    test_sample_prep_connection()
    test_process_analytical()
    test_no_fc_in_dna_peptide()
    test_total_connection_domains()
    test_mass_spec_replacement_mentioned()

    print()
    print("  All Sprint 5b tests passed.")
    print()
''')

print()
print("  Done! Updated files:")
print("    core/connections.py        (expanded: pulldown, column, cell sort, sample prep, PAT)")
print("    tests/test_sprint5b.py     (8 tests)")
print()
print("  Next steps:")
print("    python tests\\test_sprint5b.py")
print('    python main.py "lead capture from mine water"')
print()