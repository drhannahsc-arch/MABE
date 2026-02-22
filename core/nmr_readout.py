"""
core/nmr_readout.py — Sprint 29d: NMR Relaxation + Tag-Based Readout

NMR relaxation enhancement from paramagnetic metals (PRE) — predicts
T1/T2 effects that can replace mass spec detection. Also models
DNA/molecular tag readout strategies: displacement assays, FRET,
strand displacement, and barcode multiplexing.

Physics:
  Solomon-Bloembergen: 1/T1_para = C × μ_eff² × τ_c / r⁶
  Inner-sphere PRE: depends on number of coordinated waters, exchange rate
  Outer-sphere PRE: diffusion-mediated
  Tag displacement: target binding releases detectable DNA/fluorescent tag
"""
from dataclasses import dataclass
import math


@dataclass
class NMRRelaxationProfile:
    """NMR relaxation enhancement from paramagnetic metal binding."""
    metal_formula: str
    unpaired_electrons: int
    magnetic_moment_bm: float
    inner_sphere_r1_mM_s: float     # r1 relaxivity (mM⁻¹s⁻¹) inner sphere
    outer_sphere_r1_mM_s: float     # r1 relaxivity outer sphere
    total_r1_mM_s: float            # Total longitudinal relaxivity
    total_r2_mM_s: float            # Total transverse relaxivity
    t1_at_1uM_ms: float             # T1 shortening at 1 µM metal
    t2_at_1uM_ms: float             # T2 shortening at 1 µM metal
    mri_contrast_agent: bool         # Suitable as MRI contrast agent
    nmr_detection_limit_uM: float   # Minimum detectable concentration
    notes: str = ""

@dataclass
class TagReadoutStrategy:
    """Tag-based readout to replace mass spectrometry."""
    strategy_name: str
    readout_type: str               # "fluorescence", "colorimetric", "electrochemical",
                                    # "lateral_flow", "qPCR", "sequencing", "NMR"
    sensitivity: str                # "ppt", "ppb", "nM", "µM"
    multiplexing_capacity: int      # How many targets simultaneously
    time_to_result_min: float
    equipment_required: str
    field_deployable: bool
    description: str
    advantages: str
    limitations: str


# ═══════════════════════════════════════════════════════════════════════════
# NMR RELAXATION — SOLOMON-BLOEMBERGEN
# ═══════════════════════════════════════════════════════════════════════════

# Relaxivity database: (inner_r1, outer_r1, r2/r1_ratio, notes)
# r1 in mM⁻¹s⁻¹ at 20 MHz (0.47 T), 25°C
_RELAXIVITY = {
    # High-spin d5 — best relaxation agents
    "Mn2+": (7.0, 1.5, 1.2, "Optimal τ_c; 1 coordinated water; fast exchange"),
    "Fe3+": (6.0, 1.2, 1.5, "6 waters in shell; moderate exchange rate"),
    "Gd3+": (10.5, 3.0, 1.1, "9 waters; very fast exchange; clinical MRI agent"),
    # Other paramagnetic
    "Fe2+": (1.5, 0.8, 2.0, "4 unpaired; less effective than Fe3+"),
    "Co2+": (3.5, 0.5, 3.0, "3 unpaired; significant contact shift"),
    "Ni2+": (2.0, 0.4, 4.0, "2 unpaired; slow water exchange limits r1"),
    "Cu2+": (0.8, 0.3, 5.0, "1 unpaired; Jahn-Teller lability helps exchange"),
    "Cr3+": (1.5, 0.3, 2.0, "3 unpaired; INERT water exchange kills inner-sphere"),
    # Diamagnetic — no PRE
    "Zn2+": (0.0, 0.0, 0.0, "Diamagnetic: no paramagnetic relaxation enhancement"),
    "Cd2+": (0.0, 0.0, 0.0, "Diamagnetic"),
    "Pb2+": (0.0, 0.0, 0.0, "Diamagnetic"),
    "Ag+":  (0.0, 0.0, 0.0, "Diamagnetic"),
    "Au3+": (0.0, 0.0, 0.0, "Diamagnetic (low-spin d8)"),
    "Au+":  (0.0, 0.0, 0.0, "Diamagnetic (d10)"),
    "Hg2+": (0.0, 0.0, 0.0, "Diamagnetic"),
}


def predict_nmr_relaxation(metal_formula, unpaired_electrons=None,
                             field_mhz=20.0):
    """Predict NMR relaxation enhancement for paramagnetic metal detection.

    Key insight: paramagnetic metals shorten T1 and T2 of nearby water
    protons. This is detectable by NMR relaxometry, even in crude samples,
    without separation or mass spec.
    """
    data = _RELAXIVITY.get(metal_formula)

    if unpaired_electrons is not None:
        n = unpaired_electrons
    elif data:
        # Infer from relaxivity
        n = 0 if data[0] == 0 else 3  # Rough
    else:
        n = 0

    mu_bm = math.sqrt(n * (n + 2)) if n > 0 else 0.0

    if data:
        r1_inner, r1_outer, r2_r1_ratio, notes = data
    else:
        # Estimate from unpaired electrons
        r1_inner = n * 1.5 if n > 0 else 0.0
        r1_outer = n * 0.3 if n > 0 else 0.0
        r2_r1_ratio = 1.5
        notes = "Estimated from unpaired electron count"

    # Field dependence: r1 decreases at high field, r2 stays or increases
    field_factor = 1.0
    if field_mhz > 100:
        field_factor = 20.0 / field_mhz  # Approximate NMRD scaling

    r1_total = (r1_inner + r1_outer) * field_factor
    r2_total = r1_total * r2_r1_ratio

    # Detection: T1/T2 at 1 µM concentration
    # 1/T1_obs = 1/T1_water + r1 × [M] (mM)
    # At 1 µM = 0.001 mM: Δ(1/T1) = r1 × 0.001
    t1_water = 2500  # ms for pure water at 0.47T
    if r1_total > 0:
        delta_r1 = r1_total * 0.001  # s⁻¹ at 1 µM
        t1_at_1uM = 1000 / (1/t1_water*1000 + delta_r1) if delta_r1 > 0 else t1_water
        # Detection limit: where Δ(1/T1) > 5% of 1/T1_water
        det_limit = 0.05 / (t1_water/1000 * r1_total) * 1000  # µM
    else:
        t1_at_1uM = t1_water
        det_limit = 1e6  # Undetectable by NMR

    t2_water = 2500  # ms
    if r2_total > 0:
        delta_r2 = r2_total * 0.001
        t2_at_1uM = 1000 / (1/t2_water*1000 + delta_r2) if delta_r2 > 0 else t2_water
    else:
        t2_at_1uM = t2_water

    mri_suitable = r1_total > 3.0 and n >= 3

    return NMRRelaxationProfile(
        metal_formula=metal_formula,
        unpaired_electrons=n if data else (unpaired_electrons or 0),
        magnetic_moment_bm=round(mu_bm, 2),
        inner_sphere_r1_mM_s=round(r1_inner * field_factor, 2),
        outer_sphere_r1_mM_s=round(r1_outer * field_factor, 2),
        total_r1_mM_s=round(r1_total, 2),
        total_r2_mM_s=round(r2_total, 2),
        t1_at_1uM_ms=round(t1_at_1uM, 1),
        t2_at_1uM_ms=round(t2_at_1uM, 1),
        mri_contrast_agent=mri_suitable,
        nmr_detection_limit_uM=round(det_limit, 2),
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAG-BASED READOUT STRATEGIES
# Replacing mass spec with molecular recognition + signal amplification
# ═══════════════════════════════════════════════════════════════════════════

_TAG_STRATEGIES = {
    "DNA_strand_displacement": TagReadoutStrategy(
        strategy_name="DNA Strand Displacement Assay",
        readout_type="fluorescence",
        sensitivity="nM",
        multiplexing_capacity=100,
        time_to_result_min=30,
        equipment_required="Fluorescence plate reader or smartphone camera with filter",
        field_deployable=True,
        description="Target binding to aptamer/DNAzyme releases a quenched DNA strand. "
                    "Released strand activates fluorescent reporter via toehold exchange.",
        advantages="Isothermal, no enzymes needed, room temperature, multiplexable via "
                   "orthogonal toehold sequences, works in crude matrices",
        limitations="Requires aptamer/DNAzyme for each target; µL sample volumes",
    ),
    "DNAzyme_cleavage": TagReadoutStrategy(
        strategy_name="DNAzyme Catalytic Cleavage",
        readout_type="fluorescence",
        sensitivity="nM",
        multiplexing_capacity=20,
        time_to_result_min=15,
        equipment_required="UV lamp or plate reader",
        field_deployable=True,
        description="Metal-specific DNAzyme cleaves fluorogenic substrate upon target binding. "
                    "Signal is catalytically amplified: one metal ion → many cleavage events.",
        advantages="Catalytic amplification (10-100× signal), metal-specific DNAzymes known "
                   "for Pb2+, UO2²⁺, Zn2+, Cu2+, Hg2+, Ag+, and others",
        limitations="DNAzyme discovery required for new targets; RNA substrate cost",
    ),
    "lateral_flow_DNA": TagReadoutStrategy(
        strategy_name="Lateral Flow with DNA Tags",
        readout_type="colorimetric",
        sensitivity="ppb",
        multiplexing_capacity=5,
        time_to_result_min=10,
        equipment_required="None (visual) or smartphone camera for quantitative",
        field_deployable=True,
        description="Au nanoparticle-DNA conjugates aggregate or de-aggregate upon target binding. "
                    "Color change visible to naked eye (red→blue for aggregation).",
        advantages="No equipment, field-deployable, low cost, rapid, "
                   "familiar format (pregnancy test style)",
        limitations="Semi-quantitative; limited multiplexing; matrix effects",
    ),
    "qPCR_barcode": TagReadoutStrategy(
        strategy_name="qPCR Barcode Quantification",
        readout_type="qPCR",
        sensitivity="ppt",
        multiplexing_capacity=1000,
        time_to_result_min=90,
        equipment_required="qPCR thermocycler",
        field_deployable=False,
        description="Each binder carries a unique DNA barcode. Target binding releases barcode. "
                    "Released barcodes quantified by qPCR. Exponential amplification → "
                    "single-molecule sensitivity.",
        advantages="Extreme sensitivity (single molecule), massive multiplexing via "
                   "unique barcodes, quantitative, works with any binder type",
        limitations="Requires qPCR instrument; 90 min turnaround; lab setting",
    ),
    "sequencing_barcode": TagReadoutStrategy(
        strategy_name="Next-Gen Sequencing Barcode",
        readout_type="sequencing",
        sensitivity="ppt",
        multiplexing_capacity=100000,
        time_to_result_min=480,
        equipment_required="NGS sequencer (MinION for field, Illumina for lab)",
        field_deployable=True,  # MinION is portable
        description="Massively parallel barcode sequencing. Each binder has unique DNA barcode. "
                    "After capture, all barcodes sequenced simultaneously. "
                    "Count = concentration. THIS is the mass-spec replacement.",
        advantages="Unlimited multiplexing, absolute quantification from read counts, "
                   "MinION enables field deployment, digital readout (counting molecules)",
        limitations="8+ hour turnaround for Illumina; MinION lower accuracy; "
                   "bioinformatics pipeline needed",
    ),
    "FRET_proximity": TagReadoutStrategy(
        strategy_name="FRET Proximity Sensing",
        readout_type="fluorescence",
        sensitivity="nM",
        multiplexing_capacity=10,
        time_to_result_min=5,
        equipment_required="Fluorescence reader with dual-channel",
        field_deployable=True,
        description="Donor and acceptor fluorophores on binder arms. Target binding brings "
                    "arms together → FRET signal. Ratiometric (self-calibrating).",
        advantages="Ratiometric (immune to photobleaching/dilution), real-time, "
                   "reversible for continuous monitoring",
        limitations="Requires dual-labeled binder; spectral overlap constraints",
    ),
    "electrochemical_tag": TagReadoutStrategy(
        strategy_name="Electrochemical Aptamer Sensor",
        readout_type="electrochemical",
        sensitivity="nM",
        multiplexing_capacity=16,
        time_to_result_min=2,
        equipment_required="Potentiostat or handheld electrochemical reader",
        field_deployable=True,
        description="Methylene blue or ferrocene tagged aptamer on gold electrode. "
                    "Target binding changes electron transfer distance → current change. "
                    "Reagentless, reusable, continuous.",
        advantages="Reagentless, reusable (>100 cycles), real-time, continuous monitoring, "
                   "works in whole blood/environmental samples",
        limitations="Electrode fouling over time; limited multiplexing per electrode",
    ),
    "NMR_relaxometry": TagReadoutStrategy(
        strategy_name="NMR Relaxometry (Paramagnetic)",
        readout_type="NMR",
        sensitivity="µM",
        multiplexing_capacity=1,
        time_to_result_min=5,
        equipment_required="Benchtop NMR relaxometer (e.g., Bruker Minispec)",
        field_deployable=True,
        description="Paramagnetic metal binding changes T1/T2 of water protons. "
                    "Benchtop relaxometer measures directly in crude sample. "
                    "No separation, no labels, no tags needed.",
        advantages="No sample prep, no labels, works in turbid/colored samples, "
                   "non-destructive, compact benchtop instruments available",
        limitations="Only for paramagnetic metals (Fe, Mn, Co, Ni, Cu, Gd); "
                   "µM sensitivity (worse than fluorescence)",
    ),
}


def recommend_readout(metal_formula, required_sensitivity="µM",
                       field_deployable=False, multiplexing_needed=1):
    """Recommend optimal readout strategy for a given target and constraints.

    This is the mass-spec replacement logic: for each target and
    deployment scenario, what combination of binder + readout gives
    the best detection?
    """
    # Check if NMR is viable (paramagnetic metals only)
    nmr_viable = metal_formula in ("Fe3+", "Fe2+", "Mn2+", "Co2+", "Ni2+",
                                    "Cu2+", "Cr3+", "Gd3+")

    # Sensitivity ranking
    sens_rank = {"ppt": 0, "ppb": 1, "nM": 2, "µM": 3, "mM": 4}
    required_rank = sens_rank.get(required_sensitivity, 3)

    candidates = []
    for name, strat in _TAG_STRATEGIES.items():
        strat_rank = sens_rank.get(strat.sensitivity, 3)
        if strat_rank > required_rank:
            continue  # Not sensitive enough
        if field_deployable and not strat.field_deployable:
            continue
        if strat.multiplexing_capacity < multiplexing_needed:
            continue
        if name == "NMR_relaxometry" and not nmr_viable:
            continue

        # Score: sensitivity + speed + simplicity
        score = (4 - strat_rank) * 10  # Sensitivity weight
        score += max(0, 60 - strat.time_to_result_min)  # Speed bonus
        if strat.field_deployable:
            score += 20
        if strat.multiplexing_capacity >= 100:
            score += 15

        candidates.append((score, name, strat))

    candidates.sort(reverse=True)
    return [c[2] for c in candidates[:3]]  # Top 3

