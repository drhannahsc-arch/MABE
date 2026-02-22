"""
core/deployment_scoring.py — Sprint 31: Deployment Scoring Layer

After thermodynamic scoring, runs each candidate through:
cooperativity → mass_transport → surface → photostability → phonon → radiation
Produces deployment_score alongside binding_score.
"""
from dataclasses import dataclass
import math

from core.cooperativity import analyze_cooperativity
from core.mass_transport import analyze_transport, predict_capture_time
from core.surface_magnetic import get_surface_profile, compute_magnetic_properties
from core.photostability import assess_photostability
from core.phonon_thermal import analyze_phonon_stability
from core.electron_transfer import assess_radiation_stability


@dataclass
class DeploymentScore:
    """Deployment feasibility assessment for a binder design."""
    # Subscores (0-100 each)
    transport_score: float = 0.0
    capacity_score: float = 0.0
    wetting_score: float = 0.0
    thermal_score: float = 0.0
    uv_score: float = 0.0
    radiation_score: float = 0.0
    kinetics_score: float = 0.0
    # Overall
    deployment_score: float = 0.0    # Weighted composite (0-100)
    deployment_class: str = ""       # "field_ready", "lab_viable", "needs_engineering", "redesign"
    # Details
    transport_regime: str = ""
    capacity_mg_g: float = 0.0
    capture_time_90_min: float = 0.0
    max_temp_C: float = 0.0
    outdoor_lifetime_days: float = 0.0
    wettability: str = ""
    limiting_factor: str = ""
    recommendations: list = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


def score_deployment(scaffold_type, target_formula, target_charge=2,
                      target_mw=60.0, hydrated_radius_nm=0.2,
                      pore_diameter_nm=0.0, binding_energy_kj=50.0,
                      ion_mass_amu=60.0, unpaired_electrons=0,
                      is_nuclear=False, outdoor_use=False,
                      target_conc_uM=1.0):
    """Full deployment feasibility scoring."""
    recs = []

    # === TRANSPORT ===
    if pore_diameter_nm > 0:
        tp = analyze_transport(hydrated_radius_nm, pore_diameter_nm)
        transport_score = tp.effectiveness_factor * 100
        transport_regime = tp.transport_regime
        if tp.transport_regime == "excluded":
            transport_score = 0
            recs.append(f"Ion EXCLUDED from pore (λ={tp.lambda_ratio:.2f}). Need larger pore.")
        elif tp.transport_regime == "severely_hindered":
            recs.append(f"Severely hindered diffusion (H={tp.hindrance_factor:.4f}). "
                        f"Consider larger pore or smaller particle.")
    else:
        transport_score = 90  # Free solution / open scaffold
        transport_regime = "unhindered"

    # === CAPACITY ===
    coop = analyze_cooperativity(scaffold_type, target_charge, target_mw)
    capacity_mg_g = coop.capacity_mg_per_g
    capacity_score = min(100, capacity_mg_g / 2.0)  # 200 mg/g = 100
    if coop.max_practical_loading < 0.5:
        recs.append(f"Only {coop.max_practical_loading*100:.0f}% of sites usable "
                    f"(n_Hill={coop.hill_coefficient:.2f})")

    # === CAPTURE KINETICS ===
    eff = transport_score / 100.0
    ct = predict_capture_time(target_conc_uM, coop.capacity_mmol_per_g,
                               1.0, effectiveness=max(0.01, eff))
    capture_90_min = ct.time_to_90pct_s / 60.0
    kinetics_score = max(0, min(100, 100 - capture_90_min))  # <1 min = 100
    if capture_90_min > 60:
        recs.append(f"Slow capture ({capture_90_min:.0f} min to 90%). "
                    f"Increase material loading or use flow-through.")

    # === WETTING ===
    sp = get_surface_profile(scaffold_type, pore_diameter_nm)
    if sp.spontaneous_wetting:
        wetting_score = max(50, 100 - sp.contact_angle_water_deg)
    else:
        wetting_score = max(0, 30 - (sp.contact_angle_water_deg - 90))
        recs.append(f"Hydrophobic surface (θ={sp.contact_angle_water_deg}°). "
                    f"{sp.surface_treatment}")

    # === THERMAL STABILITY ===
    ph = analyze_phonon_stability(scaffold_type, binding_energy_kj,
                                    ion_mass_amu)
    max_temp = ph.max_operating_temp_C
    thermal_score = min(100, max_temp / 2.0)  # 200°C = 100
    if ph.thermal_stability_class == "unstable":
        recs.append(f"Thermally unstable at 25°C. Use stiffer scaffold "
                    f"(current Θ_D={ph.debye_temp_K} K)")

    # === UV STABILITY ===
    ps = assess_photostability(scaffold_type, outdoor_use)
    outdoor_days = ps.operational_lifetime_outdoor_days
    if outdoor_use:
        uv_score = min(100, outdoor_days / 3.0)  # 300 days = 100
        if ps.stability_class in ("UV_sensitive", "poor"):
            recs.append(f"UV-sensitive ({outdoor_days:.0f} day outdoor life). "
                        f"{ps.protection_strategy}")
    else:
        uv_score = 90  # Indoor use, UV rarely a problem

    # === RADIATION ===
    rad = assess_radiation_stability(scaffold_type, is_nuclear_application=is_nuclear)
    rad_score_map = {"excellent": 100, "good": 80, "moderate": 50,
                     "poor": 20, "unsuitable": 0, "unknown": 50}
    radiation_score = rad_score_map.get(rad.stability_rating, 50)
    if is_nuclear and rad.stability_rating in ("unsuitable", "poor"):
        recs.append(f"NOT radiation-stable ({rad.stability_rating}). "
                    f"Use: {', '.join(rad.rad_resistant_alternatives[:2])}")

    # === COMPOSITE ===
    weights = {"transport": 0.20, "capacity": 0.15, "kinetics": 0.15,
               "wetting": 0.15, "thermal": 0.10, "uv": 0.10, "radiation": 0.15}
    if not is_nuclear:
        weights["radiation"] = 0.05
        weights["capacity"] = 0.20

    composite = (transport_score * weights["transport"]
                 + capacity_score * weights["capacity"]
                 + kinetics_score * weights["kinetics"]
                 + wetting_score * weights["wetting"]
                 + thermal_score * weights["thermal"]
                 + uv_score * weights["uv"]
                 + radiation_score * weights["radiation"])

    if composite > 75:
        dep_class = "field_ready"
    elif composite > 50:
        dep_class = "lab_viable"
    elif composite > 25:
        dep_class = "needs_engineering"
    else:
        dep_class = "redesign"

    # Limiting factor
    scores = {"transport": transport_score, "capacity": capacity_score,
              "kinetics": kinetics_score, "wetting": wetting_score,
              "thermal": thermal_score, "UV": uv_score, "radiation": radiation_score}
    limiting = min(scores, key=scores.get)

    return DeploymentScore(
        transport_score=round(transport_score, 1),
        capacity_score=round(capacity_score, 1),
        wetting_score=round(wetting_score, 1),
        thermal_score=round(thermal_score, 1),
        uv_score=round(uv_score, 1),
        radiation_score=round(radiation_score, 1),
        kinetics_score=round(kinetics_score, 1),
        deployment_score=round(composite, 1),
        deployment_class=dep_class,
        transport_regime=transport_regime,
        capacity_mg_g=round(capacity_mg_g, 1),
        capture_time_90_min=round(capture_90_min, 1),
        max_temp_C=max_temp,
        outdoor_lifetime_days=round(outdoor_days, 0),
        wettability=sp.wettability,
        limiting_factor=limiting,
        recommendations=recs,
    )


