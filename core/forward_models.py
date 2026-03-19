"""
acoustic/forward_models.py — Acoustic Forward Models

Three mechanisms for sound blocking, paralleling the optical pipeline:

1. ATMM (Acoustic Transfer Matrix Method) — analog of optical TMM
   Multilayer structures: computes transmission loss spectrum for
   arbitrary layer stacks. Each layer defined by (thickness, ρ, v, loss_factor).

2. Local Resonance Model — sub-wavelength sound blocking
   Core-shell particles (heavy core + soft coating) create bandgaps
   at frequencies determined by resonance, not Bragg periodicity.
   This is the acoustic analog of plasmonic resonance.

3. Phononic Bandgap Calculator — Bragg regime
   Periodic structures: bandgap center frequency and width from
   material pair impedance contrast and layer thicknesses.

Physics sources:
  - Kinsler & Frey "Fundamentals of Acoustics" (TMM)
  - Liu et al. 2000 Science 289:1734 (local resonance)
  - Kushwaha et al. 1993 PRL 71:2022 (phononic bandgap)
  - Maldovan 2013 Nature 503:209 (phonon engineering review)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from acoustic.impedance_db import get_material, AcousticMaterial


# ═══════════════════════════════════════════════════════════════════════════
# 1. ACOUSTIC TRANSFER MATRIX METHOD (ATMM)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AcousticLayer:
    """Single layer in an acoustic multilayer stack."""
    material: str           # key into ACOUSTIC_DB
    thickness_m: float      # layer thickness
    loss_factor: float = 0.0  # η: internal damping (0 = lossless)

    @property
    def props(self) -> AcousticMaterial:
        return get_material(self.material)


@dataclass
class ATMMResult:
    """Result of ATMM calculation."""
    frequencies_Hz: np.ndarray
    transmission_loss_dB: np.ndarray      # TL (higher = more blocking)
    reflection_coefficient: np.ndarray     # |R| (0-1)
    absorption_coefficient: np.ndarray     # α (0-1, requires loss)
    peak_tl_dB: float = 0.0
    peak_tl_freq_Hz: float = 0.0
    mean_tl_dB: float = 0.0              # average over band
    stc_estimate: int = 0                  # Sound Transmission Class


def atmm_spectrum(
    layers: List[AcousticLayer],
    medium_in: str = "air",
    medium_out: str = "air",
    freq_min_Hz: float = 20.0,
    freq_max_Hz: float = 20000.0,
    n_freq: int = 500,
) -> ATMMResult:
    """
    Compute transmission loss spectrum for a multilayer acoustic stack.

    Analog of optical TMM. Uses 2×2 transfer matrices relating pressure
    and velocity at each interface.

    For a single layer of impedance Z, thickness d, at frequency f:
        M = [[cos(kd),      j Z sin(kd)],
             [j sin(kd)/Z,  cos(kd)     ]]

    where k = 2πf/v × (1 + jη/2) is the complex wavenumber.

    Total transfer matrix: M_total = M_1 × M_2 × ... × M_n

    Transmission coefficient: t = 2 / (M11 + M12/Z_out + Z_in×M21 + Z_in×M22/Z_out)
    """
    freqs = np.linspace(freq_min_Hz, freq_max_Hz, n_freq)

    z_in = get_material(medium_in).Z_longitudinal
    z_out = get_material(medium_out).Z_longitudinal

    tl_array = np.zeros(n_freq)
    r_array = np.zeros(n_freq)
    a_array = np.zeros(n_freq)

    for fi, f in enumerate(freqs):
        if f <= 0:
            continue

        # Build total transfer matrix
        M = np.eye(2, dtype=complex)

        for layer in layers:
            props = layer.props
            z = props.Z_longitudinal
            v = props.v_longitudinal_m_s
            d = layer.thickness_m
            eta = layer.loss_factor

            # Complex wavenumber (loss included)
            k = 2 * math.pi * f / v * (1 + 1j * eta / 2)

            kd = k * d
            cos_kd = np.cos(kd)
            sin_kd = np.sin(kd)

            layer_M = np.array([
                [cos_kd,        1j * z * sin_kd],
                [1j * sin_kd / z, cos_kd        ]
            ], dtype=complex)

            M = M @ layer_M

        # Transmission coefficient
        M11, M12, M21, M22 = M[0,0], M[0,1], M[1,0], M[1,1]
        denom = M11 + M12 / z_out + z_in * M21 + z_in * M22 / z_out
        t = 2.0 / denom

        # Power transmission
        tau = abs(t)**2 * z_in / z_out
        tau = min(tau, 1.0)  # physical bound

        # Reflection
        r = (M11 + M12/z_out - z_in*M21 - z_in*M22/z_out) / denom
        R = abs(r)**2

        # Absorption
        alpha = 1.0 - tau - R
        alpha = max(0.0, alpha)

        # Transmission loss
        if tau > 1e-15:
            tl_array[fi] = 10 * math.log10(1.0 / tau)
        else:
            tl_array[fi] = 150.0  # cap

        r_array[fi] = math.sqrt(R)
        a_array[fi] = alpha

    # Peak and mean TL
    peak_idx = np.argmax(tl_array)
    peak_tl = tl_array[peak_idx]
    peak_freq = freqs[peak_idx]
    mean_tl = np.mean(tl_array)

    # STC estimate (simplified: average TL in 125-4000 Hz band)
    mask = (freqs >= 125) & (freqs <= 4000)
    stc = int(round(np.mean(tl_array[mask]))) if mask.any() else 0

    return ATMMResult(
        frequencies_Hz=freqs,
        transmission_loss_dB=tl_array,
        reflection_coefficient=r_array,
        absorption_coefficient=a_array,
        peak_tl_dB=peak_tl,
        peak_tl_freq_Hz=peak_freq,
        mean_tl_dB=mean_tl,
        stc_estimate=stc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. LOCAL RESONANCE MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LocalResonanceResult:
    """Result of local resonance calculation."""
    resonance_freq_Hz: float
    bandgap_lower_Hz: float
    bandgap_upper_Hz: float
    bandgap_width_Hz: float
    bandgap_fractional: float       # Δf / f_center
    effective_mass_ratio: float     # m_eff / m_static at resonance
    peak_attenuation_dB_per_cell: float


def local_resonance(
    core_material: str,
    coating_material: str,
    core_radius_m: float,
    coating_thickness_m: float,
    matrix_material: str = "epoxy",
    filling_fraction: float = 0.4,
) -> LocalResonanceResult:
    """
    Calculate resonance frequency and bandgap for locally resonant particle.

    Physics (Liu et al. 2000):
    A heavy core (mass m) connected to the matrix by a soft coating
    (spring constant K) creates a resonance at f = (1/2π)√(K/m).

    Below resonance: effective mass increases (mass amplification).
    Above resonance: effective mass goes negative → bandgap.

    K_eff = 4π × G_coating × core_radius × (R_outer/d_coating)
    where G = shear modulus of coating.
    """
    core = get_material(core_material)
    coat = get_material(coating_material)
    matrix = get_material(matrix_material)

    r_core = core_radius_m
    d_coat = coating_thickness_m
    r_outer = r_core + d_coat

    # Core mass
    m_core = (4/3) * math.pi * r_core**3 * core.density_kg_m3

    # Coating shear modulus (from v_T if available, else estimate)
    if coat.v_transverse_m_s > 0:
        G_coat = coat.density_kg_m3 * coat.v_transverse_m_s**2
    else:
        # For fluids/soft polymers: estimate G from v_L
        # G ≈ 0 for liquids; for soft rubber, G ~ 0.1-1 MPa
        G_coat = coat.density_kg_m3 * (coat.v_longitudinal_m_s * 0.1)**2

    # Effective spring constant
    # K = 4π G_coat r_core (r_outer / d_coat)  [simplified shell model]
    if d_coat > 0 and G_coat > 0:
        K_eff = 4 * math.pi * G_coat * r_core * (r_outer / d_coat)
    else:
        K_eff = 1.0

    # Resonance frequency
    f_res = (1 / (2 * math.pi)) * math.sqrt(K_eff / m_core)

    # Bandgap width estimate (from effective medium theory)
    # Δf/f ≈ filling_fraction × (m_core/m_matrix_cell) for small ff
    m_matrix_cell = (4/3) * math.pi * r_outer**3 * matrix.density_kg_m3
    mass_ratio = m_core / m_matrix_cell if m_matrix_cell > 0 else 1.0

    gap_fractional = filling_fraction * mass_ratio * 0.5  # approximate
    gap_fractional = min(gap_fractional, 0.8)  # physical cap

    f_lower = f_res * (1 - gap_fractional / 2)
    f_upper = f_res * (1 + gap_fractional / 2)

    # Peak attenuation (very approximate)
    peak_atten = 20 * math.log10(1 + mass_ratio * filling_fraction)

    return LocalResonanceResult(
        resonance_freq_Hz=f_res,
        bandgap_lower_Hz=f_lower,
        bandgap_upper_Hz=f_upper,
        bandgap_width_Hz=f_upper - f_lower,
        bandgap_fractional=gap_fractional,
        effective_mass_ratio=mass_ratio,
        peak_attenuation_dB_per_cell=peak_atten,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. PHONONIC BANDGAP (BRAGG)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BraggBandgapResult:
    """Result of Bragg phononic bandgap calculation."""
    center_freq_Hz: float
    gap_lower_Hz: float
    gap_upper_Hz: float
    gap_width_Hz: float
    gap_fractional: float
    layer1_thickness_m: float
    layer2_thickness_m: float
    period_m: float
    impedance_ratio: float
    n_periods_for_20dB: int       # periods needed for 20 dB TL


def bragg_phononic_bandgap(
    material1: str,
    material2: str,
    target_freq_Hz: float,
) -> BraggBandgapResult:
    """
    Calculate Bragg bandgap for 1D phononic crystal.

    Quarter-wave condition: d_i = v_i / (4 × f_target)
    Bandgap width: Δf/f = (2/π) × arcsin(|Z₂-Z₁|/(Z₂+Z₁))
    """
    m1 = get_material(material1)
    m2 = get_material(material2)

    z1 = m1.Z_longitudinal
    z2 = m2.Z_longitudinal

    # Quarter-wave thicknesses
    d1 = m1.v_longitudinal_m_s / (4 * target_freq_Hz)
    d2 = m2.v_longitudinal_m_s / (4 * target_freq_Hz)
    period = d1 + d2

    # Bandgap width
    gap_frac = (2 / math.pi) * math.asin(abs(z2 - z1) / (z2 + z1))

    f_center = target_freq_Hz
    f_lower = f_center * (1 - gap_frac / 2)
    f_upper = f_center * (1 + gap_frac / 2)

    # Impedance ratio
    z_ratio = max(z1, z2) / min(z1, z2)

    # Periods needed for 20 dB TL at center
    # Each period adds ~(20 log10(Z_ratio)) dB at center frequency
    tl_per_period = 20 * math.log10(z_ratio) if z_ratio > 1 else 0.1
    n_for_20dB = max(1, int(math.ceil(20.0 / tl_per_period)))

    return BraggBandgapResult(
        center_freq_Hz=f_center,
        gap_lower_Hz=f_lower,
        gap_upper_Hz=f_upper,
        gap_width_Hz=f_upper - f_lower,
        gap_fractional=gap_frac,
        layer1_thickness_m=d1,
        layer2_thickness_m=d2,
        period_m=period,
        impedance_ratio=z_ratio,
        n_periods_for_20dB=n_for_20dB,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN API: TARGET FREQUENCY → BEST MATERIAL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SoundBlockingDesign:
    """Complete design for a sound-blocking structure."""
    mechanism: str                  # 'bragg', 'local_resonance', 'multilayer'
    target_freq_Hz: float
    materials: list
    layer_thicknesses_m: list
    total_thickness_m: float
    predicted_tl_dB: float
    bandgap_width_Hz: float = 0.0
    n_layers: int = 0
    notes: str = ""


def design_sound_blocker(
    target_freq_Hz: float,
    max_thickness_m: float = 0.1,
    mechanism: str = "auto",
    verbose: bool = False,
) -> List[SoundBlockingDesign]:
    """
    Design a sound-blocking structure for a target frequency.

    Automatically selects the best mechanism and material combination.

    Args:
        target_freq_Hz: center frequency to block
        max_thickness_m: maximum total thickness allowed
        mechanism: 'bragg', 'local_resonance', 'multilayer', or 'auto'
    """
    designs = []

    # ── Bragg designs ─────────────────────────────────────────────────
    if mechanism in ("bragg", "auto"):
        from acoustic.impedance_db import rank_material_pairs_for_bandgap
        pairs = rank_material_pairs_for_bandgap(target_freq_Hz)

        for m1, m2, gap, d1, d2, fb in pairs[:5]:
            bg = bragg_phononic_bandgap(m1, m2, target_freq_Hz)
            n_periods = bg.n_periods_for_20dB
            total = bg.period_m * n_periods

            if total <= max_thickness_m:
                designs.append(SoundBlockingDesign(
                    mechanism="bragg",
                    target_freq_Hz=target_freq_Hz,
                    materials=[m1, m2],
                    layer_thicknesses_m=[bg.layer1_thickness_m, bg.layer2_thickness_m],
                    total_thickness_m=total,
                    predicted_tl_dB=20.0,
                    bandgap_width_Hz=bg.gap_width_Hz,
                    n_layers=n_periods * 2,
                    notes=f"Bragg {m1}/{m2}, {n_periods} periods, "
                          f"gap={bg.gap_fractional*100:.1f}%",
                ))

    # ── Local resonance designs ───────────────────────────────────────
    if mechanism in ("local_resonance", "auto"):
        # Heavy core options
        cores = ["lead", "tungsten", "steel_mild", "barium_titanate"]
        coatings = ["silicone_rubber", "PDMS", "natural_rubber", "polyurethane"]

        for core_mat in cores:
            for coat_mat in coatings:
                # Solve for core radius that gives target frequency
                # f = (1/2π)√(K/m), K ∝ r, m ∝ r³ → f ∝ 1/r
                # Start with r = v_coat / (2π f) and iterate
                core = get_material(core_mat)
                coat = get_material(coat_mat)

                # Initial estimate: r ~ v_coating / (2π f_target) × correction
                if coat.v_transverse_m_s > 0:
                    v_est = coat.v_transverse_m_s
                else:
                    v_est = coat.v_longitudinal_m_s * 0.1

                r_est = v_est / (2 * math.pi * target_freq_Hz) * 2.0
                r_est = max(1e-4, min(r_est, 0.05))  # 0.1mm to 50mm

                coat_thick = r_est * 0.3  # coating ~30% of core radius

                lr = local_resonance(core_mat, coat_mat, r_est, coat_thick)

                # Check if resonance is within 50% of target
                if abs(lr.resonance_freq_Hz - target_freq_Hz) / target_freq_Hz < 0.5:
                    total = 2 * (r_est + coat_thick)  # single particle diameter
                    designs.append(SoundBlockingDesign(
                        mechanism="local_resonance",
                        target_freq_Hz=target_freq_Hz,
                        materials=[core_mat, coat_mat],
                        layer_thicknesses_m=[r_est, coat_thick],
                        total_thickness_m=total,
                        predicted_tl_dB=lr.peak_attenuation_dB_per_cell,
                        bandgap_width_Hz=lr.bandgap_width_Hz,
                        n_layers=1,
                        notes=f"Resonant: {core_mat} core r={r_est*1000:.1f}mm + "
                              f"{coat_mat} coat {coat_thick*1000:.1f}mm, "
                              f"f_res={lr.resonance_freq_Hz:.0f} Hz",
                    ))

    # Sort by TL per thickness (efficiency)
    designs.sort(key=lambda d: -d.predicted_tl_dB / max(d.total_thickness_m, 1e-6))

    if verbose:
        print(f"\nSOUND BLOCKING DESIGNS for {target_freq_Hz:.0f} Hz:")
        print(f"{'#':>3s} {'Mechanism':>15s} {'Materials':>30s} "
              f"{'Thick(mm)':>10s} {'TL(dB)':>7s} {'Gap(Hz)':>10s}")
        for i, d in enumerate(designs[:10]):
            mats = "/".join(d.materials)
            print(f"{i+1:3d} {d.mechanism:>15s} {mats:>30s} "
                  f"{d.total_thickness_m*1000:10.1f} {d.predicted_tl_dB:7.1f} "
                  f"{d.bandgap_width_Hz:10.0f}")

    return designs


if __name__ == "__main__":
    # ── ATMM Test: simple steel plate in air ──────────────────────────
    print("═" * 70)
    print("ATMM: Steel plate (10mm) in air")
    print("═" * 70)
    result = atmm_spectrum(
        [AcousticLayer("steel_mild", 0.010)],
        medium_in="air", medium_out="air",
    )
    print(f"  Peak TL: {result.peak_tl_dB:.1f} dB at {result.peak_tl_freq_Hz:.0f} Hz")
    print(f"  Mean TL: {result.mean_tl_dB:.1f} dB")
    print(f"  STC estimate: {result.stc_estimate}")

    # ── ATMM: Multilayer steel/rubber/steel ───────────────────────────
    print(f"\nATMM: Steel(5mm)/Rubber(10mm)/Steel(5mm)")
    result2 = atmm_spectrum([
        AcousticLayer("steel_mild", 0.005),
        AcousticLayer("silicone_rubber", 0.010, loss_factor=0.3),
        AcousticLayer("steel_mild", 0.005),
    ])
    print(f"  Peak TL: {result2.peak_tl_dB:.1f} dB at {result2.peak_tl_freq_Hz:.0f} Hz")
    print(f"  Mean TL: {result2.mean_tl_dB:.1f} dB")
    print(f"  STC estimate: {result2.stc_estimate}")

    # ── Local resonance: Liu 2000 reproduction ────────────────────────
    print(f"\nLOCAL RESONANCE: Lead/silicone_rubber (Liu 2000 analog)")
    lr = local_resonance("lead", "silicone_rubber",
                          core_radius_m=0.005, coating_thickness_m=0.0025)
    print(f"  Resonance: {lr.resonance_freq_Hz:.0f} Hz")
    print(f"  Bandgap: {lr.bandgap_lower_Hz:.0f} – {lr.bandgap_upper_Hz:.0f} Hz")
    print(f"  Gap width: {lr.bandgap_width_Hz:.0f} Hz ({lr.bandgap_fractional*100:.1f}%)")
    print(f"  Peak attenuation: {lr.peak_attenuation_dB_per_cell:.1f} dB/cell")

    # ── Bragg bandgap ─────────────────────────────────────────────────
    print(f"\nBRAGG: Steel/PDMS at 1 kHz")
    bg = bragg_phononic_bandgap("steel_mild", "PDMS", 1000)
    print(f"  Gap: {bg.gap_lower_Hz:.0f} – {bg.gap_upper_Hz:.0f} Hz "
          f"({bg.gap_fractional*100:.1f}%)")
    print(f"  Layer thicknesses: {bg.layer1_thickness_m*1000:.2f} / "
          f"{bg.layer2_thickness_m*1000:.2f} mm")
    print(f"  Periods for 20 dB: {bg.n_periods_for_20dB}")

    # ── Design API ────────────────────────────────────────────────────
    print(f"\n" + "═" * 70)
    for freq in [100, 500, 1000, 5000]:
        designs = design_sound_blocker(freq, max_thickness_m=0.1, verbose=True)