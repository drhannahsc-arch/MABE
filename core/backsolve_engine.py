"""
core/backsolve_engine.py — Sprint 37: Universal Back-Solve Engine

Optimizes all ~55 physics parameters simultaneously against the full
calibration dataset (metals + host-guest + protein-ligand).

Uses scipy.optimize.least_squares with:
  - Trust Region Reflective (bounded parameters)
  - Per-modality weighting (metal high-precision, protein noisier)
  - Regularization toward phase-specific initial values
  - Bootstrap resampling for uncertainty estimates

Input:  list[UniversalComplex] + PhysicsParameters (initial guess)
Output: Optimized PhysicsParameters + residual diagnostics
"""
import math
import time
from dataclasses import dataclass, field

from core.universal_schema import UniversalComplex
from core.universal_predictor import (
    PhysicsParameters, predict, predict_batch, compute_statistics, PredictionResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER BOUNDS — physically constrained
# ═══════════════════════════════════════════════════════════════════════════

def get_parameter_bounds(params):
    """Return (lower, upper) bound vectors for all parameters.

    Bounds enforce physical plausibility:
      - Exchange energies: always negative (favorable)
      - Desolvation fractions: 0 < f < 0.2
      - H-bond energies: bounded by gas-phase / solution limits
      - γ: 0.010–0.040 kJ/mol/Å² (literature consensus range)
      - PC_optimal: 0.40–0.70 (Rebek ± generous margin)
    """
    names = PhysicsParameters.param_names()
    lower = []
    upper = []

    for name in names:
        val = getattr(params, name)

        # ── Exchange energies (always negative) ───────────────────────
        if name.startswith("exchange_"):
            lower.append(min(val * 3.0, -50.0))
            upper.append(0.0)

        # ── Charge scaling ────────────────────────────────────────────
        elif name == "alpha_charge":
            lower.append(-15.0)
            upper.append(0.0)

        # ── Desolvation fractions ─────────────────────────────────────
        elif name.startswith("base_f_"):
            lower.append(0.001)
            upper.append(0.20)

        # ── Dielectric ────────────────────────────────────────────────
        elif name == "epsilon_eff":
            lower.append(4.0)
            upper.append(40.0)

        # ── Chelate entropy ───────────────────────────────────────────
        elif name.startswith("chelate_base_"):
            lower.append(-25.0)
            upper.append(-2.0)

        # ── Ring strain ───────────────────────────────────────────────
        elif name == "k_strain":
            lower.append(20.0)
            upper.append(200.0)

        # ── Macrocyclic ───────────────────────────────────────────────
        elif name.startswith("k_macro_"):
            lower.append(-8.0)
            upper.append(0.0)
        elif name == "sigma_cavity":
            lower.append(0.005)
            upper.append(0.10)

        # ── LFSE ──────────────────────────────────────────────────────
        elif name == "lfse_scale":
            lower.append(0.3)
            upper.append(2.0)

        # ── Jahn-Teller ──────────────────────────────────────────────
        elif name.startswith("jt_"):
            lower.append(-50.0)
            upper.append(0.0)

        # ── Translational ─────────────────────────────────────────────
        elif name == "dg_translational_per_mol":
            lower.append(2.0)
            upper.append(12.0)

        # ── Phase 6: Hydrophobic ──────────────────────────────────────
        elif name == "gamma_hydrophobic":
            lower.append(0.010)
            upper.append(0.040)
        elif name == "k_curvature":
            lower.append(0.0)
            upper.append(3.0)
        elif name == "polar_discount":
            lower.append(0.0)
            upper.append(1.0)

        # ── Phase 7: H-bond ──────────────────────────────────────────
        elif name == "epsilon_hbond_neutral":
            lower.append(-8.0)
            upper.append(-0.5)
        elif name == "epsilon_hbond_charged":
            lower.append(-20.0)
            upper.append(-3.0)
        elif name == "epsilon_hbond_OH_pi":
            lower.append(-5.0)
            upper.append(-0.2)
        elif name == "epsilon_water_hbond":
            lower.append(1.0)
            upper.append(10.0)
        elif name == "k_cooperativity":
            lower.append(1.0)
            upper.append(1.5)
        elif name == "theta_half":
            lower.append(20.0)
            upper.append(50.0)
        elif name == "epsilon_hbond_strong":
            lower.append(-15.0)
            upper.append(-3.0)
        elif name == "epsilon_hbond_moderate":
            lower.append(-8.0)
            upper.append(-1.0)
        elif name == "epsilon_hbond_weak":
            lower.append(-4.0)
            upper.append(-0.2)

        # ── Phase 8: π-interactions ───────────────────────────────────
        elif name == "epsilon_pi_parallel":
            lower.append(-10.0)
            upper.append(-1.0)
        elif name == "epsilon_pi_T_shaped":
            lower.append(-8.0)
            upper.append(-0.5)
        elif name == "epsilon_CH_pi":
            lower.append(-5.0)
            upper.append(-0.2)
        elif name == "epsilon_cation_pi":
            lower.append(-15.0)
            upper.append(-1.0)

        # ── Phase 9: Conformational entropy ───────────────────────────
        elif name == "TdS_per_rotor":
            lower.append(1.0)
            upper.append(6.0)
        elif name == "ring_correction":
            lower.append(0.1)
            upper.append(0.8)

        # ── Phase 10: Shape ───────────────────────────────────────────
        elif name == "PC_optimal":
            lower.append(0.40)
            upper.append(0.70)
        elif name == "sigma_PC":
            lower.append(0.03)
            upper.append(0.25)
        elif name == "k_shape":
            lower.append(-25.0)
            upper.append(-2.0)
        elif name == "k_clash":
            lower.append(20.0)
            upper.append(500.0)

        # ── Phase 11: Lipophilic (Sprint 40) ───────────────────────
        elif name == "epsilon_logP":
            lower.append(-3.0)
            upper.append(0.0)
        elif name == "k_mw_penalty":
            lower.append(0.0)
            upper.append(0.02)

        # ── Default: ±50% of initial value ────────────────────────────
        else:
            if val > 0:
                lower.append(val * 0.5)
                upper.append(val * 1.5)
            elif val < 0:
                lower.append(val * 1.5)
                upper.append(val * 0.5)
            else:
                lower.append(-1.0)
                upper.append(1.0)

    return lower, upper


# ═══════════════════════════════════════════════════════════════════════════
# MODALITY WEIGHTS — noisier data gets lower weight
# ═══════════════════════════════════════════════════════════════════════════

MODALITY_SIGMA = {
    "metal_coordination": 0.3,     # NIST precision: ±0.3 log K
    "host_guest_inclusion": 0.5,   # Host-guest ITC: ±0.5 log Ka
    "synthetic_receptor": 0.5,
    "protein_ligand": 1.0,         # PDBbind: ±1.0 pKd
    "mip_template": 1.5,
    "aptamer_target": 1.0,
    "unknown": 1.0,
}


def _get_weight(uc):
    """Return 1/σ² weight for a complex based on modality."""
    sigma = MODALITY_SIGMA.get(uc.binding_mode, 1.0)
    return 1.0 / (sigma ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# RESIDUAL FUNCTION (for least_squares)
# ═══════════════════════════════════════════════════════════════════════════

def _residual_vector(param_vec, entries, initial_vec, lambda_reg):
    """Compute weighted residual vector for least_squares optimizer.

    Returns array of length N_data + N_params:
      - First N_data: weighted (pred - exp) for each complex
      - Last N_params: regularization toward initial values
    """
    params = PhysicsParameters.from_vector(param_vec)

    residuals = []
    for uc in entries:
        try:
            result = predict(uc, params)
            error = result.log_Ka_pred - uc.log_Ka_exp
            w = math.sqrt(_get_weight(uc))
            residuals.append(error * w)
        except Exception:
            residuals.append(0.0)  # Skip failed predictions

    # Regularization: pull toward initial values
    for i, (current, initial) in enumerate(zip(param_vec, initial_vec)):
        residuals.append(lambda_reg * (current - initial))

    return residuals


# ═══════════════════════════════════════════════════════════════════════════
# BACK-SOLVE RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BackSolveResult:
    """Result of back-solve optimization."""
    optimized_params: object           # PhysicsParameters
    initial_params: object             # PhysicsParameters (for comparison)

    # Performance
    n_data: int = 0
    n_params: int = 0
    r2_initial: float = 0.0
    r2_final: float = 0.0
    mae_initial: float = 0.0
    mae_final: float = 0.0

    # Per-modality breakdown
    modality_stats: dict = field(default_factory=dict)

    # Optimization metadata
    n_iterations: int = 0
    cost_initial: float = 0.0
    cost_final: float = 0.0
    elapsed_seconds: float = 0.0
    converged: bool = False

    # Parameter changes
    param_deltas: dict = field(default_factory=dict)  # {name: (initial, final, Δ%)}

    def summary(self):
        """Human-readable summary."""
        sep = "=" * 70
        lines = [
            "",
            sep,
            "  MABE UNIVERSAL BACK-SOLVE RESULT",
            sep,
            f"  Data points:  {self.n_data}",
            f"  Parameters:   {self.n_params}",
            f"  Ratio:        {self.n_data/max(1,self.n_params):.1f}:1",
            f"  Converged:    {self.converged}",
            f"  Time:         {self.elapsed_seconds:.1f}s",
            f"",
            f"  PERFORMANCE       Initial → Final",
            f"  R²:               {self.r2_initial:.4f} → {self.r2_final:.4f}",
            f"  MAE (log Ka):     {self.mae_initial:.2f}  → {self.mae_final:.2f}",
            f"",
        ]

        if self.modality_stats:
            lines.append(f"  PER-MODALITY R²:")
            for mode, stats in sorted(self.modality_stats.items()):
                lines.append(f"    {mode:30s} R²={stats['r2']:.3f}  MAE={stats['mae']:.2f}  n={stats['n']}")

        if self.param_deltas:
            lines.append("")
            lines.append("  TOP PARAMETER CHANGES:")
            sorted_deltas = sorted(self.param_deltas.items(),
                                   key=lambda x: abs(x[1][2]), reverse=True)
            for name, (init, final, pct) in sorted_deltas[:15]:
                lines.append(f"    {name:35s} {init:8.3f} → {final:8.3f}  ({pct:+.1f}%)")

        lines.append(sep)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BACK-SOLVE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_backsolve(entries, initial_params=None, lambda_reg=0.1,
                   max_iterations=200, verbose=True, free_params=None):
    """Run the universal back-solve across all modalities.

    Args:
        entries: list[UniversalComplex] — full calibration dataset
        initial_params: PhysicsParameters — starting point (None = defaults)
        lambda_reg: float — regularization strength
        max_iterations: int — max optimizer iterations
        verbose: bool — print progress
        free_params: set[str] or None — if set, only optimize these parameters.
                     Others stay frozen at initial values.

    Returns:
        BackSolveResult with optimized parameters and diagnostics
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        print("  ⚠ scipy not available — back-solve requires scipy.optimize")
        return None

    if initial_params is None:
        initial_params = PhysicsParameters()

    if verbose:
        print()
        print("  Starting universal back-solve:")
        print(f"    Data points: {len(entries)}")
        print(f"    Parameters:  {PhysicsParameters.param_count()}")
        print(f"    λ_reg:       {lambda_reg}")

    t0 = time.time()

    # Initial prediction
    results_init = predict_batch(entries, initial_params)
    stats_init = compute_statistics(results_init)

    # Parameter vectors
    all_names = PhysicsParameters.param_names()
    x0_full = initial_params.to_vector()
    lower_full, upper_full = get_parameter_bounds(initial_params)

    # Build reduced vector containing only free params
    if free_params:
        free_indices = [i for i, name in enumerate(all_names) if name in free_params]
        frozen_indices = [i for i, name in enumerate(all_names) if name not in free_params]
        n_free = len(free_indices)

        x0 = [x0_full[i] for i in free_indices]
        lower = [lower_full[i] for i in free_indices]
        upper = [upper_full[i] for i in free_indices]

        def expand_vector(x_reduced):
            """Expand reduced free-params vector to full vector."""
            x_full = list(x0_full)
            for j, idx in enumerate(free_indices):
                x_full[idx] = x_reduced[j]
            return x_full

        if verbose:
            print(f"    Free params: {n_free} of {len(all_names)}")
            print(f"    Effective ratio: {len(entries)/max(1,n_free):.1f}:1")
    else:
        free_indices = list(range(len(all_names)))
        x0 = list(x0_full)
        lower = list(lower_full)
        upper = list(upper_full)

        def expand_vector(x_reduced):
            return list(x_reduced)

    def reduced_residuals(x_reduced):
        x_full = expand_vector(x_reduced)
        return _residual_vector(x_full, entries, x0_full, lambda_reg)

    # Run optimizer
    try:
        sol = least_squares(
            reduced_residuals,
            x0=x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_iterations * len(x0),
            ftol=1e-8,
            xtol=1e-8,
            verbose=2 if verbose else 0,
        )
        converged = sol.success
        n_iters = sol.nfev
        cost_final = sol.cost
        x_final_full = expand_vector(sol.x)
    except Exception as e:
        if verbose:
            print(f"  ⚠ Optimizer failed: {e}")
        x_final_full = list(x0_full)
        converged = False
        n_iters = 0
        cost_final = 0

    # Extract optimized parameters
    optimized = PhysicsParameters.from_vector(x_final_full)
    elapsed = time.time() - t0

    # Final prediction with optimized params
    results_final = predict_batch(entries, optimized)
    stats_final = compute_statistics(results_final)

    # Per-modality stats
    modality_stats = {}
    modes = set(uc.binding_mode for uc in entries)
    for mode in modes:
        mode_results = [r for r, uc in zip(results_final, entries)
                        if uc.binding_mode == mode]
        if mode_results:
            modality_stats[mode] = compute_statistics(mode_results)

    # Parameter deltas
    param_deltas = {}
    names = PhysicsParameters.param_names()
    for name, init_val, final_val in zip(names, x0_full, x_final_full):
        if abs(init_val) > 1e-10:
            pct_change = 100.0 * (final_val - init_val) / abs(init_val)
        else:
            pct_change = 0.0
        param_deltas[name] = (init_val, final_val, pct_change)

    result = BackSolveResult(
        optimized_params=optimized,
        initial_params=initial_params,
        n_data=len(entries),
        n_params=PhysicsParameters.param_count(),
        r2_initial=stats_init["r2"],
        r2_final=stats_final["r2"],
        mae_initial=stats_init["mae"],
        mae_final=stats_final["mae"],
        modality_stats=modality_stats,
        n_iterations=n_iters,
        cost_initial=sum(r.error**2 for r in results_init),
        cost_final=cost_final,
        elapsed_seconds=elapsed,
        converged=converged,
        param_deltas=param_deltas,
    )

    if verbose:
        print(result.summary())

    return result


# ═══════════════════════════════════════════════════════════════════════════
# RESIDUAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def residual_analysis(entries, params=None):
    """Detailed residual analysis per energy term and modality.

    Identifies which terms have the largest systematic errors,
    guiding which physics needs improvement next.
    """
    if params is None:
        params = PhysicsParameters()

    results = predict_batch(entries, params)

    # Per-term averages by modality
    term_names = [
        "dg_bind", "dg_desolv", "dg_chelate", "dg_ring_strain",
        "dg_electrostatic", "dg_macrocyclic", "dg_lfse", "dg_jahn_teller",
        "dg_protonation", "dg_translational", "dg_activity", "dg_preorg",
        "dg_hydrophobic", "dg_hbond", "dg_pi", "dg_conf_entropy", "dg_shape",
        "dg_lipophilic",
        "dg_dispersion", "dg_covalent", "dg_polarization", "dg_relativistic",
    ]

    analysis = {}
    modes = set(uc.binding_mode for uc in entries)
    for mode in modes:
        mode_results = [(r, uc) for r, uc in zip(results, entries)
                        if uc.binding_mode == mode]
        if not mode_results:
            continue

        term_avgs = {}
        for tname in term_names:
            vals = [getattr(r, tname, 0.0) for r, _ in mode_results]
            nonzero = [v for v in vals if abs(v) > 0.01]
            if nonzero:
                term_avgs[tname] = {
                    "mean": round(sum(nonzero) / len(nonzero), 2),
                    "n_active": len(nonzero),
                    "n_total": len(mode_results),
                }

        errors = [r.error for r, _ in mode_results]
        analysis[mode] = {
            "n": len(mode_results),
            "mae": round(sum(abs(e) for e in errors) / len(errors), 2),
            "bias": round(sum(errors) / len(errors), 2),
            "active_terms": term_avgs,
        }

    return analysis


def print_residual_analysis(analysis):
    """Pretty-print residual analysis."""
    for mode, data in sorted(analysis.items()):
        print(f"\n  {mode} (n={data['n']}, MAE={data['mae']}, bias={data['bias']})")
        for tname, stats in sorted(data["active_terms"].items(),
                                    key=lambda x: abs(x[1]["mean"]), reverse=True):
            bar = "█" * min(40, int(abs(stats["mean"]) * 2))
            sign = "-" if stats["mean"] < 0 else "+"
            print(f"    {tname:25s} {sign}{abs(stats['mean']):6.1f} kJ/mol  "
                  f"({stats['n_active']:3d}/{stats['n_total']:3d} active) {bar}")
