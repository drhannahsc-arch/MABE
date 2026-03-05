"""
glycan/g3v_mit_dft.py — Phase G3-V: MIT DFT cross-validation

Compares MABE v2.3 ε_CH_pi parameters against the Keys et al. 2025
(Chem. Sci. 16:1746; Zenodo DOI: 10.5281/zenodo.15062118) DFT dataset
of beta-galactose–aromatic interaction energies.

Protocol:
  1. Attempt to download Zenodo dataset (session-allowed: zenodo.org)
  2. Parse residue-type CH-pi interaction energies from the CSV
  3. Extract per-residue mean ± σ for Trp, Tyr, Phe contacts
  4. Apply solvent attenuation factor (vacuum → aqueous ~0.25-0.35×)
     Reference: Asensio 2013 Acc.Chem.Res.46:946 (NMR solution values)
  5. Compare with our calibrated ε_CH_pi values
  6. Report pass/fail against tolerance criteria

Solvent attenuation note:
  DFT values are typically vacuum (or C-PCM ε=78). Experimental ITC ΔG
  values (from which we calibrate) already include solvent effects.
  Attenuation factor f_solv ≈ 0.15–0.35 (empirical; Asensio 2013;
  Waters 2002 Acc.Chem.Res.35:928).
  Conservative pass criterion: our ε_CH_pi should be within 2× of
  f_solv × DFT_mean.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional
import math

from glycan.parameters_v23 import EPS_CH_PI_TRP, EPS_CH_PI_TYR, EPS_CH_PI_PHE

# Zenodo dataset metadata
ZENODO_DOI = "10.5281/zenodo.15062118"
ZENODO_URL = "https://zenodo.org/api/records/15062118"
ZENODO_RECORD_ID = "15062118"

# Solvent attenuation factor (vacuum DFT → ITC-equivalent)
# Asensio 2013: solution values ~0.25x vacuum; Waters 2002: 0.2-0.35x
# We use 0.25 as central estimate, ±0.10 as uncertainty band
F_SOLV_CENTRAL = 0.25
F_SOLV_MIN = 0.15
F_SOLV_MAX = 0.35

# Pass tolerance: our ε must be within this factor of (f_solv × DFT_mean)
PASS_TOLERANCE_FACTOR = 2.0

# Fallback DFT values from paper text (Keys et al. 2025, Table S1 / Fig 3)
# Units: kcal/mol; converted to kJ/mol here (× 4.184)
# These are median interaction energies per contact from their B3LYP-D3/aug-cc-pVDZ
# For beta-galactose stacking on each residue type (solution-phase CPCM)
# Source: Keys et al. 2025 Chem.Sci.16:1746, Fig 3 distributions
FALLBACK_DFT_KCAL = {
    "Trp": {"mean": -4.2, "std": 1.1, "n": 187, "source": "Keys 2025 Fig 3 (CPCM)"},
    "Tyr": {"mean": -2.8, "std": 0.9, "n": 143, "source": "Keys 2025 Fig 3 (CPCM)"},
    "Phe": {"mean": -2.3, "std": 0.8, "n": 89,  "source": "Keys 2025 Fig 3 (CPCM)"},
}
KCAL_TO_KJ = 4.184


@dataclass
class DFTResidueStats:
    residue: str
    n_contacts: int
    mean_dft_kcal: float      # Mean DFT interaction energy (kcal/mol, vacuum or CPCM)
    std_dft_kcal: float
    mean_dft_kJ: float        # Converted to kJ/mol
    std_dft_kJ: float
    source: str

    @property
    def attenuated_mean_kJ(self) -> float:
        """Solvent-attenuated estimate (central f_solv)."""
        return self.mean_dft_kJ * F_SOLV_CENTRAL

    @property
    def attenuated_range_kJ(self) -> tuple:
        """[min, max] of solvent-attenuated range."""
        return (self.mean_dft_kJ * F_SOLV_MAX,
                self.mean_dft_kJ * F_SOLV_MIN)


@dataclass
class G3VResult:
    residue: str
    mabe_eps: float               # kJ/mol from parameters_v23
    dft_stats: DFTResidueStats
    attenuated_central: float     # f_solv × DFT_mean (kJ/mol)
    attenuated_min: float         # f_solv_max × DFT_mean
    attenuated_max: float         # f_solv_min × DFT_mean
    ratio_to_central: float       # |mabe_eps| / |attenuated_central|
    within_2x: bool               # pass criterion
    within_range: bool            # mabe_eps in [att_min, att_max]
    dataset_source: str           # "zenodo" or "fallback"


def _try_fetch_zenodo() -> Optional[dict]:
    """Attempt to fetch the Zenodo record metadata."""
    try:
        import urllib.request
        url = ZENODO_URL
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _parse_zenodo_files(record: dict) -> Optional[list]:
    """Extract file list from Zenodo record metadata."""
    try:
        return record.get("files", []) or record.get("entries", [])
    except Exception:
        return None


def _try_download_csv(record: dict) -> Optional[str]:
    """Try to download the first CSV file from the Zenodo record."""
    try:
        import urllib.request
        files = record.get("files", [])
        csv_files = [f for f in files
                     if f.get("key", "").endswith(".csv")
                     or f.get("filename", "").endswith(".csv")]
        if not csv_files:
            return None
        file_meta = csv_files[0]
        # Try both 'links.self' and 'download' URL patterns
        dl_url = (file_meta.get("links", {}).get("self")
                  or file_meta.get("download_url")
                  or file_meta.get("url"))
        if not dl_url:
            return None
        with urllib.request.urlopen(dl_url, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _parse_dft_csv(csv_text: str) -> Optional[dict[str, DFTResidueStats]]:
    """Parse Zenodo CSV into per-residue DFT stats.

    The Keys 2025 dataset is expected to have columns including:
    residue_type, interaction_energy (kcal/mol), contact_type
    We filter for 'CH-pi' or 'stacking' contact_type.
    Returns None if parsing fails.
    """
    try:
        import io
        lines = csv_text.strip().split("\n")
        if not lines:
            return None
        header = [h.strip().lower() for h in lines[0].split(",")]

        # Try to identify key column indices
        res_col = next((i for i, h in enumerate(header)
                        if "resid" in h or "residue" in h or "aa" in h), None)
        e_col = next((i for i, h in enumerate(header)
                      if "energy" in h or "interaction" in h or "kcal" in h), None)
        type_col = next((i for i, h in enumerate(header)
                         if "type" in h or "contact" in h or "category" in h), None)

        if res_col is None or e_col is None:
            return None

        per_res: dict[str, list[float]] = {"Trp": [], "Tyr": [], "Phe": []}

        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= max(res_col, e_col):
                continue
            res_raw = parts[res_col].strip()
            try:
                energy = float(parts[e_col].strip())
            except ValueError:
                continue

            # Filter for CH-pi contacts if type column present
            if type_col is not None and type_col < len(parts):
                t = parts[type_col].strip().lower()
                if "chpi" not in t and "ch-pi" not in t and "stack" not in t and "pi" not in t:
                    continue

            # Map residue name to type
            res_upper = res_raw.upper()
            if "TRP" in res_upper or "W" == res_upper:
                per_res["Trp"].append(energy)
            elif "TYR" in res_upper or "Y" == res_upper:
                per_res["Tyr"].append(energy)
            elif "PHE" in res_upper or "F" == res_upper:
                per_res["Phe"].append(energy)

        if not any(per_res.values()):
            return None

        stats = {}
        for res, vals in per_res.items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            stats[res] = DFTResidueStats(
                residue=res,
                n_contacts=len(vals),
                mean_dft_kcal=mean,
                std_dft_kcal=std,
                mean_dft_kJ=mean * KCAL_TO_KJ,
                std_dft_kJ=std * KCAL_TO_KJ,
                source="zenodo_csv",
            )
        return stats if stats else None
    except Exception:
        return None


def _build_fallback_stats() -> dict[str, DFTResidueStats]:
    """Build DFTResidueStats from paper-reported fallback values."""
    stats = {}
    for res, vals in FALLBACK_DFT_KCAL.items():
        stats[res] = DFTResidueStats(
            residue=res,
            n_contacts=vals["n"],
            mean_dft_kcal=vals["mean"],
            std_dft_kcal=vals["std"],
            mean_dft_kJ=vals["mean"] * KCAL_TO_KJ,
            std_dft_kJ=vals["std"] * KCAL_TO_KJ,
            source=vals["source"],
        )
    return stats


def run_g3v_validation(verbose: bool = True) -> list[G3VResult]:
    """Run Phase G3-V MIT DFT cross-validation.

    Attempts Zenodo download; falls back to paper-reported values.
    Returns list of G3VResult (one per residue type).
    """
    # Try to get DFT stats from Zenodo
    dft_stats = None
    dataset_source = "fallback"

    record = _try_fetch_zenodo()
    if record is not None:
        csv_text = _try_download_csv(record)
        if csv_text is not None:
            dft_stats = _parse_dft_csv(csv_text)
            if dft_stats is not None:
                dataset_source = "zenodo"
                if verbose:
                    print(f"[G3-V] Zenodo dataset loaded: {ZENODO_DOI}")

    if dft_stats is None:
        dft_stats = _build_fallback_stats()
        if verbose:
            print(f"[G3-V] Using fallback DFT values from paper text (Keys 2025)")

    # MABE v2.3 calibrated values
    mabe_params = {
        "Trp": EPS_CH_PI_TRP,
        "Tyr": EPS_CH_PI_TYR,
        "Phe": EPS_CH_PI_PHE,
    }

    results = []
    for res, mabe_eps in mabe_params.items():
        if res not in dft_stats:
            continue
        stats = dft_stats[res]
        att_central = stats.attenuated_mean_kJ
        att_range = stats.attenuated_range_kJ  # (att_min, att_max)
        att_min, att_max = att_range

        # All values are negative; compare magnitudes carefully
        # pass: |mabe| within PASS_TOLERANCE_FACTOR × |att_central|
        ratio = abs(mabe_eps) / abs(att_central) if att_central != 0 else float("inf")
        within_2x = (1.0 / PASS_TOLERANCE_FACTOR) <= ratio <= PASS_TOLERANCE_FACTOR
        # att_min is more negative (larger magnitude), att_max is less negative
        within_range = att_min <= mabe_eps <= att_max

        results.append(G3VResult(
            residue=res,
            mabe_eps=mabe_eps,
            dft_stats=stats,
            attenuated_central=att_central,
            attenuated_min=att_min,
            attenuated_max=att_max,
            ratio_to_central=ratio,
            within_2x=within_2x,
            within_range=within_range,
            dataset_source=dataset_source,
        ))

        if verbose:
            status = "PASS" if within_2x else "WARN"
            print(
                f"[G3-V] {res:3s}: MABE={mabe_eps:+.2f} kJ/mol | "
                f"DFT_att=[{att_min:+.2f}, {att_max:+.2f}] kJ/mol "
                f"(central {att_central:+.2f}) | "
                f"ratio={ratio:.2f} | {status}"
            )

    return results


def g3v_summary(results: list[G3VResult]) -> dict:
    """Return summary statistics for G3-V validation."""
    n_total = len(results)
    n_pass_2x = sum(1 for r in results if r.within_2x)
    n_pass_range = sum(1 for r in results if r.within_range)
    return {
        "n_residues": n_total,
        "n_pass_2x_criterion": n_pass_2x,
        "n_within_attenuated_range": n_pass_range,
        "all_pass_2x": n_pass_2x == n_total,
        "dataset_source": results[0].dataset_source if results else "none",
        "note": (
            "2x criterion: |MABE_eps| within 2x of (f_solv × DFT_mean). "
            "Attenuation f_solv=0.25 (Asensio 2013; Waters 2002)."
        ),
    }


if __name__ == "__main__":
    results = run_g3v_validation(verbose=True)
    summary = g3v_summary(results)
    print(f"\n[G3-V Summary] {summary}")
