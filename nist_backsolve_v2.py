#!/usr/bin/env python3
"""
MABE NIST BackSolve v2 — Structural Fixes + Re-optimization
==============================================================
Changes from v1:
  1. Protonation penalty zeroed (NIST log K = thermodynamic, not conditional)
  2. Cooperative trivalent term (nonlinear anionic stabilization for z>=3)
  3. Hg/Ag CN=2 attenuation (donors beyond 2 heavily penalized)
  4. 31 parameters (29 + 2 structural)
"""

import json, math, os, sys, time
import numpy as np
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# ── Metal properties (unchanged from v1) ──
PREFERRED_CN = {
    "Li+": 4, "Na+": 6, "K+": 6, "Rb+": 8, "Cs+": 8,
    "Ag+": 2, "Cu+": 4, "Tl+": 6, "Au+": 2,
    "Mg2+": 6, "Ca2+": 6, "Sr2+": 8, "Ba2+": 8,
    "Mn2+": 6, "Fe2+": 6, "Co2+": 6, "Ni2+": 6,
    "Cu2+": 6, "Zn2+": 6, "Cd2+": 6, "Pb2+": 6,
    "Hg2+": 4, "Pd2+": 4, "Pt2+": 4, "Sn2+": 6,
    "Ru2+": 6, "Os2+": 6, "VO2+": 5,
    "Fe3+": 6, "Al3+": 6, "Cr3+": 6, "Ga3+": 6, "In3+": 6,
    "Bi3+": 6, "Sc3+": 6, "Au3+": 4, "Rh3+": 6,
    "Ir3+": 6, "Ti3+": 6, "V3+": 6, "Ru3+": 6,
    "La3+": 9, "Ce3+": 9, "Nd3+": 9, "Sm3+": 9,
    "Eu3+": 8, "Gd3+": 8, "Tb3+": 8,
    "Dy3+": 8, "Ho3+": 8, "Er3+": 8,
    "Yb3+": 8, "Lu3+": 8,
    "Th4+": 8, "U4+": 8, "UO2_2+": 5,
}
HYDRATED_RADIUS_NM = {
    "Li+": 0.38, "Na+": 0.36, "K+": 0.33, "Rb+": 0.33, "Cs+": 0.33,
    "Ag+": 0.34, "Cu+": 0.30, "Tl+": 0.33, "Au+": 0.30,
    "Mg2+": 0.43, "Ca2+": 0.41, "Sr2+": 0.41, "Ba2+": 0.40,
    "Mn2+": 0.44, "Fe2+": 0.43, "Co2+": 0.42, "Ni2+": 0.40,
    "Cu2+": 0.42, "Zn2+": 0.43, "Cd2+": 0.43, "Pb2+": 0.40,
    "Hg2+": 0.40, "Pd2+": 0.40, "Pt2+": 0.40, "Sn2+": 0.40,
    "Fe3+": 0.46, "Al3+": 0.48, "Cr3+": 0.46, "Ga3+": 0.47,
    "In3+": 0.44, "Au3+": 0.40, "Bi3+": 0.40,
    "La3+": 0.45, "Ce3+": 0.45, "Nd3+": 0.44, "Sm3+": 0.44,
    "Eu3+": 0.43, "Gd3+": 0.43, "Dy3+": 0.43,
    "Yb3+": 0.42, "Lu3+": 0.42,
    "Sc3+": 0.44, "Rh3+": 0.42, "Ir3+": 0.42,
    "Ti3+": 0.44, "V3+": 0.44, "Ru2+": 0.42, "Ru3+": 0.42,
    "Os2+": 0.42, "VO2+": 0.40, "Th4+": 0.45, "U4+": 0.45,
}

_PAIRED_DONORS = {"O_carboxylate": 2, "O_phosphoryl": 2, "O_sulfonate": 3}
DONOR_PRIORITY = {
    "O_catecholate": 10, "O_hydroxamate": 9, "O_phenolate": 8,
    "S_thiolate": 8, "N_pyridine": 7, "N_imine": 7,
    "N_amine": 7, "N_imidazole": 7, "S_thioether": 6,
    "O_carboxylate": 6, "O_hydroxyl": 5, "P_phosphine": 5,
    "O_phosphoryl": 4, "O_carbonyl": 3, "O_ether": 2,
    "O_sulfonate": 2, "S_thiosulfate": 3, "S_dithiocarbamate": 5,
}

def deduplicate_donors(ds):
    counts = Counter(ds)
    out = []
    for s in ds:
        if s in _PAIRED_DONORS:
            if sum(1 for d in out if d == s) < max(1, counts[s] // _PAIRED_DONORS[s]):
                out.append(s)
        else:
            out.append(s)
    return out

def select_donors(ds, cn):
    if len(ds) <= cn: return ds
    return sorted(ds, key=lambda d: DONOR_PRIORITY.get(d, 1), reverse=True)[:cn]

# ── Parameter layout ──
SUBTYPE_ORDER = [
    "O_ether", "O_hydroxyl", "O_carboxylate", "O_phenolate",
    "O_hydroxamate", "O_catecholate", "O_carbonyl", "O_phosphoryl", "O_sulfonate",
    "N_amine", "N_imine", "N_pyridine", "N_imidazole",
    "S_thioether", "S_thiosulfate", "S_thiolate", "S_dithiocarbamate",
    "P_phosphine",
]
SUBTYPE_IDX = {s: i for i, s in enumerate(SUBTYPE_ORDER)}
N_SUB = 18

DONOR_SOFTNESS = {
    "O_ether": 0.10, "O_hydroxyl": 0.10, "O_carboxylate": 0.15,
    "O_phenolate": 0.20, "O_hydroxamate": 0.18, "O_catecholate": 0.20,
    "O_carbonyl": 0.15, "O_phosphoryl": 0.15, "O_sulfonate": 0.15,
    "N_amine": 0.40, "N_imine": 0.50, "N_pyridine": 0.45, "N_imidazole": 0.45,
    "S_thioether": 0.75, "S_thiosulfate": 0.60, "S_thiolate": 0.80,
    "S_dithiocarbamate": 0.70, "P_phosphine": 0.75,
}
ANIONIC_SUBTYPES = {"O_carboxylate", "O_phenolate", "O_catecholate",
                    "O_hydroxamate", "S_thiolate", "S_thiosulfate", "S_dithiocarbamate"}
DONOR_RADIUS_PM = {"O": 140, "S": 184, "N": 155, "P": 195, "Cl": 181, "Br": 196, "I": 220}

# LINEAR metals that strongly prefer CN=2
LINEAR_METALS = {"Hg2+", "Ag+", "Au+"}

# [0:18] exchange, [18] α_an, [19] α_neut, [20] hsab_amp, [21] hsab_slope,
# [22] ε_eff, [23] f3, [24] f2, [25] f1, [26] chelate_d, [27] chelate_0,
# [28] trans, [29] coop_trivalent, [30] linear_atten
N_PARAMS = 31

# Starting from v1 optimized values
P0 = np.array([
    # Exchange (from v1 optimization — but will shift significantly
    # since protonation is removed, N donors carry less phantom penalty)
    -1.0, -5.0, -4.0, -10.0, -18.0, -30.0, -3.0, -4.0, -4.0,  # O
    -6.0, -8.0, -7.0, -8.0,                                      # N
    -10.0, -6.0, -18.0, -10.0,                                    # S
    -15.0,                                                         # P
    5.0, 5.5, 0.3, 1.5, 4.0, 0.015, 0.025, 0.04, -12.0, -8.0, 5.5,  # physics
    -3.0,   # coop_trivalent: extra kJ/mol per anionic donor beyond 2, for z>=3
    0.15,   # linear_atten: fraction of energy retained for donors beyond CN=2
])

LO = np.array([
    -5, -15, -15, -30, -60, -80, -10, -15, -15,
    -25, -30, -25, -30,
    -35, -25, -45, -35,
    -40,
    1, 1, 0.05, 0.5, 2.0, 0.005, 0.005, 0.01, -25, -20, 2,
    -15.0,  # coop_trivalent
    0.01,   # linear_atten (nearly zero = strong attenuation)
], dtype=float)

HI = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0,
    15, 15, 1.0, 4.0, 15, 0.06, 0.08, 0.12, -4, -2, 12,
    0.0,    # coop_trivalent (must be stabilizing)
    0.5,    # linear_atten (max 50% of full energy for excess donors)
], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def nist_to_formula(nist):
    import re
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
    m = re.match(r'^(\w+)\((\w+)\)$', nist)
    if not m: return nist
    e, r = m.groups()
    c = roman.get(r, 0)
    return f"{e}+" if c == 1 else f"{e}{c}+"

def extract_features(path, max_n=None):
    from core.physics_integration import (compute_enhanced_thermodynamics,
                                          _HYDRATION_FREE_ENERGY, _AQUA_CN)
    from core.generative_physics_adapter import (
        RecognitionChemistry, StructuralConstraint, InteriorDesign,
        Problem, TargetSpecies, Matrix)
    from core.solvation import get_hydration_profile

    with open(path) as f:
        entries = json.load(f)
    entries = [e for e in entries if e["log_K_type"] == "K1"]
    print(f"  K1: {len(entries)}")
    if max_n: entries = entries[:max_n]

    feats = []
    skip = 0
    for i, entry in enumerate(entries):
        try:
            ef = nist_to_formula(entry["metal_formula"])
            z = entry["metal_charge"]; de = entry["metal_d_electrons"]
            soft = entry["metal_softness"]; rpm = entry["metal_ionic_radius_pm"]
            hsab = entry["metal_hsab"]

            raw = entry["donor_subtypes"]
            deduped = deduplicate_donors(raw)
            cn = PREFERRED_CN.get(ef, 6)
            eff_sub = select_donors(deduped, cn)
            eff_don = [d.split("_")[0] for d in eff_sub]
            dent = len(eff_don)
            if dent == 0: skip += 1; continue

            chel = max(0, dent - 1) if dent > 1 else 0
            hr = HYDRATED_RADIUS_NM.get(ef, 0.40)
            avg_ds = sum(DONOR_SOFTNESS.get(d, 0.3) for d in eff_sub) / dent
            hm = max(0.0, 1.0 - abs(soft - avg_ds) * 2.0)

            target = TargetSpecies(identity=entry["metal_formula"], formula=ef,
                charge=z, d_electrons=de, hsab_softness=soft,
                coordination_number=dent, ionic_radius_pm=rpm, hydrated_radius_nm=hr)
            problem = Problem(target=target, matrix=Matrix(ph=7.0, temperature_c=25.0, ionic_strength_mm=100.0))
            rec = RecognitionChemistry(name=entry["entry_id"], type="chelator",
                donor_atoms=eff_don, donor_type="mixed", denticity=dent,
                hsab_match=hm, chelate_rings=chel)
            rec.donor_subtypes = eff_sub
            rec.is_macrocyclic = entry.get("is_macrocyclic", False)
            rec.cavity_radius_nm = 0.0; rec.ring_sizes = None; rec.is_cage = False
            struct = StructuralConstraint(name="free", type="free", geometry="octahedral", pore_size_nm=0.0)
            inter = InteriorDesign(description="free", num_binding_sites=1, self_binding=False)

            thermo = compute_enhanced_thermodynamics(rec, struct, inter, problem)

            # ── FIX 1: Exclude protonation from fixed terms ──
            # NIST log K = thermodynamic (fully deprotonated ligand).
            # Protonation penalty only applies to conditional K at specific pH.
            fixed = (thermo.dg_lfse_kj + thermo.dg_activity_kj
                + thermo.dg_screening_kj + thermo.dg_repulsion_kj
                + thermo.dg_dispersion_kj + thermo.dg_covalent_kj
                + thermo.dg_polarization_kj + thermo.dg_hydrophobic_kj
                + thermo.dg_relativistic_correction_kj + thermo.dg_ring_strain_kj
                + getattr(thermo, 'dg_jahn_teller_kj', 0.0)
                + thermo.dg_preorg_kj + thermo.dg_macrocyclic_kj)
            # NOTE: dg_protonation_kj deliberately EXCLUDED

            # Desolvation features
            dg_hyd = _HYDRATION_FREE_ENERGY.get(ef, None)
            cn_aqua = _AQUA_CN.get(ef, 6)
            pwk = abs(dg_hyd) / max(1, cn_aqua) if dg_hyd else 50.0
            wd = min(dent, cn)
            has_soft = any(d.split("_")[0] in ("S","P","I") for d in eff_sub)
            all_O = all(d.split("_")[0] == "O" for d in eff_sub)
            dhf = 0.7 if (soft > 0.5 and has_soft) or (soft < 0.2 and all_O) else 1.0
            hydr = get_hydration_profile(ef)
            lf = 1.3 if (hydr and hydr.lability_class == "inert") else \
                 0.85 if (hydr and hydr.lability_class == "labile") else 1.0

            # Per-donor features
            dfeat = []
            for ds in eff_sub:
                elem = ds.split("_")[0]
                dfeat.append({
                    "idx": SUBTYPE_IDX.get(ds, -1),
                    "an": ds in ANIONIC_SUBTYPES,
                    "elem": elem,
                    "ds": DONOR_SOFTNESS.get(ds, 0.3),
                    "rpm": DONOR_RADIUS_PM.get(elem, 140),
                    "zd": -1 if ds in ANIONIC_SUBTYPES else 0,
                })

            n_anionic = sum(1 for d in dfeat if d["an"])
            nsd = sum(1 for d in dfeat if d["elem"] in ("S","P","I"))
            is_linear_metal = ef in LINEAR_METALS

            feats.append({
                "eid": entry["entry_id"], "metal": entry["metal_formula"],
                "lig": entry["ligand_name"][:40], "logK": entry["log_K_exp"],
                "hsab": hsab, "z": z, "de": de, "soft": soft,
                "rpm": rpm, "cn": cn, "dent": dent, "chel": chel,
                "df": dfeat, "fix": fixed, "pwk": pwk, "wd": wd,
                "dhf": dhf, "lf": lf, "nsd": nsd,
                "n_an": n_anionic, "is_lin": is_linear_metal, "ef": ef,
            })
            if (i+1) % 1000 == 0: print(f"  [{i+1}/{len(entries)}]")
        except Exception as ex:
            skip += 1
            if skip <= 5: print(f"  SKIP: {ex}")
    print(f"  Extracted: {len(feats)}, skipped: {skip}")
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict(p, f):
    ex = p[0:N_SUB]
    a_an, a_ne = p[18], p[19]
    ha, hs = p[20], p[21]
    eps = p[22]
    f3, f2, f1 = p[23], p[24], p[25]
    ch_d, ch_0 = p[26], p[27]
    tr = p[28]
    coop_tri = p[29]     # NEW: cooperative trivalent
    lin_att = p[30]       # NEW: linear metal attenuation

    z = f["z"]; soft = f["soft"]; donors = f["df"]; nd = len(donors)

    # ── dG_bind ──
    de_list = []
    for d in donors:
        idx = d["idx"]
        dg = ex[idx] if 0 <= idx < N_SUB else -8.0
        alpha = a_an if d["an"] else a_ne
        dg += -alpha * (z**2 - 1) / max(1, nd)
        mis = abs(soft - d["ds"])
        if mis < 0.15:
            fh = 1.0 + ha * max(soft, d["ds"]) * (1.0 - mis / 0.15)
        elif mis < 0.35:
            fh = 1.0
        else:
            fh = max(0.3, 1.0 - hs * (mis - 0.35))
        dg *= fh
        de_list.append(dg)

    dg_bind = sum(de_list)

    # Soft saturation (unchanged)
    if soft > 0.5 and f["nsd"] > 3:
        si = [j for j, d in enumerate(donors) if d["elem"] in ("S","P","I")]
        if len(si) > 3:
            dg_bind -= sum(de_list[j] for j in si[3:]) * 0.4

    # ── FIX 2: Cooperative trivalent stabilization ──
    # When z>=3 and >=3 anionic donors, the electrostatic shell creates
    # cooperative enhancement beyond linear summation.
    # dG_coop = coop_tri * (n_anionic - 2) * z  for n_anionic >= 3
    if z >= 3 and f["n_an"] >= 3:
        dg_coop = coop_tri * (f["n_an"] - 2) * z
        dg_bind += dg_coop

    # ── FIX 3: Linear metal CN=2 attenuation ──
    # Hg²⁺, Ag⁺, Au⁺ prefer linear 2-coordinate. Donors beyond the
    # first 2 contribute only lin_att fraction of their exchange energy.
    if f["is_lin"] and nd > 2:
        # Sort donors by strength (most negative first)
        sorted_idx = sorted(range(nd), key=lambda j: de_list[j])
        # Keep top 2 at full strength, attenuate rest
        full_2 = sum(de_list[j] for j in sorted_idx[:2])
        rest = sum(de_list[j] for j in sorted_idx[2:])
        dg_bind = full_2 + rest * lin_att

    # ── dG_desolv ──
    bf = f3 if z >= 3 else f2 if z == 2 else f1
    dg_des = f["pwk"] * f["wd"] * bf * f["dhf"] * f["lf"]

    # ── dG_chelate ──
    cb = ch_d if f["de"] > 0 else ch_0
    if z == 1: cb *= 0.5
    dg_chel = cb * f["chel"]

    # ── dG_zz ──
    ke = 1389.4 / max(1.0, eps)
    dg_zz = 0.0
    for d in donors:
        if d["zd"] < 0:
            dg_zz -= ke * z * abs(d["zd"]) / (f["rpm"] + d["rpm"])

    # ── dG_trans ──
    nc = f["chel"]
    if nc == 0: nlm = nd
    elif nc >= nd - 1: nlm = 1
    else: nlm = max(1, nd // max(2, (nd + nc) // max(1, nc)))
    dg_tr = tr * nlm

    dg_net = dg_bind + dg_des + dg_chel + dg_zz + dg_tr + f["fix"]
    return -dg_net / 5.71


def resid_fn(p, feats, yexp):
    return np.array([predict(p, f) - yexp[i] for i, f in enumerate(feats)])


def stats(a, p):
    a, p = np.asarray(a), np.asarray(p)
    n = len(a)
    if n == 0: return {"n":0,"R2":0,"MAE":0,"RMSE":0,"bias":0}
    m = a.mean(); ssr = np.sum((a-p)**2); sst = np.sum((a-m)**2)
    return {"n":n, "R2": round(float(1-ssr/sst) if sst>0 else 0,4),
            "MAE": round(float(np.mean(np.abs(a-p))),2),
            "RMSE": round(float(np.sqrt(ssr/n)),2),
            "bias": round(float(p.mean()-m),2)}


def diagnostics(p, feats, pred):
    yexp = np.array([f["logK"] for f in feats])
    bm = defaultdict(lambda: {"a":[],"p":[]})
    bh = defaultdict(lambda: {"a":[],"p":[]})
    bd = defaultdict(lambda: {"a":[],"p":[]})
    for i, f in enumerate(feats):
        bm[f["metal"]]["a"].append(yexp[i]); bm[f["metal"]]["p"].append(pred[i])
        bh[f["hsab"]]["a"].append(yexp[i]); bh[f["hsab"]]["p"].append(pred[i])
        for d in set(dd["idx"] for dd in f["df"]):
            if 0 <= d < N_SUB:
                bd[SUBTYPE_ORDER[d]]["a"].append(yexp[i])
                bd[SUBTYPE_ORDER[d]]["p"].append(pred[i])

    print("\n" + "="*70 + "\n  OPTIMIZED PARAMETERS\n" + "="*70)
    print(f"\n  {'Subtype':<22s} {'Start':>8s} {'Opt':>8s} {'Δ':>8s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for i, nm in enumerate(SUBTYPE_ORDER):
        print(f"  {nm:<22s} {P0[i]:8.2f} {p[i]:8.2f} {p[i]-P0[i]:+8.2f}")

    names = ["α_an","α_ne","HSAB_amp","HSAB_slp","ε_eff",
             "f3+","f2+","f1+","chel_d","chel_0","trans",
             "coop_tri","lin_att"]
    print(f"\n  {'Param':<22s} {'Start':>8s} {'Opt':>8s} {'Δ':>8s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for j, nm in enumerate(names):
        idx = N_SUB + j
        print(f"  {nm:<22s} {P0[idx]:8.4f} {p[idx]:8.4f} {p[idx]-P0[idx]:+8.4f}")

    print(f"\n  {'Metal':12s} {'n':>5s} {'R2':>7s} {'MAE':>6s} {'bias':>7s}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*7}")
    for m in sorted(bm, key=lambda x: -len(bm[x]["a"])):
        d = bm[m]
        if len(d["a"]) >= 10:
            s = stats(d["a"], d["p"])
            print(f"  {m:12s} {s['n']:5d} {s['R2']:7.3f} {s['MAE']:6.2f} {s['bias']:+7.2f}")

    print(f"\n  {'HSAB':12s} {'n':>5s} {'R2':>7s} {'MAE':>6s} {'bias':>7s}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*7}")
    for h in ["hard","borderline","soft"]:
        if h in bh:
            s = stats(bh[h]["a"], bh[h]["p"])
            print(f"  {h:12s} {s['n']:5d} {s['R2']:7.3f} {s['MAE']:6.2f} {s['bias']:+7.2f}")

    print(f"\n  {'Donor':20s} {'n':>5s} {'MAE':>6s} {'bias':>7s}")
    print(f"  {'-'*20} {'-'*5} {'-'*6} {'-'*7}")
    for ds in sorted(bd, key=lambda x: -len(bd[x]["a"])):
        d = bd[ds]
        if len(d["a"]) >= 20:
            s = stats(d["a"], d["p"])
            print(f"  {ds:20s} {s['n']:5d} {s['MAE']:6.2f} {s['bias']:+7.2f}")

    resid = [(i, pred[i]-yexp[i]) for i in range(len(feats))]
    resid.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  WORST 20:")
    print(f"  {'Metal':10s} {'Ligand':40s} {'exp':>6s} {'pred':>6s} {'resid':>7s}")
    print(f"  {'-'*10} {'-'*40} {'-'*6} {'-'*6} {'-'*7}")
    for idx, r in resid[:20]:
        f = feats[idx]
        print(f"  {f['metal']:10s} {f['lig']:40s} {yexp[idx]:6.1f} {pred[idx]:6.1f} {r:+7.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from scipy.optimize import least_squares

    jp = os.path.join(BASE, "knowledge", "nist_calibration_entries.json")
    print("="*70)
    print("  MABE NIST BackSolve v2 — Structural Fixes")
    print("="*70)

    feats = extract_features(jp)
    yexp = np.array([f["logK"] for f in feats])

    p0_pred = np.array([predict(P0, f) for f in feats])
    s0 = stats(yexp, p0_pred)
    print(f"\n  BEFORE: R2={s0['R2']:.4f} MAE={s0['MAE']:.2f} RMSE={s0['RMSE']:.2f} bias={s0['bias']:+.2f}")

    print(f"\n  Optimizing {N_PARAMS} params on {len(feats)} entries...")
    t0 = time.time()
    res = least_squares(resid_fn, P0, args=(feats, yexp),
        bounds=(LO, HI), method='trf', loss='soft_l1', f_scale=3.0,
        max_nfev=500, verbose=2)
    dt = time.time() - t0
    print(f"\n  Done in {dt:.1f}s, nfev={res.nfev}")

    op = res.x
    pred = np.array([predict(op, f) for f in feats])
    sf = stats(yexp, pred)
    print(f"\n  AFTER: R2={sf['R2']:.4f} MAE={sf['MAE']:.2f} RMSE={sf['RMSE']:.2f} bias={sf['bias']:+.2f}")

    diagnostics(op, feats, pred)

    # Per-metal offsets
    from collections import defaultdict
    by_m = defaultdict(list)
    for i, f in enumerate(feats):
        by_m[f["metal"]].append(pred[i] - yexp[i])
    offsets = {m: round(float(np.mean(r)), 3) for m, r in by_m.items()}

    pred_off = np.array([pred[i] - offsets[feats[i]["metal"]] for i in range(len(feats))])
    sf2 = stats(yexp, pred_off)
    print(f"\n  WITH OFFSETS: R2={sf2['R2']:.4f} MAE={sf2['MAE']:.2f} RMSE={sf2['RMSE']:.2f} bias={sf2['bias']:+.2f}")

    # Save
    pd = {nm: round(float(op[i]), 3) for i, nm in enumerate(SUBTYPE_ORDER)}
    pd.update({
        "alpha_anionic": round(float(op[18]),3), "alpha_neutral": round(float(op[19]),3),
        "hsab_amp": round(float(op[20]),4), "hsab_slope": round(float(op[21]),4),
        "epsilon_eff": round(float(op[22]),2), "f_trivalent": round(float(op[23]),5),
        "f_divalent": round(float(op[24]),5), "f_monovalent": round(float(op[25]),5),
        "chelate_d": round(float(op[26]),2), "chelate_0": round(float(op[27]),2),
        "trans": round(float(op[28]),2),
        "coop_trivalent": round(float(op[29]),3),
        "linear_attenuation": round(float(op[30]),4),
    })
    with open(os.path.join(BASE, "nist_v2_params.json"), 'w') as f:
        json.dump(pd, f, indent=2)

    with open(os.path.join(BASE, "nist_v2_offsets.json"), 'w') as f:
        json.dump(offsets, f, indent=2)

    print(f"\n  Saved: nist_v2_params.json, nist_v2_offsets.json")
    print("="*70 + "\n  DONE\n" + "="*70)