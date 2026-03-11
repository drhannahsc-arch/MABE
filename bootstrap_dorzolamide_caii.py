#!/usr/bin/env python3
"""
MABE Physics-Based Energy Decomposition: Dorzolamide + Carbonic Anhydrase II

Uses MABE's calibrated parameters from Phase 11b (644-complex joint calibration)
applied to the dorzolamide-CA II binding interaction from PDB 4M2U/6BC9.

This is an EXTRAPOLATION test — parameters calibrated on small-molecule
metal coordination (NIST SRD 46) and host-guest inclusion (CD/CB) are applied
to a protein-ligand system with NO additional fitting.

Convention: negative ΔG = favorable binding
"""

import math
import json

# ============================================================
# MABE CALIBRATED PARAMETERS (Phase 11b, 644-complex joint)
# Source: MABE_Universal_BackSolve_Protocol_v2.md
# ============================================================

# Metal: Exchange energies per donor subtype (kJ/mol)
EXCHANGE = {
    "O_carboxylate": +2.666,
    "O_phenolate": -8.322,
    "O_hydroxamate": -4.337,
    "O_catecholate": -12.862,
    "O_ether": +4.078,
    "O_hydroxyl": +1.444,
    "N_amine": -6.570,
    "N_pyridine": -5.081,
    "N_imine": -4.532,
    "N_imidazole": -7.729,
    "N_amide": -1.057,
    "S_thiolate": -8.454,
    "S_thioether": +1.042,
}

# Metal: Irving-Williams offsets (kJ/mol)
IW_OFFSET = {
    "Zn2+": -6.90,
    "Cu2+": -8.42,
    "Ni2+": +1.05,
    "Fe2+": +6.23,
    "Mn2+": +10.69,
    "Mg2+": None,  # not in calibration set
    "Ca2+": None,
}

# Metal: Global physics parameters
METAL_PARAMS = {
    "charge_scale": -1.974,        # per z^2
    "chelate_ring_d": -11.348,     # per chelate ring (divalent)
    "elec_zz_k": -2.941,           # electrostatic coupling
    "lfse_amp": 0.749,             # LFSE amplitude
    "rotor_cost": 1.542,           # per frozen rotor (kJ/mol)
    "freeze_chelate": 0.950,       # chelate freeze fraction
    "trans_entropy": 0.000,        # translational entropy (calibrated to 0)
    "jt_strong": -12.365,          # Jahn-Teller strong (d9)
    "jt_moderate": -7.382,         # Jahn-Teller moderate (d4)
}

# Host-Guest: Interaction parameters
HG_PARAMS = {
    "gamma_flat": 0.0100,              # hydrophobic transfer (kJ/mol/Å²)
    "dg_dehydr_per_A2": -0.0725,       # host dehydration (kJ/mol/Å²)
    "dehydr_other": 1.7815,            # multiplier for non-CD/CB hosts
    "eps_neutral": -8.027,             # neutral H-bond (kJ/mol)
    "eps_charge_assisted": -7.173,     # charge-assisted H-bond (kJ/mol)
    "water_penalty_per_hb": 3.995,     # water displacement penalty per H-bond
    "water_displacement": 1.332,       # water displacement bonus
    "eps_rotor": 2.481,                # conformational entropy per rotor (kJ/mol)
    "f_partial": 0.534,               # partial freezing fraction
    "k_shape": -4.997,                # shape complementarity (kJ/mol)
    "PC_optimal": 0.591,              # optimal packing coefficient
    "sigma_PC": 0.050,                # PC Gaussian width
}

RT = 8.314e-3 * 298.15  # kJ/mol at 25°C
RT_LN10 = RT * math.log(10)  # 5.708 kJ/mol

# ============================================================
# DORZOLAMIDE + CA-II BINDING CONTACTS
# Source: PDB 4M2U, 6BC9 (neutron structure), literature
# ============================================================

print("=" * 70)
print("MABE ENERGY DECOMPOSITION: DORZOLAMIDE + CARBONIC ANHYDRASE II")
print("=" * 70)
print()

# --- EXPERIMENTAL TARGET ---
Ki_nM = 9.1  # nM, from Casini et al. 2013 (PDB 4M2U paper)
Ki_M = Ki_nM * 1e-9
Ka_M = 1.0 / Ki_M
dG_exp = RT * math.log(Ki_M)  # negative = favorable
log_Ka_exp = -math.log10(Ki_M)
pKi_exp = -math.log10(Ki_M)

print(f"EXPERIMENTAL (Casini et al. 2013)")
print(f"  Ki = {Ki_nM} nM")
print(f"  Ka = {Ka_M:.2e} M⁻¹")
print(f"  pKi = {pKi_exp:.2f}")
print(f"  ΔG_exp = {dG_exp:.1f} kJ/mol")
print()

# ============================================================
# TERM-BY-TERM DECOMPOSITION
# ============================================================

terms = {}

print("-" * 70)
print("METAL COORDINATION TERMS")
print("-" * 70)
print()

# --- Term 1: Donor exchange ---
# Sulfonamide N⁻ coordinates Zn²⁺, replacing the 4th water ligand.
# The deprotonated sulfonamide nitrogen (R-SO₂-NH⁻) is an anionic N-donor.
# MABE subtype assignment: The pKa of sulfonamide NH₂ is ~9-10, but
# zinc coordination shifts it to <7, so at physiological pH it's deprotonated.
# Electronic character: N adjacent to electron-withdrawing SO₂.
# Best match: N_amide (adjacent to S=O, analogous to C=O) for neutral form,
# but the anionic form (N⁻) is more like a deprotonated amine.
#
# We compute BOTH assignments for transparency:
donor_n_amide = EXCHANGE["N_amide"]  # -1.057 (if neutral-like)
donor_n_amine = EXCHANGE["N_amine"]  # -6.570 (if anionic-like)

# Primary assignment: N_amine (anionic, deprotonated — Zn-N distance 1.76 Å
# is characteristic of strong coordinate bond to anionic N)
donor_exchange = donor_n_amine
donor_label = "N_amine (deprotonated sulfonamide N⁻)"

terms["exchange"] = donor_exchange
print(f"  1. Donor exchange ({donor_label})")
print(f"     Value: {donor_exchange:+.3f} kJ/mol")
print(f"     [Alt: N_amide assignment = {donor_n_amide:+.3f} kJ/mol]")
print()

# --- Term 2: Irving-Williams offset for Zn²⁺ ---
iw = IW_OFFSET["Zn2+"]
terms["iw_offset"] = iw
print(f"  2. Irving-Williams offset (Zn²⁺)")
print(f"     Value: {iw:+.3f} kJ/mol")
print(f"     Physics: d¹⁰ configuration, borderline Lewis acid")
print()

# --- Term 3: Charge scaling ---
z = 2  # Zn²⁺ charge
charge_term = METAL_PARAMS["charge_scale"] * z**2
terms["charge_scale"] = charge_term
print(f"  3. Charge scaling (z={z}, z²={z**2})")
print(f"     Value: {charge_term:+.3f} kJ/mol")
print()

# --- Term 4: Chelate effect ---
n_chelate_rings = 0  # Dorzolamide is monodentate to Zn
chelate = METAL_PARAMS["chelate_ring_d"] * n_chelate_rings
terms["chelate"] = chelate
print(f"  4. Chelate effect ({n_chelate_rings} rings)")
print(f"     Value: {chelate:+.3f} kJ/mol")
print(f"     Note: Sulfonamide is monodentate — no chelate ring")
print()

# --- Term 5: LFSE ---
# Zn²⁺ is d¹⁰ — no LFSE
lfse = 0.0
terms["lfse"] = lfse
print(f"  5. Ligand field stabilization (Zn²⁺ = d¹⁰)")
print(f"     Value: {lfse:+.3f} kJ/mol")
print()

# --- Term 6: Electrostatic ---
# Zn²⁺ (z=+2) binding anionic N⁻ (z=-1)
# Attractive: charge product = 2 × 1 = 2 (using absolute values for attraction)
z_ligand_abs = 1  # sulfonamide anion
elec = METAL_PARAMS["elec_zz_k"] * z * z_ligand_abs
terms["electrostatic"] = elec
print(f"  6. Electrostatic (Zn²⁺ × sulfonamide N⁻)")
print(f"     z_M × |z_L| = {z} × {z_ligand_abs} = {z * z_ligand_abs}")
print(f"     Value: {elec:+.3f} kJ/mol")
print()

# --- Term 7: Jahn-Teller ---
jt = 0.0  # Zn²⁺ is d¹⁰, not d⁹ or d⁴
terms["jt"] = jt

metal_subtotal = sum([terms["exchange"], terms["iw_offset"],
                      terms["charge_scale"], terms["chelate"],
                      terms["lfse"], terms["electrostatic"], terms["jt"]])

print(f"  METAL SUBTOTAL: {metal_subtotal:+.3f} kJ/mol")
print(f"  (log Ka contribution: {-metal_subtotal / RT_LN10:+.2f})")
print()

print("-" * 70)
print("HYDROGEN BOND TERMS (HG Scorer)")
print("-" * 70)
print()

# --- H-bond inventory from crystal structure ---
# PDB 4M2U/6BC9 contacts:
# 1. Sulfonamide NH → Thr199 Oγ (neutral donor → neutral acceptor)
# 2. Sulfonamide SO₂ oxygen → Thr199 backbone NH (neutral)
# 3. Ethylamino NH → His64 Nε (proton shuttle residue) or water-mediated
# 4. Ring sulfone SO₂ → possible weak contact
#
# Conservatively counting 3 well-defined direct H-bonds.
n_hbonds_neutral = 3
n_hbonds_charge = 0  # No charge-assisted H-bonds in this system

hbond_favorable = n_hbonds_neutral * HG_PARAMS["eps_neutral"] + \
                  n_hbonds_charge * HG_PARAMS["eps_charge_assisted"]
hbond_penalty = n_hbonds_neutral * HG_PARAMS["water_penalty_per_hb"]
hbond_net = hbond_favorable + hbond_penalty

terms["hbond_favorable"] = hbond_favorable
terms["hbond_penalty"] = hbond_penalty
terms["hbond_net"] = hbond_net

print(f"  H-bonds formed: {n_hbonds_neutral} neutral, {n_hbonds_charge} charge-assisted")
print(f"  Contacts: sulfonamide NH→Thr199 Oγ, SO₂ O→Thr199 NH, ethylamino NH→His64/water")
print(f"     Favorable: {n_hbonds_neutral} × {HG_PARAMS['eps_neutral']:.3f} = {hbond_favorable:+.3f} kJ/mol")
print(f"     Water penalty: {n_hbonds_neutral} × {HG_PARAMS['water_penalty_per_hb']:.3f} = {hbond_penalty:+.3f} kJ/mol")
print(f"     NET H-BOND: {hbond_net:+.3f} kJ/mol")
print()

print("-" * 70)
print("HYDROPHOBIC BURIAL TERMS (HG Scorer)")
print("-" * 70)
print()

# --- Hydrophobic burial ---
# Dorzolamide MW = 324 Da. The thienothiopyran ring system sits in the
# hydrophobic half of the active site (Val121, Val143, Leu198, Trp209).
# The active site is a ~15 Å deep, ~12 Å wide cone.
#
# Estimated buried nonpolar SASA from literature analysis:
# - Ring system (thienothiopyran): ~120 Å²
# - Methyl group: ~25 Å²
# - Ethyl group (partial): ~30 Å²
# Total estimated nonpolar SASA buried: ~175 Å²

sasa_buried = 175.0  # Å²

# Hydrophobic transfer (guest burial)
# Using standard transfer free energy: ~0.1 kJ/(mol·Å²) for nonpolar burial
# MABE's gamma_flat = 0.01 is the PER-ANGSTROM coefficient in the scorer's
# internal units. Combined with host dehydration:
dg_hydrophobic = -(HG_PARAMS["gamma_flat"] * sasa_buried)  # guest desolvation gain
dg_host_dehydr = HG_PARAMS["dg_dehydr_per_A2"] * HG_PARAMS["dehydr_other"] * sasa_buried

# Net hydrophobic: guest burial + host cavity dehydration
hydrophobic_net = dg_hydrophobic + dg_host_dehydr
# Note: For protein cavities, the dehydration term should be smaller than for
# synthetic hosts (protein active sites are not as heavily hydrated as CD cavities).
# Apply a conservative 0.5× scaling for protein vs synthetic host.
protein_dehydr_factor = 0.5
dg_host_dehydr_adj = dg_host_dehydr * protein_dehydr_factor
hydrophobic_net_adj = dg_hydrophobic + dg_host_dehydr_adj

terms["hydrophobic_guest"] = dg_hydrophobic
terms["hydrophobic_host"] = dg_host_dehydr_adj
terms["hydrophobic_net"] = hydrophobic_net_adj

print(f"  Estimated nonpolar SASA buried: {sasa_buried:.0f} Å²")
print(f"  (Ring system ~120, methyl ~25, ethyl ~30 Å²)")
print(f"     Guest desolvation gain: {dg_hydrophobic:+.3f} kJ/mol")
print(f"     Host dehydration: {dg_host_dehydr:+.3f} kJ/mol (synthetic host)")
print(f"     Protein adjustment (×{protein_dehydr_factor}): {dg_host_dehydr_adj:+.3f} kJ/mol")
print(f"     NET HYDROPHOBIC: {hydrophobic_net_adj:+.3f} kJ/mol")
print()

print("-" * 70)
print("CONFORMATIONAL ENTROPY (HG Scorer)")
print("-" * 70)
print()

# --- Conformational entropy ---
# Dorzolamide rotatable bonds that freeze on binding:
# 1. Ethylamino C-N bond (partially frozen)
# 2. NH-CH₂ bond (partially frozen)
# 3. Methyl on ring (free rotation likely maintained)
# Conservatively: 2 rotors with partial freezing
n_rotors = 2
conf_entropy = HG_PARAMS["eps_rotor"] * n_rotors * HG_PARAMS["f_partial"]
terms["conf_entropy"] = conf_entropy

print(f"  Rotatable bonds frozen: {n_rotors}")
print(f"  Partial freezing fraction: {HG_PARAMS['f_partial']:.3f}")
print(f"     Value: {n_rotors} × {HG_PARAMS['eps_rotor']:.3f} × {HG_PARAMS['f_partial']:.3f}")
print(f"     CONF ENTROPY PENALTY: +{conf_entropy:+.3f} kJ/mol")
print()

print("-" * 70)
print("SHAPE COMPLEMENTARITY (HG Scorer)")
print("-" * 70)
print()

# --- Shape complementarity ---
# The dorzolamide fits snugly into the CA-II active site cone.
# Packing coefficient = V_guest / V_cavity
# V_dorzolamide ≈ 280 Å³ (from RDKit or estimation: MW 324, density ~1.4)
# V_CA-II active site ≈ 520 Å³ (15 Å deep × ~12 Å diameter cone)
# PC ≈ 280/520 = 0.54

guest_vol = 280.0  # Å³
cavity_vol = 520.0  # Å³
PC = guest_vol / cavity_vol

# Gaussian shape score
shape = HG_PARAMS["k_shape"] * math.exp(
    -(PC - HG_PARAMS["PC_optimal"])**2 / (2 * HG_PARAMS["sigma_PC"]**2)
)
terms["shape"] = shape

print(f"  Guest volume: ~{guest_vol:.0f} Å³")
print(f"  Cavity volume: ~{cavity_vol:.0f} Å³")
print(f"  Packing coefficient: {PC:.3f} (optimal: {HG_PARAMS['PC_optimal']:.3f})")
print(f"  Deviation from optimal: {abs(PC - HG_PARAMS['PC_optimal']):.3f}")
print(f"     SHAPE SCORE: {shape:+.3f} kJ/mol")
print()

# ============================================================
# TOTAL AND COMPARISON
# ============================================================

print("=" * 70)
print("ENERGY DECOMPOSITION SUMMARY")
print("=" * 70)
print()

# Organize by physics channel
metal_terms = {
    "Donor exchange (N⁻→Zn²⁺)": terms["exchange"],
    "Irving-Williams (Zn²⁺)": terms["iw_offset"],
    "Charge scaling (z²=4)": terms["charge_scale"],
    "Chelate effect (0 rings)": terms["chelate"],
    "LFSE (d¹⁰ = 0)": terms["lfse"],
    "Electrostatic (2+ × 1-)": terms["electrostatic"],
}

hg_terms = {
    "H-bonds (3 neutral)": terms["hbond_favorable"],
    "Water penalty (3 displaced)": terms["hbond_penalty"],
    "Hydrophobic burial (175 Å²)": terms["hydrophobic_net"],
    "Conf. entropy (+2 rotors)": terms["conf_entropy"],
    "Shape (PC=0.54)": terms["shape"],
}

print(f"{'Term':<35} {'ΔG (kJ/mol)':>12}  {'log Ka':>8}")
print("-" * 58)

running_total = 0.0

print("METAL COORDINATION:")
for name, val in metal_terms.items():
    log_k_contrib = -val / RT_LN10
    running_total += val
    print(f"  {name:<33} {val:>+10.3f}  {log_k_contrib:>+8.2f}")

print(f"  {'Metal subtotal':<33} {metal_subtotal:>+10.3f}  {-metal_subtotal/RT_LN10:>+8.2f}")
print()

hg_total = 0.0
print("PROTEIN POCKET (non-covalent):")
for name, val in hg_terms.items():
    log_k_contrib = -val / RT_LN10
    hg_total += val
    running_total += val
    print(f"  {name:<33} {val:>+10.3f}  {log_k_contrib:>+8.2f}")

print(f"  {'Pocket subtotal':<33} {hg_total:>+10.3f}  {-hg_total/RT_LN10:>+8.2f}")
print()

dG_pred = running_total
log_Ka_pred = -dG_pred / RT_LN10

print("-" * 58)
print(f"  {'PREDICTED TOTAL':<33} {dG_pred:>+10.3f}  {log_Ka_pred:>+8.2f}")
print(f"  {'EXPERIMENTAL':<33} {dG_exp:>+10.1f}  {pKi_exp:>+8.2f}")
print(f"  {'ERROR (pred - exp)':<33} {dG_pred - dG_exp:>+10.1f}  {log_Ka_pred - pKi_exp:>+8.2f}")
print()

# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

print("=" * 70)
print("SENSITIVITY: DONOR SUBTYPE ASSIGNMENT")
print("=" * 70)
print()

# If we use N_amide instead of N_amine:
alt_exchange = EXCHANGE["N_amide"]
alt_total = dG_pred - terms["exchange"] + alt_exchange
alt_log_Ka = -alt_total / RT_LN10

print(f"  Primary (N_amine):  ΔG = {dG_pred:+.1f} kJ/mol, log Ka = {log_Ka_pred:.2f}")
print(f"  Alt (N_amide):      ΔG = {alt_total:+.1f} kJ/mol, log Ka = {alt_log_Ka:.2f}")
print(f"  Experimental:       ΔG = {dG_exp:+.1f} kJ/mol, pKi = {pKi_exp:.2f}")
print()
print(f"  Donor assignment shifts prediction by {abs(alt_exchange - terms['exchange']):.1f} kJ/mol")
print(f"  ({abs(alt_log_Ka - log_Ka_pred):.1f} log Ka units)")
print()

# ============================================================
# CHANNEL CONTRIBUTIONS (for pie chart / bar chart)
# ============================================================

print("=" * 70)
print("BINDING ENERGY BY PHYSICS CHANNEL")
print("=" * 70)
print()

channels = {
    "Zn²⁺ coordination": terms["exchange"] + terms["iw_offset"] + 
                          terms["charge_scale"] + terms["electrostatic"],
    "H-bond network": terms["hbond_favorable"] + terms["hbond_penalty"],
    "Hydrophobic burial": terms["hydrophobic_net"],
    "Shape complementarity": terms["shape"],
    "Conformational cost": terms["conf_entropy"],  # unfavorable
}

total_favorable = sum(v for v in channels.values() if v < 0)
total_unfavorable = sum(v for v in channels.values() if v > 0)

for name, val in channels.items():
    pct = (val / dG_pred) * 100 if dG_pred != 0 else 0
    bar_len = int(abs(val) / 2)
    bar_char = "█" if val < 0 else "░"
    direction = "favorable" if val < 0 else "UNFAVORABLE"
    print(f"  {name:<25} {val:>+8.2f} kJ/mol ({pct:>5.1f}%)  {direction}")

print(f"\n  Total favorable:    {total_favorable:+.1f} kJ/mol")
print(f"  Total unfavorable:  {total_unfavorable:+.1f} kJ/mol")
print(f"  Net:                {dG_pred:+.1f} kJ/mol")

# ============================================================
# KEY OBSERVATIONS
# ============================================================
print()
print("=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)
print()
print("1. MIXED-MODE is essential: Metal coordination alone predicts")
print(f"   log Ka ≈ {-metal_subtotal/RT_LN10:.1f}; pocket interactions add {-hg_total/RT_LN10:.1f} more.")
print(f"   Neither channel alone explains pKi = {pKi_exp:.1f}.")
print()
print("2. The Zn²⁺ bond is NOT the whole story. H-bonds and hydrophobic")
print("   burial contribute comparable energy to metal coordination.")
print()
print("3. No fitted parameters were added for this protein system.")
print("   Metal params: calibrated on 500 NIST complexes (R²=0.895)")
print("   HG params: calibrated on 80 synthetic host-guest pairs (R²=0.850)")
print("   This is pure physics extrapolation to a drug-target context.")
print()

# Save results as JSON for figure generation
results = {
    "system": "Dorzolamide + Carbonic Anhydrase II",
    "pdb_ids": ["4M2U", "6BC9", "1CIL"],
    "experimental": {
        "Ki_nM": Ki_nM,
        "pKi": round(pKi_exp, 2),
        "dG_kJ_mol": round(dG_exp, 1),
        "source": "Casini et al. 2013"
    },
    "predicted": {
        "dG_kJ_mol": round(dG_pred, 1),
        "log_Ka": round(log_Ka_pred, 2),
    },
    "error": {
        "dG_kJ_mol": round(dG_pred - dG_exp, 1),
        "log_Ka": round(log_Ka_pred - pKi_exp, 2),
    },
    "channels": {k: round(v, 2) for k, v in channels.items()},
    "terms": {k: round(v, 3) for k, v in terms.items()},
    "parameters_source": "MABE Phase 11b, 644-complex joint calibration",
}

with open("/home/claude/dorzolamide_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to dorzolamide_results.json")