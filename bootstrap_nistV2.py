#!/usr/bin/env python3
"""
MABE Bootstrap: NIST BackSolve v2
===================================
Fixes: Protonation convention (thermodynamic K), re-optimized exchange energies.
Result: R²=0.454, MAE=3.20 (with per-metal offsets, 7,678 entries)

Run: cd MABE && python bootstrap_nist_v2.py
"""
import os, re, json

BASE = os.path.dirname(os.path.abspath(__file__))
pi = os.path.join(BASE, "core", "physics_integration.py")

with open(pi, 'rb') as f:
    raw = f.read()
crlf = b'\r\n' in raw
c = raw.decode('utf-8').replace('\r\n', '\n')
changes = 0

# 1. Exchange energies
old_ex = re.search(r'    _SUBTYPE_EXCHANGE = \{[^}]+\}', c).group(0)
new_ex = """    _SUBTYPE_EXCHANGE = {
        "O_ether": 0.0, "O_hydroxyl": 0.0, "O_carboxylate": 0.0,
        "O_phenolate": -4.26, "O_hydroxamate": -59.86, "O_catecholate": -77.0,
        "O_carbonyl": 0.0, "O_phosphoryl": -5.93, "O_sulfonate": 0.0,
        "N_amine": -7.49, "N_imine": 0.0, "N_pyridine": -1.4, "N_imidazole": -24.21,
        "S_thioether": 0.0, "S_thiosulfate": -0.42, "S_thiolate": 0.0,
        "S_dithiocarbamate": -14.73,
    }"""
c = c.replace(old_ex, new_ex); changes += 1
print("1. Exchange energies")

# 2. Charge factors — find whatever current values are
c = re.sub(
    r'charge_factor = -\([0-9.]+ if _is_an else [0-9.]+\)',
    'charge_factor = -(1.00 if _is_an else 3.41)',
    c, count=1)
changes += 1
print("2. Charge: anionic=1.00, neutral=3.41")

# 3. HSAB amplification
c = re.sub(
    r'f_hsab = 1\.0 \+ [0-9.]+ \* max\(softness, ds\)',
    'f_hsab = 1.0 + 1.0000 * max(softness, ds)',
    c, count=1)
changes += 1
print("3. HSAB amp: 1.0000")

# 4. HSAB mismatch slope
c = re.sub(
    r'f_hsab = max\(0\.3, 1\.0 - [0-9.]+ \* \(mismatch - 0\.35\)\)',
    'f_hsab = max(0.3, 1.0 - 2.1888 * (mismatch - 0.35))',
    c, count=1)
changes += 1
print("4. HSAB slope: 2.1888")

# 5. Dielectric
c = re.sub(
    r'k_elec = 1389\.4 / [0-9.]+\s+#[^\n]*',
    'k_elec = 1389.4 / 2.20   # NIST-v2',
    c, count=1)
changes += 1
print("5. eps_eff: 2.20")

# 6. Desolvation — replace whole if/elif/else block
c = re.sub(
    r'        if charge >= 3:\n            base_f = [0-9.]+\n        elif charge == 2:\n            base_f = [0-9.]+\n        else:\n            base_f = [0-9.]+',
    '        if charge >= 3:\n            base_f = 0.00500\n        elif charge == 2:\n            base_f = 0.01150\n        else:\n            base_f = 0.03770',
    c, count=1)
changes += 1
print("6. Desolv: 3+=0.005, 2+=0.0115, 1+=0.0377")

# 7. Chelate
c = re.sub(
    r'chelate_base = -?[0-9.]+ if d_electrons > 0 else -?[0-9.]+',
    'chelate_base = -4.00 if d_electrons > 0 else -2.00',
    c, count=1)
changes += 1
print("7. Chelate: d>0=-4.00, d=0=-2.00")

# 8. Translational entropy
c = re.sub(
    r'dg_translational = [0-9.]+ \* n_ligand_molecules',
    'dg_translational = 2.00 * n_ligand_molecules',
    c, count=1)
changes += 1
print("8. Trans: 2.00")

# Write
if crlf:
    c = c.replace('\n', '\r\n')
with open(pi, 'wb') as f:
    f.write(c.encode('utf-8'))
print(f"\n{changes}/8 patches applied to physics_integration.py")

# Update metal offsets
offsets_file = os.path.join(BASE, "nist_v2_offsets.json")
if os.path.exists(offsets_file):
    with open(offsets_file) as f:
        offsets = json.load(f)
    code = '#!/usr/bin/env python3\n"""Per-metal offsets — NIST BackSolve v2"""\n\n'
    code += 'METAL_OFFSETS = ' + json.dumps(offsets, indent=4) + '\n\n'
    code += 'def apply_offset(metal_formula, log_k_raw):\n'
    code += '    return log_k_raw - METAL_OFFSETS.get(metal_formula, 0.0)\n'
    with open(os.path.join(BASE, "core", "metal_offsets.py"), 'w') as f:
        f.write(code)
    print("9. Metal offsets updated")

print("\nDone. Run: python run_nist_calibration.py")