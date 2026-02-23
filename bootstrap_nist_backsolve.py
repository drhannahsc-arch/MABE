#!/usr/bin/env python3
"""
MABE Bootstrap: NIST BackSolve Phase 4
Fitted: 7678 entries, R2=0.2616, MAE=3.59
"""
import os, re

BASE = os.path.dirname(os.path.abspath(__file__))
pi = os.path.join(BASE, "core", "physics_integration.py")

def patch(fp, reps):
    with open(fp, 'rb') as f: raw = f.read()
    crlf = b'\r\n' in raw
    c = raw.decode('utf-8').replace('\r\n', '\n')
    n = 0
    for old, new in reps:
        o = old.replace('\r\n', '\n')
        if o in c: c = c.replace(o, new, 1); n += 1
        else: print(f"  ⚠ Not found: {o.split(chr(10))[0][:70]}")
    if crlf: c = c.replace('\n', '\r\n')
    with open(fp, 'wb') as f: f.write(c.encode('utf-8'))
    return n

print("NIST BackSolve Phase 4 — Optimized parameters")

# 1. Exchange table
with open(pi, 'rb') as f: content = f.read().decode('utf-8').replace('\r\n', '\n')
old_ex = re.search(r'    _SUBTYPE_EXCHANGE = \{[^}]+\}', content).group(0)
new_ex = """    _SUBTYPE_EXCHANGE = {
        "O_ether": -0.0,
        "O_hydroxyl": -0.0,
        "O_carboxylate": -0.0,
        "O_phenolate": -2.19,
        "O_hydroxamate": -49.88,
        "O_catecholate": -67.2,
        "O_carbonyl": -0.0,
        "O_phosphoryl": -4.0,
        "O_sulfonate": -0.0,
        "N_amine": -20.0,
        "N_imine": -13.31,
        "N_pyridine": -16.77,
        "N_imidazole": -24.85,
        "S_thioether": -0.0,
        "S_thiosulfate": -5.01,
        "S_thiolate": -0.0,
        "S_dithiocarbamate": -22.24,
        "P_phosphine": -0.0,
    }"""
t = 0
t += patch(pi, [(old_ex, new_ex)])
print("1. Exchange energies")

# 2-8. Other params
t += patch(pi, [('charge_factor = -(5.0 if _is_an else 5.5)',
    'charge_factor = -(1.00 if _is_an else 4.99)')])
print(f"2. Charge: an=1.00 ne=4.99")

t += patch(pi, [('f_hsab = 1.0 + 0.3 * max(softness, ds)',
    'f_hsab = 1.0 + 0.9699 * max(softness, ds)')])
print(f"3. HSAB amp: 0.9699")

t += patch(pi, [('f_hsab = max(0.3, 1.0 - 1.5 * (mismatch - 0.35))',
    'f_hsab = max(0.3, 1.0 - 0.5000 * (mismatch - 0.35))')])
print(f"4. HSAB slope: 0.5000")

t += patch(pi, [('k_elec = 1389.4 / 4.0   # = 347.4',
    'k_elec = 1389.4 / 2.00   # NIST-optimized')])
print(f"5. eps_eff: 2.00")

t += patch(pi, [('            base_f = 0.015', '            base_f = 0.00500')])
t += patch(pi, [('            base_f = 0.025', '            base_f = 0.00500')])
t += patch(pi, [('            base_f = 0.04', '            base_f = 0.01000')])
print(f"6. Desolv: 3+=0.00500 2+=0.00500 1+=0.01000")

t += patch(pi, [('chelate_base = -12.0 if d_electrons > 0 else -8.0',
    'chelate_base = -4.61 if d_electrons > 0 else -2.09')])
print(f"7. Chelate: d>0=-4.61 d=0=-2.09")

t += patch(pi, [('dg_translational = 5.5 * n_ligand_molecules',
    'dg_translational = 2.00 * n_ligand_molecules')])
print(f"8. Trans: 2.00")

print(f"\n{t}/11 patches applied")