#!/usr/bin/env python3
"""
MABE Bootstrap: NIST Physics Patch
====================================
Applies ON TOP of Sprint 36 (physics_integration.py must exist in core/).

Fixes informed by NIST SRD 46 calibration (7,705 K1 entries):
  1. ε_eff: 10.0 → 4.0  (inner-sphere Coulombic 2.5× stronger)
  2. HSAB amplification cap: 1.5× → 1.3×
  3. Charge factor: flat → anionic/neutral split (α=5.0 / 5.5)
  4. Soft-soft donor saturation (>3 soft donors → 40% attenuation)
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(BASE, "core")


def patch_file(filepath, replacements):
    """CRLF-safe patching: normalizes to LF for matching, preserves original endings."""
    with open(filepath, 'rb') as f:
        raw = f.read()

    # Detect line ending style
    has_crlf = b'\r\n' in raw
    content = raw.decode('utf-8')

    # Normalize to LF for matching
    normalized = content.replace('\r\n', '\n')

    applied = 0
    for old, new in replacements:
        # Normalize the search pattern too
        old_norm = old.replace('\r\n', '\n')
        if old_norm in normalized:
            normalized = normalized.replace(old_norm, new, 1)
            applied += 1
        else:
            print(f"  ⚠ Pattern not found: {old_norm.split(chr(10))[0][:70]}...")

    # Restore original line ending style
    if has_crlf:
        normalized = normalized.replace('\n', '\r\n')

    with open(filepath, 'wb') as f:
        f.write(normalized.encode('utf-8'))

    return applied


pi_path = os.path.join(CORE, "physics_integration.py")
print("NIST Physics Patch — core/physics_integration.py")
print()

total = 0

# ═══════════════════════════════════════════════════════════════════════════
# PATCH 1: ε_eff  10.0 → 4.0
# ═══════════════════════════════════════════════════════════════════════════
print("1. Dielectric ε_eff: 10.0 → 4.0")
total += patch_file(pi_path, [(
    '    # Effective dielectric for inner-sphere: ~10 (between vacuum=1 and water=78)\n'
    '    # k_elec = 1389.4 / ε_eff  (Coulomb\'s law in kJ·pm/mol)\n'
    '    k_elec = 1389.4 / 10.0   # = 138.94',
    #
    '    # Effective dielectric for inner-sphere: ~4 (Warshel 4-8 range;\n'
    '    # calibrated against 47 training + 7705 NIST entries)\n'
    '    k_elec = 1389.4 / 4.0   # = 347.4',
)])
print("  ✅ Coulombic term 2.5× stronger for anionic donors")


# ═══════════════════════════════════════════════════════════════════════════
# PATCH 2: HSAB amplification  1.5× → 1.3×
# ═══════════════════════════════════════════════════════════════════════════
print("2. HSAB cap: 1.5× → 1.3×")
total += patch_file(pi_path, [(
    '            f_hsab = 1.0 + 0.5 * max(softness, ds) * (1.0 - mismatch / 0.15)',
    '            f_hsab = 1.0 + 0.3 * max(softness, ds) * (1.0 - mismatch / 0.15)',
)])
print("  ✅ Soft-soft amplification capped")


# ═══════════════════════════════════════════════════════════════════════════
# PATCH 3: Charge factor anionic/neutral split
# ═══════════════════════════════════════════════════════════════════════════
print("3. Charge factor: anionic α=5.0, neutral α=5.5")
total += patch_file(pi_path, [(
    '        # Charge scaling: z² dependence (electrostatic field strength)\n'
    '        charge_factor = -5.0 * (charge**2 - 1) / max(1, len(donors))\n'
    '        dg_exchange += charge_factor',
    #
    '        # Charge scaling: anionic α=5.0, neutral α=5.5\n'
    '        _AN = {"O_carboxylate","O_phenolate","O_catecholate","O_hydroxamate",\n'
    '               "S_thiolate","S_thiosulfate","S_dithiocarbamate"}\n'
    '        _AN_EL = {"Cl","Br","I"}\n'
    '        _is_an = (donor_subtypes and i < len(donor_subtypes) and donor_subtypes[i] in _AN) or da in _AN_EL\n'
    '        charge_factor = -(5.0 if _is_an else 5.5) * (charge**2 - 1) / max(1, len(donors))\n'
    '        dg_exchange += charge_factor',
)])
print("  ✅ Split charge factor applied")


# ═══════════════════════════════════════════════════════════════════════════
# PATCH 4: Soft-soft saturation
# ═══════════════════════════════════════════════════════════════════════════
print("4. Soft-soft saturation (>3 soft donors → 40% attenuated)")
total += patch_file(pi_path, [(
    '    dg_bind = sum(donor_energies)',
    #
    '    dg_bind = sum(donor_energies)\n'
    '\n'
    '    # Soft-soft saturation: diminishing returns beyond 3 soft donors\n'
    '    if softness > 0.5:\n'
    '        _SOFT_D = {"S", "P", "I"}\n'
    '        _si = [j for j, d in enumerate(donors) if d in _SOFT_D]\n'
    '        if len(_si) > 3:\n'
    '            _excess = sum(donor_energies[j] for j in _si[3:] if j < len(donor_energies))\n'
    '            dg_bind -= _excess * 0.4',
)])
print("  ✅ Excess soft donors attenuated")


# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
if total == 4:
    print(f"  All 4 patches applied successfully")
else:
    print(f"  ⚠ {total}/4 patches applied — check warnings above")
print("=" * 60)
print("""
  Now run:  python run_nist_calibration.py
""")