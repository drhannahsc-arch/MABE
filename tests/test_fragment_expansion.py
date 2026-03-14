"""
tests/test_fragment_expansion.py — Tests for Fragment Library Expansion

Validates 14 new backbones + 16 new arms + 6 new bioisostere groups.
All SMILES must parse in RDKit, all new fragments must assemble with
existing fragments, and bioisostere lookups must return correct groups.
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem

from core.de_novo_generator import (
    BACKBONE_LIBRARY, ARM_LIBRARY, BIOISOSTERE_GROUPS,
    assemble, sa_score, passes_filter, PropertyFilter,
    get_bioisosteres, _ARM_BY_NAME, _ARM_TO_GROUP,
    hsab_compatible,
)


# ═══════════════════════════════════════════════════════════════════════════
# NEW BACKBONE NAMES (must exist in BACKBONE_LIBRARY)
# ═══════════════════════════════════════════════════════════════════════════

NEW_BACKBONES = [
    "12-crown-4", "15-crown-5-open", "18-crown-6-open",
    "cryptand-222-open",
    "terpyridine", "phenanthroline", "triazine", "bipyrazole",
    "tripeptide", "DKP", "cyclic-tripeptide",
    "BPA", "IDA-core", "DTPA-open",
]

NEW_ARMS = [
    "cysteinyl", "seryl-OH", "threonyl-OH", "tyrosyl",
    "phenylboronic", "methylboronic",
    "urea", "guanidinium", "sulfonamide",
    "methoxy", "methoxyethyl", "ethoxyethyl",
    "NHC-imidazolyl", "isocyanide",
    "tetrazolyl", "pyrazolyl",
]

NEW_BIOISOSTERE_GROUPS = [
    "boronic_acid", "anion_donor", "mono_O_ether",
    "mono_C_soft", "peptidic_multi",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. LIBRARY SIZE
# ═══════════════════════════════════════════════════════════════════════════

class TestLibrarySize:

    def test_backbone_count(self):
        assert len(BACKBONE_LIBRARY) == 42

    def test_arm_count(self):
        assert len(ARM_LIBRARY) == 51

    def test_bioisostere_group_count(self):
        assert len(BIOISOSTERE_GROUPS) >= 15


# ═══════════════════════════════════════════════════════════════════════════
# 2. ALL NEW BACKBONES EXIST AND HAVE VALID SMILES
# ═══════════════════════════════════════════════════════════════════════════

class TestNewBackbones:

    def _get_bb(self, name):
        matches = [b for b in BACKBONE_LIBRARY if b.name == name]
        assert len(matches) == 1, f"Backbone '{name}' not found"
        return matches[0]

    @pytest.mark.parametrize("name", NEW_BACKBONES)
    def test_backbone_exists(self, name):
        self._get_bb(name)

    @pytest.mark.parametrize("name", NEW_BACKBONES)
    def test_backbone_valid_smiles(self, name):
        bb = self._get_bb(name)
        test_smi = bb.smiles
        for i in range(1, 5):
            test_smi = test_smi.replace(f"[{i}*]", "[*]")
        mol = Chem.MolFromSmiles(test_smi)
        assert mol is not None, f"Invalid SMILES for {name}: {bb.smiles}"

    @pytest.mark.parametrize("name", NEW_BACKBONES)
    def test_backbone_n_sites_matches(self, name):
        bb = self._get_bb(name)
        test_smi = bb.smiles
        for i in range(1, 5):
            test_smi = test_smi.replace(f"[{i}*]", "[*]")
        mol = Chem.MolFromSmiles(test_smi)
        n_wild = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
        assert n_wild == bb.n_sites, \
            f"{name}: n_sites={bb.n_sites} but SMILES has {n_wild} wildcards"

    def test_crown_ethers_are_macrocyclic(self):
        for name in ["12-crown-4", "15-crown-5-open", "18-crown-6-open"]:
            bb = self._get_bb(name)
            assert bb.category == "macrocyclic"

    def test_terpyridine_is_aromatic(self):
        bb = self._get_bb("terpyridine")
        assert bb.category == "aromatic"

    def test_peptidic_categories(self):
        assert self._get_bb("tripeptide").category == "linear"
        assert self._get_bb("DKP").category == "macrocyclic"
        assert self._get_bb("cyclic-tripeptide").category == "macrocyclic"


# ═══════════════════════════════════════════════════════════════════════════
# 3. ALL NEW ARMS EXIST AND HAVE VALID SMILES
# ═══════════════════════════════════════════════════════════════════════════

class TestNewArms:

    @pytest.mark.parametrize("name", NEW_ARMS)
    def test_arm_exists(self, name):
        assert name in _ARM_BY_NAME, f"Arm '{name}' not found"

    @pytest.mark.parametrize("name", NEW_ARMS)
    def test_arm_valid_smiles(self, name):
        arm = _ARM_BY_NAME[name]
        mol = Chem.MolFromSmiles(arm.smiles)
        assert mol is not None, f"Invalid SMILES for {name}: {arm.smiles}"

    @pytest.mark.parametrize("name", NEW_ARMS)
    def test_arm_single_attachment(self, name):
        arm = _ARM_BY_NAME[name]
        mol = Chem.MolFromSmiles(arm.smiles)
        n_wild = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
        assert n_wild == 1, f"{name}: expected 1 wildcard, got {n_wild}"

    @pytest.mark.parametrize("name", NEW_ARMS)
    def test_arm_has_donor_element(self, name):
        arm = _ARM_BY_NAME[name]
        assert arm.donor_element != "", f"{name}: missing donor_element"

    @pytest.mark.parametrize("name", NEW_ARMS)
    def test_arm_has_hardness(self, name):
        arm = _ARM_BY_NAME[name]
        assert arm.hardness in ("hard", "borderline", "soft"), \
            f"{name}: invalid hardness '{arm.hardness}'"

    def test_boronic_acids_are_hard(self):
        for name in ["phenylboronic", "methylboronic"]:
            assert _ARM_BY_NAME[name].hardness == "hard"

    def test_cysteinyl_is_soft(self):
        assert _ARM_BY_NAME["cysteinyl"].hardness == "soft"

    def test_nhc_is_soft(self):
        assert _ARM_BY_NAME["NHC-imidazolyl"].hardness == "soft"


# ═══════════════════════════════════════════════════════════════════════════
# 4. ASSEMBLY: NEW BACKBONES × EXISTING ARMS
# ═══════════════════════════════════════════════════════════════════════════

class TestAssembly:

    def _get_bb(self, name):
        return [b for b in BACKBONE_LIBRARY if b.name == name][0]

    def test_crown_ether_with_acetic_acid(self):
        bb = self._get_bb("12-crown-4")
        arm = _ARM_BY_NAME["acetic-acid"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None, "Crown ether + acetic acid assembly failed"

    def test_terpyridine_with_aminoethyl(self):
        bb = self._get_bb("terpyridine")
        arm = _ARM_BY_NAME["aminoethyl"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None, "Terpyridine + aminoethyl assembly failed"

    def test_triazine_with_phenol(self):
        bb = self._get_bb("triazine")
        arm = _ARM_BY_NAME["phenol"]
        smi, mol = assemble(bb, [arm, arm, arm])
        assert smi is not None, "Triazine + phenol assembly failed"

    def test_tripeptide_with_thiol(self):
        bb = self._get_bb("tripeptide")
        arm = _ARM_BY_NAME["thiol-methyl"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None

    def test_DKP_with_pyridyl(self):
        bb = self._get_bb("DKP")
        arm = _ARM_BY_NAME["2-pyridyl"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None

    def test_BPA_with_acetic_acid(self):
        bb = self._get_bb("BPA")
        arm = _ARM_BY_NAME["acetic-acid"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None

    def test_DTPA_with_ethanol(self):
        bb = self._get_bb("DTPA-open")
        arm = _ARM_BY_NAME["ethanol"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None


# ═══════════════════════════════════════════════════════════════════════════
# 5. ASSEMBLY: EXISTING BACKBONES × NEW ARMS
# ═══════════════════════════════════════════════════════════════════════════

class TestNewArmAssembly:

    def test_ethylenediamine_with_boronic(self):
        bb = [b for b in BACKBONE_LIBRARY if b.name == "ethylenediamine"][0]
        arm = _ARM_BY_NAME["phenylboronic"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None, "EDA + phenylboronic assembly failed"

    def test_ethylenediamine_with_urea(self):
        bb = [b for b in BACKBONE_LIBRARY if b.name == "ethylenediamine"][0]
        arm = _ARM_BY_NAME["urea"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None

    def test_ethylenediamine_with_methoxy(self):
        bb = [b for b in BACKBONE_LIBRARY if b.name == "ethylenediamine"][0]
        arm = _ARM_BY_NAME["methoxy"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None

    def test_ethylenediamine_with_NHC(self):
        bb = [b for b in BACKBONE_LIBRARY if b.name == "ethylenediamine"][0]
        arm = _ARM_BY_NAME["NHC-imidazolyl"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None

    def test_ethylenediamine_with_tetrazolyl(self):
        bb = [b for b in BACKBONE_LIBRARY if b.name == "ethylenediamine"][0]
        arm = _ARM_BY_NAME["tetrazolyl"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None

    def test_new_bb_with_new_arm(self):
        """Cross-test: new backbone + new arm."""
        bb = [b for b in BACKBONE_LIBRARY if b.name == "phenanthroline"][0]
        arm = _ARM_BY_NAME["guanidinium"]
        smi, mol = assemble(bb, [arm, arm])
        assert smi is not None


# ═══════════════════════════════════════════════════════════════════════════
# 6. BIOISOSTERE GROUPS
# ═══════════════════════════════════════════════════════════════════════════

class TestBioisostereGroups:

    @pytest.mark.parametrize("group_name", NEW_BIOISOSTERE_GROUPS)
    def test_new_group_exists(self, group_name):
        group_names = [g[0] for g in BIOISOSTERE_GROUPS]
        assert group_name in group_names

    def test_all_members_in_arm_library(self):
        for gname, members in BIOISOSTERE_GROUPS:
            for m in members:
                assert m in _ARM_BY_NAME, \
                    f"Bioisostere '{m}' in group '{gname}' not in ARM_LIBRARY"

    def test_tetrazolyl_in_mono_O_acid(self):
        assert _ARM_TO_GROUP.get("tetrazolyl") == "mono_O_acid"

    def test_pyrazolyl_in_mono_N_aromatic(self):
        assert _ARM_TO_GROUP.get("pyrazolyl") == "mono_N_aromatic"

    def test_seryl_in_mono_O_hydroxyl(self):
        assert _ARM_TO_GROUP.get("seryl-OH") == "mono_O_hydroxyl"

    def test_phenylboronic_in_boronic_acid(self):
        assert _ARM_TO_GROUP.get("phenylboronic") == "boronic_acid"

    def test_urea_in_anion_donor(self):
        assert _ARM_TO_GROUP.get("urea") == "anion_donor"

    def test_methoxy_in_mono_O_ether(self):
        assert _ARM_TO_GROUP.get("methoxy") == "mono_O_ether"

    def test_NHC_in_mono_C_soft(self):
        assert _ARM_TO_GROUP.get("NHC-imidazolyl") == "mono_C_soft"

    def test_bioisostere_lookup_works(self):
        """get_bioisosteres returns correct replacements."""
        bios = get_bioisosteres("phenylboronic")
        names = [b.name for b in bios]
        assert "methylboronic" in names
        assert "phenylboronic" not in names  # excludes self

    def test_tetrazolyl_bioisosteres_include_carboxylates(self):
        bios = get_bioisosteres("tetrazolyl")
        names = [b.name for b in bios]
        assert "acetic-acid" in names

    def test_seryl_bioisosteres_include_ethanol(self):
        bios = get_bioisosteres("seryl-OH")
        names = [b.name for b in bios]
        assert "ethanol" in names


# ═══════════════════════════════════════════════════════════════════════════
# 7. HSAB COMPATIBILITY FOR NEW ARMS
# ═══════════════════════════════════════════════════════════════════════════

class TestHSABNewArms:

    def test_boronic_compatible_with_hard_metal(self):
        arm = _ARM_BY_NAME["phenylboronic"]
        assert hsab_compatible("Fe3+", arm) is True

    def test_boronic_incompatible_with_soft_metal(self):
        arm = _ARM_BY_NAME["phenylboronic"]
        assert hsab_compatible("Au+", arm) is False

    def test_NHC_compatible_with_soft_metal(self):
        arm = _ARM_BY_NAME["NHC-imidazolyl"]
        assert hsab_compatible("Pd2+", arm) is True

    def test_NHC_incompatible_with_hard_metal(self):
        arm = _ARM_BY_NAME["NHC-imidazolyl"]
        assert hsab_compatible("Fe3+", arm) is False

    def test_urea_compatible_with_hard_metal(self):
        arm = _ARM_BY_NAME["urea"]
        assert hsab_compatible("Ca2+", arm) is True

    def test_cysteinyl_compatible_with_soft_metal(self):
        arm = _ARM_BY_NAME["cysteinyl"]
        assert hsab_compatible("Hg2+", arm) is True


# ═══════════════════════════════════════════════════════════════════════════
# 8. NO DUPLICATE NAMES
# ═══════════════════════════════════════════════════════════════════════════

class TestNoDuplicates:

    def test_no_duplicate_backbone_names(self):
        names = [b.name for b in BACKBONE_LIBRARY]
        assert len(names) == len(set(names)), \
            f"Duplicate backbones: {[n for n in names if names.count(n) > 1]}"

    def test_no_duplicate_arm_names(self):
        names = [a.name for a in ARM_LIBRARY]
        assert len(names) == len(set(names)), \
            f"Duplicate arms: {[n for n in names if names.count(n) > 1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
