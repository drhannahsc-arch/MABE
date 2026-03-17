"""
tests/test_pocket_analyzer.py — Tests for PDB pocket analyzer
"""

import pytest
import sys
import os

_mabe_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _mabe_root not in sys.path:
    sys.path.insert(0, _mabe_root)

from knowledge.pocket_analyzer import (
    PocketAnalyzer, PocketDescriptors, parse_pdb, populate_uc_from_pocket,
)

THROMBIN_PDB = """ATOM      1  N   ASP A 189       8.000   9.000  10.000  1.00 15.00           N
ATOM      2  CA  ASP A 189       8.500   9.500  10.000  1.00 15.00           C
ATOM      3  C   ASP A 189       9.000  10.000  10.000  1.00 15.00           C
ATOM      4  O   ASP A 189       9.500  10.500  10.000  1.00 15.00           O
ATOM      5  OD1 ASP A 189       7.500   8.500  10.000  1.00 15.00           O
ATOM      6  OD2 ASP A 189       7.000   9.000  10.500  1.00 15.00           O
ATOM     10  N   TRP A 215      12.000  10.000  10.000  1.00 12.00           N
ATOM     11  CA  TRP A 215      12.500  10.500  10.000  1.00 12.00           C
ATOM     12  C   TRP A 215      13.000  11.000  10.000  1.00 12.00           C
ATOM     13  O   TRP A 215      13.500  11.500  10.000  1.00 12.00           O
ATOM     14  NE1 TRP A 215      11.500   9.500  10.000  1.00 12.00           N
ATOM     20  N   SER A 214      11.000  11.000   9.000  1.00 14.00           N
ATOM     21  CA  SER A 214      11.500  11.500   9.000  1.00 14.00           C
ATOM     22  C   SER A 214      12.000  12.000   9.000  1.00 14.00           C
ATOM     23  O   SER A 214      12.500  12.500   9.000  1.00 14.00           O
ATOM     24  OG  SER A 214      10.500  11.000   9.500  1.00 14.00           O
ATOM     30  N   GLY A 216      10.000  12.000  11.000  1.00 18.00           N
ATOM     31  CA  GLY A 216      10.500  12.500  11.000  1.00 18.00           C
ATOM     32  C   GLY A 216      11.000  13.000  11.000  1.00 18.00           C
ATOM     33  O   GLY A 216      11.500  13.500  11.000  1.00 18.00           O
HETATM   50  C1  BEN A 500      10.000  10.000  10.000  1.00 10.00           C
HETATM   51  C2  BEN A 500      10.500  10.500  10.000  1.00 10.00           C
HETATM   52  N1  BEN A 500       9.500   9.500  10.000  1.00 10.00           N
HETATM   60  O   HOH A 601       9.000  10.000  11.000  1.00 25.00           O
HETATM   61  O   HOH A 602      11.000   9.000  10.000  1.00 35.00           O
HETATM   62  O   HOH A 603      10.000  11.000   8.500  1.00 20.00           O
END"""

CAII_PDB = """ATOM      1  N   HIS A  94       8.500  10.000  10.000  1.00 12.00           N
ATOM      2  CA  HIS A  94       8.000  10.500  10.000  1.00 12.00           C
ATOM      3  C   HIS A  94       7.500  11.000  10.000  1.00 12.00           C
ATOM      4  O   HIS A  94       7.000  11.500  10.000  1.00 12.00           O
ATOM      5  ND1 HIS A  94       9.000   9.500  10.500  1.00 12.00           N
ATOM      6  NE2 HIS A  94       9.500  10.500  10.500  1.00 12.00           N
ATOM     10  N   HIS A  96       9.000  11.000   9.000  1.00 11.00           N
ATOM     11  CA  HIS A  96       8.500  11.500   9.000  1.00 11.00           C
ATOM     12  C   HIS A  96       8.000  12.000   9.000  1.00 11.00           C
ATOM     13  O   HIS A  96       7.500  12.500   9.000  1.00 11.00           O
ATOM     14  ND1 HIS A  96       9.500  10.500   9.500  1.00 11.00           N
ATOM     20  N   HIS A 119      11.000   9.000  10.500  1.00 10.00           N
ATOM     21  CA  HIS A 119      11.500   8.500  10.500  1.00 10.00           C
ATOM     22  C   HIS A 119      12.000   8.000  10.500  1.00 10.00           C
ATOM     23  O   HIS A 119      12.500   7.500  10.500  1.00 10.00           O
ATOM     24  ND1 HIS A 119      10.500   9.500  10.000  1.00 10.00           N
ATOM     30  N   GLU A 106      12.000  10.000  11.000  1.00 14.00           N
ATOM     31  CA  GLU A 106      12.500  10.500  11.000  1.00 14.00           C
ATOM     32  C   GLU A 106      13.000  11.000  11.000  1.00 14.00           C
ATOM     33  O   GLU A 106      13.500  11.500  11.000  1.00 14.00           O
ATOM     34  OE1 GLU A 106      11.500  10.000  11.500  1.00 14.00           O
ATOM     35  OE2 GLU A 106      11.000  10.500  11.500  1.00 14.00           O
ATOM     40  N   THR A 199      11.000  11.000   8.000  1.00 15.00           N
ATOM     41  CA  THR A 199      11.500  11.500   8.000  1.00 15.00           C
ATOM     42  C   THR A 199      12.000  12.000   8.000  1.00 15.00           C
ATOM     43  O   THR A 199      12.500  12.500   8.000  1.00 15.00           O
ATOM     44  OG1 THR A 199      10.500  11.000   8.500  1.00 15.00           O
HETATM   50  ZN  ZN  A 301      10.000  10.000  10.000  1.00  8.00          ZN
HETATM   60  S1  SLF A 400       9.000  10.000   9.000  1.00 10.00           S
HETATM   61  N1  SLF A 400       8.500   9.500   9.000  1.00 10.00           N
HETATM   62  O1  SLF A 400       8.000  10.500   9.000  1.00 10.00           O
HETATM   63  C1  SLF A 400       9.500   9.000   8.500  1.00 10.00           C
HETATM   70  O   HOH A 501      10.000   8.500   9.000  1.00 30.00           O
HETATM   71  O   HOH A 502      11.000  11.000   9.000  1.00 18.00           O
END"""


class TestPDBParsing:
    def test_parse_atom_count(self):
        atoms = parse_pdb(THROMBIN_PDB)
        assert len(atoms) > 10

    def test_parse_hetatm(self):
        atoms = parse_pdb(THROMBIN_PDB)
        hetatm = [a for a in atoms if a.is_hetatm]
        assert len(hetatm) >= 3

    def test_parse_coordinates(self):
        atoms = parse_pdb(THROMBIN_PDB)
        ligand = [a for a in atoms if a.resname == "BEN"]
        assert len(ligand) >= 2
        assert abs(ligand[0].x - 10.0) < 0.1

    def test_parse_waters(self):
        atoms = parse_pdb(THROMBIN_PDB)
        waters = [a for a in atoms if a.resname == "HOH"]
        assert len(waters) == 3

    def test_parse_metals(self):
        atoms = parse_pdb(CAII_PDB)
        metals = [a for a in atoms if a.element.upper() == "ZN"]
        assert len(metals) == 1


class TestThrombinPocket:
    @pytest.fixture
    def pocket(self):
        return PocketAnalyzer.from_text(THROMBIN_PDB).analyze_pocket(ligand_resname="BEN", cutoff_A=6.0)

    def test_finds_pocket_residues(self, pocket):
        assert pocket.n_pocket_residues >= 3

    def test_detects_asp(self, pocket):
        assert pocket.n_negative_residues >= 1

    def test_detects_trp(self, pocket):
        assert pocket.n_trp >= 1

    def test_detects_waters(self, pocket):
        assert pocket.n_waters >= 1

    def test_backbone_hbonds(self, pocket):
        assert pocket.n_hbd_backbone >= 1
        assert pocket.n_hba_backbone >= 1

    def test_sidechain_donors(self, pocket):
        assert pocket.n_hbd_sidechain >= 1

    def test_negative_charge(self, pocket):
        assert pocket.net_charge <= 0

    def test_pocket_desolv_computed(self, pocket):
        assert pocket.pocket_desolv_kJ != 0.0

    def test_preorg_computed(self, pocket):
        assert pocket.preorganization_kJ < 0


class TestCAIIPocket:
    @pytest.fixture
    def pocket(self):
        return PocketAnalyzer.from_text(CAII_PDB).analyze_pocket(ligand_resname="SLF", cutoff_A=6.0)

    def test_finds_zinc(self, pocket):
        assert len(pocket.metal_ions) >= 1
        assert pocket.metal_ions[0][0] == "ZN"
        assert pocket.metal_ions[0][1] == 2

    def test_zinc_charge(self, pocket):
        assert pocket.metal_charge_total >= 2

    def test_net_charge_positive(self, pocket):
        assert pocket.net_charge >= 1

    def test_finds_his(self, pocket):
        assert pocket.n_his >= 2

    def test_finds_glu(self, pocket):
        assert pocket.n_negative_residues >= 1

    def test_waters_present(self, pocket):
        assert pocket.n_waters >= 1

    def test_water_displacement_energy(self, pocket):
        assert isinstance(pocket.water_displacement_kJ, float)

    def test_sidechain_acceptors(self, pocket):
        assert pocket.n_hba_sidechain >= 3


class TestUCPopulation:
    def test_populate_sets_host_charge(self):
        from core.universal_schema import UniversalComplex
        pocket = PocketAnalyzer.from_text(CAII_PDB).analyze_pocket(ligand_resname="SLF")
        uc = UniversalComplex(name="test", binding_mode="protein_ligand_physics", guest_smiles="NS(=O)(=O)c1ccccc1")
        populate_uc_from_pocket(uc, pocket)
        assert uc.host_charge >= 1

    def test_populate_sets_metal(self):
        from core.universal_schema import UniversalComplex
        pocket = PocketAnalyzer.from_text(CAII_PDB).analyze_pocket(ligand_resname="SLF")
        uc = UniversalComplex(name="test", binding_mode="protein_ligand_physics", guest_smiles="NS(=O)(=O)c1ccccc1")
        populate_uc_from_pocket(uc, pocket)
        assert uc.metal_formula == "ZN"

    def test_populate_sets_hbond_counts(self):
        from core.universal_schema import UniversalComplex
        pocket = PocketAnalyzer.from_text(THROMBIN_PDB).analyze_pocket(ligand_resname="BEN")
        uc = UniversalComplex(name="test", binding_mode="protein_ligand_physics", guest_smiles="NC(=N)c1ccccc1")
        populate_uc_from_pocket(uc, pocket)
        assert uc.n_hbond_donors_host > 0
        assert uc.n_hbond_acceptors_host > 0


class TestEnergyEstimates:
    def test_water_displacement_range(self):
        pocket = PocketAnalyzer.from_text(THROMBIN_PDB).analyze_pocket(ligand_resname="BEN")
        assert -30 < pocket.water_displacement_kJ < 30

    def test_pocket_desolv_small(self):
        pocket = PocketAnalyzer.from_text(THROMBIN_PDB).analyze_pocket(ligand_resname="BEN")
        assert -30 < pocket.pocket_desolv_kJ < 20

    def test_preorg_always_favorable(self):
        for pdb, lig in [(THROMBIN_PDB, "BEN"), (CAII_PDB, "SLF")]:
            pocket = PocketAnalyzer.from_text(pdb).analyze_pocket(ligand_resname=lig)
            assert pocket.preorganization_kJ < 0, f"{lig}: preorg={pocket.preorganization_kJ}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
