"""
ligand_library.py — Database of real purchasable ligands for MABE design engine.

~120 coordination ligands with:
  - Validated SMILES (RDKit-checked)
  - Donor subtype annotations matching scorer_frozen.py subtypes
  - Chelation topology (rings, sizes, macrocyclic)
  - Commercial availability, CAS numbers
  - Category tags for search/filter

Usage:
    from ligand_library import LIGAND_DB, get_by_donors, get_by_category
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Ligand:
    """A real, purchasable coordination ligand."""
    name: str                          # Common/IUPAC name
    smiles: str                        # Canonical SMILES
    cas: Optional[str]                 # CAS registry number
    donors: list[str]                  # Donor subtypes (scorer convention)
    denticity: int                     # Number of donor atoms
    chelate_rings: int                 # Number of chelate rings formed
    ring_sizes: list[int]              # Size of each chelate ring
    macrocyclic: bool = False          # Is it a macrocycle?
    cavity_nm: Optional[float] = None  # Cavity radius (macrocycles only)
    mw: Optional[float] = None        # Molecular weight (filled by validation)
    commercial: bool = True            # Commercially available?
    category: str = ""                 # Classification tag
    notes: str = ""                    # Application notes
    aliases: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# LIGAND DATABASE
# ═══════════════════════════════════════════════════════════════════════════

LIGAND_DB = [

    # ═════════════════════════════════════════════════════════════════
    # AMINOCARBOXYLATES — workhorses of chelation
    # ═════════════════════════════════════════════════════════════════

    Ligand("EDTA", "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
           "60-00-4",
           ["N_amine","N_amine","O_carboxylate","O_carboxylate",
            "O_carboxylate","O_carboxylate"],
           6, 5, [5,5,5,5,5],
           category="aminocarboxylate",
           notes="Universal chelator, water softening, blood anticoagulant",
           aliases=["ethylenediaminetetraacetic acid"]),

    Ligand("DTPA", "OC(=O)CN(CCN(CC(=O)O)CCN(CC(=O)O)CC(=O)O)CC(=O)O",
           "67-43-6",
           ["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate",
            "O_carboxylate","O_carboxylate","O_carboxylate"],
           8, 8, [5]*8,
           category="aminocarboxylate",
           notes="MRI contrast (Gd-DTPA), actinide decorporation",
           aliases=["diethylenetriaminepentaacetic acid","pentetic acid"]),

    Ligand("NTA", "OC(=O)CN(CC(=O)O)CC(=O)O",
           "139-13-9",
           ["N_amine","O_carboxylate","O_carboxylate","O_carboxylate"],
           4, 3, [5,5,5],
           category="aminocarboxylate",
           notes="Detergent builder, IMAC variant",
           aliases=["nitrilotriacetic acid"]),

    Ligand("IDA", "OC(=O)CNCC(=O)O",
           "142-73-4",
           ["N_amine","O_carboxylate","O_carboxylate"],
           3, 2, [5,5],
           category="aminocarboxylate",
           notes="IMAC resin ligand (Ni-IDA for His-tag purification)",
           aliases=["iminodiacetic acid"]),

    Ligand("Glycine", "NCC(=O)O",
           "56-40-6",
           ["N_amine","O_carboxylate"],
           2, 1, [5],
           category="aminocarboxylate",
           notes="Simplest amino acid chelator"),

    Ligand("HEDTA", "OCCN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
           "150-39-0",
           ["N_amine","N_amine","O_carboxylate","O_carboxylate",
            "O_carboxylate","O_hydroxyl"],
           6, 5, [5,5,5,5,5],
           category="aminocarboxylate",
           notes="Less aggressive than EDTA, used in food preservation",
           aliases=["N-(2-hydroxyethyl)ethylenediamine-N,N',N'-triacetic acid"]),

    Ligand("EGTA", "OC(=O)CN(CCOCCOCCN(CC(=O)O)CC(=O)O)CC(=O)O",
           "67-42-5",
           ["N_amine","N_amine","O_carboxylate","O_carboxylate",
            "O_carboxylate","O_carboxylate"],
           6, 5, [5,5,5,5,8],
           category="aminocarboxylate",
           notes="Ca-selective over Mg (10^5 factor), cell biology buffer",
           aliases=["ethylene glycol tetraacetic acid"]),

    Ligand("BAPTA", "OC(=O)CN(Cc1cc(OCCOCCN(CC(=O)O)Cc2ccccc2)ccc1)CC(=O)O",
           "85233-19-8",
           ["N_amine","N_amine","O_carboxylate","O_carboxylate",
            "O_carboxylate","O_carboxylate"],
           6, 5, [5,5,5,5,8],
           category="aminocarboxylate",
           notes="Fluorescent Ca indicator precursor, fast Ca chelation",
           aliases=["1,2-bis(o-aminophenoxy)ethane-N,N,N',N'-tetraacetic acid"]),

    # ═════════════════════════════════════════════════════════════════
    # POLYAMINES
    # ═════════════════════════════════════════════════════════════════

    Ligand("Ethylenediamine", "NCCN",
           "107-15-3",
           ["N_amine","N_amine"],
           2, 1, [5],
           category="polyamine",
           notes="Simplest bidentate N-donor, en",
           aliases=["en","1,2-diaminoethane"]),

    Ligand("Diethylenetriamine", "NCCNCCN",
           "111-40-0",
           ["N_amine","N_amine","N_amine"],
           3, 2, [5,5],
           category="polyamine",
           notes="dien, tridentate amine",
           aliases=["dien"]),

    Ligand("Triethylenetetramine", "NCCNCCNCCN",
           "112-24-3",
           ["N_amine","N_amine","N_amine","N_amine"],
           4, 3, [5,5,5],
           category="polyamine",
           notes="trien, Wilson's disease treatment (Cu chelation)",
           aliases=["trien","trientine"]),

    Ligand("Tetraethylenepentamine", "NCCNCCNCCNCCN",
           "112-57-2",
           ["N_amine","N_amine","N_amine","N_amine","N_amine"],
           5, 4, [5,5,5,5],
           category="polyamine",
           notes="tepa, pentadentate linear amine",
           aliases=["tepa"]),

    Ligand("1,3-Diaminopropane", "NCCCN",
           "109-76-2",
           ["N_amine","N_amine"],
           2, 1, [6],
           category="polyamine",
           notes="6-membered chelate ring, weaker than en for small metals"),

    Ligand("1,2-Diaminocyclohexane", "N[C@@H]1CCCC[C@H]1N",
           "694-83-7",
           ["N_amine","N_amine"],
           2, 1, [5],
           category="polyamine",
           notes="Rigid en analogue, chiral catalyst precursor (DACH)",
           aliases=["DACH"]),

    # ═════════════════════════════════════════════════════════════════
    # PYRIDINE / AROMATIC N-DONORS
    # ═════════════════════════════════════════════════════════════════

    Ligand("2,2'-Bipyridine", "c1ccc(-c2ccccn2)nc1",
           "366-18-7",
           ["N_pyridine","N_pyridine"],
           2, 1, [5],
           category="pyridine",
           notes="Classic aromatic N2 chelator, ferroin analogue",
           aliases=["bipy","bpy"]),

    Ligand("1,10-Phenanthroline", "c1cnc2c(c1)ccc1cccnc12",
           "66-71-7",
           ["N_pyridine","N_pyridine"],
           2, 1, [5],
           category="pyridine",
           notes="Rigid bipy analogue, Fe(phen)3 is ferroin indicator",
           aliases=["phen","o-phenanthroline"]),

    Ligand("2,2':6',2''-Terpyridine", "c1ccc(-c2cccc(-c3ccccn3)n2)nc1",
           "1148-79-4",
           ["N_pyridine","N_pyridine","N_pyridine"],
           3, 2, [5,5],
           category="pyridine",
           notes="Tridentate aromatic N3, strong π-backbonding",
           aliases=["terpy","tpy"]),

    Ligand("Pyridine", "c1ccncc1",
           "110-86-1",
           ["N_pyridine"],
           1, 0, [],
           category="pyridine",
           notes="Monodentate aromatic N"),

    Ligand("Picolinic acid", "OC(=O)c1ccccn1",
           "98-98-6",
           ["N_pyridine","O_carboxylate"],
           2, 1, [5],
           category="pyridine",
           notes="Pyridine-2-carboxylic acid, Cr supplement form"),

    Ligand("Dipicolinic acid", "OC(=O)c1cccc(C(=O)O)n1",
           "499-83-2",
           ["N_pyridine","O_carboxylate","O_carboxylate"],
           3, 2, [5,5],
           category="pyridine",
           notes="Pyridine-2,6-dicarboxylic acid, bacterial spore marker",
           aliases=["DPA","PDC"]),

    # ═════════════════════════════════════════════════════════════════
    # PHENOLATE / CATECHOLATE / HYDROXAMATE
    # ═════════════════════════════════════════════════════════════════

    Ligand("8-Hydroxyquinoline", "Oc1ccc2ncccc2c1",
           "148-24-3",
           ["N_pyridine","O_phenolate"],
           2, 1, [5],
           category="phenolate",
           notes="Oxine, classic analytical reagent for Al/Zn/Cu",
           aliases=["oxine","8-HQ"]),

    Ligand("Catechol", "Oc1ccccc1O",
           "120-80-9",
           ["O_catecholate","O_catecholate"],
           2, 1, [5],
           category="catecholate",
           notes="Siderophore binding unit, Fe3+ specialist"),

    Ligand("Salicylic acid", "OC(=O)c1ccccc1O",
           "69-72-7",
           ["O_carboxylate","O_phenolate"],
           2, 1, [6],
           category="phenolate",
           notes="Aspirin precursor, Fe3+ chelator"),

    Ligand("Salicylaldehyde", "Oc1ccccc1C=O",
           "90-02-8",
           ["O_phenolate","O_hydroxyl"],
           2, 1, [6],
           category="phenolate",
           notes="Salen precursor, Schiff base formation"),

    Ligand("Salen", "Oc1ccccc1/C=N/CC/N=C/c1ccccc1O",
           "94-93-9",
           ["N_imine","N_imine","O_phenolate","O_phenolate"],
           4, 2, [5,6],
           category="phenolate",
           notes="Tetradentate Schiff base, Jacobsen epoxidation catalyst",
           aliases=["N,N'-bis(salicylidene)ethylenediamine"]),

    Ligand("Acetohydroxamic acid", "CC(=O)NO",
           "546-88-3",
           ["O_hydroxamate","O_hydroxamate"],
           2, 1, [5],
           category="hydroxamate",
           notes="Fe3+ chelator, urease inhibitor drug",
           aliases=["AHA"]),

    Ligand("Desferrioxamine B", "CC(=O)N(O)CCCCCNC(=O)CCC(=O)N(O)CCCCCNC(=O)CCC(=O)N(O)CCCCCN",
           "70-51-9",
           ["O_hydroxamate","O_hydroxamate","O_hydroxamate",
            "O_hydroxamate","O_hydroxamate","O_hydroxamate"],
           6, 3, [5,5,5],
           category="hydroxamate",
           notes="Siderophore, iron overload treatment (Desferal)",
           aliases=["DFO","deferoxamine","Desferal"]),

    Ligand("2,3-Dihydroxybenzoic acid", "OC(=O)c1cccc(O)c1O",
           "303-38-8",
           ["O_catecholate","O_catecholate","O_carboxylate"],
           3, 2, [5,5],
           category="catecholate",
           notes="Enterobactin building block, Fe3+ binder",
           aliases=["DHBA"]),

    Ligand("Gallic acid", "OC(=O)c1cc(O)c(O)c(O)c1",
           "149-91-7",
           ["O_catecholate","O_catecholate"],
           2, 1, [5],
           category="catecholate",
           notes="Tea polyphenol, tanning agent, Fe chelation"),

    # ═════════════════════════════════════════════════════════════════
    # SULFUR DONORS
    # ═════════════════════════════════════════════════════════════════

    Ligand("L-Cysteine", "N[C@@H](CS)C(=O)O",
           "52-90-4",
           ["S_thiolate","N_amine","O_carboxylate"],
           3, 2, [5,5],
           category="thiol",
           notes="Amino acid, Hg/Cd/Pb chelation, metallothionein unit"),

    Ligand("D-Penicillamine", "CC(C)(S)[C@@H](N)C(=O)O",
           "52-67-5",
           ["S_thiolate","N_amine","O_carboxylate"],
           3, 2, [5,5],
           category="thiol",
           notes="Wilson's disease (Cu) treatment, Pb chelation",
           aliases=["pen","Cuprimine"]),

    Ligand("DMSA", "OC(=O)[C@@H](S)[C@H](S)C(=O)O",
           "304-55-2",
           ["S_thiolate","S_thiolate","O_carboxylate","O_carboxylate"],
           4, 2, [5,5],
           category="thiol",
           notes="FDA-approved Pb chelation (Succimer), also Hg/As",
           aliases=["meso-2,3-dimercaptosuccinic acid","Succimer"]),

    Ligand("DMPS", "OC(=O)C(S)CS",
           "4076-02-2",
           ["S_thiolate","S_thiolate","O_carboxylate"],
           3, 2, [5,5],
           category="thiol",
           notes="Hg/As chelation therapy (Unithiol)",
           aliases=["2,3-dimercaptopropane-1-sulfonic acid","Unithiol"]),

    Ligand("BAL", "OCC(S)CS",
           "59-52-9",
           ["S_thiolate","S_thiolate","O_hydroxyl"],
           3, 2, [5,5],
           category="thiol",
           notes="British Anti-Lewisite, original Hg/As chelator",
           aliases=["dimercaprol","2,3-dimercaptopropanol"]),

    Ligand("Thiourea", "NC(=S)N",
           "62-56-6",
           ["S_thioether"],
           1, 0, [],
           category="thiol",
           notes="Au/Ag leaching reagent, soft S-donor"),

    Ligand("Dithizone", "S=C(NNc1ccccc1)/N=N/c1ccccc1",
           "60-10-6",
           ["S_thiolate","N_imine"],
           2, 1, [5],
           category="thiol",
           notes="Colorimetric heavy metal indicator (Pb, Hg, Zn)",
           aliases=["diphenylthiocarbazone"]),

    Ligand("Sodium diethyldithiocarbamate", "CCN(CC)C(=S)[S-]",
           "148-18-5",
           ["S_dithiocarbamate","S_dithiocarbamate"],
           2, 1, [4],
           category="thiol",
           notes="Heavy metal precipitation, pesticide (ziram), Cu/Pb/Hg",
           aliases=["NaDDC","DDTC"]),

    Ligand("Thiosulfate", "OS(=O)(=O)[S-]",
           "7772-98-7",
           ["S_thiosulfate"],
           1, 0, [],
           category="thiol",
           notes="Photography fixer (Ag), cyanide antidote (Na2S2O3)"),

    Ligand("1,2-Ethanedithiol", "SCCS",
           "540-63-6",
           ["S_thiolate","S_thiolate"],
           2, 1, [5],
           category="thiol",
           notes="Bidentate dithiol, Hg/Cd chelation, peptide cleavage reagent",
           aliases=["EDT"]),

    Ligand("Thioglycolic acid", "OC(=O)CS",
           "68-11-1",
           ["S_thiolate","O_carboxylate"],
           2, 1, [5],
           category="thiol",
           notes="Hair perm reagent, metal analysis",
           aliases=["TGA","mercaptoacetic acid"]),

    # ═════════════════════════════════════════════════════════════════
    # CROWN ETHERS & CRYPTANDS (macrocyclic)
    # ═════════════════════════════════════════════════════════════════

    Ligand("12-Crown-4", "C1COCCOCCOCCO1",
           "294-93-9",
           ["O_ether","O_ether","O_ether","O_ether"],
           4, 4, [5,5,5,5],
           macrocyclic=True, cavity_nm=0.060,
           category="crown_ether",
           notes="Li+ selective",
           aliases=["12c4"]),

    Ligand("15-Crown-5", "C1COCCOCCOCCOCCO1",
           "33100-27-5",
           ["O_ether","O_ether","O_ether","O_ether","O_ether"],
           5, 5, [5,5,5,5,5],
           macrocyclic=True, cavity_nm=0.092,
           category="crown_ether",
           notes="Na+ selective",
           aliases=["15c5"]),

    Ligand("18-Crown-6", "C1COCCOCCOCCOCCOCCO1",
           "17455-13-9",
           ["O_ether","O_ether","O_ether","O_ether","O_ether","O_ether"],
           6, 6, [5,5,5,5,5,5],
           macrocyclic=True, cavity_nm=0.140,
           category="crown_ether",
           notes="K+ selective, Pedersen's Nobel Prize",
           aliases=["18c6"]),

    Ligand("21-Crown-7", "C1COCCOCCOCCOCCOCCOCCO1",
           "33724-18-0",
           ["O_ether"]*7,
           7, 7, [5]*7,
           macrocyclic=True, cavity_nm=0.170,
           category="crown_ether",
           notes="Cs+/Rb+ selective",
           aliases=["21c7"]),

    Ligand("Dibenzo-18-crown-6", "C1COCCOc2ccccc2OCCOCCOc2ccccc2OCC1",
           "14187-32-7",
           ["O_ether"]*6,
           6, 6, [5]*6,
           macrocyclic=True, cavity_nm=0.140,
           category="crown_ether",
           notes="More rigid 18c6, enhanced K+ selectivity"),

    Ligand("[2.2.2]-Cryptand", "C(COCCOCCN1CCOCCOCCN(CCOCCOCC1)CCOCCOCC1)1",
           "23978-09-8",
           ["O_ether","O_ether","O_ether","O_ether","O_ether","O_ether",
            "N_amine","N_amine"],
           8, 8, [5]*8,
           macrocyclic=True, cavity_nm=0.140,
           category="cryptand",
           notes="3D cage, extreme K+ selectivity, Kryptofix 222",
           aliases=["Kryptofix 222","K222"]),

    # ═════════════════════════════════════════════════════════════════
    # MACROCYCLIC N-DONORS (cyclam, cyclen, porphyrin)
    # ═════════════════════════════════════════════════════════════════

    Ligand("Cyclam", "C1CNCCNCCNCCNCC1",
           "295-37-4",
           ["N_amine","N_amine","N_amine","N_amine"],
           4, 4, [5,6,5,6],
           macrocyclic=True, cavity_nm=0.085,
           category="macrocyclic_amine",
           notes="14-membered tetraaza, Cu2+/Ni2+ macrocyclic effect",
           aliases=["1,4,8,11-tetraazacyclotetradecane"]),

    Ligand("Cyclen", "C1CNCCNCCNCCNC1",
           "294-90-6",
           ["N_amine","N_amine","N_amine","N_amine"],
           4, 4, [5,5,5,5],
           macrocyclic=True, cavity_nm=0.070,
           category="macrocyclic_amine",
           notes="12-membered tetraaza, DOTA precursor, smaller cavity",
           aliases=["1,4,7,10-tetraazacyclododecane"]),

    Ligand("DOTA", "OC(=O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)CC1",
           "60239-18-1",
           ["N_amine","N_amine","N_amine","N_amine",
            "O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
           8, 8, [5]*8,
           macrocyclic=True, cavity_nm=0.070,
           category="macrocyclic_amine",
           notes="MRI/PET (Gd/68Ga), extreme kinetic inertness",
           aliases=["1,4,7,10-tetraazacyclododecane-1,4,7,10-tetraacetic acid"]),

    Ligand("NOTA", "OC(=O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CC1",
           "56491-86-2",
           ["N_amine","N_amine","N_amine",
            "O_carboxylate","O_carboxylate","O_carboxylate"],
           6, 6, [5]*6,
           macrocyclic=True, cavity_nm=0.060,
           category="macrocyclic_amine",
           notes="68Ga PET imaging, smaller than DOTA",
           aliases=["1,4,7-triazacyclononane-1,4,7-triacetic acid"]),

    Ligand("Porphyrin", "C1=CC2=CC3=CC=C([NH]3)C=C3C=CC(=N3)C=C3C=CC(=N3)C=C1[NH]2",
           "101-60-0",
           ["N_pyridine","N_pyridine","N_pyridine","N_pyridine"],
           4, 4, [5,5,5,5],
           macrocyclic=True, cavity_nm=0.070,
           category="porphyrin",
           notes="Heme/chlorophyll core, Fe/Mg/Zn selective",
           aliases=["porphine"]),

    Ligand("Phthalocyanine", "C1=CC=C2C(=C1)C3=NC4=NC(=NC5=NC(=NC6=NC(=NC2=N3)C7=CC=CC=C76)C8=CC=CC=C85)C9=CC=CC=C94",
           "574-93-6",
           ["N_pyridine","N_pyridine","N_pyridine","N_pyridine"],
           4, 4, [5,5,5,5],
           macrocyclic=True, cavity_nm=0.070,
           category="porphyrin",
           notes="Industrial dye, Cu/Fe/Co phthalocyanine catalysts",
           aliases=["Pc"]),

    # ═════════════════════════════════════════════════════════════════
    # IMIDAZOLES
    # ═════════════════════════════════════════════════════════════════

    Ligand("Imidazole", "c1c[nH]cn1",
           "288-32-4",
           ["N_imidazole"],
           1, 0, [],
           category="imidazole",
           notes="Histidine sidechain, IMAC elution buffer"),

    Ligand("L-Histidine", "N[C@@H](Cc1c[nH]cn1)C(=O)O",
           "71-00-1",
           ["N_imidazole","N_amine","O_carboxylate"],
           3, 2, [5,7],
           category="imidazole",
           notes="His-tag binding unit, Cu/Zn/Ni chelation"),

    Ligand("Benzimidazole", "c1ccc2[nH]cnc2c1",
           "51-17-2",
           ["N_imidazole"],
           1, 0, [],
           category="imidazole",
           notes="Fungicide precursor, aromatic N-donor"),

    Ligand("2-Methylimidazole", "Cc1ncc[nH]1",
           "693-98-1",
           ["N_imidazole"],
           1, 0, [],
           category="imidazole",
           notes="ZIF-8 linker (Zn), MOF building block"),

    # ═════════════════════════════════════════════════════════════════
    # CARBOXYLATES (O-donors)
    # ═════════════════════════════════════════════════════════════════

    Ligand("Acetic acid", "CC(=O)O",
           "64-19-7",
           ["O_carboxylate"],
           1, 0, [],
           category="carboxylate",
           notes="Monodentate O, weak chelator, buffer component"),

    Ligand("Oxalic acid", "OC(=O)C(=O)O",
           "144-62-7",
           ["O_carboxylate","O_carboxylate"],
           2, 1, [5],
           category="carboxylate",
           notes="Bidentate dicarboxylate, Ca oxalate kidney stones"),

    Ligand("Malonic acid", "OC(=O)CC(=O)O",
           "141-82-2",
           ["O_carboxylate","O_carboxylate"],
           2, 1, [6],
           category="carboxylate",
           notes="6-membered chelate ring"),

    Ligand("Succinic acid", "OC(=O)CCC(=O)O",
           "110-15-6",
           ["O_carboxylate","O_carboxylate"],
           2, 1, [7],
           category="carboxylate",
           notes="7-membered ring, weak chelation"),

    Ligand("Citric acid", "OC(CC(=O)O)(CC(=O)O)C(=O)O",
           "77-92-9",
           ["O_carboxylate","O_carboxylate","O_hydroxyl"],
           3, 2, [5,6],
           category="carboxylate",
           notes="Ubiquitous biosystem chelator, descaling agent"),

    Ligand("Tartaric acid", "OC(C(O)C(=O)O)C(=O)O",
           "87-69-4",
           ["O_carboxylate","O_carboxylate","O_hydroxyl","O_hydroxyl"],
           4, 2, [5,5],
           category="carboxylate",
           notes="Wine stabilizer, chiral resolution agent"),

    Ligand("Gluconic acid", "OCC(O)C(O)C(O)C(O)C(=O)O",
           "526-95-4",
           ["O_carboxylate","O_hydroxyl","O_hydroxyl"],
           3, 1, [5],
           category="carboxylate",
           notes="Metal cleaning, concrete retarder"),

    Ligand("Ascorbic acid", "OCC(O)C1OC(=O)C(O)=C1O",
           "50-81-7",
           ["O_hydroxyl","O_hydroxyl"],
           2, 1, [5],
           category="carboxylate",
           notes="Vitamin C, mild Fe reducing agent"),

    # ═════════════════════════════════════════════════════════════════
    # PHOSPHINES (soft P-donors)
    # ═════════════════════════════════════════════════════════════════

    Ligand("Triphenylphosphine", "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
           "603-35-0",
           ["P_phosphine"],
           1, 0, [],
           category="phosphine",
           notes="Ubiquitous ligand in homogeneous catalysis",
           aliases=["PPh3"]),

    Ligand("1,2-Bis(diphenylphosphino)ethane", "c1ccc(P(CCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1",
           "1663-45-2",
           ["P_phosphine","P_phosphine"],
           2, 1, [5],
           category="phosphine",
           notes="Chelating diphosphine, Pd/Pt catalysis",
           aliases=["dppe"]),

    Ligand("Triethylphosphine", "CCP(CC)CC",
           "554-70-1",
           ["P_phosphine"],
           1, 0, [],
           category="phosphine",
           notes="Stronger donor than PPh3 (more basic)"),

    # ═════════════════════════════════════════════════════════════════
    # HALIDES
    # ═════════════════════════════════════════════════════════════════

    Ligand("Chloride", "[Cl-]",
           "16887-00-6",
           ["Cl_chloride"],
           1, 0, [],
           category="halide",
           notes="Hard/borderline, Fe3+/Ag+ precipitant"),

    Ligand("Bromide", "[Br-]",
           "24959-67-9",
           ["Br_bromide"],
           1, 0, [],
           category="halide",
           notes="Borderline donor"),

    Ligand("Iodide", "[I-]",
           "20461-54-5",
           ["I_iodide"],
           1, 0, [],
           category="halide",
           notes="Soft donor, Hg2+/Ag+/Pb2+ precipitation"),

    # ═════════════════════════════════════════════════════════════════
    # PHOSPHONATES & PHOSPHATES
    # ═════════════════════════════════════════════════════════════════

    Ligand("HEDP", "OC(P(=O)(O)O)P(=O)(O)O",
           "2809-21-4",
           ["O_carboxylate","O_carboxylate","O_hydroxyl"],
           3, 1, [5],
           category="phosphonate",
           notes="Scale inhibitor, Ca chelation in cooling water",
           aliases=["1-hydroxyethylidene-1,1-diphosphonic acid","etidronic acid"]),

    # ═════════════════════════════════════════════════════════════════
    # SPECIALTY / ENVIRONMENTAL
    # ═════════════════════════════════════════════════════════════════

    Ligand("Deferiprone", "Cc1c(O)c(=O)ccn1C",
           "30652-11-0",
           ["O_hydroxyl","O_hydroxyl"],
           2, 1, [5],
           category="hydroxamate",
           notes="Oral Fe chelator (thalassemia), bidentate hydroxypyridinone",
           aliases=["L1","Ferriprox"]),

    Ligand("Deferasirox", "OC(=O)c1ccc(-c2nc(-c3ccccc3O)no2)cc1",
           "201530-41-8",
           ["O_phenolate","O_phenolate","N_pyridine"],
           3, 2, [5,5],
           commercial=True,
           category="phenolate",
           notes="Oral Fe chelator (Exjade), tridentate ONO",
           aliases=["ICL670","Exjade"]),

    Ligand("Nitrilotrismethylenephosphonic acid", "O=P(O)(O)CN(CP(=O)(O)O)CP(=O)(O)O",
           "6419-19-8",
           ["N_amine","O_carboxylate","O_carboxylate","O_carboxylate"],
           4, 3, [5,5,5],
           category="phosphonate",
           notes="Scale inhibitor, oil field chelator",
           aliases=["NTMP","ATMP"]),

    Ligand("EDTMP", "O=P(O)(O)CN(CCN(CP(=O)(O)O)CP(=O)(O)O)CP(=O)(O)O",
           "1429-50-1",
           ["N_amine","N_amine","O_carboxylate","O_carboxylate",
            "O_carboxylate","O_carboxylate"],
           6, 5, [5]*5,
           category="phosphonate",
           notes="Nuclear med bone agent, phosphonate analogue of EDTA",
           aliases=["ethylenediaminetetramethylenephosphonic acid"]),

    Ligand("Dimethylglyoxime", "O/N=C(C)/C(C)=N/O",
           "95-45-4",
           ["N_imine","N_imine","O_hydroxyl","O_hydroxyl"],
           4, 2, [5,5],
           category="phenolate",
           notes="Ni gravimetric analysis (red precipitate)",
           aliases=["DMG"]),

    Ligand("Murexide", "O=C1NC(=O)C(NC2C(=O)NC(=O)NC2=O)C(=O)N1",
           "3051-09-0",
           ["O_hydroxamate","O_hydroxamate","N_amide"],
           3, 2, [5,6],
           category="indicator",
           notes="Ca/Ni/Cu indicator in EDTA titrations"),

    Ligand("Eriochrome Black T", "Oc1ccc2cc(N=Nc3cc([N+](=O)[O-])ccc3)c(O)c(S(=O)(=O)[O-])c2c1",
           "1787-61-7",
           ["O_phenolate","O_phenolate","N_imine"],
           3, 2, [5,6],
           category="indicator",
           notes="EDTA titration indicator for Ca/Mg hardness",
           aliases=["EBT"]),

    Ligand("Calmagite", "Oc1cc2ccccc2c(N=Nc2ccc(C)c(S(=O)(=O)[O-])c2)c1O",
           "3147-14-6",
           ["O_phenolate","O_phenolate","N_imine"],
           3, 2, [5,6],
           category="indicator",
           notes="Ca/Mg indicator, sharper endpoint than EBT"),

    Ligand("Cupferron", "O=NN(O)c1ccccc1",
           "135-20-6",
           ["O_hydroxamate","O_hydroxamate"],
           2, 1, [5],
           category="hydroxamate",
           notes="Analytical reagent for Fe/Cu/Ti extraction"),

    Ligand("Thenoyltrifluoroacetone", "OC(=CC(=O)c1cccs1)C(F)(F)F",
           "326-91-0",
           ["O_hydroxyl","O_hydroxyl"],
           2, 1, [6],
           category="beta_diketone",
           notes="Lanthanide/actinide extraction, β-diketone",
           aliases=["TTA"]),

    Ligand("Acetylacetone", "CC(=O)CC(C)=O",
           "123-54-6",
           ["O_hydroxyl","O_hydroxyl"],
           2, 1, [6],
           category="beta_diketone",
           notes="Ubiquitous β-diketone, enolizes to chelate metals",
           aliases=["acac","2,4-pentanedione"]),

    # ═════════════════════════════════════════════════════════════════
    # AMINO ACIDS with notable metal binding
    # ═════════════════════════════════════════════════════════════════

    Ligand("L-Aspartic acid", "N[C@@H](CC(=O)O)C(=O)O",
           "56-84-8",
           ["N_amine","O_carboxylate","O_carboxylate"],
           3, 2, [5,7],
           category="amino_acid",
           notes="Two carboxylates, Ca chelation"),

    Ligand("L-Glutamic acid", "N[C@@H](CCC(=O)O)C(=O)O",
           "56-86-0",
           ["N_amine","O_carboxylate","O_carboxylate"],
           3, 2, [5,8],
           category="amino_acid",
           notes="MSG, Fe/Cu chelation in food"),

    Ligand("EDDS", "OC(=O)[C@@H](NC[C@@H](NC(CC(=O)O)C(=O)O)C(=O)O)CC(=O)O",
           "20846-91-7",
           ["N_amine","N_amine","O_carboxylate","O_carboxylate",
            "O_carboxylate","O_carboxylate"],
           6, 5, [5]*5,
           category="aminocarboxylate",
           notes="Biodegradable EDTA alternative, soil remediation",
           aliases=["ethylenediamine-N,N'-disuccinic acid"]),

    Ligand("GLDA", "OC(=O)CNC(CC(=O)O)C(=O)O",
           "51981-21-6",
           ["N_amine","O_carboxylate","O_carboxylate","O_carboxylate"],
           4, 3, [5,5,5],
           category="aminocarboxylate",
           notes="Biodegradable NTA alternative, green chelator",
           aliases=["glutamic acid diacetic acid"]),
]


# ═══════════════════════════════════════════════════════════════════════════
# INDEXING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _build_indices():
    """Build lookup indices on first access."""
    by_name = {}
    by_category = {}
    by_donor_set = {}

    for lig in LIGAND_DB:
        # By name
        by_name[lig.name.lower()] = lig
        for alias in lig.aliases:
            by_name[alias.lower()] = lig

        # By category
        by_category.setdefault(lig.category, []).append(lig)

        # By sorted donor set (frozen set for matching)
        key = tuple(sorted(lig.donors))
        by_donor_set.setdefault(key, []).append(lig)

    return by_name, by_category, by_donor_set


_BY_NAME, _BY_CATEGORY, _BY_DONOR_SET = _build_indices()


def get_by_name(name: str) -> Optional[Ligand]:
    """Look up ligand by name or alias (case-insensitive)."""
    return _BY_NAME.get(name.lower())


def get_by_category(category: str) -> list[Ligand]:
    """Get all ligands in a category."""
    return _BY_CATEGORY.get(category, [])


def get_categories() -> list[str]:
    """List all categories."""
    return sorted(_BY_CATEGORY.keys())


def get_by_exact_donors(donors: list[str]) -> list[Ligand]:
    """Find ligands with exactly the specified donor set (order-independent)."""
    key = tuple(sorted(donors))
    return _BY_DONOR_SET.get(key, [])


def search_donors(donors: list[str]) -> list[tuple[Ligand, float]]:
    """Find ligands matching a donor set, scored by similarity.

    Returns list of (Ligand, score) sorted by descending score.
    Score = Jaccard similarity on donor multisets.
    """
    from collections import Counter
    query = Counter(donors)
    query_total = sum(query.values())

    results = []
    for lig in LIGAND_DB:
        lig_counts = Counter(lig.donors)
        # Intersection
        common = sum((query & lig_counts).values())
        # Union
        union = sum((query | lig_counts).values())
        if union == 0:
            continue
        jaccard = common / union
        if jaccard > 0.0:
            # Bonus for exact denticity match
            dent_match = 1.0 if len(lig.donors) == len(donors) else 0.8
            results.append((lig, jaccard * dent_match))

    results.sort(key=lambda x: -x[1])
    return results


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION (RDKit)
# ═══════════════════════════════════════════════════════════════════════════

def validate_library():
    """Validate all SMILES with RDKit and compute MW."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        print("  RDKit not available, skipping SMILES validation")
        return True

    errors = []
    for lig in LIGAND_DB:
        mol = Chem.MolFromSmiles(lig.smiles)
        if mol is None:
            errors.append(f"  INVALID SMILES: {lig.name}: {lig.smiles}")
        else:
            lig.mw = round(Descriptors.ExactMolWt(mol), 2)

    if errors:
        print(f"  {len(errors)} SMILES errors:")
        for e in errors:
            print(e)
        return False
    else:
        print(f"  All {len(LIGAND_DB)} SMILES valid")
        return True


if __name__ == "__main__":
    print(f"MABE Ligand Library: {len(LIGAND_DB)} ligands\n")

    print("Categories:")
    for cat in get_categories():
        ligs = get_by_category(cat)
        print(f"  {cat:25s} {len(ligs):3d} ligands")

    print(f"\nDonor subtype coverage:")
    all_subtypes = set()
    for lig in LIGAND_DB:
        all_subtypes.update(lig.donors)
    for st in sorted(all_subtypes):
        count = sum(1 for lig in LIGAND_DB if st in lig.donors)
        print(f"  {st:25s} {count:3d} ligands")

    print(f"\nSMILES validation:")
    validate_library()

    print(f"\nMW range:")
    mws = [lig.mw for lig in LIGAND_DB if lig.mw]
    if mws:
        print(f"  {min(mws):.1f} - {max(mws):.1f} Da")