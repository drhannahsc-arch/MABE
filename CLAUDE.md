# MABE — Modality-Agnostic Binder Engine
## Claude Context File

This repo is the MABE platform for AI-driven molecular binder discovery,
including aptamer virtual SELEX, β-cyclodextrin inclusion modeling,
MIP monomer screening, and environmental remediation targets (e.g. 6PPD-quinone).

---

## Scientific Skills

Skill documentation lives in `skills/`. Each subdirectory contains a `SKILL.md`
with usage instructions, code examples, and dependency notes.

Before executing any computational task, check the relevant skill directory first.

| Skill | Directory | Primary Use in MABE |
|---|---|---|
| RDKit | `skills/rdkit/` | Molecular manipulation, SAR, SMILES/SDF handling |
| Datamol | `skills/datamol/` | Analog generation, fragment ops, mol cleaning |
| DeepChem | `skills/deepchem/` | ADMET prediction, ML scoring, featurization |
| DiffDock | `skills/diffdock/` | Molecular docking, binding pose prediction |
| ChEMBL | `skills/chembl/` | Bioactivity queries, IC50 data, target lookup |
| PubChem | `skills/pubchem/` | Compound lookup, CID resolution, property data |
| BioPython | `skills/biopython/` | Aptamer sequence analysis, SELEX simulation |
| ESM | `skills/esm/` | Protein language model embeddings, pocket scoring |
| AlphaFold DB | `skills/alphafold-db/` | Structure retrieval for target proteins |
| UniProt | `skills/uniprot/` | Protein annotation, target characterization |
| Molfeat | `skills/molfeat/` | Molecular featurization for ML pipelines |
| Scientific Writing | `skills/scientific-writing/` | ChemRxiv preprint preparation |
| bioRxiv | `skills/biorxiv/` | Literature search, preprint retrieval |

---

## Repo Structure

```
MABE/
├── CLAUDE.md              ← you are here
├── skills/                ← K-Dense scientific skill docs (read-only reference)
├── core/                  ← core MABE engine modules
├── adapters/              ← modality adapters (small molecule, aptamer, MIP, etc.)
├── knowledge/             ← curated knowledge base
├── conversation/          ← conversation/session management
├── tests/                 ← test suite
├── main.py                ← entry point
├── design_binder.py       ← binder design pipeline
├── protein_pockets.py     ← pocket detection
├── fast_enrich.py         ← enrichment scoring
└── bootstrap_sprint*.py   ← sprint development scripts
```

---

## Active Development Targets

- **6PPD-quinone capture** — aptamer + MIP dual modality for environmental remediation
- **Virtual SELEX** — aptamer candidate enrichment simulation
- **β-cyclodextrin inclusion modeling** — host-guest binding geometry
- **MIP monomer screening** — computational template matching
- **ChemRxiv preprint** — 5-week solo dev timeline

---

## Skill Usage Pattern

When working on a task, reference the skill like this:

```
Read skills/rdkit/SKILL.md, then help me [task].
```

Or for multi-skill workflows:

```
Read skills/chembl/SKILL.md and skills/rdkit/SKILL.md,
then query ChEMBL for 6PPD-quinone analogs and compute RDKit descriptors.
```

---

## Dependencies

Skills use `uv` as the package manager. Install it once:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# then install a skill's deps
uv pip install rdkit datamol deepchem
```

Individual skill dependency lists are in each `SKILL.md`.

---

## License

MABE core: see repo LICENSE.
K-Dense skills in `skills/`: MIT (repo-level); check individual `SKILL.md`
`license` field for per-skill terms.