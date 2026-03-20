#!/usr/bin/env python
"""
glycan/demo_precompute.py -- Pre-compute all binder predictor data for poster demo.

Generates de novo candidates for all 8 sugar targets, scores known binders,
and dumps everything to JSON for the interactive HTML demo.

Usage:
    python glycan/demo_precompute.py [--out binder_data.json] [--max-candidates 300]

Requires: glycan module, core/de_novo_generator.py, core/pareto.py, RDKit
"""

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from glycan.de_novo_binder import (
    score_glycan_binder,
    compute_receptor_descriptor,
    generate_glycan_binders,
    GlycanBinderSpec,
    list_sugar_targets,
    get_sugar_target,
)

# Known synthetic binder pharmacophores
KNOWN_BINDERS = {
    "Davis anth.+diurea": "NC(=O)Nc1ccc2cc3ccc(NC(N)=O)cc3cc2c1",
    "Ke anthracene": "c1ccc2cc3ccccc3cc2c1",
    "Phenylboronic acid": "OB(O)c1ccccc1",
    "Diboronic biphenyl": "OB(O)c1ccc(-c2ccc(B(O)O)cc2)cc1",
    "Anth.+boronic": "OB(O)c1ccc2cc3ccccc3cc2c1",
    "Pyridine+diurea": "NC(=O)Nc1cccc(NC(N)=O)n1",
    "Indole+amide": "NC(=O)c1c[nH]c2ccccc12",
    "Catechol+amine": "NCc1ccc(O)c(O)c1",
}


def precompute(max_candidates=300, top_n=20):
    targets = list_sugar_targets()
    data = {
        "targets": targets,
        "known_names": list(KNOWN_BINDERS.keys()),
        "sugar_info": {},
        "known_scores": {},
        "results": {},
    }

    # Sugar descriptors
    for t in targets:
        s = get_sugar_target(t)
        data["sugar_info"][t] = {
            "eq_OH": s.n_eq_OH, "ax_OH": s.n_ax_OH,
            "primary_OH": s.n_primary_OH, "NAc": s.n_NAc,
            "axial_CH": s.n_axial_CH, "cis12": s.has_cis_12_diol,
            "ring": s.ring_type, "vol": s.molecular_volume_A3,
        }

    # Known binder descriptors (compute once)
    known_descs = {}
    for name, smi in KNOWN_BINDERS.items():
        known_descs[name] = compute_receptor_descriptor(smi)

    # Score known binders against all targets
    for name, smi in KNOWN_BINDERS.items():
        data["known_scores"][name] = {}
        for t in targets:
            bs = score_glycan_binder(smi, t, known_descs[name])
            data["known_scores"][name][t] = {
                "dG": round(bs.dG_total, 2),
                "chpi": round(bs.dG_chpi, 2),
                "hb": round(bs.dG_hb, 2),
                "desolv": round(bs.dG_desolv, 2),
                "boronic": round(bs.dG_boronic, 2),
                "shape": round(bs.dG_shape, 2),
                "flex": round(bs.dG_flexibility, 2),
            }

    # De novo generation per target
    for t in targets:
        t0 = time.time()
        spec = GlycanBinderSpec(target=t, max_candidates=max_candidates)
        result = generate_glycan_binders(spec=spec)
        elapsed = time.time() - t0

        top = []
        for c in result.candidates[:top_n]:
            top.append({
                "bb": c.backbone_name,
                "arms": c.arm_names,
                "smiles": c.smiles[:80],
                "dG": round(c.score.dG_total, 2),
                "chpi": round(c.score.dG_chpi, 2),
                "hb": round(c.score.dG_hb, 2),
                "desolv": round(c.score.dG_desolv, 2),
                "boronic": round(c.score.dG_boronic, 2),
                "shape": round(c.score.dG_shape, 2),
                "sel": round(c.selectivity_ddG, 2),
                "sa": round(c.sa_score, 2),
                "front": c.pareto_rank,
            })

        affinities = [round(c.score.dG_total, 1) for c in result.candidates]

        data["results"][t] = {
            "n_scored": result.n_scored,
            "n_pareto": result.n_pareto_front,
            "elapsed": round(elapsed, 1),
            "top20": top,
            "affinities": affinities,
        }
        print(f"  {t}: {result.n_scored} scored, "
              f"{result.n_pareto_front} Pareto, {elapsed:.1f}s")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute glycan binder predictor data")
    parser.add_argument("--out", default="binder_predictor_data.json",
                        help="Output JSON path")
    parser.add_argument("--max-candidates", type=int, default=300,
                        help="Max candidates per target")
    args = parser.parse_args()

    print("MABE Glycan Binder Predictor -- Pre-computation")
    print(f"  Max candidates per target: {args.max_candidates}")
    print()

    data = precompute(max_candidates=args.max_candidates)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"\nDone. Written to {args.out} ({os.path.getsize(args.out)} bytes)")
    print(f"Targets: {data['targets']}")
    print(f"Known binders: {len(data['known_names'])}")


if __name__ == "__main__":
    main()