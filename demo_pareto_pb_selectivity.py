"""
demo_pareto_pb_selectivity.py — End-to-End Pareto Demo

Target: Pb²⁺ selective over Ca²⁺, Mg²⁺, Fe³⁺
(Elk Valley coal mine selenium remediation context — Pb is a proxy
for heavy metal selectivity; same physics applies to Se capture agents)

Runs the full pipeline:
  1. Enumerate chelator candidates (backbone × arm combinatorics)
  2. Score each against Pb²⁺ (target) and Ca²⁺/Mg²⁺/Fe³⁺ (interferents)
  3. Pareto-rank on 3 objectives: affinity × selectivity × synthesizability
  4. Generate figures: 3 pairwise projections + candidate summary table

Usage:
    cd C:\\dev\\MABE
    python demo_pareto_pb_selectivity.py

Output:
    figures/pareto_pb_affinity_vs_selectivity.png
    figures/pareto_pb_affinity_vs_SA.png
    figures/pareto_pb_selectivity_vs_SA.png
    figures/pareto_pb_summary_table.png
    figures/pareto_pb_3d_scatter.png
    Console: ranked candidate table

No new physics. Pure pipeline exercise using calibrated MABE scorer.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("=" * 70)
    print("MABE Pareto Demo: Pb²⁺ Selective Chelator Design")
    print("=" * 70)
    print()

    from core.de_novo_generator import generate_and_screen

    # ── Step 1: Generate + Screen ─────────────────────────────────────────

    target = "Pb2+"
    interferents = ["Ca2+", "Mg2+", "Fe3+"]
    print(f"Target: {target}")
    print(f"Interferents: {', '.join(interferents)}")
    print(f"Ranking: 3-objective Pareto (affinity × selectivity × SA)")
    print()

    t0 = time.time()
    result = generate_and_screen(
        target_metal=target,
        interferents=interferents,
        max_candidates=500,
        max_scored=100,
        ranking_mode="pareto",
    )
    elapsed = time.time() - t0

    print(f"Enumerated: {result.n_enumerated}")
    print(f"Scored:     {result.n_scored}")
    print(f"Failed:     {result.n_failed}")
    print(f"Elapsed:    {elapsed:.1f}s")
    print()

    if result.n_scored == 0:
        print("ERROR: No candidates scored. Check errors:")
        for smi, err in result.errors[:5]:
            print(f"  {smi[:40]}: {err}")
        return

    # ── Step 2: Extract Pareto Data ───────────────────────────────────────

    candidates = result.candidates
    n_pareto = sum(1 for c in candidates if c.pareto_front_idx == 0)

    affinities = [c.log_Ka_pred for c in candidates]
    selectivities = [c.min_gap for c in candidates]
    sa_scores = [c.sa_score_val for c in candidates]
    is_pareto = [c.pareto_front_idx == 0 for c in candidates]

    print(f"Pareto-optimal candidates: {n_pareto} / {len(candidates)}")
    print()

    # ── Step 3: Print Top Candidates ──────────────────────────────────────

    print(f"{'Rank':>4s}  {'Front':>5s}  {'logKa':>6s}  {'MinGap':>7s}  "
          f"{'SA':>4s}  {'Grade':>5s}  {'Scaffold':<30s}  SMILES")
    print("-" * 110)

    for c in candidates[:25]:
        front_str = "P*" if c.pareto_front_idx == 0 else f"F{c.pareto_front_idx}"
        scaffold = c.backbone_name if hasattr(c, 'backbone_name') else ""
        print(f"{c.rank:4d}  {front_str:>5s}  {c.log_Ka_pred:+6.1f}  "
              f"{c.min_gap:+7.1f}  {c.sa_score_val:4.1f}  "
              f"{c.grade:>5s}  {scaffold:<30s}  {c.smiles[:50]}")

    # ── Step 4: Figures ───────────────────────────────────────────────────

    os.makedirs("figures", exist_ok=True)

    # Color scheme
    c_pareto = "#E63946"   # red for Pareto front
    c_other = "#457B9D"    # steel blue for dominated
    c_edge = "#1D3557"     # dark navy for edges

    def pareto_colors(is_p):
        return [c_pareto if p else c_other for p in is_p]

    def pareto_sizes(is_p):
        return [80 if p else 30 for p in is_p]

    def pareto_zorder(is_p):
        return [10 if p else 1 for p in is_p]

    # ── Fig 1: Affinity vs Selectivity ────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(candidates)):
        ax.scatter(affinities[i], selectivities[i],
                   c=c_pareto if is_pareto[i] else c_other,
                   s=80 if is_pareto[i] else 30,
                   edgecolors=c_edge if is_pareto[i] else "none",
                   linewidths=0.8 if is_pareto[i] else 0,
                   alpha=0.9 if is_pareto[i] else 0.5,
                   zorder=10 if is_pareto[i] else 1)

    # Connect Pareto front
    pareto_pts = [(affinities[i], selectivities[i])
                  for i in range(len(candidates)) if is_pareto[i]]
    if pareto_pts:
        pareto_pts.sort()
        px, py = zip(*pareto_pts)
        ax.plot(px, py, '--', color=c_pareto, alpha=0.5, linewidth=1.2, zorder=5)

    ax.set_xlabel("Predicted log Ka (Pb²⁺)", fontsize=12)
    ax.set_ylabel(f"Min selectivity gap (log Ka units)", fontsize=12)
    ax.set_title(f"Pb²⁺ Chelator Design: Affinity vs Selectivity\n"
                 f"{n_pareto} Pareto-optimal / {len(candidates)} total",
                 fontsize=13, fontweight="bold")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c_pareto,
                       markersize=10, markeredgecolor=c_edge, label=f'Pareto front ({n_pareto})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c_other,
                       markersize=7, label=f'Dominated ({len(candidates)-n_pareto})'),
        ],
        loc='lower right', fontsize=10
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pareto_pb_affinity_vs_selectivity.png", dpi=150)
    plt.close(fig)
    print("\n  Saved: figures/pareto_pb_affinity_vs_selectivity.png")

    # ── Fig 2: Affinity vs SA ─────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(candidates)):
        ax.scatter(affinities[i], sa_scores[i],
                   c=c_pareto if is_pareto[i] else c_other,
                   s=80 if is_pareto[i] else 30,
                   edgecolors=c_edge if is_pareto[i] else "none",
                   linewidths=0.8 if is_pareto[i] else 0,
                   alpha=0.9 if is_pareto[i] else 0.5,
                   zorder=10 if is_pareto[i] else 1)

    ax.set_xlabel("Predicted log Ka (Pb²⁺)", fontsize=12)
    ax.set_ylabel("Synthetic Accessibility Score (lower = easier)", fontsize=12)
    ax.set_title("Pb²⁺ Chelator Design: Affinity vs Synthesizability", fontsize=13,
                 fontweight="bold")
    ax.invert_yaxis()  # lower SA is better → put "good" at top
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pareto_pb_affinity_vs_SA.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/pareto_pb_affinity_vs_SA.png")

    # ── Fig 3: Selectivity vs SA ──────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(candidates)):
        ax.scatter(selectivities[i], sa_scores[i],
                   c=c_pareto if is_pareto[i] else c_other,
                   s=80 if is_pareto[i] else 30,
                   edgecolors=c_edge if is_pareto[i] else "none",
                   linewidths=0.8 if is_pareto[i] else 0,
                   alpha=0.9 if is_pareto[i] else 0.5,
                   zorder=10 if is_pareto[i] else 1)

    ax.set_xlabel(f"Min selectivity gap (log Ka units)", fontsize=12)
    ax.set_ylabel("Synthetic Accessibility Score (lower = easier)", fontsize=12)
    ax.set_title("Pb²⁺ Chelator Design: Selectivity vs Synthesizability", fontsize=13,
                 fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pareto_pb_selectivity_vs_SA.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/pareto_pb_selectivity_vs_SA.png")

    # ── Fig 4: Summary Table ──────────────────────────────────────────────

    top_n = min(15, len(candidates))
    top = candidates[:top_n]

    fig, ax = plt.subplots(figsize=(14, 0.5 + 0.4 * top_n))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top_n + 1.5)

    # Header
    headers = ["Rank", "Front", "log Ka", "Min Gap", "SA", "Grade", "Scaffold"]
    x_positions = [0.03, 0.10, 0.18, 0.28, 0.38, 0.46, 0.56]
    y_header = top_n + 0.8

    for x, h in zip(x_positions, headers):
        ax.text(x, y_header, h, fontsize=9, fontweight="bold", va="center",
                fontfamily="monospace")

    ax.axhline(y=top_n + 0.3, xmin=0.02, xmax=0.95, color="black", linewidth=0.8)

    for row_idx, c in enumerate(top):
        y = top_n - row_idx - 0.2
        front_str = "P*" if c.pareto_front_idx == 0 else f"F{c.pareto_front_idx}"
        scaffold = c.backbone_name[:25] if hasattr(c, 'backbone_name') else ""
        bg_color = "#FFEAEA" if c.pareto_front_idx == 0 else "white"

        ax.fill_between([0.01, 0.95], y - 0.25, y + 0.25,
                        color=bg_color, alpha=0.7, zorder=0)

        vals = [
            f"{c.rank}",
            front_str,
            f"{c.log_Ka_pred:+.1f}",
            f"{c.min_gap:+.1f}",
            f"{c.sa_score_val:.1f}",
            c.grade,
            scaffold,
        ]
        for x, v in zip(x_positions, vals):
            ax.text(x, y, v, fontsize=8, va="center", fontfamily="monospace")

    ax.set_title(f"Top {top_n} Pb²⁺-Selective Chelators (Pareto-Ranked)\n"
                 f"Target: Pb²⁺ | Interferents: Ca²⁺, Mg²⁺, Fe³⁺ | "
                 f"{n_pareto} Pareto-optimal",
                 fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig("figures/pareto_pb_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figures/pareto_pb_summary_table.png")

    # ── Fig 5: 3D-ish scatter (affinity vs selectivity, color=SA) ─────────

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        affinities, selectivities,
        c=sa_scores, cmap="RdYlGn_r",
        s=[80 if p else 30 for p in is_pareto],
        edgecolors=[c_edge if p else "none" for p in is_pareto],
        linewidths=[0.8 if p else 0 for p in is_pareto],
        alpha=0.85,
    )
    cbar = fig.colorbar(scatter, ax=ax, label="SA Score (lower = easier)")
    ax.set_xlabel("Predicted log Ka (Pb²⁺)", fontsize=12)
    ax.set_ylabel(f"Min selectivity gap (log Ka units)", fontsize=12)
    ax.set_title("Pb²⁺ Chelator Design: 3-Objective Landscape\n"
                 "Position = affinity × selectivity | Color = synthesizability",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/pareto_pb_3d_scatter.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/pareto_pb_3d_scatter.png")

    # ── Summary ───────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Target:            {target}")
    print(f"  Interferents:      {', '.join(interferents)}")
    print(f"  Candidates scored: {result.n_scored}")
    print(f"  Pareto-optimal:    {n_pareto}")
    if candidates:
        best = candidates[0]
        print(f"  Best candidate:")
        print(f"    SMILES:      {best.smiles}")
        print(f"    Scaffold:    {best.backbone_name}")
        print(f"    log Ka(Pb):  {best.log_Ka_pred:+.2f}")
        print(f"    Min gap:     {best.min_gap:+.2f} log Ka units")
        print(f"    SA score:    {best.sa_score_val:.1f}")
        print(f"    Grade:       {best.grade}")
    print(f"  Elapsed:           {elapsed:.1f}s")
    print(f"  Figures:           figures/pareto_pb_*.png (5 files)")
    print()


if __name__ == "__main__":
    main()