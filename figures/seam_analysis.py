"""
Seam analysis for P1 — computed directly from data/cka_matrix_v2.npy.

Reproduces, deterministically and without the excluded raw tensor:
  1. Optimal contiguous segmentations for k = 3..6 blocks (exact enumeration,
     min block size 3), using the sealed pipeline's cost
     c(a,b) = 2/(m-1) * sum_{i<j in block} (1 - CKA_ij),  Q = sum(c)/42.
  2. Marginal weight of each seam in the 5-block optimum {L9, L15, L23, L40}:
     delta-Q when that seam is removed.
  3. Intra- vs cross-block CKA for the L40-42 segment (emission-seam caveat).
  4. Figure 6: Q(k) curve + seam-weight hierarchy bar chart.

Published-value validation: the 3-block optimum reproduces Q=0.1109 (L15/L23),
the L9 3-block candidate reproduces Q=0.1153, and the 4-block optimum
reproduces Q=0.0993 at {L9, L15, L23} — matching report_v2.md to rounding.
"""
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

N = 42
cka = np.load(DATA / "cka_matrix_v2.npy")
D = 1.0 - cka


def block_cost(a, b):
    m = b - a + 1
    if m < 2:
        return 0.0
    sub = D[a:b + 1, a:b + 1]
    return 2.0 / (m - 1) * np.triu(sub, k=1).sum()


def Q(cuts):
    bounds = [0] + list(cuts) + [N]
    return sum(block_cost(bounds[i], bounds[i + 1] - 1)
               for i in range(len(bounds) - 1)) / N


def optimal(k, min_size=3):
    best_q, best_c = np.inf, None
    for cuts in combinations(range(min_size, N - min_size + 1), k - 1):
        b = [0] + list(cuts) + [N]
        if min(b[i + 1] - b[i] for i in range(k)) < min_size:
            continue
        q = Q(cuts)
        if q < best_q:
            best_q, best_c = q, cuts
    return best_q, best_c


def main():
    results = {}
    print("Optimal segmentations (exact enumeration, min block size 3):")
    for k in (3, 4, 5, 6):
        q, cuts = optimal(k)
        results[k] = {"Q": round(float(q), 4),
                      "boundaries_1idx": [c + 1 for c in cuts]}
        print(f"  k={k}: Q={q:.4f}  first-layer-of-block boundaries "
              f"= {[c + 1 for c in cuts]}")

    # marginal seam weights in the 5-block optimum
    full = list(optimal(5)[1])
    q5 = Q(full)
    weights = {}
    print("\nMarginal seam weights (delta-Q on removal from 5-block optimum):")
    for c in full:
        dq = Q([x for x in full if x != c]) - q5
        weights[f"L{c + 1}"] = round(float(dq), 4)
        print(f"  L{c + 1}: +{dq:.4f}")

    # emission-seam caveat
    motor = cka[22:39, 22:39]
    out = cka[39:, 39:]
    cross = cka[22:39, 39:]
    tri = lambda M: M[np.triu_indices_from(M, 1)].mean()
    caveat = {"intra_L23_39": round(float(tri(motor)), 3),
              "intra_L40_42": round(float(tri(out)), 3),
              "cross": round(float(cross.mean()), 3)}
    print(f"\nEmission-seam caveat: intra L23-39 = {caveat['intra_L23_39']}, "
          f"intra L40-42 = {caveat['intra_L40_42']}, cross = {caveat['cross']}")

    with open(DATA / "seam_analysis.json", "w") as f:
        json.dump({"segmentations": results, "seam_weights": weights,
                   "emission_caveat": caveat}, f, indent=1)

    # --- Figure 6 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    ks = sorted(results)
    ax1.plot(ks, [results[k]["Q"] for k in ks], "-o", color=BLUE, lw=1.6,
             ms=5, mfc=BLUE, mec=SURFACE)
    for k in ks:
        ax1.annotate(f"{results[k]['Q']:.4f}", (k, results[k]["Q"]),
                     textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9, color=INK_SECONDARY)
    ax1.set_xticks(ks)
    ax1.set_xlabel("number of blocks $k$")
    ax1.set_ylabel("segmentation cost $Q$")
    ax1.set_title("Q decreases monotonically in k —\nno clean elbow",
                  fontsize=11, color=INK_PRIMARY)
    ax1.spines[["top", "right"]].set_visible(False)

    seams = list(weights)
    vals = [weights[s] for s in seams]
    colors = [ORANGE if s == "L23" else BLUE for s in seams]
    ax2.bar(seams, vals, color=colors, width=0.55)
    for s, v in zip(seams, vals):
        ax2.annotate(f"+{v:.4f}", (s, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=9,
                     color=INK_SECONDARY)
    ax2.set_ylabel("marginal cost of removal  $\\Delta Q$")
    ax2.set_title("Seam hierarchy: L23 dominates",
                  fontsize=11, color=INK_PRIMARY)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_seam_hierarchy.png", dpi=200)
    print(f"\nWrote {OUT / 'fig6_seam_hierarchy.png'} and "
          f"{DATA / 'seam_analysis.json'}")


if __name__ == "__main__":
    main()
