"""
Regenerates the P1 figures from data/ (cka_matrix_v2.npy, curves_v2.npz,
bootstrap_v2.json, bootstrap_4blocks.json). Deterministic, no dependency on
J_bar_v2.pt (the excluded raw tensor) — everything here is derivable from the
small artifacts already shipped in data/.

Q(tau) reconstruction: cost c(a,b) = 2/(m-1) * sum_{i<j in block} (1 - CKA_ij),
Q = sum of block costs / n_layers. This reproduces the sealed pipeline's
published values (Q=0.1109 at onset L15, Q=0.1153 at onset L9, Q=0.0993 for
the 4-block cut) to within rounding, which is the validation that the
reconstruction formula matches the original (undocumented-in-full) pipeline.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent

# --- palette (dataviz skill default, light mode) ---
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"      # categorical slot 1 / sequential base
GREEN = "#008300"     # categorical slot 2
YELLOW = "#eda100"    # categorical slot 4
ORANGE = "#eb6834"    # categorical slot 6
RED = "#e34948"       # categorical slot 8 (status accent only)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

N = 42  # layers, lambda = 1..42


def block_cost(D, a, b):
    m = b - a + 1
    if m < 2:
        return 0.0
    sub = D[a:b + 1, a:b + 1]
    return 2.0 / (m - 1) * np.triu(sub, k=1).sum()


# ---------------------------------------------------------------------------
# Figure 1 — CKA heatmap with 3-block and 4-block boundaries
# ---------------------------------------------------------------------------
def fig1_cka_heatmap():
    cka = np.load(DATA / "cka_matrix_v2.npy")
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(cka, cmap=cmap, vmin=cka.min(), vmax=1.0, origin="upper")

    for boundary, label in [(8.5, None), (14.5, "L15"), (22.5, "L23")]:
        ax.axvline(boundary, color=INK_PRIMARY, lw=1.4, ls=(0, (4, 2)))
        ax.axhline(boundary, color=INK_PRIMARY, lw=1.4, ls=(0, (4, 2)))

    ax.axvline(8.5, color=ORANGE, lw=1.4, ls=(0, (1, 1.5)))
    ax.axhline(8.5, color=ORANGE, lw=1.4, ls=(0, (1, 1.5)))

    ticks = [0, 8, 14, 22, 41]
    ax.set_xticks(ticks)
    ax.set_xticklabels([t + 1 for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([t + 1 for t in ticks])
    ax.set_xlabel("layer $\\lambda$")
    ax.set_ylabel("layer $\\lambda$")
    ax.set_title("CKA similarity across layers — Gemma-4-E4B-it (confirmatory v2)",
                 color=INK_PRIMARY, fontsize=12, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("linear CKA (centered, admitted vocabulary)", color=INK_SECONDARY)
    cbar.outline.set_edgecolor(BASELINE)

    handles = [
        plt.Line2D([0], [0], color=INK_PRIMARY, lw=1.4, ls=(0, (4, 2)), label="3-block cut: L15 / L23 (confirmatory, sealed)"),
        plt.Line2D([0], [0], color=ORANGE, lw=1.4, ls=(0, (1, 1.5)), label="4-block cut adds: L9 (post-hoc, exploratory)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, -0.30),
              frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_cka_heatmap.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — reconstructed cost landscape Q(onset) for the 3-block segmentation
# ---------------------------------------------------------------------------
def fig2_cost_landscape():
    cka = np.load(DATA / "cka_matrix_v2.npy")
    D = 1.0 - cka

    onsets, Qvals = [], []
    for tau1 in range(3, N - 5 - 3 + 1):        # last layer of block 1 (0-indexed cut)
        best_Q = None
        for tau2 in range(tau1 + 5, N - 3 + 1):
            Q = (block_cost(D, 0, tau1 - 1) + block_cost(D, tau1, tau2 - 1)
                 + block_cost(D, tau2, N - 1)) / N
            if best_Q is None or Q < best_Q:
                best_Q = Q
        onsets.append(tau1 + 1)  # onset = first layer of block 2, 1-indexed
        Qvals.append(best_Q)

    onsets = np.array(onsets)
    Qvals = np.array(Qvals)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(onsets, Qvals, color=BLUE, lw=2.2, zorder=3)

    ymin, ymax = Qvals.min(), Qvals.max()
    pad = (ymax - ymin) * 0.18
    ax.set_ylim(ymin - pad * 1.9, ymax + pad * 0.6)

    top = ax.get_ylim()[1]
    ax.axvspan(13, 17, color=GREEN, alpha=0.10, lw=0)
    ax.text(15, top, "P-EXTRAP\nL13–17", color=GREEN, fontsize=9,
            ha="center", va="bottom", linespacing=1.3)
    ax.axvspan(19, 23, color=RED, alpha=0.08, lw=0)
    ax.text(21, top, "P-LSGOT\nL19–23", color=RED, fontsize=9,
            ha="center", va="bottom", linespacing=1.3)

    i15 = np.where(onsets == 15)[0][0]
    i9 = np.where(onsets == 9)[0][0]
    for i, label, xoff, ha in [
        (i15, "L15\nglobal min.\nQ=0.1109", 28, "left"),
        (i9, "L9\nnear-degenerate\nQ=0.1153", -28, "right"),
    ]:
        ax.scatter([onsets[i]], [Qvals[i]], color=ORANGE, s=48, zorder=4, edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(label, (onsets[i], Qvals[i]), textcoords="offset points",
                    xytext=(xoff, -38), ha=ha, va="top", fontsize=8.5,
                    color=INK_SECONDARY, linespacing=1.35)

    ax.set_xlabel("candidate onset $\\tau_1$ (layer)")
    ax.set_ylabel("segmentation cost $Q$ (lower = better fit)")
    ax.set_title("3-block cost landscape is bimodal: two near-degenerate minima (Δ=3.9%)",
                 color=INK_PRIMARY, fontsize=11.5, pad=28)
    ax.grid(True, color=GRIDLINE, lw=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_cost_landscape.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — bootstrap onset distribution (3-block), bimodal, sealed verdict
# ---------------------------------------------------------------------------
def fig3_bootstrap_3block():
    with open(DATA / "bootstrap_v2.json") as f:
        boot = json.load(f)

    dist = boot["onset_distribution"]
    layers = sorted(int(k) for k in dist)
    counts = [dist[str(l)] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(layers, counts, color=BLUE, width=0.8, zorder=3)

    lo, hi = boot["stability_interval_95"]
    ax.axvspan(lo - 0.5, hi + 0.5, color=INK_MUTED, alpha=0.12, lw=0, zorder=1)
    ax.text((lo + hi) / 2, max(counts) * 1.02, f"95% stability interval [{lo},{hi}] (width {hi-lo}, threshold ≤4)",
            ha="center", color=INK_SECONDARY, fontsize=9)

    ax.axvline(boot["point_onset"], color=RED, lw=1.6, ls=(0, (4, 2)), zorder=4)
    ax.text(boot["point_onset"] + 0.3, max(counts) * 0.9, "point estimate\nL15 (59%)",
            color=RED, fontsize=9, va="top")

    ax.set_xlabel("bootstrap onset $\\hat{\\tau}_1$ (layer, B=500 document resamples)")
    ax.set_ylabel("replicate count")
    ax.set_title(f"3-block bootstrap is bimodal → sealed verdict: {boot['verdict']}",
                 color=INK_PRIMARY, fontsize=11.5, pad=24)
    ax.grid(True, axis="y", color=GRIDLINE, lw=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_bootstrap_onset_3block.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — 4-block bootstrap distributions (exploratory), small multiples
# ---------------------------------------------------------------------------
def fig4_bootstrap_4block():
    with open(DATA / "bootstrap_4blocks.json") as f:
        boot4 = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    labels = {
        "tau1": ("$\\hat{\\tau}_1$ (sensory → intermediate)", "point ≈ L9"),
        "tau2": ("$\\hat{\\tau}_2$ (intermediate → workspace)", "point ≈ L15"),
        "tau3": ("$\\hat{\\tau}_3$ (workspace → motor)", "point ≈ L23, bimodal L23/L40"),
    }
    for ax, key in zip(axes, ["tau1", "tau2", "tau3"]):
        dist = boot4[key]["dist"]
        layers = sorted(int(k) for k in dist)
        counts = [dist[str(l)] for l in layers]
        ax.bar(layers, counts, color=BLUE, width=1.0, zorder=3)
        lo, hi = boot4[key]["interval95"]
        ax.axvspan(lo - 0.5, hi + 0.5, color=INK_MUTED, alpha=0.12, lw=0, zorder=1)
        title, note = labels[key]
        ax.set_title(title, fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel(f"layer · 95% interval [{lo},{hi}]\n{note}", fontsize=8.5, color=INK_SECONDARY)
        ax.grid(True, axis="y", color=GRIDLINE, lw=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("replicate count (B=500)")
    fig.suptitle("Post-hoc 4-block bootstrap (exploratory, not sealed) — boundaries at ≈ L9 / L15 / L23",
                 color=INK_PRIMARY, fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig4_bootstrap_4block.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — corroborative per-layer diagnostics (addendum §7), small multiples
# ---------------------------------------------------------------------------
def fig5_corroborative_diagnostics():
    npz = np.load(DATA / "curves_v2.npz")
    layers = np.arange(1, N + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    ax.plot(layers, npz["eff_dim"], color=BLUE, lw=2)
    ax.set_title("effective dimension ($d_{.99}/d$)", fontsize=10.5, color=INK_PRIMARY)
    ax.set_ylabel("fraction of full rank")

    ax = axes[0, 1]
    ax.plot(layers, npz["kurtosis"], color=GREEN, lw=2)
    ax.set_title("median kurtosis of readout logits", fontsize=10.5, color=INK_PRIMARY)

    ax = axes[1, 0]
    for key, color, lab in [("acc_top1", BLUE, "top-1"), ("acc_top5", GREEN, "top-5"),
                             ("acc_top10", YELLOW, "top-10"), ("acc_top25", ORANGE, "top-25")]:
        ax.plot(layers, npz[key], color=color, lw=1.8, label=lab)
    ax.set_title("readout accuracy vs. model's own argmax", fontsize=10.5, color=INK_PRIMARY)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("layer $\\lambda$")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="upper left")

    ax = axes[1, 1]
    ax.plot(layers, npz["autocorr_logratio"], color=BLUE, lw=2)
    ax.set_title("log-ratio autocorrelation vs. permutation", fontsize=10.5, color=INK_PRIMARY)
    ax.set_xlabel("layer $\\lambda$")

    for ax in axes.flat:
        ax.axvline(15, color=INK_PRIMARY, lw=1.1, ls=(0, (4, 2)), alpha=0.6)
        ax.axvline(23, color=ORANGE, lw=1.1, ls=(0, (1, 1.5)), alpha=0.7)
        ax.grid(True, color=GRIDLINE, lw=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Corroborative diagnostics (addendum §7, 500 positions) — dashed: L15 onset, dotted: L23 exit",
                 color=INK_PRIMARY, fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig5_corroborative_diagnostics.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_cka_heatmap()
    fig2_cost_landscape()
    fig3_bootstrap_3block()
    fig4_bootstrap_4block()
    fig5_corroborative_diagnostics()
    print("done:", sorted(p.name for p in OUT.glob("*.png")))
