"""
Visualise training results: loss curve & friction-parameter convergence.

Usage
-----
    python plot_results.py                               # default paths
    python plot_results.py --logs results/obs500_lr0.1/log.txt results/obs100_lr0.1/log.txt
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────
GROUND_TRUTH = [0.0, 0.5, 0.1, 0.2]   # μ_0 … μ_3
BAND_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "^", "s", "D"]
# ──────────────────────────────────────────────────────────────────────


def parse_log(path):
    """Extract epoch, loss, and μ values from a log file."""
    pattern = re.compile(
        r"(\d+)\s+([\d.eE+\-]+)\s+\[([\d.\s]+)\]"
    )
    epochs, losses, mus = [], [], []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                vals = [float(v) for v in m.group(3).split()]
                mus.append(vals)
    return np.array(epochs), np.array(losses), np.array(mus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+",
                    default=["results/obs500_lr0.1/log.txt",
                             "results/obs100_lr0.1/log.txt"])
    ap.add_argument("--labels", nargs="+",
                    default=["$N_s=500$", "$N_s=100$"])
    ap.add_argument("-o", "--output", default="loss_and_friction.png")
    args = ap.parse_args()

    fig, (ax_loss, ax_mu) = plt.subplots(1, 2, figsize=(13, 5))

    legend_mu = []

    for idx, (log_path, label) in enumerate(zip(args.logs, args.labels)):
        if not os.path.exists(log_path):
            print(f"⚠  Skipping {log_path} (not found)")
            continue

        epochs, losses, mus = parse_log(log_path)

        # ── left panel: loss ──
        ax_loss.plot(epochs, losses, label=f"Loss ({label})")

        # ── right panel: μ_i convergence ──
        for i in range(min(mus.shape[1], 4)):
            ax_mu.plot(
                epochs, mus[:, i],
                color=BAND_COLORS[i],
                marker=MARKERS[idx],
                markevery=20,
                linewidth=1,
                label=f"$\\mu_{{{i}}}$ ({label})",
            )

    # Ground-truth dashed lines
    for i, gt in enumerate(GROUND_TRUTH):
        ax_mu.axhline(gt, color=BAND_COLORS[i], linestyle="--", linewidth=1.5)

    # ── formatting ──
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch", fontsize=14)
    ax_loss.set_ylabel("Training Loss", fontsize=14)
    ax_loss.legend(fontsize=11)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.tick_params(labelsize=12)

    ax_mu.set_xlabel("Epoch", fontsize=14)
    ax_mu.set_ylabel(r"$\mu_i$", fontsize=14)
    ax_mu.grid(True, alpha=0.3)
    ax_mu.tick_params(labelsize=12)

    # Build combined legend
    handles, labels = ax_mu.get_legend_handles_labels()
    for i, gt in enumerate(GROUND_TRUTH):
        handles.append(Line2D([0], [0], color=BAND_COLORS[i], linestyle="--",
                              label=f"True $\\mu_{{{i}}}={gt}$"))
        labels.append(f"True $\\mu_{{{i}}}={gt}$")
    ax_mu.legend(handles, labels, fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
