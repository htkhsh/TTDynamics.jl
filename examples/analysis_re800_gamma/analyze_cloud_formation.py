#!/usr/bin/env python3
"""Describe buildup of the equal-time Froehlich electron-lattice correlation.

These finite-window threshold times are not equilibrium relaxation constants
or measurements of the lag of a cloud behind a moving electron.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

OUT = Path(__file__).resolve().parent
ICM2IFS = 2 * np.pi * 2.99792458e-5


def main():
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for gamma, color in [(50, "#0072B2"), (300, "#D55E00")]:
        path = (OUT.parent / "lattice_frohlich" / f"output-n15-re800-g{gamma}"
                / "lattice_frohlich_electron_lattice_correlation.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        times = np.unique(data[:, 0])
        dt = np.diff(times)[0]
        assert np.allclose(np.diff(times), dt)
        n = len(np.unique(data[:, 1]))
        corr = data[:, 2].reshape(-1, n) / ICM2IFS
        corr = (corr + corr[:, (-np.arange(n)) % n]) / 2
        late = (times >= 600) & (times <= 700)
        reference = trapezoid(corr[late], x=times[late], axis=0) / 100
        assert np.all(reference[:3] < 0)
        for window in [10, 20, 40]:
            size = round(window / dt) + 1
            assert size % 2 == 1
            half = size // 2
            smooth_times = times[half:-half]
            for r in [0, 1, 2]:
                smooth = np.convolve(corr[:, r], np.ones(size) / size, mode="valid")
                fraction = smooth / reference[r]
                row = dict(gamma_cm=gamma, absolute_distance=r,
                           smoothing_span_fs=window, reference_mean_600_700_cm=reference[r],
                           ratio_to_onsite_reference=reference[r] / reference[0])
                for target in [0.5, 0.8, 0.9]:
                    below = np.flatnonzero(fraction < target)
                    index = below[-1] + 1 if len(below) else 0
                    reached = index < len(fraction)
                    if reached:
                        assert np.all(fraction[index:] >= target)
                    row[f"sustained_t{round(target * 100)}_fs"] = (
                        float(smooth_times[index]) if reached else "")
                rows.append(row)
                if window == 20:
                    label = f"gamma={gamma} cm^-1"
                    axes[0, r].plot(smooth_times, smooth, color=color, label=label)
                    axes[1, r].plot(smooth_times, fraction, color=color, label=label)
                    crossing = row["sustained_t50_fs"]
                    if crossing != "":
                        axes[1, r].plot(crossing, np.interp(crossing, smooth_times, fraction),
                                        "o", color=color, ms=5)
        print(f"gamma={gamma}: C1/C0={reference[1]/reference[0]:.4f}, "
              f"C2/C0={reference[2]/reference[0]:.4f}")
    for r in [0, 1, 2]:
        axes[0, r].set(title=f"|r|={r}", ylabel="Connected CnB (cm^-1)")
        axes[1, r].set(ylabel="C(r,t) / mean C(r,600-700 fs)")
        axes[1, r].axhline(0.5, color="gray", ls="--", lw=0.8)
        axes[1, r].axhline(1, color="gray", ls=":", lw=0.8)
    for ax in axes.flat:
        ax.set_xlabel("Time since localized, factorized preparation (fs)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Froehlich cloud buildup: equal-time correlation, 20 fs smoothing\n"
                 "Reference: 600-700 fs average; dots: sustained 50% threshold")
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"cloud_formation.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (OUT / "cloud_formation_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["smoothing_span_fs"] == 20:
            print(row)


if __name__ == "__main__":
    main()
