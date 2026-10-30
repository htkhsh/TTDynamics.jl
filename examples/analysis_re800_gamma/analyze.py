#!/usr/bin/env python3
"""Reproduce the re800 gamma comparison from saved CSVs, without HEOM propagation.

Run: python TTDynamics/examples/analysis_re800_gamma/analyze.py
Requires Python >= 3.11, numpy, scipy, matplotlib. Input files are read only.
"""
from pathlib import Path
import csv
import hashlib
import json
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid

OUT = Path(__file__).resolve().parent
EXAMPLES = OUT.parent
ICM2IFS = 2 * np.pi * 2.99792458e-5
KT_OVER_HBAR = 1.380649e-23 * 300 / 1.054571817e-34 * 1e-15
COLORS = {50: "#0072B2", 300: "#D55E00", 600: "#009E73"}
NAMES = {"holstein": "Holstein", "lattice_frohlich": "Lattice Froehlich"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22,
                     "savefig.dpi": 180, "axes.spines.top": False,
                     "axes.spines.right": False})


def table(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def average(t, values, lo, hi):
    """Time average with trapezoidal endpoint weights; endpoints must be present."""
    mask = (t >= lo) & (t <= hi)
    assert t[mask][0] == lo and t[mask][-1] == hi
    return trapezoid(values[mask], x=t[mask], axis=0) / (hi - lo)


def rho_from_upper(eq, n):
    # CSV stores real populations and only upper-triangular coherences.
    # This is a Hermitian completion, not a measurement of rho-rho^dagger.
    rho = np.zeros((len(eq), n, n), complex)
    for j in range(n):
        rho[:, j, j] = eq[f"population_site_{j + 1}"]
        for k in range(j + 1, n):
            rho[:, j, k] = (eq[f"coherence_{j + 1}_{k + 1}_real"]
                            + 1j * eq[f"coherence_{j + 1}_{k + 1}_imag"])
            rho[:, k, j] = rho[:, j, k].conj()
    return rho


def main():
    runs, manifest, profiles, quality, currents, fit_rows = [], [], [], [], [], []
    for model in NAMES:
        for directory in sorted((EXAMPLES / model).glob("*re800-g*")):
            paths = [directory / f"{model}_{suffix}" for suffix in
                     ["equilibrium_metadata.toml", "electron_lattice_correlation.csv",
                      "equilibration.csv", "current_correlation.csv"]]
            meta = tomllib.loads(paths[0].read_text())
            params = meta["physical_parameters"]
            n = params["site_count"]
            gamma = params["brownian_damping_cm"]
            omega = params["brownian_frequency_cm"]
            assert n == 15 and params["reorganization_energy_cm"] == 800
            assert params["hopping_cm"] == 4000 and params["temperature_K"] == 300
            assert params["decomposition_method"] == "psd" and params["pade_order"] == 0
            if model == "lattice_frohlich":
                assert params["frohlich_range"] == 1
            el = np.loadtxt(paths[1], delimiter=",", skiprows=1)
            assert np.isfinite(el).all()
            times = np.unique(el[:, 0])
            assert len(el) == len(times) * n and np.all(np.diff(times) > 0)
            assert np.array_equal(el[:, 1].reshape(-1, n),
                                  np.broadcast_to(np.arange(n), (len(times), n)))
            assert np.array_equal(el[:, 0].reshape(-1, n),
                                  np.broadcast_to(times[:, None], (len(times), n)))
            real = el[:, 2].reshape(-1, n)
            imag = el[:, 3].reshape(-1, n)
            uncentered = el[:, 4].reshape(-1, n)
            assert np.max(abs(real.sum(axis=1))) < 1e-10
            assert np.max(abs(imag.sum(axis=1))) < 1e-10
            eq = np.genfromtxt(paths[2], delimiter=",", names=True)
            assert np.array_equal(eq["time_fs"], times)
            rho = rho_from_upper(eq, n)
            assert np.isfinite(rho).all()
            eigmin = np.linalg.eigvalsh(rho)[:, 0]
            populations = rho.diagonal(axis1=1, axis2=2).real
            signed_r = np.where(np.arange(n) <= n // 2, np.arange(n), np.arange(n) - n)
            w = np.zeros(n)
            w[0] = 1
            if model == "lattice_frohlich":
                w = (1 + abs(signed_r) ** 2.) ** -1.5
                w /= np.linalg.norm(w)
            assert np.isclose(w @ w, 1)
            velocity = np.zeros((n, n), complex)
            for j in range(n):
                velocity[(j + 1) % n, j] = 1j * params["hopping_cm"] * ICM2IFS
                velocity[j, (j + 1) % n] = -1j * params["hopping_cm"] * ICM2IFS
            mean_current = np.einsum("ij,tji->t", velocity, rho).real
            kinetic_cm = -2 * params["hopping_cm"] * (
                sum(rho[:, j, j + 1].real for j in range(n - 1)) + rho[:, 0, -1].real)
            late = (times >= 600) & (times <= 700)
            mean = average(times, real, 600, 700)
            prev = average(times, real, 500, 600)
            std = real[late].std(axis=0)
            label = f"gamma={gamma:g}"
            if omega != 160:
                label += f", omega0={omega:g}"
            run = dict(model=model, directory=directory, params=params, gamma=gamma,
                       omega=omega, label=label, times=times, real=real, imag=imag,
                       mean=mean, std=std, w=w, signed_r=signed_r, eigmin=eigmin,
                       eq=eq, populations=populations, kinetic_cm=kinetic_cm)
            summary = dict(model=model, directory=directory.name, gamma_cm=gamma,
                           omega0_cm=omega, current_csv_present=paths[3].exists(),
                           CnB0_mean_600_700_cm=mean[0] / ICM2IFS,
                           CnB0_std_600_700_cm=std[0] / ICM2IFS,
                           CnB0_drift_vs_500_600_percent=100 * (mean[0] / prev[0] - 1),
                           CnB_nearest_mean_cm=(mean[1] + mean[-1]) / (2 * ICM2IFS),
                           CnB_next_nearest_mean_cm=(mean[2] + mean[-2]) / (2 * ICM2IFS),
                           connected_interaction_mean_cm=float(mean @ w / ICM2IFS),
                           interaction_mean_cm=float(average(times, uncentered @ w, 600, 700) / ICM2IFS),
                           CnB_late_max_imag_over_max_real=float(abs(imag[late]).max() / abs(real[late]).max()),
                           CnB_sum_rule_max_abs=float(abs(real.sum(axis=1)).max()),
                           trace_max_abs_error=float(abs(eq["trace"] - 1).max()),
                           rho_hermitian_completion_min_eigenvalue_final=float(eigmin[-1]),
                           mean_current_hermitian_completion_final=float(mean_current[-1]),
                           kinetic_energy_600_cm=float(kinetic_cm[times == 600][0]),
                           kinetic_energy_700_cm=float(kinetic_cm[-1]),
                           late_population_max_deviation_uniform=float(abs(populations[late] - 1 / n).max()),
                           bath_envelope_time_fs=2 / (gamma * ICM2IFS),
                           bath_damped_period_fs=2 * np.pi / (np.sqrt(omega ** 2 - (gamma / 2) ** 2) * ICM2IFS),
                           adjacent_potential_noise_relative=float(1 - w @ np.roll(w, 1)))
            for r in range(n):
                profiles.append(dict(model=model, gamma_cm=gamma, omega0_cm=omega,
                                     distance=r, signed_distance=int(signed_r[r]),
                                     mean_600_700_cm=mean[r] / ICM2IFS,
                                     std_600_700_cm=std[r] / ICM2IFS,
                                     final_700_cm=real[-1, r] / ICM2IFS,
                                     coupling_weight=w[r]))
            for j, t in enumerate(times):
                quality.append(dict(model=model, gamma_cm=gamma, time_fs=t,
                                    trace_error=eq["trace"][j] - 1,
                                    min_eigenvalue_hermitian_completion=eigmin[j],
                                    max_population_deviation_uniform=abs(populations[j] - 1/n).max(),
                                    mean_current_hermitian_completion=mean_current[j],
                                    kinetic_energy_cm=kinetic_cm[j]))
            if paths[3].exists():
                cc = np.loadtxt(paths[3], delimiter=",", skiprows=1)
                assert np.isfinite(cc).all() and cc[0, 0] == 0 and np.all(np.diff(cc[:, 0]) > 0)
                t, cr, ci = cc[:, 0], cc[:, 1], cc[:, 2]
                d = cumulative_trapezoid(cr, t, initial=0)
                dim = -2 * KT_OVER_HBAR * cumulative_trapezoid(t * ci, t, initial=0)
                # Independent summation verifies the cumulative integration implementation.
                assert np.isclose(d[-1], sum(np.diff(t) * (cr[1:] + cr[:-1]) / 2), rtol=1e-12)
                run.update(ct=t, cr=cr, ci=ci, diffusion=d, diffusion_imag=dim)
                summary.update(Cjj0_real=cr[0], Cjj0_imag=ci[0],
                               Cjj_min_real=cr.min(), Cjj_min_time_fs=t[cr.argmin()],
                               current_end_fs=t[-1], D_end_a2_per_fs=d[-1],
                               D_500_a2_per_fs=float(np.interp(500, t, d)),
                               D_600_a2_per_fs=float(np.interp(600, t, d)),
                               D_imag_600_a2_per_fs=float(np.interp(600, t, dim)),
                               D_imag_end_a2_per_fs=dim[-1],
                               Cjj_end_real=cr[-1],
                               quadrature_simpson_minus_trapezoid=float(simpson(cr, x=t) - d[-1]))
                for j, tt in enumerate(t):
                    currents.append(dict(model=model, gamma_cm=gamma, time_fs=tt,
                                         Cjj_real_fs_minus2=cr[j], Cjj_imag_fs_minus2=ci[j],
                                         D_real_a2_per_fs=d[j], D_imag_check_a2_per_fs=dim[j]))
                if model == "lattice_frohlich" and gamma == 50:
                    for start in [400, 500, 600, 700]:
                        mask = t >= start
                        assert np.all(cr[mask] > 0)
                        slope, intercept = np.polyfit(t[mask], np.log(cr[mask]), 1)
                        tau = -1 / slope
                        fit_rows.append(dict(model=model, gamma_cm=gamma, start_fs=start,
                                             end_fs=t[-1], tau_fs=tau,
                                             rms_log_residual=float(np.sqrt(np.mean((np.log(cr[mask]) - slope*t[mask] - intercept)**2))),
                                             endpoint_anchored_Dinf_a2_per_fs=d[-1]+cr[-1]*tau))
            for path in paths:
                if path.exists():
                    manifest.append(dict(path=str(path.relative_to(EXAMPLES)),
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         size_bytes=path.stat().st_size))
            run["summary"] = summary
            runs.append(run)

    all_keys = list(dict.fromkeys(k for r in runs for k in r["summary"]))
    table("summary.csv", [{k: r["summary"].get(k, "") for k in all_keys} for r in runs])
    table("electron_lattice_profiles.csv", profiles)
    table("current_and_diffusion.csv", currents)
    table("equilibration_diagnostics.csv", quality)
    table("frohlich_g50_tail_fits.csv", fit_rows)
    table("input_manifest.csv", manifest)
    (OUT / "summary.json").write_text(json.dumps([r["summary"] for r in runs], indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, model in enumerate(NAMES):
        for run in runs:
            if run["model"] != model or "ct" not in run:
                continue
            kw = dict(color=COLORS[run["gamma"]], label=run["label"], lw=1.7)
            axes[row, 0].plot(run["ct"], run["cr"], **kw)
            axes[row, 1].plot(run["ct"], run["ci"], **kw)
        for col, component in enumerate(["Re", "Im"]):
            axes[row, col].set(title=f"{NAMES[model]}: {component} Cjj(t)",
                               xlabel="Time after current insertion (fs)", ylabel="Cjj (1/fs^2)")
            axes[row, col].axhline(0, color="gray", lw=0.7)
            axes[row, col].legend()
    fig.suptitle("N=15, lambda=800, omega0=160 cm^-1; T=300 K")
    save(fig, "current_correlations")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for run in runs:
        if "ct" not in run:
            continue
        row = list(NAMES).index(run["model"])
        kw = dict(color=COLORS[run["gamma"]], label=run["label"], lw=1.8)
        for ax in axes[row]:
            ax.plot(run["ct"], run["diffusion"], **kw)
            ax.set(xlabel="Time (fs)", ylabel="Integral Re Cjj (a^2/fs)")
        axes[row, 0].set_title(NAMES[run["model"]])
        axes[row, 1].set_title(NAMES[run["model"]] + ": late times")
        axes[row, 1].set_xlim(300, 800 if row == 1 else 600)
    axes[0, 1].set_ylim(13.7, 16.6)
    axes[1, 1].set_ylim(42, 90)
    for ax in axes.flat:
        ax.legend()
    fig.suptitle("Running diffusion integral; Froehlich gamma=50 still grows at 800 fs")
    save(fig, "diffusion")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, model in enumerate(NAMES):
        for col, gamma in enumerate([50, 300]):
            run = next(r for r in runs if r["model"] == model and r["gamma"] == gamma)
            order = np.argsort(run["signed_r"])
            img = axes[row, col].pcolormesh(run["times"], run["signed_r"][order],
                                          run["real"][:, order].T / ICM2IFS,
                                          cmap="RdBu_r", vmin=-140, vmax=140, shading="auto", rasterized=True)
            axes[row, col].set(title=f"{NAMES[model]}, gamma={gamma}",
                               xlabel="Equilibration time (fs)", ylabel="Signed distance (sites)")
    fig.colorbar(img, ax=axes.ravel().tolist(), label="Connected CnB (cm^-1)")
    save(fig, "electron_lattice_heatmaps")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, model in enumerate(NAMES):
        for col, r in enumerate([0, 1, 2]):
            for run in runs:
                if run["model"] != model:
                    continue
                values = run["real"][:, r] if r == 0 else (run["real"][:, r] + run["real"][:, -r]) / 2
                axes[row, col].plot(run["times"], values / ICM2IFS,
                                    color=COLORS[run["gamma"]], label=run["label"], lw=1.3)
            axes[row, col].set(title=f"{NAMES[model]}, |r|={r}",
                               xlabel="Equilibration time (fs)", ylabel="Connected CnB (cm^-1)")
            axes[row, col].axhline(0, color="gray", lw=0.7)
            axes[row, col].legend(fontsize=8)
    save(fig, "electron_lattice_short_distances")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for col, model in enumerate(NAMES):
        for run in runs:
            if run["model"] != model:
                continue
            order = np.argsort(run["signed_r"])
            # Shade temporal standard deviation, not a statistical confidence interval.
            x = run["signed_r"][order]
            y = run["mean"][order] / ICM2IFS
            std = run["std"][order] / ICM2IFS
            axes[col].plot(x, y, "o-", ms=3, color=COLORS[run["gamma"]], label=run["label"])
            axes[col].fill_between(x, y - std, y + std, color=COLORS[run["gamma"]], alpha=0.15)
        axes[col].set(title=NAMES[model] + ": 600-700 fs average", xlabel="Signed distance (sites)",
                      ylabel="Connected CnB (cm^-1)")
        axes[col].legend(fontsize=8)
    save(fig, "electron_lattice_profiles")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for run in runs:
        row = list(NAMES).index(run["model"])
        axes[row, 0].plot(run["times"], run["eigmin"], color=COLORS[run["gamma"]], label=run["label"])
        axes[row, 2].plot(run["times"], run["kinetic_cm"], color=COLORS[run["gamma"]], label=run["label"])
        if "ct" in run:
            color = COLORS[run["gamma"]]
            axes[row, 1].plot(run["ct"], run["diffusion"], color=color, label=run["label"] + " Re integral")
            axes[row, 1].plot(run["ct"], run["diffusion_imag"], color=color, ls="--", label=run["label"] + " Im check")
    for row, model in enumerate(NAMES):
        axes[row, 0].set(title=NAMES[model] + ": Hermitian completion of saved upper triangle",
                          xlabel="Equilibration time (fs)", ylabel="Minimum eigenvalue")
        axes[row, 0].axhline(0, color="black", lw=0.8)
        axes[row, 1].set(title=NAMES[model] + ": equilibrium consistency check",
                          xlabel="Current correlation time (fs)", ylabel="a^2/fs")
        axes[row, 2].set(title=NAMES[model] + ": electronic kinetic energy",
                          xlabel="Equilibration time (fs)", ylabel="Energy (cm^-1)")
        for ax in axes[row]:
            ax.legend(fontsize=8)
    save(fig, "quality_diagnostics")
    print(f"Analyzed {len(runs)} runs; {len(currents)} current samples; {len(manifest)} input files.")
    for r in runs:
        s = r["summary"]
        print(f"{r['model']:17s} {r['label']:28s} CnB(0)={s['CnB0_mean_600_700_cm']:.3f} cm^-1; "
              f"D(600)={s.get('D_600_a2_per_fs', 'missing')}")


if __name__ == "__main__":
    main()
