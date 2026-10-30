# Periodic Holstein HEOM-TT example

This example is intentionally procedural: `config.jl` defines the default
controls, `model.jl` builds the Hamiltonian and HEOM problem, `utils.jl` holds
the propagation/measurement helpers, and the two short entry scripts run the
calculation.

From the repository root, install the example environment and run:

```bash
julia --project=examples/holstein examples/holstein/equilibrate.jl
julia --project=examples/holstein examples/holstein/current_correlation.jl
```

The first command writes an equilibrated state, minimal metadata, CSV
diagnostics, a population/coherence plot, and an equilibration-rank plot to
`examples/holstein/output/`.
The second command reloads that state, checks its layout against the current
configuration, and writes the current correlation and rank plots.
Set `output_directory` in `config.jl` to change the directory; relative paths
are resolved from `examples/holstein/`. The default is `"output"`.

The default initial state is localized on `initial_site`. Use a merged
`NamedTuple` to select the bare electronic Gibbs state:

```julia
include("examples/holstein/equilibrate.jl")
config = merge(DEFAULT_CONFIG, (initial_condition=:gibbs,))
equilibrate_main(config)
```

The solver tolerances are independent: use
`equilibration_tamen_tolerance` and `correlation_tamen_tolerance` to tune the
two calculations separately.

Set `blas_threads` in the configuration to control the process-wide BLAS
thread count. The default is `1`; this avoids oversubscription with Julia/TT
work. `BLAS.set_num_threads` is applied at the beginning of each entry point.

Set `decomposition_method` to `:tpsd` (default) or `:psd` to select QFiND's
truncated or full Padé spectral decomposition. `tpsd_tolerance` is used only
for `:tpsd`.

tAMEn propagation prints step, time, trace error/correlation, and TT-rank
diagnostics at step zero, the final step, and every `config.progress_interval`
steps.

All HEOM auxiliary modes start in their vacuum component. Each Crank–Nicolson
tAMEn step preserves the supplied HEOM trace observable within the requested
solver tolerance (`correct_2_norm=false`); no post-step normalization is
performed. Equilibration reports every site population, every unique
site-pair coherence, the trace, and maximum/mean TT rank.

The current source is `J ρ_eq`, and the reported unsymmetrized correlation is
`Tr[J exp(L t)(J ρ_eq)]`. Equilibrium metadata records the Hamiltonian, bath,
temperature, and spectral-decomposition parameters. The correlation script
checks them against the current config and reports every mismatch before
building the correlation problem. For older checkpoints without this metadata,
the script warns and continues without that physical-parameter check. The
correlation time step may still be changed independently because it is not a
physical parameter.
To overlay the saved `N=11`, `N=15`, and `N=19` current correlations, run
`julia --project=examples/holstein examples/holstein/plot_current_correlation_sizes.jl`.
The comparison is written to `examples/holstein/output-n19/`.

## Equal-time electron–lattice correlation

`equilibrate.jl` also measures, at every time step including the initial state,

```text
C_nB(r,t) = sum_j [<n_j(t) B_((j+r) mod N)(t)> - <n_j(t)><B_((j+r) mod N)(t)>]
```

All directed periodic distances `r = 0,...,N-1` are retained; there is no
division by `N`. `B_l` is the bath coordinate in `H_int = sum_l n_l B_l`,
including its coupling strength. Its unit, and that of `C_nB`, is `fs^-1`.
Multiply by `1 / HEOMKit.icm2ifs` to express values in `cm^-1`; this is not
a bare dimensionless phonon displacement.

The first-tier ADO contribution is
`R_l = sum_(k in bath l) sqrt(noise.abs_coeff[k]) * rho[e_k]`.
Both the joint moment `R_l[j,j]` and bath mean `Tr(R_l)` are divided by
the root trace before subtracting the disconnected contribution. Thus a
factorized state gives zero even with a nonzero bath mean. All hierarchy
local sizes must be at least two.

The configured output directory gains `holstein_electron_lattice_correlation.csv`
and `holstein_electron_lattice_correlation.png`. The CSV contains one row per
`(time_fs, distance)`, with real and imaginary connected and uncentered
correlations. The plot shows the real connected correlation as a heatmap.
The additional `holstein_electron_lattice_correlation_r0_r1_r2.png` overlays
the real time traces for `r=0`, `r=1`, and `r=2`. The CSV retains the
imaginary parts for numerical diagnostics.
The returned equilibration result contains `lattice_distances`,
`electron_lattice_correlation` and `electron_lattice_uncentered_correlation`
(distance × time), and `bath_mean` (bath × time).

A saved physical HEOM state can be measured without propagation:

```julia
include("examples/holstein/equilibrate.jl")
decomposition = decompose_bath(DEFAULT_CONFIG)
problem = build_problem(DEFAULT_CONFIG, decomposition)
state = load_tt_binary(joinpath("examples/holstein", DEFAULT_CONFIG.output_directory,
                               "holstein_equilibrium.ttbin"))
measurement = electron_lattice_correlation(state, problem.system)
measurement.correlation  # entries correspond to measurement.distances
```

Use the same model and bath decomposition as the checkpoint. This observable
uses the physical density trajectory from equilibration; the `J rho_eq`
trajectory in `current_correlation.jl` computes a different correlation.
