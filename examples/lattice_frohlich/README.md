# Periodic lattice Fröhlich HEOM-TT example

This example follows the same two-stage workflow and configuration style as
`examples/holstein`: first equilibrate a HEOM-TT state, then reload it for a
current-correlation calculation. The only model-specific part is the
nonlocal Fröhlich coupling.

## Setup and run

From the TTDynamics repository root, instantiate the example environment with
the local sibling packages when necessary:

```bash
julia --project=examples/lattice_frohlich -e 'using Pkg; Pkg.instantiate()'
```

Run the stages in order:

```bash
julia --project=examples/lattice_frohlich examples/lattice_frohlich/equilibrate.jl
julia --project=examples/lattice_frohlich examples/lattice_frohlich/current_correlation.jl
```

Equilibration writes a TT checkpoint, metadata, CSV diagnostics, and a plot.
It also writes a separate equilibration-rank plot.
The correlation stage reloads the checkpoint and writes the current
correlation and rank diagnostics. Both stages use `output/` by default.
Set `output_directory` in `config.jl` to change the directory; relative paths
are resolved from `examples/lattice_frohlich/`.

## Configuration

`config.jl` exposes one `DEFAULT_CONFIG` named tuple, matching the Holstein
example. The physical and numerical defaults are:

```text
output_directory = "output"
site_count = 7
site_energies_cm = zeros(7)
hopping_cm = 1000.0
brownian_frequency_cm = 100.0
brownian_damping_cm = 180.0
reorganization_energy_cm = 200.0
frohlich_range = 1.0
temperature_K = 300.0
blas_threads = 1
decomposition_method = :tpsd       # or :psd
initial_condition = :localized       # or :gibbs
initial_site = 1
time_step_fs = 1.0
pade_order = 8
tpsd_tolerance = 2e-2
pade_type = :Nm1
hierarchy_local_size = 4
temporal_basis_size = 3
equilibration_tamen_tolerance = 2e-2
correlation_tamen_tolerance = 2e-2
operator_tolerance = 1e-10
state_rounding_tolerance = 1e-10
sweep_count = 3
local_iterations = 5
kick_rank = 4
progress_interval = 10
```

The stage durations are selected by the entry points through
`EQUILIBRATION_TIME_FS` and `CORRELATION_TIME_FS`; they are not duplicated in
the model configuration. `blas_threads` controls the process-wide BLAS thread
count. Set `decomposition_method` to `:tpsd` (default) or `:psd`; the
`tpsd_tolerance` setting is used only by `:tpsd`. The two propagation stages
intentionally have separate tAMEn tolerances.

## Fröhlich interaction

For one LO branch, the translated coupling profile is

```text
w_r = [1 + (r / frohlich_range)^2]^(-3/2)
f_r = w_r / sqrt(sum_r w_r^2)
```

where `r` is the shortest periodic lattice distance. For every translated
bath index `l`, the electronic coupling operator is

```text
S_l = sum_j f_(j-l) n_j
```

Thus `frohlich_range` is the only interaction-specific control. The local
Holstein limit is obtained by taking a very short range. Each translated
index has an independent bath copy with the same Brownian spectral density.

The Brownian spectral density is decomposed with QFiND's TPSD routine and
passed to HEOMKit as exponential bath terms. `reorganization_energy_cm`,
`temperature_K`, and the Brownian frequency and damping determine that bath.

## Diagnostics

Equilibration reports step, time, subsystem trace error, maximum TT rank, and
mean TT rank. Trace error is `abs(trace - 1)`; the trace-preserving observable
is supplied to tAMEn as a conserved observable. Both stages use
`correct_2_norm=false` and do not apply post-step normalization; the requested
tAMEn tolerance controls the numerical conservation error. Population and
all unique coherence channels are included in the equilibration plot. The
correlation CSV and plots report the complex current correlation together with
rank diagnostics.

For production calculations, converge the physical discretization and the
HEOM-TT controls (`pade_order`, `tpsd_tolerance`, hierarchy size, time step,
and both tAMEn tolerances) rather than relying on the defaults.

## Equal-time electron–lattice correlation

`equilibrate.jl` also samples

```text
C_nB(r,t) = sum_j [<n_j(t) B_((j+r) mod N)(t)> - <n_j(t)><B_((j+r) mod N)(t)>]
```

for all directed periodic distances `r = 0,...,N-1`, including the initial
time. There is no division by `N`. Here `B_l` is the independent translated
bath coordinate in `H_int = sum_l S_l B_l`, including its coupling strength;
the Fröhlich weights in `S_l` are not multiplied into this observable again.
This measures the bath displacement around the electron, rather than the
weighted potential `sum_l f_(j-l) B_l` at an electronic site.

The bath moment is reconstructed from first-tier scaled ADOs as
`R_l = sum_(k in bath l) sqrt(noise.abs_coeff[k]) * rho[e_k]`.
The joint moment `R_l[j,j]` and bath mean `Tr(R_l)` are normalized by
the root trace, and their disconnected product is subtracted at each time.
Every hierarchy local size must be at least two. Units are `fs^-1` for both
`B_l` and `C_nB`; divide by `HEOMKit.icm2ifs` for `cm^-1`. These coordinates
are not bare dimensionless phonon displacements.

The output directory gains `lattice_frohlich_electron_lattice_correlation.csv`
and `lattice_frohlich_electron_lattice_correlation.png`. The CSV has one row
per `(time_fs, distance)` and retains real and imaginary connected and
uncentered correlations. The heatmap displays the real connected part.
The additional `lattice_frohlich_electron_lattice_correlation_r0_r1_r2.png`
overlays the real time traces for `r=0`, `r=1`, and `r=2`. The CSV retains the
imaginary parts for numerical diagnostics.
The equilibration result exposes `lattice_distances`,
`electron_lattice_correlation`, `electron_lattice_uncentered_correlation`
(distance × time), and `bath_mean` (bath × time).

Equilibrium metadata records the Hamiltonian, bath, temperature, Fröhlich
range, and spectral-decomposition parameters. The current-correlation script
checks them against the current config and reports every mismatch before
building the correlation problem. For older checkpoints without these
parameters, the script warns and continues without that check. The correlation
time step may be changed independently.

To measure a saved physical HEOM checkpoint without propagation:

```julia
include("examples/lattice_frohlich/equilibrate.jl")
decomposition = decompose_bath(DEFAULT_CONFIG)
problem = build_problem(DEFAULT_CONFIG, decomposition)
state = load_tt_binary(joinpath("examples/lattice_frohlich", DEFAULT_CONFIG.output_directory,
                               "lattice_frohlich_equilibrium.ttbin"))
measurement = electron_lattice_correlation(state, problem.system)
measurement.correlation  # entries correspond to measurement.distances
```

Use the checkpoint's model and decomposition. Equal-time measurements use
the physical equilibration trajectory, whereas `current_correlation.jl`
propagates the current-inserted source `J rho_eq`.
