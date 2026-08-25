# Quartic local-oscillator charge transfer with multi-core HEOM-TT

This example propagates one electron on an open tight-binding chain. Site
`n` has its own retained quartic oscillator and critically damped Brownian
bath. The electron is one Hilbert-space core of dimension `N`; each local
oscillator is a separate core of dimension `d_keep`. The implementation uses
units with `hbar = k_B = 1`, so energies, inverse times, and temperatures use
one consistent unit and times use its reciprocal.

The system Hamiltonian is

```text
H = sum_n epsilon_n |n><n|
    - J sum_n (|n><n+1| + |n+1><n|)
    + sum_n [Omega_p p_n^2/2 + K2 q_n^2/2 + K4 q_n^4/4]
    + g sum_n |n><n| q_n .
```

Open boundary conditions are used. No bath counter term is added: changing
`bath_lambda`, `bath_gamma`, or `temperature` does not alter the system MPO.

## Environment setup

Julia 1.11 is supported. The authoritative development setup uses sibling
working copies of TTDynamics, TTSolver, HEOMKit, QFiND, and ExpFit. From the
TTDynamics repository root, run:

```bash
DEV_ROOT="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
julia --project=examples/quartic -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=path) for path in ARGS]); Pkg.instantiate()' "$PWD" "$DEV_ROOT/TTSolver" "$DEV_ROOT/HEOMKit" "$DEV_ROOT/QFiND" "$DEV_ROOT/ExpFit"
```

The example-only packages (`ExpFit`, `QuadGK`, and `CairoMakie`) stay in
`examples/quartic/Project.toml`; they are not dependencies of the TTDynamics
library environment.

## Running the examples

The full one-site relaxation and four-case two-site comparison are:

```bash
julia --project=examples/quartic examples/quartic/one_site_relaxation.jl
julia --project=examples/quartic examples/quartic/two_site_transfer.jl
```

For a short two-site smoke run:

```bash
julia --project=examples/quartic -e 'include("examples/quartic/two_site_transfer.jl"); two_site_main(QuarticConfig(final_time=0.1, time_step=0.1, d_raw=6, d_keep=2))'
```

Default outputs are below `examples/quartic/output/`. Existing files are
never replaced unless `overwrite=true` is explicitly passed. For an isolated
quick check without plotting, use a new temporary directory:

```bash
QUARTIC_OUT="$(mktemp -d)"
julia --project=examples/quartic -e 'include("examples/quartic/two_site_transfer.jl"); two_site_main(QuarticConfig(final_time=0.1, time_step=0.1, d_raw=6, d_keep=2, hierarchy_nmax=0, tamen_tolerance=1e-5, sweep_count=4, local_iterations=8, kick_rank=2, trace_tolerance=1e-4, hermiticity_tolerance=1e-4, electron_number_tolerance=1e-4); output_directory=ARGS[1], plotter=nothing)' "$QUARTIC_OUT"
```

Here `hierarchy_nmax=0` checks the complete fit/build/propagate/output plumbing
but suppresses nonzero auxiliary occupations; it is not a finite-bath
convergence result. Use positive hierarchy cutoffs for physical bath dynamics.

The default configurations are demonstrations, not evidence of numerical
convergence.

## Tensor ordering and hierarchy cutoff

Julia column-major vectorization is used locally. Physical Liouville cores
come first, followed by hierarchy cores:

```text
electron: N^2
oscillator 1: d_keep^2, ..., oscillator N: d_keep^2
hierarchy (site 1, pole 1), ..., (site 1, pole K),
          (site 2, pole 1), ..., (site N, pole K)
```

Every hierarchy core has local size `hierarchy_nmax + 1`. This is a product
cutoff, not a global total-tier cutoff: with `N` sites and `K` fitted poles,
the uncompressed hierarchy product contains
`(hierarchy_nmax + 1)^(N*K)` index combinations. TT compression avoids storing
that product densely but does not remove the need to converge the local
cutoff and inspect TT ranks.

## Outputs and diagnostics

The one-site workflow writes `one_site.csv` plus population, `q`, `q2`, bare
oscillator-energy, trace, and TT-rank PNG files. The two-site workflow writes
one CSV for each of `harmonic_no_bath`, `harmonic_bath`, `quartic_no_bath`,
and `quartic_bath`, along with `two_site_population_comparison.png` and
`two_site_metadata.txt`.

Each CSV has these columns:

- `time` and `population_site_n`;
- real and imaginary parts of the root-ADO trace and electron number;
- `hermiticity_error`;
- `oscillator_q_site_n`, `oscillator_q2_site_n`, and
  `oscillator_energy_site_n`;
- `maximum_rank`, `mean_rank`, and the available tAMEn residual/truncation
  fields (blank when the solver does not expose them).

The two-site metadata file records every `QuarticConfig` field for every case,
the common fitted rates, forward/backward coefficients and scaling factors,
and all correlation-fit metadata. Standard output also reports training and
holdout absolute/relative fit errors, the Vandermonde condition number, and
the minimum pole separation. A run should be rejected or refined when values
are nonfinite or when trace, Hermiticity, electron number, fit error, or TT
rank diagnostics violate the requested tolerances.

## Convergence sweeps

`run_quartic_convergence(base_config; parameter, values, runner)` calls the
runner with a newly constructed configuration for every value and returns
labeled final populations, final trace, validation fit error, and maximum TT
rank. For example:

```julia
include("examples/quartic/one_site_relaxation.jl")

base = QuarticConfig(
    site_count=1,
    site_energies=[0.0],
    hopping=0.0,
    final_time=0.2,
    time_step=0.1,
)
runner = config -> one_site_main(
    config;
    output_directory=mktempdir(),
    plotter=nothing,
)
runs = run_quartic_convergence(
    base;
    parameter=:d_keep,
    values=[4, 6, 8],
    runner,
)
```

Converge all seven supported controls independently:

- `d_raw`: raw harmonic-basis size; rebuilds the quartic eigensystem.
- `d_keep`: retained oscillator dimension; rebuilds the mode and all
  downstream TT objects. Every value must satisfy `d_keep <= d_raw`.
- `basis_frequency`: auxiliary raw-basis frequency; rebuilds the projected
  mode and all downstream objects.
- `hierarchy_nmax`: local hierarchy occupation cutoff; rebuilds the HEOM
  system, generator, and initial state.
- `fit_rank`: requested ESPRIT candidate rank (`0` selects by threshold);
  resamples/refits the correlation and rebuilds bath-dependent TT objects.
- `tt_cutoff`: a convergence alias that sets both `operator_tolerance` and
  `state_rounding_tolerance`, rebuilding rounded operators and states.
- `time_step`: Crank--Nicolson step size; `final_time / time_step` must be an
  integer and the dynamics is rerun.

The harness deliberately invokes the complete supplied runner for each case,
so changed basis, fit, hierarchy, generator, initial state, and propagation
objects cannot be accidentally reused. It never mutates `base_config`.

Also vary `tamen_tolerance`, sweep/local-iteration settings, the fit time
window and sampling interval when diagnosing a difficult run, even though
they are not dispatcher axes. Compare populations and oscillator observables
as well as trace/Hermiticity/electron-number errors and maximum ranks.

## ESPRIT and model limitations

The critically damped Brownian correlation is sampled by numerical
quadrature only during preparation. ESPRIT uses a separate validation grid and
fits forward and backward coefficients independently on one conjugate-closed
pole basis. A critical double pole is approximated by nearby simple poles over
the finite fit window. Nearby poles can make the Vandermonde problem poorly
conditioned; they are diagnosed by minimum pole separation and condition
number and are not merged or converted to a Jordan block. Increase the fit
window/resolution and compare ranks rather than interpreting a single fit as
unique.

There is no correlated-equilibrium preparation, analytic Brownian
decomposition, global total-tier truncation, dynamic core reordering, or GPU
path in this example.

The low-level model supports an optional open-chain `N = 3` extension even
though no dedicated executable is supplied:

```julia
include("examples/quartic/one_site_relaxation.jl")
config = QuarticConfig(site_count=3, site_energies=zeros(3))
fit = build_quartic_correlation_fit(config)
mode = build_quartic_mode(config)
model = build_quartic_model(config, mode, fit; initial_site=1)
result = propagate_quartic_heom(model, config)
```

Converge this extension especially carefully because both the hierarchy
product and attainable TT ranks grow with site count and fitted pole count.
