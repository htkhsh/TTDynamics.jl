# Reloaded periodic lattice Fröhlich current correlation

This Julia 1.11-or-newer workflow first propagates a periodic lattice
Fröhlich HEOM-TT state for a fixed equilibration time, then reloads its
checkpoint to calculate an unsymmetrized particle-current correlation. Its
separate environment keeps QFiND and CairoMakie out of the `TTDynamics`
library dependencies.

## Setup

Run the following from the TTDynamics repository root. For local sibling
checkouts, develop the primary checkout and the three sibling repositories in
one `Pkg.develop` transaction, then instantiate the example environment:

```bash
DEV_ROOT="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
julia --project=examples/lattice_frohlich_current_correlation -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=path) for path in ARGS]); Pkg.instantiate()' "$PWD" "$DEV_ROOT/TTSolver" "$DEV_ROOT/HEOMKit" "$DEV_ROOT/QFiND"
```

This creates an environment-local `Manifest.toml`, which is ignored and must
not be committed.

## Run

Run equilibration before the correlation calculation. Both stages use the
same `output/` directory:

```bash
julia --project=examples/lattice_frohlich_current_correlation \
  examples/lattice_frohlich_current_correlation/equilibrate.jl
julia --project=examples/lattice_frohlich_current_correlation \
  examples/lattice_frohlich_current_correlation/current_correlation.jl
```

Equilibration writes these four outputs:

- `output/lattice_frohlich_equilibration.csv`
- `output/lattice_frohlich_equilibrium.ttbin`
- `output/lattice_frohlich_equilibrium_metadata.toml`
- `output/lattice_frohlich_equilibration_populations.png`

The correlation stage reloads and validates the TT checkpoint and metadata,
then writes these three outputs:

- `output/lattice_frohlich_current_correlation.csv`
- `output/lattice_frohlich_current_correlation.png`
- `output/lattice_frohlich_current_correlation_ranks.png`

## Definition and units

The periodic Fröhlich kernel starts from
`k_mn = (d_mn^2 + 1)^(-3/2)`, where `d_mn` is the periodic lattice distance.
Each electronic-site column is normalized, so
`sum_m f_mn^2 = 1`, and the coupling operators are
`S_m = sum_n f_mn |n><n|`. This preserves the site-local interpretation of
the reorganization energy.

For hopping `h` in cm^-1, the Hermitian particle-current operator is expressed
in fs^-1 as `J = i h * icm2ifs (|n+1><n| - |n><n+1|)` on each periodic
nearest-neighbor bond (with the duplicate two-site bond omitted). The reported
complex, unsymmetrized current correlation is

```text
C(t) = Tr[J exp(L t)(J rho_eq)].
```

The CSV stores time in fs; real and imaginary `C(t)` components in fs^-2; and
maximum and mean TT ranks. The current-correlation PNG uses the same fs^-2
vertical units, while the rank PNG reports those rank diagnostics over time.
`J rho_eq` is a current source rather than a density matrix, so it is not
trace-normalized.

The checkpoint is equilibrated for a fixed 1000 fs, not a state automatically
certified to be stationary. Before using results as a production calculation,
converge the equilibration time, Padé order, TPSD tolerance, hierarchy local
size, time step, and TT truncation/solver tolerances.
