# Reloaded periodic Holstein current correlation

This Julia 1.11-or-newer example first prepares a fixed-time Holstein HEOM-TT
state, then reloads and validates that state to calculate an unsymmetrized
current correlation. Its separate environment keeps QFiND and CairoMakie out
of the `TTDynamics` library dependencies.

## Setup

Run these commands from the TTDynamics repository root. For pinned remote
sources, add the compatible TTDynamics and unregistered dependencies in one
transaction, then instantiate the environment:

```bash
julia --project=examples/holstein_current_correlation -e '
using Pkg
Pkg.add([
    Pkg.PackageSpec(url="https://github.com/htkhsh/TTDynamics.jl.git", rev="0722677b16c526c0e58aa8c4c25bcc0e218f6e9b"),
    Pkg.PackageSpec(url="https://github.com/htkhsh/TTSolver.jl.git", rev="7a2eb169648257b9619c14bab349d63798bd220a"),
    Pkg.PackageSpec(url="https://github.com/DOC-Package/HEOMKit.jl.git", rev="06f7939557bce802ea967b6028e9899f55a120dd"),
    Pkg.PackageSpec(url="https://github.com/DOC-Package/QFiND.jl.git", rev="e5d1186cb2f07195c8dd07429861cbe898c9b4da"),
])
Pkg.instantiate()
'
```

For local sibling checkouts, develop the primary checkout and the three
sibling repositories instead. `DEV_ROOT` works from either the primary checkout
or a linked Git worktree:

```bash
DEV_ROOT="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
julia --project=examples/holstein_current_correlation -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=path) for path in ARGS]); Pkg.instantiate()' "$PWD" "$DEV_ROOT/TTSolver" "$DEV_ROOT/HEOMKit" "$DEV_ROOT/QFiND"
```

Both commands create an environment-local `Manifest.toml`; it is ignored and
must not be committed.

The pinned TTDynamics revision is after the twin-space HEOM migration and
exports the APIs this example needs: `save_tt_binary`, `root_density_matrix`,
and `heom_tt_dimensions`.

## Run

Run equilibration before the correlation calculation. Both commands use the
same `output/` directory:

```bash
julia --project=examples/holstein_current_correlation examples/holstein_current_correlation/equilibrate.jl
julia --project=examples/holstein_current_correlation examples/holstein_current_correlation/current_correlation.jl
```

Equilibration propagates for 1000 fs by default and produces:

- `output/holstein_equilibration.csv`
- `output/holstein_equilibrium.ttbin`
- `output/holstein_equilibrium_metadata.toml`
- `output/holstein_equilibration_populations.png`

The PNG plots each site's population against time and is generated automatically
by `equilibrate.jl`.

The correlation executable requires the latter binary/TOML pair, reconstructs
the decomposition and HEOM problem, validates the state layout, every saved
configuration field, hierarchy dimensions, and complex decomposition data,
then propagates for 200 fs by default. It produces:

- `output/holstein_current_correlation.csv`
- `output/holstein_current_correlation.png`
- `output/holstein_current_correlation_ranks.png`

During propagation, the executable reports step 0, every
`config.progress_interval` steps (10 fs with the defaults), and the final
step. Each line includes the physical time, maximum and mean TT rank, and
elapsed wall-clock time. The complete loaded-equilibrium and final-source TT
rank vectors are also printed.

Direct execution replaces existing example outputs by default, so repeated
runs refresh the CSV, TT/TOML, and PNG files. Programmatic use can pass
`overwrite=false` to `equilibrate_main` or `current_correlation_main` to request
no-clobber checks. Low-level writer functions remain no-clobber by default.
Use a fresh output directory when transactional recovery of every previous
file is required, since a later write failure cannot restore a file already
replaced.

To import both executables without propagating or writing outputs:

```bash
julia --project=examples/holstein_current_correlation -e 'include("examples/holstein_current_correlation/equilibrate.jl"); include("examples/holstein_current_correlation/current_correlation.jl")'
```

## Definition and units

For hopping `h` in cm^-1, the current operator is expressed in fs^-1 as
`J = i h * icm2ifs (|n+1><n| - |n><n+1|)` on each periodic nearest-neighbor
bond (with the duplicate two-site bond omitted). The reported complex
unsymmetrized correlation is

```text
C(t) = Tr[J exp(L t)(J rho_eq)].
```

The CSV columns are time in fs, the real and imaginary parts of `C(t)` in
fs^-2, and maximum/mean TT rank. The PNG shows the real and imaginary
components with the same fs^-2 vertical units. `J rho_eq` is a current source,
not a density matrix, so it is deliberately not trace-normalized.
The separate rank PNG shows the maximum and mean TT rank at every saved time;
these are the same rank diagnostics stored in the CSV.

The saved input is a state equilibrated for a fixed 1000 fs, not a state
automatically certified to be stationary. Before treating either result as a
production calculation, converge the equilibration time, Padé order, TPSD
tolerance, hierarchy local size, time step, and TT truncation/solver
tolerances.
