# Periodic Holstein Brownian HEOM-TT example

Run all commands below from the TTDynamics repository root with Julia 1.11 or
newer. The example has its own environment so its decomposition and plotting
packages do not become dependencies of the TTDynamics library.

## Setup

Add known-compatible revisions of TTDynamics and the three unregistered
dependencies in one transaction, then instantiate the remaining registered
dependencies. The remote workflow uses `Pkg.add` because Julia's `Pkg.develop`
does not accept pinned `rev` specifications:

```bash
julia --project=examples/holstein -e '
using Pkg
Pkg.add([
    Pkg.PackageSpec(url="https://github.com/htkhsh/TTDynamics.jl.git", rev="ace749d14ed8dd7eeef74cd21ba00dc11a4ade7b"),
    Pkg.PackageSpec(url="https://github.com/htkhsh/TTSolver.jl.git", rev="7a2eb169648257b9619c14bab349d63798bd220a"),
    Pkg.PackageSpec(url="https://github.com/DOC-Package/HEOMKit.jl.git", rev="06f7939557bce802ea967b6028e9899f55a120dd"),
    Pkg.PackageSpec(url="https://github.com/DOC-Package/QFiND.jl.git", rev="e5d1186cb2f07195c8dd07429861cbe898c9b4da"),
])
Pkg.instantiate()
'
```

`Project.toml` declares every direct dependency and its compatibility range;
the setup command records the local/unregistered sources in the generated
Manifest. If those three repositories are checked out beside the primary
TTDynamics checkout and you want the example to use the local working copies,
develop the four paths instead. `DEV_ROOT` works from either the primary
checkout or a linked Git worktree:

```bash
DEV_ROOT="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
julia --project=examples/holstein -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=path) for path in ARGS]); Pkg.instantiate()' "$PWD" "$DEV_ROOT/TTSolver" "$DEV_ROOT/HEOMKit" "$DEV_ROOT/QFiND"
```

Both setup commands create an environment-local `Manifest.toml`; it is ignored
by this repository and should remain uncommitted.

## Run

After setup, the executable remains a single command:

```bash
julia --project=examples/holstein examples/holstein/holstein_brownian_heomtt.jl
```

The default calculation propagates for 100 fs and writes one CSV file and three
PNG files beside the script. For a quick import check that does not start the
simulation or write outputs, run:

```bash
julia --project=examples/holstein -e 'include("examples/holstein/holstein_brownian_heomtt.jl"); @assert DEFAULT_CONFIG.temporal_basis_size == 3'
```

The TPSD defaults are `pade_order=8`, `tpsd_tolerance=5e-2`, and
`pade_type=:Nm1`. Before treating results as production calculations, converge
the Padé order, TPSD tolerance, hierarchy local size, time step, and TT
tolerances.

## HEOM-TT checkpoint migration

Any application that saves a Holstein HEOM-TT checkpoint must record and
validate this application metadata alongside the TT file:

```text
heom_representation = "twin-space-v1"
```

The generic `TTTensor`/`TTMatrix` binary format is unchanged and does not store
this HEOM-specific field itself. HEOM-TT checkpoints created before
`twin-space-v1` used one `N^2` system core and must be regenerated rather than
relabeled or automatically converted; generic non-HEOM TT binary files remain
unaffected.

## Migrating from the ESPRIT example

Replace `bcf_final_time_fs` with `validation_final_time_fs` and
`bcf_sample_count` with `validation_sample_count`. `bcf_fit_tolerance` has no
one-to-one replacement; choose `pade_order` and `tpsd_tolerance` for the TPSD
decomposition instead.
