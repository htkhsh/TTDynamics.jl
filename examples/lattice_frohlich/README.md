# Periodic lattice Fröhlich Brownian HEOM-TT example

This example replaces each local Holstein coupling projector with
`S_m = sum_n f_mn |n><n|`, where the periodic-distance kernel is
`k_mn = (d_mn^2 + 1)^(-3/2)` and each electronic-site column is normalized so
`sum_m f_mn^2 = 1`. Therefore `reorganization_energy_cm` retains the same
site-local meaning as in the Holstein example.

For a remote setup, add the known-compatible revisions and instantiate:

```bash
julia --project=examples/lattice_frohlich -e '
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

From the TTDynamics repository root, develop the sibling working copies:

```bash
DEV_ROOT="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
julia --project=examples/lattice_frohlich -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=path) for path in ARGS]); Pkg.instantiate()' "$PWD" "$DEV_ROOT/TTSolver" "$DEV_ROOT/HEOMKit" "$DEV_ROOT/QFiND"
```

Run the calculation with:

```bash
julia --project=examples/lattice_frohlich examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl
```

The calculation writes one CSV and three PNG diagnostics with the
`lattice_frohlich_brownian_` prefix. Converge the Padé order, TPSD tolerance,
hierarchy local size, time step, and TT tolerances before using results as
production data.
