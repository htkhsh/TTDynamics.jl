# Periodic lattice Fröhlich Brownian HEOM-TT example

This example replaces each local Holstein coupling projector with
`S_m = sum_n f_mn |n><n|`, where the periodic-distance kernel is
`k_mn = (d_mn^2 + 1)^(-3/2)` and each electronic-site column is normalized so
`sum_m f_mn^2 = 1`. Therefore `reorganization_energy_cm` retains the same
site-local meaning as in the Holstein example.

The authoritative setup currently uses sibling working copies. From the
TTDynamics repository root, develop all four local packages:

```bash
DEV_ROOT="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
julia --project=examples/lattice_frohlich -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=path) for path in ARGS]); Pkg.instantiate()' "$PWD" "$DEV_ROOT/TTSolver" "$DEV_ROOT/HEOMKit" "$DEV_ROOT/QFiND"
```

A remote-only setup requires published, mutually compatible post-rename
revisions of TTDynamics, TTSolver, HEOMKit, and QFiND. This example does not
claim a compatible remote revision set until those revisions are published;
older pinned revisions predate the HEOMKit rename.

Run the calculation with:

```bash
julia --project=examples/lattice_frohlich examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl
```

The calculation writes one CSV and three PNG diagnostics with the
`lattice_frohlich_brownian_` prefix. Converge the Padé order, TPSD tolerance,
hierarchy local size, time step, and TT tolerances before using results as
production data.
