# Lattice Fröhlich Brownian HEOM-TT Example Design

## Goal

Add a standalone `examples/lattice_frohlich` calculation that extends the
existing periodic single-excitation Holstein example to a one-dimensional
lattice Fröhlich model with long-range diagonal electron–phonon coupling.
Keep the electronic Hamiltonian, independent Brownian baths, TPSD
decomposition, HEOM-TT representation, observables, and tAMEn propagation
consistent with the Holstein example so the coupling operators are the only
physical model change.

## Scope

The implementation adds:

- a new `examples/lattice_frohlich` environment and executable;
- periodic-distance, normalized Fröhlich-kernel, and coupling-operator
  utilities;
- model documentation and separate output names;
- focused unit and construction tests in the TTDynamics test suite.

The existing Holstein example remains behaviorally unchanged. Current
uncommitted Holstein data, plots, and unrelated package-renaming work are not
modified by this feature.

## Model

The electronic system is a periodic one-dimensional lattice with one
excitation, site energies `epsilon_n`, and nearest-neighbor hopping `J`. Its
Hamiltonian is the existing periodic Holstein Hamiltonian.

For bath site `m` and electronic site `n`, define the periodic distance

```math
d_{mn} = \min\left(|m-n|, N-|m-n|\right)
```

and raw lattice Fröhlich kernel

```math
k_{mn} = \left(d_{mn}^2 + 1\right)^{-3/2}.
```

Normalize the kernel in the bath-site direction:

```math
f_{mn} = \frac{k_{mn}}{\sqrt{\sum_{m'} k_{m'n}^2}},
\qquad
\sum_m f_{mn}^2 = 1.
```

Translation symmetry makes the normalization factor independent of `n`.
This convention preserves the meaning of `reorganization_energy_cm` as the
total reorganization energy experienced by one localized electronic site,
making it directly comparable to the Holstein example.

Each independent Brownian bath at lattice site `m` couples through

```math
S_m = \sum_n f_{mn} |n\rangle\langle n|.
```

The matrices `S_m` are real diagonal Hermitian matrices related by cyclic
translation. The Brownian spectral density and its TPSD exponential expansion
are identical for every bath.

## File Structure and Reuse

Create the following files:

- `examples/lattice_frohlich/Project.toml`
- `examples/lattice_frohlich/README.md`
- `examples/lattice_frohlich/utils.jl`
- `examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl`
- `test/lattice_frohlich.jl`

The new example includes the existing `examples/holstein/utils.jl` and
`examples/holstein/holstein_brownian_heomtt.jl` definitions. It reuses
`HolsteinConfig`, the periodic Hamiltonian, Brownian TPSD decomposition,
measurement, time propagation, CSV generation, and plotting patterns.

The Fröhlich executable provides a distinct problem builder that replaces the
Holstein site projectors with `S_m`. It must not override or modify
`build_holstein_heomtt`. The executable guard uses
`abspath(PROGRAM_FILE) == @__FILE__`, so including the file only defines its
API and performs no simulation or output writes.

## Data Flow

1. Construct and validate `HolsteinConfig` using the existing defaults.
2. Decompose the configured Brownian bath correlation function with the
   existing `decompose_brownian_bcf` function.
3. Build the periodic electronic Hamiltonian.
4. Generate the normalized kernel matrix and one diagonal `S_m` per bath.
5. Construct `BathExp` objects with the shared exponential expansion and
   individual `S_m` operators.
6. Combine the baths in `NoiseExp`, construct `HEOMTTSystem`, and build the
   Liouvillian and observables.
7. Propagate the initial localized state with the existing tAMEn workflow.
8. Write Fröhlich-specific CSV and PNG diagnostics.

Output basenames use the `lattice_frohlich_brownian_` prefix, including
population, trace, and TT-rank diagnostics, so they cannot overwrite Holstein
outputs.

## Validation and Errors

The Fröhlich utilities reject site counts below two and non-finite kernel
parameters. Construction validates that:

- every kernel value is finite and positive;
- every normalization factor is finite and strictly positive;
- each electronic-site column has unit squared norm within floating-point
  tolerance;
- coupling matrices are diagonal and Hermitian;
- the coupling matrices obey cyclic translation symmetry.

Errors use `ArgumentError` for invalid caller input and explicit construction
errors for violated internal invariants. Existing configuration and TPSD
validation remains authoritative for all shared physical and numerical
parameters.

## Testing

Focused tests cover:

- periodic distances, including wraparound at lattice endpoints;
- positive, symmetric, circulant raw and normalized kernels;
- unit column squared norms of the normalized kernel;
- diagonal, Hermitian, cyclically shifted coupling operators;
- two-site and representative odd/even lattice sizes;
- rejection of invalid sizes and non-finite inputs;
- the local-kernel limit, which reproduces the Holstein site projectors;
- construction of a small Fröhlich `HEOMTTSystem` from an artificial
  exponential bath decomposition;
- expected HEOM-TT dimensions, initial state, and observables;
- an executable include guard that produces no output.

Verification runs the focused test first, then the full TTDynamics suite, and
finally imports the new example environment with local `TTDynamics`,
`TTSolver`, `HEOMKit`, and `QFiND` development paths.

## Completion Criteria

The feature is complete when the new directory is independently runnable,
the normalized long-range operators implement the equations above, output
names are isolated from Holstein outputs, focused tests pass, the existing
test suite has no feature-caused regression, and the new environment imports
all local dependencies successfully.
