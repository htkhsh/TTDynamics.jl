# Holstein Equilibration and Current Correlation Design

## Goal

Add a separate, readable HEOM-TT example that:

1. propagates the existing periodic one-dimensional Brownian Holstein model
   for a fixed 1000 fs equilibration interval;
2. normalizes and saves the final HEOM-TT state in TTDynamics' versioned
   binary format; and
3. loads that state in a separate process and computes the unsymmetrized
   equilibrium current correlation function for 0:1:200 fs.

The example will live in `examples/holstein_current_correlation/`, separate
from the existing population-dynamics example. It reuses the existing
Holstein configuration, QFiND TPSD decomposition, and HEOM-TT model builder by
including `examples/holstein/holstein_brownian_heomtt.jl`. That file's program
guard prevents its 100 fs population calculation from running on include.

The fixed-time final state is described as a "1000 fs equilibrated state," not
as a numerically proven stationary or exact thermal state. Automatic
steady-state detection is out of scope.

## Physical Definition

With unit lattice spacing and charge, the single-particle current operator for
the default ring (`site_count > 2`) is

```text
J_op = i J_fs sum_n (|n+1><n| - |n><n+1|),
J_fs = hopping_cm * icm2ifs.
```

The periodic bond from the final site to the first site is included exactly
once. For the existing model's special two-site case, there is one `1 -> 2`
bond and no duplicate closing bond, matching
`periodic_holstein_hamiltonian`. `J_op` is Hermitian. Since it is expressed in
inverse femtoseconds, the current correlation has units of `fs^-2` under these
conventions.

The calculated quantity is the unsymmetrized correlation

```text
C_JJ(t) = Tr[J_op exp(L t)(J_op rho_eq)].
```

TTDynamics uses a system density-vector convention in which left
multiplication by `J_op` is represented on the system core by
`kron(J_op, I)`. The HEOM-space source state is therefore produced by a
TT-matrix with that system superoperator and identity matrices on every
hierarchy core. The current observable uses `vec(J_op)` on the system core
and the hierarchy vacuum basis vector on every hierarchy core, so it measures
only the root ADO. Because `tt_dot` conjugates its first argument and `J_op` is
Hermitian, this contraction equals `Tr(J_op rho_root)`.

## Directory and Components

### `examples/holstein_current_correlation/equilibrate.jl`

This executable will:

- construct the existing default five-site Holstein Brownian HEOM-TT problem;
- start from the existing localized density state at site 1;
- propagate with the existing Crank-Nicolson tAMEn settings in 1 fs steps for
  1000 fs;
- record time, populations, trace, maximum TT rank, and mean TT rank;
- divide the complete HEOM-TT state, including all ADOs, by its final root
  trace;
- reject a zero or nonfinite normalization trace;
- save the normalized state to
  `output/holstein_equilibrium.ttbin` with `save_tt_binary`; and
- write `output/holstein_equilibrium_metadata.toml` plus an equilibration CSV.

The executable refuses to overwrite the binary state by default. A clearly
named configuration flag or command entry point may enable explicit
replacement; silent replacement is prohibited.

### `examples/holstein_current_correlation/current_correlation.jl`

This executable independently reconstructs the same bath decomposition,
HEOM-TT system, Liouvillian, current source operator, and current observable.
It loads the saved binary state and metadata, validates them, constructs
`J_op rho_eq`, rounds the source state with the configured state tolerance,
and propagates it for 200 steps of 1 fs.

It writes:

- `output/holstein_current_correlation.csv`, with columns `time_fs`,
  `correlation_real_fs^-2`, `correlation_imag_fs^-2`, `max_rank`, and
  `mean_rank`; and
- `output/holstein_current_correlation.png`, showing real and imaginary parts
  without discarding either component.

The returned in-memory result contains the time vector, complex correlation
vector, and TT-rank diagnostics.

### `examples/holstein_current_correlation/utils.jl`

Focused helpers will own:

- one Crank-Nicolson tAMEn propagation step;
- fixed-step state propagation with an observation callback;
- construction of the periodic dense current operator;
- construction of the HEOM-TT left-current operator and root-current
  observable;
- metadata serialization and exact compatibility checks; and
- equilibration and correlation CSV output.

Helpers will accept the problem/config they need rather than relying on mutable
global state. Plotting and executable orchestration stay in the two scripts.

### Environment and Documentation

The directory will contain its own `Project.toml` and `README.md`. Its direct
dependencies match what its scripts import. The README will give setup and two
separate run commands, state the physical current convention and correlation
definition, list output files, and warn users to converge the hierarchy,
Padé/TPSD, time-step, equilibration-time, and TT truncation settings before
production use.

## State Metadata and Validation

The binary TT format intentionally contains structural data only. A TOML
sidecar will record the physical and hierarchy identity needed to avoid using a
valid TT state with the wrong Liouvillian:

- format identifier and metadata version;
- equilibration duration and time step;
- site count and site energies;
- hopping, Brownian frequency, QFiND damping, reorganization energy, and
  temperature;
- Padé order/type and TPSD tolerance;
- hierarchy local size;
- the number of fitted exponential terms and their exponents and
  coefficients; and
- the saved TT dimensions.

Complex exponents and coefficients will be represented as parallel real and
imaginary arrays because TOML has no native complex scalar.

The correlation loader will reject:

- missing state or metadata files;
- an object that is not a `TTTensor`;
- a metadata identifier or version it does not support;
- any physical, TPSD, hierarchy, decomposition, or TT-dimension mismatch with
  the reconstructed problem; and
- nonfinite metadata values.

Floating-point fields produced by the same configuration and decomposition are
compared with a strict documented numerical tolerance, rather than formatted
decimal strings. Error messages name the mismatched field.

## Propagation and Output Behavior

Both executables use the existing tAMEn Crank-Nicolson construction:

1. tensor the current spatial state with the odd temporal basis;
2. solve one interval with `config.time_step_fs * liouvillian`;
3. extract the endpoint snapshot; and
4. TT-round the result with `state_rounding_tolerance`.

The common propagator returns the actual final `TTTensor`; unlike the existing
population example, the state is not discarded after diagnostics are
collected. Progress output is limited by `progress_interval`.

The equilibration CSV is diagnostic evidence only. The correlation script does
not infer convergence from it. Correlation values remain complex throughout
the computation and are split only when written or plotted.

## Error Handling

Configuration validation and the TPSD validation from the existing Holstein
example run before propagation. New helpers use `ArgumentError` for invalid
durations, mismatched TT dimensions, wrong loaded object type, invalid
metadata, and nonfinite or zero trace. Filesystem errors and binary-format
errors from `save_tt_binary` and `load_tt_binary` retain their existing
exception behavior.

Output directories are created explicitly. A failed binary save must not
publish a partial state, as guaranteed by the existing binary API. Metadata is
written only after the state save succeeds, through a sibling temporary file
followed by publication, so a partially written TOML file is not presented as
valid metadata.

## Tests

The package test suite will include focused utility tests without running the
default 1000 fs or 200 fs production calculations:

- the periodic current operator has the required sign, includes the closing
  bond once, and is Hermitian;
- for a small two-site HEOM layout, the TT left-current action matches dense
  `J_op * rho` on the root and auxiliary hierarchy components;
- the TT current observable matches dense `Tr(J_op * rho_root)` for a complex
  density matrix;
- a short, small-system propagation returns a final TT state that survives an
  exact binary save/load round trip;
- state trace normalization produces unit root trace and rejects zero or
  nonfinite traces;
- metadata round trips and rejects a changed physical parameter,
  decomposition, or TT dimension;
- the correlation CSV preserves time plus real and imaginary values; and
- importing either executable does not start the 1000 fs or 200 fs run or
  write output files.

The full existing TTDynamics suite will be run after the focused tests.

## Non-Goals

- automatic steady-state convergence detection;
- imaginary-time equilibrium preparation;
- symmetrized, Kubo-transformed, or Fourier-transformed conductivity;
- charge or lattice-spacing unit conversion;
- embedding physical metadata into the generic TT binary format; and
- running the expensive default production trajectories as part of tests.
