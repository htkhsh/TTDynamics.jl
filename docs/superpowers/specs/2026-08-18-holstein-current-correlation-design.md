# Holstein Equilibration and Current Correlation Design

## Goal

Add a separate, readable HEOM-TT example that:

1. propagates the existing five-site periodic one-dimensional Brownian
   Holstein model for a fixed 1000 fs equilibration interval;
2. normalizes and saves the final twin-space HEOM-TT state in TTDynamics'
   versioned binary format; and
3. loads that state in a separate process and computes the unsymmetrized
   current correlation function for 0:1:200 fs.

The example lives in `examples/holstein_current_correlation/`, separate from
the population-dynamics example. It reuses the existing Holstein
configuration, QFiND TPSD decomposition, and HEOM-TT model construction. A
program guard prevents long propagation or file output when either executable
is included by tests or another script.

The fixed-time final state is described as a "1000 fs equilibrated state," not
as a numerically proven stationary or exact thermal state. Automatic
steady-state detection is out of scope.

## Physical Definition

With unit lattice spacing and charge, the single-particle current operator is

```text
J_op = i J_fs sum_n (|n+1><n| - |n><n+1|),
J_fs = hopping_cm * icm2ifs.
```

The periodic bond from the final site to the first site is included exactly
once. For the model's special two-site case, there is one `1 -> 2` bond and no
duplicate closing bond, matching `periodic_holstein_hamiltonian`. `J_op` is
Hermitian. Since it is expressed in inverse femtoseconds, the current
correlation has units of `fs^-2` under these conventions.

The calculated quantity is the unsymmetrized correlation

```text
C_JJ(t) = Tr[J_op exp(L t)(J_op rho_eq)].
```

The canonical HEOM-TT dimensions are
`[ket(N), bra(N), hierarchy...]`. Left multiplication by `J_op` is represented
by a factorized TT matrix containing `J_op` on the ket core and identities on
the bra and hierarchy cores. No dense `N^2 x N^2` superoperator is formed.

The current observable is a root-ADO twin-space TT tensor. Its ket and bra
cores have bond rank `N` and contract the root density as `Tr(J_op rho_root)`;
every hierarchy core selects its vacuum entry. Correlation values remain
complex throughout propagation and are split into real and imaginary parts
only for tabular and graphical output.

## Directory and Components

### `examples/holstein_current_correlation/equilibrate.jl`

This executable will:

- construct the existing default five-site Holstein Brownian HEOM-TT problem;
- start from the localized density state at site 1;
- propagate with the existing Crank-Nicolson tAMEn settings in 1 fs steps for
  1000 fs;
- record time, populations, trace, maximum TT rank, and mean TT rank;
- divide the complete HEOM-TT state, including all ADOs, by its final root
  trace;
- reject a zero or nonfinite normalization trace;
- save the normalized state to `output/holstein_equilibrium.ttbin` with
  `save_tt_binary`; and
- write `output/holstein_equilibrium_metadata.toml` and
  `output/holstein_equilibration.csv`.

The executable refuses to overwrite the binary state by default. Replacement
requires an explicit option; silent replacement is prohibited.

### `examples/holstein_current_correlation/current_correlation.jl`

This executable independently reconstructs the bath decomposition, HEOM-TT
system, Liouvillian, current source operator, and current observable. It loads
the saved binary state and metadata, validates them, constructs
`J_op rho_eq`, rounds the source state with the configured state tolerance,
and propagates it for 200 steps of 1 fs.

It writes:

- `output/holstein_current_correlation.csv`, with time, real and imaginary
  correlation, maximum rank, and mean rank; and
- `output/holstein_current_correlation.png`, showing both correlation
  components.

The returned in-memory result contains the time vector, complex correlation
vector, and TT-rank diagnostics.

### `examples/holstein_current_correlation/utils.jl`

Focused helpers own:

- one Crank-Nicolson tAMEn propagation step;
- fixed-step propagation with an observation callback;
- construction of the periodic dense current operator;
- construction of the twin-space TT left-current operator and root-current
  observable;
- root-trace normalization;
- metadata serialization and compatibility checks; and
- equilibration and correlation CSV output.

Helpers accept their dependencies explicitly rather than relying on mutable
global state. Plotting and executable orchestration remain in the two scripts.

### Environment and Documentation

The directory contains its own `Project.toml` and `README.md`. Direct
dependencies match script imports. The README gives setup and two separate run
commands, states the current convention and correlation definition, lists
output files, and warns users to converge the hierarchy, Padé/TPSD, time step,
equilibration time, and TT truncation settings before production use.

## State Metadata and Validation

The binary TT format intentionally stores structural data only. A TOML sidecar
records the physical and hierarchy identity needed to avoid using a valid TT
state with the wrong Liouvillian:

- format identifier and metadata version;
- `heom_representation = "twin-space-v1"`;
- equilibration duration and time step;
- site count and site energies;
- hopping, Brownian frequency, QFiND damping, reorganization energy, and
  temperature;
- Padé order/type and TPSD tolerance;
- hierarchy local size;
- the number of TPSD exponential terms and their exponents and coefficients;
  and
- the saved TT dimensions.

Complex exponents and coefficients use parallel real and imaginary arrays
because TOML has no native complex scalar.

The correlation loader rejects:

- missing state or metadata files;
- an object that is not a `TTTensor`;
- an unsupported metadata identifier, version, or HEOM representation;
- any physical, TPSD, hierarchy, decomposition, or TT-dimension mismatch with
  the reconstructed problem;
- the old `[N^2, hierarchy...]` HEOM layout; and
- nonfinite metadata values.

Floating-point fields produced by the same configuration and decomposition are
compared with a strict documented numerical tolerance. Error messages identify
the mismatched field.

## Propagation and Output Behavior

Both executables use the existing tAMEn Crank-Nicolson construction:

1. tensor the current spatial state with the odd temporal basis;
2. solve one interval with `config.time_step_fs * liouvillian`;
3. extract the endpoint snapshot; and
4. TT-round the result with `state_rounding_tolerance`.

The common propagator returns the actual final `TTTensor`. Progress output is
limited by `progress_interval`.

The equilibration CSV is diagnostic evidence only. The correlation script does
not infer convergence from it. Output directories are created explicitly. A
failed binary save cannot publish a partial state, as guaranteed by the binary
API. Metadata is published through a sibling temporary file only after the
state save succeeds.

## Error Handling

Existing Holstein configuration and TPSD validation run before propagation.
New helpers use `ArgumentError` for invalid durations, mismatched twin-space TT
dimensions, wrong loaded object type, invalid metadata, and nonfinite or zero
trace. Filesystem and binary-format errors retain the behavior of
`save_tt_binary` and `load_tt_binary`.

## Tests

The package test suite includes focused tests without running the default
1000 fs or 200 fs trajectories:

- the periodic current operator has the required sign, includes the closing
  bond once, and is Hermitian;
- for a small HEOM layout, the factorized TT left-current action matches dense
  `J_op * rho` on root and auxiliary hierarchy components;
- the twin-space current observable matches dense
  `Tr(J_op * rho_root)` for a complex density matrix;
- non-root ADO entries do not contribute to the current observable;
- a short propagation returns a final TT state that survives an exact binary
  save/load round trip;
- state normalization produces unit root trace and rejects zero or nonfinite
  traces;
- metadata round trips and rejects a changed parameter, decomposition,
  representation, or TT dimension;
- the correlation CSV preserves time plus real and imaginary values; and
- importing either executable does not start a long run or write output.

The full existing TTDynamics suite runs after the focused tests.

## Non-Goals

- automatic steady-state convergence detection;
- imaginary-time equilibrium preparation;
- symmetrized, Kubo-transformed, or Fourier-transformed conductivity;
- charge or lattice-spacing unit conversion;
- embedding physical metadata into the generic TT binary format; and
- running the expensive default trajectories as part of tests.
