# Holstein Example Simplification Design

## Goal

Make `examples/holstein` read like an executable scientific example rather
than a reusable application framework. A reader should be able to follow the
physical and numerical flow from configuration through propagation and output
without tracing callback injection, transactional publication, or exhaustive
checkpoint validation.

The split into `config.jl`, `model.jl`, `utils.jl`, `equilibrate.jl`,
`current_correlation.jl`, and `plotting.jl` remains. The simplification applies
only to the Holstein example; lattice Fröhlich is out of scope.

## Preserved Behavior

- Build the periodic Holstein Hamiltonian and independent site-local Brownian
  HEOM baths from a QFiND TPSD decomposition.
- Select a localized electronic initial state or the bare-system Gibbs state.
- Initialize every HEOM auxiliary mode in its vacuum component.
- Propagate with Crank--Nicolson tAMEn and reuse a `TAMEnWorkspace`.
- Pass the root trace observable through `conserved_observables` on every
  density-state solve, with `correct_2_norm=false`.
- Do not normalize or otherwise correct the propagated state after a step.
- Record all site populations, every unique site-pair coherence, the root
  trace, and TT-rank diagnostics during equilibration.
- Save and reload the equilibrium TT state before calculating the
  unsymmetrized current correlation.
- Preserve the existing CSV and PNG output names and the two-panel
  population/coherence equilibration figure.
- Keep both entry scripts safe to include without starting a calculation.

## Configuration

`config.jl` defines one `DEFAULT_CONFIG` named tuple. It contains the physical
parameters, initial-condition selector, time step, HEOM sizes, and tAMEn
controls. Callers can make a modified configuration with ordinary named-tuple
merge syntax, for example:

```julia
config = merge(DEFAULT_CONFIG, (; initial_condition=:gibbs))
```

There is no custom configuration type and no general-purpose validation API.
Entry points check only conditions needed to avoid an invalid calculation:
the initial-condition symbol, positive stage duration and time step, integral
step count, positive solver sizes, and compatible state dimensions.

## File Responsibilities

### `config.jl`

Defines `DEFAULT_CONFIG`, `EQUILIBRATION_TIME_FS`, and
`CORRELATION_TIME_FS`. It contains no constructors or validation framework.

### `model.jl`

Defines the periodic Hamiltonian, site projectors, TPSD decomposition, HEOM
problem construction, and periodic particle-current operator. It contains
only model construction, not propagation or persistence.

### `utils.jl`

Contains the small set of reusable mechanics needed by both stages:

- localized/Gibbs HEOM initial-state construction;
- one tAMEn Crank--Nicolson step;
- a fixed-step loop with a reused workspace;
- root-density measurement and rank diagnostics;
- minimal checkpoint metadata read/write;
- straightforward CSV writers.

It does not implement atomic file replacement, rollback, overwrite policy,
dependency injection, lazy plotting, world-age workarounds, post-step
normalization, or exhaustive metadata type checking.

### `equilibrate.jl`

Shows the workflow directly: load configuration, decompose the bath, build
the HEOM problem, select the initial state, propagate, print progress, and
write the TT state, minimal TOML, CSV, and PNG. A single small `main` function
and a program-file guard keep inclusion safe.

### `current_correlation.jl`

Shows the reload workflow directly: rebuild the same problem, load the TT
state and TOML, check compatibility, construct `J rho`, propagate, and write
the correlation/rank CSV and PNG files. It also uses a single `main` function
and program-file guard.

### `plotting.jl`

Provides only the equilibration, current-correlation, and rank figures. Entry
scripts include it normally; there is no lazy loader.

## Checkpoint Contract

The equilibrium state remains a TTDynamics TT binary. The TOML sidecar keeps
only:

- `identifier`;
- `version`;
- `site_count`;
- `initial_condition`;
- `equilibration_time_fs`;
- `hierarchy_sizes`;
- `tt_dimensions`.

The reader ignores additional keys, so checkpoints written by the existing
example remain readable when these required values and dimensions agree. The
reader reconstructs the model from the current configuration and verifies the
site count, hierarchy sizes, and TT dimensions. It does not compare every
physical or numerical setting. The README therefore instructs users to rerun
equilibration after changing the model configuration.

Output files are written directly and replace existing files. The example
does not promise transactional recovery if a later output fails.

## Error Handling

Checks sit at workflow boundaries rather than inside every helper. Errors are
reported for an unsupported initial condition, invalid or non-integral time
grid, invalid solver sizes, missing checkpoint files, non-TT checkpoint data,
or dimensions incompatible with the reconstructed HEOM problem. Internal
arrays are constructed together and are not repeatedly length-checked.

## Tests

Tests focus on scientific behavior and the two runnable workflows:

- localized and Gibbs initial densities;
- trace-conserving one-step tAMEn propagation without post-normalization;
- populations and all unique site-pair coherences;
- small-system equilibration, checkpoint reload, and current correlation;
- minimal TOML contents and backward-compatible handling of extra keys;
- CSV columns and real plot generation;
- include guards for both entry scripts.

Tests for deleted implementation machinery are removed: atomic publication,
rollback after injected writer failures, no-clobber races, builder/runner/
plotter injection, lazy loading, and world-age handling.

## Success Criteria

- The entry scripts expose the numerical workflow in execution order.
- `utils.jl` contains only computation and minimal persistence helpers rather
  than an application framework.
- No propagated density state is normalized after a tAMEn step.
- The trace observable is supplied to every density-state tAMEn solve.
- The small-system scientific tests and the package test suite pass.
- The README describes the simplified configuration and checkpoint contract.
