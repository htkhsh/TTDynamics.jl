# Self-Contained Holstein-Family Examples Design

## Goal

Refactor the periodic Holstein, lattice Fröhlich, and Holstein current-
correlation examples so their executable scripts are as easy to read as the
spin-boson HEOM-TT example. Each example owns its configuration and
model-specific implementation; no example includes files from another
example directory.

## Scope

This refactor covers:

- `examples/holstein`;
- `examples/lattice_frohlich`;
- `examples/holstein_current_correlation`;
- tests that exercise these examples.

The refactor preserves current numerical behavior, output basenames,
checkpoint metadata, overwrite behavior, and include guards. Existing
generated CSV, PNG, checkpoint, Manifest, and output-directory contents are
not modified or committed.

## Design Priorities

1. A reader can understand each executable from imports, includes, defaults,
   and a short `main` flow.
2. Each example directory can be copied, modified, and executed without
   including source from another example directory.
3. Configuration and model construction remain model-specific.
4. Duplication between examples is accepted deliberately; independence is
   more important than deduplicating example code.
5. The TTDynamics package API is not expanded solely to support this
   refactor.

## Directory Structure

### Periodic Holstein

`examples/holstein` contains:

- `config.jl`: `HolsteinConfig`, its constructor, and validation;
- `model.jl`: periodic single-excitation Hamiltonian, site projectors,
  Brownian TPSD decomposition, and Holstein HEOM-TT problem construction;
- `dynamics.jl`: measurement, progress reporting, tAMEn propagation, and CSV
  serialization;
- `plotting.jl`: population, trace, and TT-rank figures;
- `holstein_brownian_heomtt.jl`: imports, local includes, visible default
  configuration, short orchestration, and executable guard.

The existing `utils.jl` is retired after its responsibilities have moved to
the focused files above.

### Lattice Fröhlich

`examples/lattice_frohlich` contains:

- `config.jl`: `LatticeFrohlichConfig` and validation;
- `model.jl`: periodic Hamiltonian, normalized Fröhlich kernel and coupling
  operators, Brownian TPSD decomposition, and Fröhlich HEOM-TT construction;
- `dynamics.jl`: Fröhlich-owned measurement, progress, propagation, and CSV;
- `plotting.jl`: Fröhlich-owned diagnostics;
- `lattice_frohlich_brownian_heomtt.jl`: imports, local includes, visible
  defaults, short orchestration, and executable guard.

It no longer includes either `../holstein/utils.jl` or
`../holstein/holstein_brownian_heomtt.jl`.

### Holstein Current Correlation

`examples/holstein_current_correlation` contains:

- `config.jl`: `HolsteinCurrentCorrelationConfig` and validation;
- `model.jl`: the periodic Holstein Hamiltonian, projectors, Brownian TPSD,
  and HEOM-TT construction required by equilibration and correlation;
- `utils.jl`: checkpoint metadata, correlation-specific measurement,
  serialization, progress, and validation helpers;
- `equilibrate.jl`: thin fixed-time equilibration orchestration;
- `current_correlation.jl`: thin state reload and correlation orchestration;
- `plotting.jl`: current-correlation-owned population, current, and rank
  plots.

It no longer includes any file under `examples/holstein`.

## Configuration Ownership

The three setting types are distinct:

- `HolsteinConfig`;
- `LatticeFrohlichConfig`;
- `HolsteinCurrentCorrelationConfig`.

They may contain structurally similar fields, but no type, constructor, or
default instance is shared across example directories. Each model builder
accepts only its local configuration type.

The current uncommitted values in `examples/holstein/utils.jl` are the
behavioral authority for this migration. In particular, the refactor retains
the working-tree values such as `tpsd_tolerance = 2e-2` and
`validation_sample_count = 200`, rather than restoring older expectations
from tests. Tests are updated to assert the current values. Other current
working-tree defaults are likewise preserved exactly.

## Interfaces and Naming

Public example functions use model-specific names to avoid collisions when
tests load multiple examples:

- `decompose_holstein_bath`, `build_holstein_model`,
  `run_holstein_dynamics`, and `save_holstein_results`;
- `decompose_lattice_frohlich_bath`, `build_lattice_frohlich_model`,
  `run_lattice_frohlich_dynamics`, and
  `save_lattice_frohlich_results`;
- `decompose_current_correlation_bath`,
  `build_current_correlation_model`, plus existing equilibration and
  correlation entry points with their configuration type updated.

Internal helpers are also model-qualified where loading multiple examples in
one Julia process would otherwise redefine methods or constants.

Each executable follows the same visible flow:

```julia
include("config.jl")
include("model.jl")
include("dynamics.jl")
include("plotting.jl")

const DEFAULT_CONFIG = ModelSpecificConfig(...)

function main(config=DEFAULT_CONFIG)
    decomposition = decompose_model_bath(config)
    model = build_model(config, decomposition)
    result = run_model_dynamics(config, model)
    save_model_results(config, result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

The exact function names are model-qualified as listed above. Executable
scripts target roughly 50–100 lines; clarity of orchestration is the binding
requirement rather than a hard line-count limit.

## Behavior Preservation

The refactor does not intentionally change:

- Hamiltonian matrix elements or unit conversions;
- Brownian spectral-density or TPSD semantics;
- bath coupling operators;
- HEOM-TT dimensions or hierarchy sizes;
- initial states and observables;
- tAMEn options and time grids;
- current-correlation equilibrium metadata and checkpoint validation;
- CSV schemas, plot contents, output basenames, and overwrite behavior;
- executable include guards.

Current-correlation checkpoints retain
`heom_representation = "twin-space-v1"` and remain compatible with files
created before the source split.

## Error Handling

Each local configuration constructor validates its own physical and numerical
inputs. Model builders validate bath decomposition output and model-specific
invariants. File publishing and checkpoint validation retain their current
errors and atomicity behavior.

Including any executable defines functions and constants only. It must not
start a simulation, create directories, or write data or plots.

## Migration Strategy

1. Add characterization tests for current working-tree defaults, matrices,
   coupling operators, HEOM-TT dimensions, initial states, metadata, output
   paths, and include guards.
2. Split the periodic Holstein example into local focused files without
   changing behavior.
3. Make the lattice Fröhlich example fully local, copying the small required
   Holstein-like configuration and dynamics logic under Fröhlich-specific
   names.
4. Make current correlation fully local, copying its required Holstein model
   configuration and construction under current-correlation-specific names.
5. Remove cross-example includes and retired utility files.
6. Run isolated subprocess imports, focused characterization tests, and the
   package suite.

## Testing

Tests verify:

- all current configuration defaults, including working-tree adjustments;
- Hamiltonians, site projectors, Fröhlich coupling operators, bath
  decompositions, HEOM-TT dimensions, and initial-state traces;
- short deterministic propagation or equivalent matrix/state baselines for
  Holstein and lattice Fröhlich;
- current-correlation metadata, checkpoints, equilibration, propagation, and
  overwrite behavior;
- unchanged CSV schemas, output paths, and plot publication contracts;
- isolated subprocess inclusion of each executable;
- absence of output side effects during inclusion;
- absence of cross-example include paths such as `../holstein/`;
- no obsolete references to retired `utils.jl` files.

Generated user outputs remain untouched during testing. Tests use temporary
directories for writes.

## Completion Criteria

The refactor is complete when each of the three example directories is
self-contained, each executable reads as a short orchestration script, current
working-tree defaults and numerical behavior are preserved, cross-example
includes are absent, focused tests pass, and no failure in broader
verification is caused by the refactor.
