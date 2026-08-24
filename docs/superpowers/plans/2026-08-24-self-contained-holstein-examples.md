# Self-Contained Holstein Examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the periodic Holstein, lattice Fröhlich, and Holstein current-correlation examples into short orchestration scripts whose configuration, model construction, dynamics, and plotting code are owned locally by each example directory.

**Architecture:** Each example gets a distinct configuration type and model-qualified functions. Small, focused local files are included by a thin executable; cross-example includes are removed even where that intentionally duplicates example code. Existing numerical behavior, files, checkpoint metadata, and executable guards remain unchanged.

**Tech Stack:** Julia, TTDynamics, TTSolver, HEOMKit, QFiND, CairoMakie, Julia `Test`

**Spec:** `docs/superpowers/specs/2026-08-24-self-contained-holstein-examples-design.md`

## Global Constraints

- Do not expand the TTDynamics package API for this refactor.
- Preserve Hamiltonian elements, units, TPSD semantics, bath operators, HEOM-TT sizes, initial states, tAMEn settings, time grids, CSV schemas, plot basenames, overwrite behavior, and include guards.
- Preserve the current working-tree Holstein defaults exactly, including `final_time_fs=500.0`, `tpsd_tolerance=2e-2`, and `validation_sample_count=200`.
- Keep `heom_representation = "twin-space-v1"` and version-1 checkpoint compatibility.
- Do not modify, delete, stage, or commit generated CSV/PNG/checkpoint files, `output/` directories, or example `Manifest.toml` files.
- Tests that write files must use `mktempdir()`.
- Each executable may define functions and constants when included, but must not simulate, create directories, or write outputs.

## File Structure

- `examples/holstein/config.jl`: `HolsteinConfig` and `validate_holstein_config`.
- `examples/holstein/model.jl`: Hamiltonian, projectors, bath decomposition, and HEOM-TT construction.
- `examples/holstein/dynamics.jl`: measurement, tAMEn propagation, and CSV writing.
- `examples/holstein/plotting.jl`: Holstein diagnostic plots and output paths.
- `examples/holstein/holstein_brownian_heomtt.jl`: imports, local includes, defaults, and orchestration only.
- `examples/lattice_frohlich/config.jl`: `LatticeFrohlichConfig` and local validation.
- `examples/lattice_frohlich/model.jl`: periodic model, Fröhlich kernel/operators, bath decomposition, HEOM-TT construction.
- `examples/lattice_frohlich/dynamics.jl`: locally owned measurement, tAMEn propagation, and CSV writing.
- `examples/lattice_frohlich/plotting.jl`: Fröhlich diagnostic plots and output paths.
- `examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl`: imports, local includes, defaults, and orchestration only.
- `examples/holstein_current_correlation/config.jl`: `HolsteinCurrentCorrelationConfig` and local validation.
- `examples/holstein_current_correlation/model.jl`: local periodic Holstein model, bath decomposition, and HEOM-TT construction.
- `examples/holstein_current_correlation/utils.jl`: existing checkpoint and propagation helpers, retargeted to the local type.
- `examples/holstein_current_correlation/equilibrate.jl`: local includes and equilibration orchestration.
- `examples/holstein_current_correlation/current_correlation.jl`: state reload and correlation orchestration.
- `examples/holstein_current_correlation/plotting.jl`: unchanged local plotting responsibility.
- Retire `examples/holstein/utils.jl` and `examples/lattice_frohlich/utils.jl` only after their contents exist in the new focused files.

---

### Task 1: Split the Periodic Holstein Example

**Files:**
- Create: `examples/holstein/config.jl`
- Create: `examples/holstein/model.jl`
- Create: `examples/holstein/dynamics.jl`
- Create: `examples/holstein/plotting.jl`
- Modify: `examples/holstein/holstein_brownian_heomtt.jl`
- Delete: `examples/holstein/utils.jl`
- Modify: `test/runtests.jl`
- Modify: `test/heom_twin_space.jl`

**Interfaces:**
- Produces: `HolsteinConfig(; kwargs...)`, `validate_holstein_config(::HolsteinConfig)`, `periodic_holstein_hamiltonian(::AbstractVector, ::Real)`, `holstein_site_projectors(::Integer)`, `decompose_holstein_bath(::HolsteinConfig)`, `build_holstein_model(::HolsteinConfig, decomposition)`, `measure_holstein_state`, `run_holstein_dynamics(::HolsteinConfig, problem)`, `write_holstein_population_csv(path, result)`, `holstein_output_paths(directory)`, and `save_holstein_results(directory, config, result)`.
- Compatibility is intentional only for file formats and numerics; tests and callers migrate away from `validate_config`, `decompose_brownian_bcf`, `build_holstein_heomtt`, `run_dynamics`, `write_population_csv`, and `save_diagnostic_plots`.

- [ ] **Step 1: Add failing layout, defaults, and side-effect characterization tests**

Replace the Holstein utility include in `test/runtests.jl` with the four focused includes, then assert the current defaults and model-qualified API:

```julia
include("../examples/holstein/config.jl")
include("../examples/holstein/model.jl")
include("../examples/holstein/dynamics.jl")
include("../examples/holstein/plotting.jl")

@testset "Periodic Holstein example layout" begin
    config = HolsteinConfig()
    @test config.final_time_fs == 500.0
    @test config.tpsd_tolerance == 2e-2
    @test config.validation_sample_count == 200
    @test validate_holstein_config(config) === config
    @test length(holstein_site_projectors(config.site_count)) == config.site_count
    @test holstein_output_paths("out").csv ==
          joinpath("out", "holstein_brownian_populations.csv")
end
```

Extend the subprocess example test in `test/heom_twin_space.jl` to snapshot the directory before and after inclusion:

```julia
expression = "before = Set(readdir($(repr(example_directory)))); " *
             "include($(repr(example_path))); " *
             "after = Set(readdir($(repr(example_directory)))); " *
             "@assert before == after; " *
             "@assert isdefined(Main, :run_holstein_dynamics); " *
             "print($(repr(marker)))"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `julia --project=. test/runtests.jl`

Expected: FAIL because `examples/holstein/config.jl` and the model-qualified functions do not exist. Existing unrelated current-correlation output-guard failures may appear later, but the first new failure must be the missing split file/API.

- [ ] **Step 3: Move configuration without changing field order or values**

Move the `HolsteinConfig` struct and keyword constructor verbatim from the current working-tree `examples/holstein/utils.jl` into `config.jl`. Rename validation and the constructor call:

```julia
function HolsteinConfig(; site_count=5, site_energies_cm=zeros(site_count),
    hopping_cm=400.0, brownian_frequency_cm=1400.0,
    brownian_damping_cm=200.0, reorganization_energy_cm=600.0,
    temperature_K=300.0, initial_site=1, final_time_fs=500.0,
    time_step_fs=1.0, pade_order=8, tpsd_tolerance=2e-2,
    pade_type=:Nm1, validation_final_time_fs=100.0,
    validation_sample_count=200, bcf_upper_bound_cm=10_000.0,
    hierarchy_local_size=4, temporal_basis_size=3, tamen_tolerance=2e-2,
    operator_tolerance=1e-10, state_rounding_tolerance=1e-10,
    sweep_count=3, local_iterations=5, kick_rank=4,
    progress_interval=10)
    config = HolsteinConfig(
        Int(site_count), Float64.(site_energies_cm), Float64(hopping_cm),
        Float64(brownian_frequency_cm), Float64(brownian_damping_cm),
        Float64(reorganization_energy_cm), Float64(temperature_K),
        Int(initial_site), Float64(final_time_fs), Float64(time_step_fs),
        Int(pade_order), Float64(tpsd_tolerance), Symbol(pade_type),
        Float64(validation_final_time_fs), Int(validation_sample_count),
        Float64(bcf_upper_bound_cm), Int(hierarchy_local_size),
        Int(temporal_basis_size), Float64(tamen_tolerance),
        Float64(operator_tolerance), Float64(state_rounding_tolerance),
        Int(sweep_count), Int(local_iterations), Int(kick_rank),
        Int(progress_interval))
    return validate_holstein_config(config)
end
```

Keep every existing validation predicate and error message, changing only the signature to `validate_holstein_config(config::HolsteinConfig)`.

- [ ] **Step 4: Move model, dynamics, and plotting responsibilities**

In `model.jl`, move the existing Hamiltonian unchanged, rename `site_projectors` to `holstein_site_projectors`, rename `decompose_brownian_bcf` to `decompose_holstein_bath`, and rename `build_holstein_heomtt` to `build_holstein_model`. The builder must use:

```julia
baths = [BathExp(decomposition.exponents, decomposition.coefficients, projector)
         for projector in holstein_site_projectors(config.site_count)]
system = HEOMTTSystem(H_fs, NoiseExp(baths), config.hierarchy_local_size)
liouvillian, trace_observable, population_observables =
    build_heom_liouvillian(system; tol=config.operator_tolerance)
return (; system, liouvillian, trace_observable, population_observables)
```

In `dynamics.jl`, move the current measurement/progress/propagation/CSV bodies unchanged and rename them to `measure_holstein_state`, `_print_holstein_progress`, `run_holstein_dynamics`, and `write_holstein_population_csv`. In `plotting.jl`, move the plot bodies unchanged and add:

```julia
holstein_output_paths(directory::AbstractString) = (
    csv=joinpath(directory, "holstein_brownian_populations.csv"),
    populations=joinpath(directory, "holstein_brownian_populations.png"),
    trace=joinpath(directory, "holstein_brownian_trace.png"),
    rank=joinpath(directory, "holstein_brownian_rank.png"),
)
```

`save_holstein_results` writes the CSV with `write_holstein_population_csv`, saves the three existing figures, and returns `(; csv_path=paths.csv, plot_paths=(paths.populations, paths.trace, paths.rank))`.

- [ ] **Step 5: Rewrite the executable as orchestration and retire `utils.jl`**

Use local guarded includes and a short main:

```julia
using LinearAlgebra, Statistics
using TTSolver, TTDynamics, QFiND, HEOMKit, CairoMakie
using HEOMKit: icm2ifs

isdefined(@__MODULE__, :HolsteinConfig) || include("config.jl")
isdefined(@__MODULE__, :build_holstein_model) || include("model.jl")
isdefined(@__MODULE__, :run_holstein_dynamics) || include("dynamics.jl")
isdefined(@__MODULE__, :save_holstein_results) || include("plotting.jl")

const DEFAULT_CONFIG = HolsteinConfig()

function main(config::HolsteinConfig=DEFAULT_CONFIG; output_directory=@__DIR__)
    validate_holstein_config(config)
    decomposition = decompose_holstein_bath(config)
    problem = build_holstein_model(config, decomposition)
    result = run_holstein_dynamics(config, problem)
    paths = save_holstein_results(output_directory, config, result)
    return result
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
```

Retain the current informative `println` calls around those four orchestration stages. Delete `examples/holstein/utils.jl` only after `rg 'holstein/utils\.jl|include\("utils\.jl"\)' examples/holstein test` returns no live reference.

- [ ] **Step 6: Run Holstein tests and commit only source/test files**

Run: `julia --project=. -e 'using Test; include("test/heom_twin_space.jl")'`

Run: `julia --project=. test/runtests.jl`

Expected: Holstein configuration/model/import assertions PASS. If the full runner reaches the pre-existing current-correlation output-directory guard, record it without changing user output files.

```bash
git add examples/holstein/config.jl examples/holstein/model.jl \
  examples/holstein/dynamics.jl examples/holstein/plotting.jl \
  examples/holstein/holstein_brownian_heomtt.jl examples/holstein/utils.jl \
  test/runtests.jl test/heom_twin_space.jl
git commit -m "refactor: split periodic Holstein example"
```

### Task 2: Make the Lattice Fröhlich Example Self-Contained

**Files:**
- Create: `examples/lattice_frohlich/config.jl`
- Create: `examples/lattice_frohlich/model.jl`
- Create: `examples/lattice_frohlich/dynamics.jl`
- Create: `examples/lattice_frohlich/plotting.jl`
- Modify: `examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl`
- Delete: `examples/lattice_frohlich/utils.jl`
- Modify: `test/lattice_frohlich.jl`

**Interfaces:**
- Consumes: only installed packages and files in `examples/lattice_frohlich`.
- Produces: `LatticeFrohlichConfig`, `validate_lattice_frohlich_config`, `periodic_lattice_frohlich_hamiltonian`, `periodic_lattice_distance`, `normalized_frohlich_kernel`, `frohlich_coupling_operators`, `decompose_lattice_frohlich_bath`, `build_lattice_frohlich_model`, `run_lattice_frohlich_dynamics`, `write_lattice_frohlich_population_csv`, `lattice_frohlich_output_paths`, and `save_lattice_frohlich_results`.

- [ ] **Step 1: Change tests first to require local ownership**

Replace the Holstein/lattice utility includes at the top of `test/lattice_frohlich.jl` with:

```julia
include("../examples/lattice_frohlich/config.jl")
include("../examples/lattice_frohlich/model.jl")
include("../examples/lattice_frohlich/dynamics.jl")
include("../examples/lattice_frohlich/plotting.jl")
include("../examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl")
```

Replace `HolsteinConfig` with `LatticeFrohlichConfig`, `build_lattice_frohlich_heomtt` with `build_lattice_frohlich_model`, and add:

```julia
@test DEFAULT_LATTICE_FROHLICH_CONFIG isa LatticeFrohlichConfig
@test LatticeFrohlichConfig !== HolsteinConfig
source = read(joinpath(@__DIR__, "..", "examples", "lattice_frohlich",
                       "lattice_frohlich_brownian_heomtt.jl"), String)
@test !occursin("../holstein", source)
@test !occursin("HolsteinConfig", source)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `julia --project=. -e 'using Test; include("test/lattice_frohlich.jl")'`

Expected: FAIL because `config.jl` and `LatticeFrohlichConfig` do not exist.

- [ ] **Step 3: Add the local configuration and complete model**

Copy the complete Holstein field list, constructor conversions, defaults, and validation predicates into `config.jl`, renaming the type and validator:

```julia
struct LatticeFrohlichConfig
    site_count::Int
    site_energies_cm::Vector{Float64}
    hopping_cm::Float64
    brownian_frequency_cm::Float64
    brownian_damping_cm::Float64
    reorganization_energy_cm::Float64
    temperature_K::Float64
    initial_site::Int
    final_time_fs::Float64
    time_step_fs::Float64
    pade_order::Int
    tpsd_tolerance::Float64
    pade_type::Symbol
    validation_final_time_fs::Float64
    validation_sample_count::Int
    bcf_upper_bound_cm::Float64
    hierarchy_local_size::Int
    temporal_basis_size::Int
    tamen_tolerance::Float64
    operator_tolerance::Float64
    state_rounding_tolerance::Float64
    sweep_count::Int
    local_iterations::Int
    kick_rank::Int
    progress_interval::Int
end
```

The keyword defaults and conversions must exactly match Task 1. Move every function from the old lattice `utils.jl` unchanged into `model.jl`, add a locally named periodic Hamiltonian, and copy the complete Brownian decomposition algorithm under `decompose_lattice_frohlich_bath`. Implement the builder as:

```julia
function build_lattice_frohlich_model(config::LatticeFrohlichConfig,
                                       decomposition;
                                       kernel=default_frohlich_kernel)
    validate_lattice_frohlich_config(config)
    H_fs = periodic_lattice_frohlich_hamiltonian(
        config.site_energies_cm, config.hopping_cm) * icm2ifs
    baths = [BathExp(decomposition.exponents, decomposition.coefficients, V)
             for V in frohlich_coupling_operators(config.site_count; kernel)]
    system = HEOMTTSystem(H_fs, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end
```

- [ ] **Step 4: Add local dynamics/plotting and thin driver**

Copy the Task 1 dynamics bodies, qualifying all definitions and calls with `lattice_frohlich`; do not include Holstein files. Move the existing lattice plotting and path bodies into `plotting.jl`, and make `save_lattice_frohlich_results` write the existing CSV and three plots. Rewrite the driver with only local includes:

```julia
isdefined(@__MODULE__, :LatticeFrohlichConfig) || include("config.jl")
isdefined(@__MODULE__, :build_lattice_frohlich_model) || include("model.jl")
isdefined(@__MODULE__, :run_lattice_frohlich_dynamics) || include("dynamics.jl")
isdefined(@__MODULE__, :save_lattice_frohlich_results) || include("plotting.jl")

const DEFAULT_LATTICE_FROHLICH_CONFIG = LatticeFrohlichConfig()

function lattice_frohlich_main(config::LatticeFrohlichConfig=
                               DEFAULT_LATTICE_FROHLICH_CONFIG;
                               output_directory=@__DIR__)
    decomposition = decompose_lattice_frohlich_bath(config)
    problem = build_lattice_frohlich_model(config, decomposition)
    result = run_lattice_frohlich_dynamics(config, problem)
    save_lattice_frohlich_results(output_directory, config, result)
    return result
end
```

Delete the old `utils.jl` after all references are gone.

- [ ] **Step 5: Verify isolation and commit**

Run: `julia --project=. -e 'using Test; include("test/lattice_frohlich.jl")'`

Run: `rg -n '\.\./holstein|HolsteinConfig|decompose_brownian_bcf|run_dynamics' examples/lattice_frohlich test/lattice_frohlich.jl`

Expected: focused tests PASS; `rg` prints no matches.

```bash
git add examples/lattice_frohlich/config.jl examples/lattice_frohlich/model.jl \
  examples/lattice_frohlich/dynamics.jl examples/lattice_frohlich/plotting.jl \
  examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl \
  examples/lattice_frohlich/utils.jl test/lattice_frohlich.jl
git commit -m "refactor: isolate lattice Frohlich example"
```

### Task 3: Make Holstein Current Correlation Self-Contained

**Files:**
- Create: `examples/holstein_current_correlation/config.jl`
- Create: `examples/holstein_current_correlation/model.jl`
- Modify: `examples/holstein_current_correlation/utils.jl`
- Modify: `examples/holstein_current_correlation/equilibrate.jl`
- Modify: `examples/holstein_current_correlation/current_correlation.jl`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: only installed packages and files in `examples/holstein_current_correlation`.
- Produces: `HolsteinCurrentCorrelationConfig`, `validate_current_correlation_config`, `periodic_current_correlation_hamiltonian`, `current_correlation_site_projectors`, `decompose_current_correlation_bath`, and `build_current_correlation_model`; existing metadata, equilibration, correlation, serialization, and plot interfaces retain their names.

- [ ] **Step 1: Retarget tests to the new local type and builders**

Replace the cross-example include with local configuration/model includes:

```julia
include("../examples/holstein_current_correlation/config.jl")
include("../examples/holstein_current_correlation/model.jl")
include("../examples/holstein_current_correlation/utils.jl")
include("../examples/holstein_current_correlation/equilibrate.jl")
include("../examples/holstein_current_correlation/current_correlation.jl")
```

Mechanically replace `HolsteinConfig` with `HolsteinCurrentCorrelationConfig` throughout this test. In the lazy-builder subprocess fixtures, define `decompose_current_correlation_bath` and `build_current_correlation_model`, and assert their received config is the new local type. Add:

```julia
@test DEFAULT_CURRENT_CORRELATION_CONFIG isa HolsteinCurrentCorrelationConfig
@test HolsteinCurrentCorrelationConfig !== HolsteinConfig
for file in ("equilibrate.jl", "current_correlation.jl", "utils.jl", "model.jl")
    source = read(joinpath(@__DIR__, "..", "examples",
                           "holstein_current_correlation", file), String)
    @test !occursin("../holstein", source)
end
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `julia --project=. -e 'using Test; include("test/holstein_current_correlation.jl")'`

Expected: FAIL because the local configuration and model files do not exist.

- [ ] **Step 3: Add the local configuration and model**

Copy the complete 25-field configuration, constructor defaults/conversions, and validation predicates from Task 1, renaming the type to `HolsteinCurrentCorrelationConfig` and validator to `validate_current_correlation_config`. In `model.jl`, copy the periodic Hamiltonian, projectors, and complete Brownian TPSD/BCF validation algorithm under the model-qualified names. Implement:

```julia
function build_current_correlation_model(
    config::HolsteinCurrentCorrelationConfig, decomposition)
    H_fs = periodic_current_correlation_hamiltonian(
        config.site_energies_cm, config.hopping_cm) * icm2ifs
    baths = [BathExp(decomposition.exponents, decomposition.coefficients, V)
             for V in current_correlation_site_projectors(config.site_count)]
    system = HEOMTTSystem(H_fs, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end
```

- [ ] **Step 4: Retarget the existing correlation implementation**

In `utils.jl`, change every `::HolsteinConfig` annotation to `::HolsteinCurrentCorrelationConfig`; leave metadata keys, `_EQUILIBRIUM_METADATA_VERSION = 1`, `_EQUILIBRIUM_HEOM_REPRESENTATION = "twin-space-v1"`, propagation, file publication, and atomicity bodies unchanged.

At the top of `equilibrate.jl`, use only local guards:

```julia
isdefined(@__MODULE__, :HolsteinCurrentCorrelationConfig) || include("config.jl")
isdefined(@__MODULE__, :build_current_correlation_model) || include("model.jl")
isdefined(@__MODULE__, :run_equilibration) || include("utils.jl")

const DEFAULT_CURRENT_CORRELATION_CONFIG = HolsteinCurrentCorrelationConfig()
```

Delete `_load_holstein_builders`. The default injected builders become direct, world-age-safe calls:

```julia
_default_decomposition_builder(config::HolsteinCurrentCorrelationConfig) =
    decompose_current_correlation_bath(config)
_default_problem_builder(config::HolsteinCurrentCorrelationConfig, decomposition) =
    build_current_correlation_model(config, decomposition)
```

Change both `equilibrate_main` and `current_correlation_main` defaults from `DEFAULT_CONFIG` to `DEFAULT_CURRENT_CORRELATION_CONFIG`, and call `validate_current_correlation_config`.

- [ ] **Step 5: Verify checkpoint/output behavior and commit**

Run: `julia --project=. -e 'using Test; include("test/holstein_current_correlation.jl")'`

Expected: all current-correlation tests PASS, including metadata round trips, legacy checkpoint validation, overwrite rollback, injected builders, and include guards.

Run: `rg -n '\.\./holstein|HolsteinConfig|decompose_brownian_bcf|build_holstein_heomtt' examples/holstein_current_correlation test/holstein_current_correlation.jl`

Expected: no matches.

```bash
git add examples/holstein_current_correlation/config.jl \
  examples/holstein_current_correlation/model.jl \
  examples/holstein_current_correlation/utils.jl \
  examples/holstein_current_correlation/equilibrate.jl \
  examples/holstein_current_correlation/current_correlation.jl \
  test/holstein_current_correlation.jl
git commit -m "refactor: isolate Holstein current correlation example"
```

### Task 4: Enforce Cross-Example Isolation and Run Final Verification

**Files:**
- Modify: `test/runtests.jl`
- Modify: `test/heom_twin_space.jl`
- Modify: `test/lattice_frohlich.jl`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: all three self-contained directory layouts from Tasks 1–3.
- Produces: regression checks that reject cross-example includes, retired utilities, import side effects, and config-type reuse.

- [ ] **Step 1: Add one repository-level source-layout test**

Add to `test/runtests.jl`:

```julia
@testset "Holstein-family examples are self-contained" begin
    roots = [
        joinpath(@__DIR__, "..", "examples", "holstein"),
        joinpath(@__DIR__, "..", "examples", "lattice_frohlich"),
        joinpath(@__DIR__, "..", "examples", "holstein_current_correlation"),
    ]
    for root in roots
        for file in filter(path -> endswith(path, ".jl"), readdir(root; join=true))
            source = read(file, String)
            @test !occursin(r"include\([^\n]*(?:\.\./holstein|lattice_frohlich|holstein_current_correlation)", source)
        end
    end
    @test !isfile(joinpath(roots[1], "utils.jl"))
    @test !isfile(joinpath(roots[2], "utils.jl"))
    @test HolsteinConfig !== LatticeFrohlichConfig
    @test HolsteinConfig !== HolsteinCurrentCorrelationConfig
    @test LatticeFrohlichConfig !== HolsteinCurrentCorrelationConfig
end
```

- [ ] **Step 2: Run static checks before expensive Julia tests**

Run: `rg -n '\.\./holstein|examples/holstein/utils\.jl|examples/lattice_frohlich/utils\.jl|decompose_brownian_bcf|build_holstein_heomtt|run_dynamics\(' examples test`

Expected: no obsolete example references; unrelated package symbols are acceptable only if inspected and confirmed outside these three examples.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 3: Include each executable in an isolated subprocess**

Run:

```bash
julia --project=examples/holstein -e 'include("examples/holstein/holstein_brownian_heomtt.jl"); print("holstein-import-ok")'
julia --project=examples/lattice_frohlich -e 'include("examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl"); print("frohlich-import-ok")'
julia --project=examples/holstein_current_correlation -e 'include("examples/holstein_current_correlation/equilibrate.jl"); include("examples/holstein_current_correlation/current_correlation.jl"); print("correlation-import-ok")'
```

Expected: each marker appears, no simulation output is printed, and no files are created. If an example environment is not instantiated, run `julia --project=<example> -e 'using Pkg; Pkg.instantiate()'` only with the required package/network approval, and do not stage the resulting Manifest changes.

- [ ] **Step 4: Run focused and package tests**

Run:

```bash
julia --project=. -e 'using Test; include("test/lattice_frohlich.jl")'
julia --project=. -e 'using Test; include("test/holstein_current_correlation.jl")'
julia --project=. test/runtests.jl
```

Expected: focused tests PASS and no package-suite failure is caused by the refactor. Do not remove or overwrite `examples/holstein_current_correlation/output/` to satisfy its intentional clean-output guard; if that existing user directory remains the only failure, report it separately with the exact failing assertion.

- [ ] **Step 5: Inspect the diff boundary and commit test hardening**

Run: `git status --short`

Expected: source/test changes from this plan plus the pre-existing user-generated CSV/PNG/Manifest/output entries. Stage only the source/test paths named in this plan.

```bash
git add test/runtests.jl test/heom_twin_space.jl \
  test/lattice_frohlich.jl test/holstein_current_correlation.jl
git commit -m "test: enforce self-contained Holstein examples"
```

- [ ] **Step 6: Final verification report**

Record the exact commands, pass counts, any pre-existing output-guard failure, and `git status --short`. Confirm explicitly that generated CSV/PNG/checkpoint files, output directories, and example Manifests were neither modified by the refactor nor included in its commits.
