# Lattice Fröhlich Current-Correlation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained two-stage lattice Fröhlich HEOM-TT example that checkpoints a fixed-time equilibrium state and reloads it to calculate the unsymmetrized periodic particle-current correlation.

**Architecture:** The new example owns a distinct configuration type, normalized long-range diagonal Fröhlich model, propagation/checkpoint utilities, and two guarded executables. It follows the proven Holstein current-correlation workflow while using a model-specific metadata identifier and storing the normalized Fröhlich weight matrix so incompatible checkpoints are rejected.

**Tech Stack:** Julia 1.11+, TTDynamics, TTSolver, HEOMKit, QFiND, CairoMakie, TOML, Julia `Test`

**Spec:** `docs/superpowers/specs/2026-08-24-lattice-frohlich-current-correlation-design.md`

## Global Constraints

- The directory is `examples/lattice_frohlich_current_correlation` and includes no source from another example directory.
- The configuration type is `LatticeFrohlichCurrentCorrelationConfig`; it is not shared with another example.
- Use metadata identifier `TTDynamics.LatticeFrohlichEquilibrium`, version `1`, and HEOM representation `twin-space-v1`.
- Store and validate the complete normalized Fröhlich weight matrix in checkpoint metadata.
- The particle-current operator contains only the periodic nearest-neighbor hopping current; diagonal electron-phonon coupling adds no separate current term.
- Preserve the exact defaults in the spec, including `time_step_fs=1.0`, `hierarchy_local_size=4`, `tamen_tolerance=2e-2`, `progress_interval=10`, equilibration `1000.0 fs`, and correlation `200.0 fs`.
- `overwrite=false` is no-clobber; failed non-overwrite publication rolls back newly published files.
- Including executables does not propagate, create directories, or write files.
- Tests write only under `mktempdir()`.
- Do not modify or commit existing generated CSV/PNG/checkpoint files, any `output/` directory, or any `Manifest.toml`.

---

### Task 1: Local Configuration and Fröhlich Model

**Files:**
- Create: `examples/lattice_frohlich_current_correlation/config.jl`
- Create: `examples/lattice_frohlich_current_correlation/model.jl`
- Create: `test/lattice_frohlich_current_correlation.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `LatticeFrohlichCurrentCorrelationConfig(; kwargs...)`, `validate_lattice_frohlich_current_correlation_config(config)`, `periodic_lattice_frohlich_current_hamiltonian(site_energies, hopping)`, `periodic_lattice_frohlich_distance(m,n,N)`, `default_lattice_frohlich_current_kernel(distance)`, `normalized_lattice_frohlich_current_kernel(N; kernel)`, `lattice_frohlich_current_coupling_operators(N; kernel)`, `decompose_lattice_frohlich_current_bath(config)`, and `build_lattice_frohlich_current_model(config,decomposition; kernel)`.
- The model builder returns `(; system, liouvillian, trace_observable, population_observables, frohlich_weights)`.

- [ ] **Step 1: Write failing configuration and kernel tests**

Create the test file and include it from `test/runtests.jl`:

```julia
using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

include("../examples/lattice_frohlich_current_correlation/config.jl")
include("../examples/lattice_frohlich_current_correlation/model.jl")

@testset "Lattice Frohlich current configuration" begin
    config = LatticeFrohlichCurrentCorrelationConfig()
    expected = (
        site_count=5, site_energies_cm=zeros(5), hopping_cm=400.0,
        brownian_frequency_cm=1400.0, brownian_damping_cm=200.0,
        reorganization_energy_cm=600.0, temperature_K=300.0,
        initial_site=1, final_time_fs=500.0, time_step_fs=1.0,
        pade_order=8, tpsd_tolerance=2e-2, pade_type=:Nm1,
        validation_final_time_fs=100.0, validation_sample_count=200,
        bcf_upper_bound_cm=10_000.0, hierarchy_local_size=4,
        temporal_basis_size=3, tamen_tolerance=2e-2,
        operator_tolerance=1e-10, state_rounding_tolerance=1e-10,
        sweep_count=3, local_iterations=5, kick_rank=4,
        progress_interval=10,
    )
    @test fieldnames(typeof(config)) == keys(expected)
    @test all(field -> getproperty(config, field) == getproperty(expected, field), keys(expected))
    @test_throws ArgumentError LatticeFrohlichCurrentCorrelationConfig(site_count=1)
    @test_throws ArgumentError LatticeFrohlichCurrentCorrelationConfig(temporal_basis_size=4)
end

@testset "Lattice Frohlich current model" begin
    @test periodic_lattice_frohlich_distance(1, 3, 4) == 2
    @test periodic_lattice_frohlich_distance(1, 4, 4) == 1
    weights = normalized_lattice_frohlich_current_kernel(4)
    @test size(weights) == (4, 4)
    @test all(n -> sum(abs2, weights[:, n]) ≈ 1, 1:4)
    @test weights ≈ transpose(weights)
    @test weights[:, 2] ≈ circshift(weights[:, 1], 1)
    operators = lattice_frohlich_current_coupling_operators(4)
    @test all(isdiag, operators)
    @test all(ishermitian, operators)
end
```

- [ ] **Step 2: Run RED**

Run: `julia --project=. -e 'using Test; include("test/lattice_frohlich_current_correlation.jl")'`

Expected: FAIL because the local configuration and model files do not exist.

- [ ] **Step 3: Implement the distinct configuration type**

Define these fields in this exact order:

```julia
struct LatticeFrohlichCurrentCorrelationConfig
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

The keyword constructor uses every literal from `expected` above, converts to the declared types, and calls `validate_lattice_frohlich_current_correlation_config`. Validation requires: at least two sites; matching finite site energies; valid initial site; nonnegative hopping; positive physical/numerical scales; underdamped Brownian poles; positive Padé order; `:N` or `:Nm1`; at least two validation samples; positive iteration controls; odd temporal basis at least three; and an integral final-time/time-step ratio.

- [ ] **Step 4: Implement the local Fröhlich model**

Implement the kernel exactly as:

```julia
periodic_lattice_frohlich_distance(m::Integer, n::Integer, N::Integer) =
    min(abs(Int(m) - Int(n)), Int(N) - abs(Int(m) - Int(n)))

default_lattice_frohlich_current_kernel(distance::Integer) =
    (Float64(distance)^2 + 1.0)^(-1.5)
```

Validate indices before computing distance. Build `raw[m,n] = kernel(distance(m,n,N))`, reject nonfinite/negative values, verify symmetry and circulant columns, then normalize each column by `sqrt(sum(abs2, raw[:,n]))`. Recheck unit norms, symmetry, and circulant structure. Build one `Matrix(Diagonal(ComplexF64.(weights[m,:])))` per bath and verify diagonal, Hermitian, cyclic translation.

Use the periodic Hamiltonian matrix with diagonal site energies, `-hopping` nearest-neighbor entries, and one last-to-first bond only when `N > 2`.

`decompose_lattice_frohlich_current_bath` constructs `BrownianSD`, calls `tpsd`, converts values to `ComplexF64`, verifies finite positive-real exponents, and computes the relative BCF error against `BosonicBCF` on the configured validation grid.

Build the problem with:

```julia
function build_lattice_frohlich_current_model(
    config::LatticeFrohlichCurrentCorrelationConfig,
    decomposition;
    kernel=default_lattice_frohlich_current_kernel,
)
    H = periodic_lattice_frohlich_current_hamiltonian(
        config.site_energies_cm, config.hopping_cm) * icm2ifs
    frohlich_weights = normalized_lattice_frohlich_current_kernel(
        config.site_count; kernel)
    couplings = lattice_frohlich_current_coupling_operators(
        config.site_count; kernel)
    baths = [BathExp(decomposition.exponents, decomposition.coefficients, V)
             for V in couplings]
    system = HEOMTTSystem(H, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable,
            population_observables, frohlich_weights)
end
```

- [ ] **Step 5: Run GREEN and commit**

Run: `julia --project=. -e 'using Test; include("test/lattice_frohlich_current_correlation.jl")'`

Expected: configuration and kernel/model testsets PASS.

```bash
git add examples/lattice_frohlich_current_correlation/config.jl \
  examples/lattice_frohlich_current_correlation/model.jl \
  test/lattice_frohlich_current_correlation.jl test/runtests.jl
git commit -m "feat: add lattice Frohlich current model"
```

### Task 2: Propagation and Particle-Current Utilities

**Files:**
- Create: `examples/lattice_frohlich_current_correlation/utils.jl`
- Modify: `test/lattice_frohlich_current_correlation.jl`

**Interfaces:**
- Consumes: the Task 1 configuration and problem tuple.
- Produces: `lattice_frohlich_particle_current(config)`, `build_lattice_frohlich_current_operators(config,problem)`, `propagate_lattice_frohlich_cn_step`, `propagate_lattice_frohlich_fixed_steps`, `measure_lattice_frohlich_heom_state`, `normalize_lattice_frohlich_heom_state`, `run_lattice_frohlich_equilibration`, `run_lattice_frohlich_current_correlation`, `_print_lattice_frohlich_equilibration_progress`, and `_print_lattice_frohlich_current_progress`.

- [ ] **Step 1: Write failing current and propagation tests**

Append:

```julia
include("../examples/lattice_frohlich_current_correlation/utils.jl")

@testset "Lattice Frohlich particle current" begin
    config = LatticeFrohlichCurrentCorrelationConfig(
        site_count=3, site_energies_cm=zeros(3), hopping_cm=20.0,
        final_time_fs=2.0, hierarchy_local_size=2)
    current = lattice_frohlich_particle_current(config)
    scale = 20.0 * HEOMKit.icm2ifs
    @test current == ComplexF64[0 -im*scale im*scale;
                               im*scale 0 -im*scale;
                               -im*scale im*scale 0]
    @test ishermitian(current)
end
```

Use a one-term decomposition
`(; exponents=ComplexF64[0.4], coefficients=ComplexF64[0.03-0.02im])`, build a three-site problem, and assert current action dimensions equal the Liouvillian dimensions, observable dimensions equal `heom_tt_dimensions(problem.system)`, and a zero-step current run returns exactly one sample whose first value equals `tt_dot(operators.observable, operators.left_action * state)`.

For progress, use `progress_interval=2`, five steps, collect callback records, and assert steps `[0,2,4,5]`, physical times `[0.0,2.0,4.0,5.0]`, finite nonnegative elapsed time, and ranks matching returned arrays for both equilibration and current propagation.

- [ ] **Step 2: Run RED**

Run the focused test. Expected: FAIL because `utils.jl` and the local current APIs do not exist.

- [ ] **Step 3: Implement fixed-step propagation and measurements**

Use Crank-Nicolson tAMEn options:

```julia
options = Dict(
    :verb => 0,
    :nswp => config.sweep_count,
    :local_iters => config.local_iterations,
    :kickrank => config.kick_rank,
    :time_scheme => "CN",
    :time_error_damp => 100.0,
)
```

Add `:obs => [trace_observable]` when supplied. Advance through `tkron(state, tt_ones(config.temporal_basis_size; T=ComplexF64))`, `tamen`, `extract_snapshot(..., 1.0, "CN")`, and `tt_round(..., config.state_rounding_tolerance)`. `propagate_lattice_frohlich_fixed_steps` records the initial observation before its loop and rejects negative step counts.

Measurement returns populations, real trace, maximum rank, and mean rank. Normalization divides the entire TT state by its finite nonzero complex root trace.

- [ ] **Step 4: Implement the current and two high-level runners**

Construct the Hermitian periodic current with `scale = config.hopping_cm * icm2ifs`, adding `+im*scale` to `[n+1,n]` and `-im*scale` to `[n,n+1]`, plus the periodic bond for `N > 2`.

Embed the ket-side action using local matrices `[current, I_sys, hierarchy_identities...]`. Build the observable with the diagonal-selector ket core, `reshape(current,N,N,1)` bra core, and hierarchy-vacuum cores. Round both with `operator_tolerance` and validate dimensions.

Equilibration observes every state and notifies only step zero, multiples of `progress_interval`, and the final step. Current propagation forms `exact_source = left_action * equilibrium_state`, rounds only the propagated source, and overwrites `correlation[1]` with the exact-source value.

Progress messages use these exact fields:

```julia
"Step $(p.step)/$(p.step_count) | time=$(p.time_fs) fs | " *
"trace=$(p.trace) | max rank=$(p.maximum_rank) | " *
"mean rank=$(p.mean_rank) | elapsed=$(round(p.elapsed_seconds; digits=2)) s"
```

The current message omits `trace`.

- [ ] **Step 5: Run GREEN and commit**

Run the focused test and expect current matrix, TT dimensions, zero-step correlation, fixed-step propagation, and progress tests to PASS.

```bash
git add examples/lattice_frohlich_current_correlation/utils.jl \
  test/lattice_frohlich_current_correlation.jl
git commit -m "feat: add lattice Frohlich current propagation"
```

### Task 3: Fröhlich Checkpoint Metadata and Equilibration Stage

**Files:**
- Modify: `examples/lattice_frohlich_current_correlation/utils.jl`
- Create: `examples/lattice_frohlich_current_correlation/equilibrate.jl`
- Create: `examples/lattice_frohlich_current_correlation/plotting.jl`
- Modify: `test/lattice_frohlich_current_correlation.jl`

**Interfaces:**
- Produces: `lattice_frohlich_equilibrium_metadata`, `write_lattice_frohlich_equilibrium_metadata`, `read_lattice_frohlich_equilibrium_metadata`, `validate_lattice_frohlich_equilibrium_state`, binary/CSV/PNG publication helpers, `_lattice_frohlich_equilibration_output_paths`, `save_lattice_frohlich_equilibration_outputs`, and `lattice_frohlich_equilibrate_main`.

- [ ] **Step 1: Write failing metadata and equilibration tests**

Assert metadata literals:

```julia
@test metadata["identifier"] == "TTDynamics.LatticeFrohlichEquilibrium"
@test metadata["version"] == 1
@test metadata["heom_representation"] == "twin-space-v1"
@test metadata["frohlich_weights"] == [collect(row) for row in eachrow(problem.frohlich_weights)]
```

Round-trip metadata through a temporary TOML, reload a TT binary, and validate it. Mutate one weight, one exponent, one hierarchy size, and the identifier in separate copies and assert `ArgumentError` for each. Assert the output basenames exactly:

```julia
(
 state_path="lattice_frohlich_equilibrium.ttbin",
 metadata_path="lattice_frohlich_equilibrium_metadata.toml",
 csv_path="lattice_frohlich_equilibration.csv",
 png_path="lattice_frohlich_equilibration_populations.png",
)
```

Run one equilibration step with injected decomposition/problem builders and a temporary output directory; verify all four files, normalized saved trace, progress output, and replacement on a second `overwrite=true` run. Add failure writers and assert non-overwrite rollback leaves no new files.

- [ ] **Step 2: Run RED**

Expected: FAIL because metadata and equilibration-stage APIs do not exist.

- [ ] **Step 3: Implement metadata with Fröhlich weights**

Define constants:

```julia
const _LATTICE_FROHLICH_METADATA_IDENTIFIER =
    "TTDynamics.LatticeFrohlichEquilibrium"
const _LATTICE_FROHLICH_METADATA_VERSION = 1
const _LATTICE_FROHLICH_HEOM_REPRESENTATION = "twin-space-v1"
```

Serialize every config field, real/imaginary TPSD arrays, term count, hierarchy sizes, TT dimensions, equilibration time, and:

```julia
"frohlich_weights" => [collect(row) for row in eachrow(problem.frohlich_weights)]
```

Validation first enforces TOML-safe string keys and finite scalar/array values. It reconstructs `Matrix{Float64}` from the nested rows, checks its shape against `site_count`, then compares it with `problem.frohlich_weights` using the caller's `rtol`/`atol`. Validate all configuration fields, complex TPSD reconstruction, hierarchy sizes, problem dimensions, and loaded-state dimensions.

Use the existing TTDynamics `save_tt_binary`/`load_tt_binary` API. Metadata and CSV writes use temporary files in the target directory; no-clobber publication uses hard links, overwrite uses the local atomic rename helper, and `finally` removes staging paths.

- [ ] **Step 4: Implement plotting and equilibration orchestration**

`plotting.jl` provides `_save_lattice_frohlich_equilibration_plot(path,result)` with one line per population. `equilibrate.jl` locally includes `config.jl`, `model.jl`, and `utils.jl`, lazily includes plotting, declares the four exact output paths, and defines:

```julia
function lattice_frohlich_equilibrate_main(
    config=DEFAULT_LATTICE_FROHLICH_CURRENT_CONFIG;
    equilibration_time_fs=1000.0,
    output_directory=joinpath(@__DIR__, "output"),
    overwrite=true,
    decomposition_builder=decompose_lattice_frohlich_current_bath,
    problem_builder=build_lattice_frohlich_current_model,
    equilibration_runner=run_lattice_frohlich_equilibration,
    plotter=_default_lattice_frohlich_equilibration_plotter,
    progress_io=stdout,
)
```

Connect the runner callback to `_print_lattice_frohlich_equilibration_progress`, save outputs, print trace drift/final ranks/paths, return result/problem/decomposition and paths, and guard direct execution with `abspath(PROGRAM_FILE) == @__FILE__`.

- [ ] **Step 5: Run GREEN and commit**

Run focused tests. Expected: metadata round trip/mismatch rejection, one-step equilibration, progress, output paths, overwrite, and rollback PASS.

```bash
git add examples/lattice_frohlich_current_correlation/utils.jl \
  examples/lattice_frohlich_current_correlation/equilibrate.jl \
  examples/lattice_frohlich_current_correlation/plotting.jl \
  test/lattice_frohlich_current_correlation.jl
git commit -m "feat: checkpoint lattice Frohlich equilibrium"
```

### Task 4: Reloaded Current-Correlation Stage

**Files:**
- Modify: `examples/lattice_frohlich_current_correlation/utils.jl`
- Modify: `examples/lattice_frohlich_current_correlation/plotting.jl`
- Create: `examples/lattice_frohlich_current_correlation/current_correlation.jl`
- Modify: `test/lattice_frohlich_current_correlation.jl`

**Interfaces:**
- Produces: `write_lattice_frohlich_current_csv`, `_lattice_frohlich_current_output_paths`, `save_lattice_frohlich_current_outputs`, and `lattice_frohlich_current_correlation_main`.

- [ ] **Step 1: Write failing reload/output tests**

In a temporary directory, first save a one-step equilibrium checkpoint. Call the current main with `correlation_time_fs=0.0` and injected builders, assert that it reloads the same state/metadata paths and writes exactly:

```julia
lattice_frohlich_current_correlation.csv
lattice_frohlich_current_correlation.png
lattice_frohlich_current_correlation_ranks.png
```

Assert CSV header:

```text
time_fs,correlation_real_fs^-2,correlation_imag_fs^-2,max_rank,mean_rank
```

Delete either equilibrium input in separate test directories and assert `ArgumentError`. Precreate each correlation output and assert `overwrite=false` rejects before changing any file. Inject a failing second plotter and assert newly published non-overwrite outputs are removed.

- [ ] **Step 2: Run RED**

Expected: FAIL because the current executable/output APIs do not exist.

- [ ] **Step 3: Implement CSV, plots, and atomic publication**

CSV rows contain `time`, `real(correlation)`, `imag(correlation)`, maximum rank, and mean rank. Add plotting functions for real/imaginary correlation in `fs⁻²` and maximum/mean rank. The output publisher checks all three targets before writing when `overwrite=false`, tracks published paths, and removes them in reverse order if a later publication fails.

- [ ] **Step 4: Implement reload orchestration**

`current_correlation.jl` includes `equilibrate.jl`, lazily loads both current plotters, declares `DEFAULT_LATTICE_FROHLICH_CORRELATION_TIME_FS = 200.0`, and defines:

```julia
function lattice_frohlich_current_correlation_main(
    config=DEFAULT_LATTICE_FROHLICH_CURRENT_CONFIG;
    correlation_time_fs=200.0,
    output_directory=joinpath(@__DIR__, "output"),
    overwrite=true,
    decomposition_builder=decompose_lattice_frohlich_current_bath,
    problem_builder=build_lattice_frohlich_current_model,
    correlation_runner=run_lattice_frohlich_current_correlation,
    plotter=_default_lattice_frohlich_current_plotter,
    rank_plotter=_default_lattice_frohlich_current_rank_plotter,
    progress_io=stdout,
)
```

Require both inputs, reconstruct the problem, load state/metadata, validate all checkpoint contracts including weights, print loaded ranks, run with progress callback, publish outputs, print final source ranks and paths, return result/problem/decomposition/input/output paths, and retain the executable guard.

- [ ] **Step 5: Run GREEN and commit**

Run focused tests. Expected: zero-step reload, exact `t=0`, CSV schema, plots, missing-input errors, no-clobber checks, rollback, and progress PASS.

```bash
git add examples/lattice_frohlich_current_correlation/utils.jl \
  examples/lattice_frohlich_current_correlation/plotting.jl \
  examples/lattice_frohlich_current_correlation/current_correlation.jl \
  test/lattice_frohlich_current_correlation.jl
git commit -m "feat: calculate lattice Frohlich current correlation"
```

### Task 5: Standalone Environment, Documentation, and Isolation Verification

**Files:**
- Create: `examples/lattice_frohlich_current_correlation/Project.toml`
- Create: `examples/lattice_frohlich_current_correlation/README.md`
- Modify: `.gitignore`
- Modify: `test/lattice_frohlich_current_correlation.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: both complete executables.
- Produces: documented standalone setup and permanent regression guards for source isolation and include-time side effects.

- [ ] **Step 1: Add failing isolated-import and source-isolation tests**

Read every `.jl` file in the new directory and reject `include` paths containing `../holstein`, `../lattice_frohlich`, or `../holstein_current_correlation`. In fresh subprocesses include `equilibrate.jl`, then include both executables; capture stdout separately from stderr, assert exact marker-only stdout, successful exit, unchanged example-directory snapshot, and an empty temporary working directory.

Use a platform-aware load-path separator:

```julia
separator = Sys.iswindows() ? ';' : ':'
load_path = join((repository_root, sibling_ttsolver, sibling_heomkit,
                  sibling_qfind, "@", "@stdlib"), separator)
```

- [ ] **Step 2: Run RED**

Expected: FAIL because the Project/README/ignore contract and isolated imports are incomplete.

- [ ] **Step 3: Add standalone project and ignore rules**

Create `Project.toml` with these UUIDs and Julia 1.11-compatible bounds:

```toml
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
HEOMKit = "e865c079-6a0d-426d-afc2-450809b0c699"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
QFiND = "af16a7c1-792b-4481-9b88-c9c438329a9c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
TTDynamics = "98f59f52-e016-4068-afe5-90126cbc5c1c"
TTSolver = "6fd66e5e-56e0-488c-9f68-e25b72b7e6b0"

[compat]
CairoMakie = "0.12 - 0.15"
HEOMKit = "1"
LinearAlgebra = "1.11"
QFiND = "1"
Statistics = "1.11"
TOML = "1"
TTDynamics = "1"
TTSolver = "1"
julia = "1.11"
```

Add exact `.gitignore` entries for the new `Manifest.toml` and `output/`.

- [ ] **Step 4: Write the README**

Document local sibling setup with one `Pkg.develop` transaction, then the two exact commands:

```bash
julia --project=examples/lattice_frohlich_current_correlation \
  examples/lattice_frohlich_current_correlation/equilibrate.jl
julia --project=examples/lattice_frohlich_current_correlation \
  examples/lattice_frohlich_current_correlation/current_correlation.jl
```

List all seven outputs, define the normalized kernel and current correlation with units, explain that equilibration is fixed-time rather than automatically stationary, and list the physical/numerical convergence parameters.

- [ ] **Step 5: Run final verification**

Run:

```bash
julia --project=. -e 'using Test; include("test/lattice_frohlich_current_correlation.jl")'
rg -n '\.\./(holstein|lattice_frohlich|holstein_current_correlation)' \
  examples/lattice_frohlich_current_correlation test/lattice_frohlich_current_correlation.jl
git diff --check
julia --project=. test/runtests.jl
```

Expected: focused tests PASS, `rg` has no cross-example source include, whitespace check is clean, and the package suite exits zero. If pre-existing user output directories trigger an unrelated guard in the main checkout, rerun the suite in the clean implementation worktree and report both outcomes without deleting user files.

- [ ] **Step 6: Verify artifact boundaries and commit**

Run `git status --short` and verify no generated output or Manifest is staged. Then:

```bash
git add .gitignore \
  examples/lattice_frohlich_current_correlation/Project.toml \
  examples/lattice_frohlich_current_correlation/README.md \
  test/lattice_frohlich_current_correlation.jl test/runtests.jl
git commit -m "docs: add lattice Frohlich current workflow"
```
