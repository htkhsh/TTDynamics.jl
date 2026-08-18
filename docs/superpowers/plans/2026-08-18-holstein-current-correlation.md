# Holstein Twin-Space Current Correlation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add separate executables that save a fixed-1000-fs equilibrated periodic Holstein HEOM-TT state and reload it to calculate the unsymmetrized current correlation for 0:1:200 fs.

**Architecture:** `examples/holstein_current_correlation/` contains two guarded executables plus focused shared utilities. It includes the existing Brownian Holstein example to reuse `HolsteinConfig`, QFiND TPSD decomposition, and HEOM construction, while its own helpers own state-returning propagation, exact factorized twin-space current operators, metadata validation, and output formatting.

**Tech Stack:** Julia 1.11+, TTDynamics/TTSolver, KaisouEOM, QFiND TPSD, TOML stdlib, CairoMakie, `Test`; TTDynamics versioned TT binary I/O.

## Global Constraints

- Production equilibration is fixed at `1000.0` fs with `1.0` fs steps; it is not claimed to prove stationarity.
- Production current correlation is the unsymmetrized `C_JJ(t) = Tr[J exp(Lt)(J rho_eq)]` on `0.0:1.0:200.0` fs.
- `J = im * hopping_cm * icm2ifs * sum(|n+1><n| - |n><n+1|)` includes the closing bond once for `N > 2`; `N = 2` has one bond, matching `periodic_holstein_hamiltonian`.
- Unit lattice spacing and charge are assumed, so current is in `fs^-1` and correlation in `fs^-2`.
- Canonical HEOM-TT dimensions are `[ket(N), bra(N), hierarchy...]`; old `[N^2, hierarchy...]` states are rejected.
- Left current action uses `J` on the ket core and identities on the bra and hierarchy cores; no dense `N^2 x N^2` superoperator is formed.
- The current observable uses ket and bra cores connected by exact bond rank `N`, plus hierarchy vacuum vectors, and measures only `Tr(J rho_root)`.
- Metadata records and validates `heom_representation = "twin-space-v1"`.
- The complete HEOM state is divided by its finite nonzero root trace before saving.
- State output is `output/holstein_equilibrium.ttbin`; physical/decomposition identity is stored separately in `output/holstein_equilibrium_metadata.toml`.
- Correlation output preserves both real and imaginary components in CSV and PNG.
- Default executable paths do not overwrite an existing equilibrium binary; explicit `overwrite=true` is required.
- Tests must not run the default 1000 fs equilibration or 200 fs correlation trajectory.
- Do not add QFiND, CairoMakie, or TOML to the root TTDynamics package dependencies; the example owns its environment.

---

## File Structure

- Create `examples/holstein_current_correlation/utils.jl`: current construction, state-returning CN propagation, normalization, metadata, validation, and CSV helpers.
- Create `examples/holstein_current_correlation/equilibrate.jl`: guarded 1000 fs orchestration and binary/metadata output.
- Create `examples/holstein_current_correlation/current_correlation.jl`: guarded reload, source preparation, 200 fs propagation, CSV, and plot.
- Create `examples/holstein_current_correlation/Project.toml`: direct dependencies and Julia 1.11 compatibility.
- Create `examples/holstein_current_correlation/README.md`: setup, run order, definitions, outputs, and convergence warning.
- Create `test/holstein_current_correlation.jl`: inexpensive focused unit/integration tests.
- Modify `test/runtests.jl`: include the new focused tests.

### Task 1: Current Operators and State-Returning Propagation

**Files:**
- Create: `examples/holstein_current_correlation/utils.jl`
- Create: `test/holstein_current_correlation.jl`
- Modify: `test/runtests.jl:10-15`

**Interfaces:**
- Consumes: `HolsteinConfig`, a problem named tuple with `system`, `liouvillian`, `trace_observable`, and `population_observables`, plus TTSolver TT operations.
- Produces: `periodic_current_operator(config)::Matrix{ComplexF64}`.
- Produces: `build_current_heom_operators(config, problem)::NamedTuple{(:current,:left_action,:observable)}`.
- Produces: `propagate_cn_step(rho::TTTensor, liouvillian::TTMatrix, config)::TTTensor`.
- Produces: `propagate_fixed_steps(rho, problem, config, step_count; observe)::NamedTuple{(:state,:observations)}`.
- Produces: `normalize_heom_state(rho, trace_observable; atol=1e-14)::TTTensor`.
- Produces: `measure_heom_state(rho, trace_observable, population_observables)::NamedTuple`.

- [ ] **Step 1: Write failing current-operator tests**

Create `test/holstein_current_correlation.jl`, import `LinearAlgebra`,
`Statistics`, `TTDynamics`, `TTSolver`, `KaisouEOM`, and `Test`, include
`examples/holstein/utils.jl` for `HolsteinConfig`, then include the new utility
file. Do not include the existing executable here because QFiND and CairoMakie
belong only to the example environment. Add:

```julia
@testset "Holstein current correlation utilities" begin
    config = HolsteinConfig(
        site_count=3,
        site_energies_cm=zeros(3),
        hopping_cm=2.0,
        final_time_fs=1.0,
    )
    current = periodic_current_operator(config)
    scale = config.hopping_cm * KaisouEOM.icm2ifs

    @test ishermitian(current)
    @test current[2, 1] == 1im * scale
    @test current[1, 2] == -1im * scale
    @test current[1, 3] == 1im * scale
    @test current[3, 1] == -1im * scale
    @test diag(current) == zeros(3)

    config2 = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
    )
    current2 = periodic_current_operator(config2)
    @test current2 == ComplexF64[0 -1im * scale; 1im * scale 0]
end
```

Add `include("holstein_current_correlation.jl")` to `test/runtests.jl` after
`include("tt_io.jl")`.

- [ ] **Step 2: Run the focused test and confirm the utility is absent**

Run:

```bash
julia --project=. test/holstein_current_correlation.jl
```

Expected: FAIL because `examples/holstein_current_correlation/utils.jl` or
`periodic_current_operator` does not exist.

- [ ] **Step 3: Implement the dense current and HEOM-TT embeddings**

In `utils.jl`, validate `site_count >= 2` and finite nonnegative hopping. Build
nearest-neighbor directed bonds for `1:(N-1)` and add `(N,1)` only for `N > 2`:

```julia
function periodic_current_operator(config::HolsteinConfig)
    N = config.site_count
    scale = config.hopping_cm * icm2ifs
    current = zeros(ComplexF64, N, N)
    for site in 1:(N - 1)
        current[site + 1, site] += 1im * scale
        current[site, site + 1] -= 1im * scale
    end
    if N > 2
        current[1, N] += 1im * scale
        current[N, 1] -= 1im * scale
    end
    current
end
```

Implement `build_current_heom_operators`. Let `I_sys` be the `N x N`
identity and construct the left action directly from factorized local cores:

```julia
local_matrices = [current, I_sys, hierarchy_identities...]
left_action = TTMatrix([
    reshape(ComplexF64.(matrix), 1, size(matrix, 1), size(matrix, 2), 1)
    for matrix in local_matrices
])
left_action = tt_round(left_action, config.operator_tolerance)
```

Construct the observable with an exact ket-bra bond of rank `N`:

```julia
ket_core = zeros(ComplexF64, 1, N, N)
for ket in 1:N
    ket_core[1, ket, ket] = 1
end
bra_core = reshape(current, N, N, 1)
observable = TTTensor([ket_core, bra_core, vacuum_cores...])
observable = tt_round(observable, config.operator_tolerance)
```

Because `tt_dot` conjugates the observable and `current` is Hermitian, this
contracts as `sum(conj(current[k,b]) * rho[k,b]) = Tr(current * rho)`.
Validate `tt_dims(left_action)` against both sides of `problem.liouvillian`,
validate `tt_dims(observable) == heom_tt_dimensions(problem.system)`, and
return the dense `current` together with both TT objects.

- [ ] **Step 4: Test TT left action and observable against dense algebra**

Build a two-site, one-bath-term-per-site problem with local hierarchy size 2,
as in the existing HEOM test. Construct a twin-space TT state whose first two
cores encode a nonsymmetric complex `rho` and whose hierarchy cores contain
nontrivial vectors. Assert:

```julia
ket_core = reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2)
bra_core = reshape(rho, 2, 2, 1)
state = TTTensor([
    ket_core,
    bra_core,
    reshape(h1, 1, 2, 1),
    reshape(h2, 1, 2, 1),
])
source = operators.left_action * state
expected = kron(vec(current * rho), kron(h1, h2))
@test vec(tt_full(source)) ≈ expected

root_state = TTTensor([
    ket_core,
    bra_core,
    reshape(ComplexF64[1, 0], 1, 2, 1),
    reshape(ComplexF64[1, 0], 1, 2, 1),
])
@test tt_dot(operators.observable, root_state) ≈ tr(current * rho)
```

Add a state with a zero root-vacuum entry and nonzero auxiliary entries and
assert the observable returns zero. These fixtures distinguish ket action from
bra action, transpose, and conjugation mistakes.

- [ ] **Step 5: Implement and test propagation plus normalization**

Extract the existing one-step tAMEn sequence into `propagate_cn_step`, with
the exact options currently used by `run_dynamics`. `propagate_fixed_steps`
must reject a negative `step_count`, call `observe(step,time,state)` at step
zero and every completed step, and return both the final state and collected
observations. Keep observation storage generic by returning the callback
values in a vector.

`normalize_heom_state` computes `trace_value = tt_dot(trace_observable,rho)`,
rejects nonfinite real/imaginary parts and `abs(trace_value) <= atol`, divides
the complete TT tensor by that complex trace, rounds only when the caller asks
afterward, and verifies the normalized trace is approximately one.

Implement `measure_heom_state` in the new utility rather than depending on the
existing executable's `measure_state`; it returns real site populations, real
root trace, maximum rank, and mean rank. This keeps root tests independent of
QFiND and CairoMakie.

Test propagation with a two-core twin-space state and zero Liouvillian:

```julia
rho = TTTensor([
    reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2),
    reshape(ComplexF64[1 2im; -3im 4], 2, 2, 1),
])
zero_liouvillian = TTMatrix([
    zeros(ComplexF64, 1, 2, 2, 1),
    reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1),
])
```

For one step, the returned dense state must equal the input. Test that the
callback records steps `[0,1]`. Test normalization on a scaled twin-space HEOM
initial state and reject a zero TT state.

- [ ] **Step 6: Run focused and full tests**

```bash
julia --project=. test/holstein_current_correlation.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: focused tests and the complete existing suite pass.

- [ ] **Step 7: Commit the current and propagation utilities**

```bash
git add examples/holstein_current_correlation/utils.jl test/holstein_current_correlation.jl test/runtests.jl
git commit -m "feat: add Holstein current TT utilities"
```

### Task 2: Versioned Metadata and State Compatibility

**Files:**
- Modify: `examples/holstein_current_correlation/utils.jl`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: `HolsteinConfig`, TPSD decomposition with `exponents` and `coefficients`, `problem.system.nb`, and a saved `TTTensor`.
- Produces: `equilibrium_metadata(config, decomposition, problem, state; equilibration_time_fs)::Dict{String,Any}`.
- Produces: `write_equilibrium_metadata(path, metadata; overwrite=false)::String`.
- Produces: `read_equilibrium_metadata(path)::Dict{String,Any}`.
- Produces: `validate_equilibrium_state(state, metadata, config, decomposition, problem; rtol=1e-12, atol=1e-14)::TTTensor`.

- [ ] **Step 1: Add failing metadata round-trip tests**

Construct a deterministic fake decomposition with two complex exponents and
coefficients, a small `HEOMTTSystem`, and its initial state. Assert metadata
contains identifier `"TTDynamics.HolsteinEquilibrium"`, version `1`,
`heom_representation = "twin-space-v1"`, all
physical fields, parallel real/imaginary decomposition arrays, hierarchy
sizes, and `tt_dims(state)`. In `mktempdir`, write/read metadata and compare
the dictionaries. Assert a second default write throws and `overwrite=true`
replaces it.

- [ ] **Step 2: Run focused tests to see missing metadata functions**

```bash
julia --project=. test/holstein_current_correlation.jl
```

Expected: FAIL with `UndefVarError: equilibrium_metadata not defined`.

- [ ] **Step 3: Implement TOML-safe metadata and atomic publication**

Add `using TOML` in the example utility. Use only TOML-supported strings,
integers, floats, booleans, arrays, and dictionaries. Store symbols such as
`pade_type` as strings. Encode complex arrays as:

```julia
"exponents_real" => real.(decomposition.exponents)
"exponents_imag" => imag.(decomposition.exponents)
"coefficients_real" => real.(decomposition.coefficients)
"coefficients_imag" => imag.(decomposition.coefficients)
```

`write_equilibrium_metadata` must validate the dictionary first, write with
`TOML.print` to `mktemp(dirname(abspath(path)))`, close successfully, then
publish without silent overwrite. Implement private example-local publication
helpers with `hardlink(temp,target)` for no-clobber and the Julia-runtime
`ccall(:jl_fs_rename, ...)` boundary for explicit atomic replacement, matching
the Julia 1.11-safe semantics of `save_tt_binary`; do not call destructive
`mv(...; force=true)` on a directory. Clean the temporary file in `finally`.

- [ ] **Step 4: Implement exact compatibility validation**

Validate identifier/version/HEOM representation, finite numeric arrays, equal
real/imaginary array lengths, reconstructed complex decomposition, every
physical/config field, `problem.system.nb`, `heom_tt_dimensions(problem.system)`,
and `tt_dims(state)`. First reject non-`TTTensor` objects, then call
`root_density_matrix(state, problem.system)` as the public semantic validation
boundary so old `[N^2, hierarchy...]` checkpoints receive the repository's
migration error and nonfinite reconstructed roots are rejected.
For floats and complex arrays use `isapprox(...; rtol, atol)` and include the
field name in each `ArgumentError`. For integers, strings, vectors of integers,
and dimensions use exact equality.

Add tests that independently alter `heom_representation`, `hopping_cm`, one
exponent imaginary part, one hierarchy size, and one TT dimension. Add a
synthetic `[N^2, hierarchy...]` state and verify its error contains
`"old vectorized HEOM state is unsupported"`. Each changed metadata field must
throw `ArgumentError` whose message contains its name. Add missing file,
unsupported identifier, unsupported version, and nonfinite metadata cases.

- [ ] **Step 5: Test binary save/load together with metadata validation**

In a temporary directory, save a small HEOM initial state with
`save_tt_binary`, write its metadata, reload both, validate, and assert exact
core equality. Replace the binary with a `TTMatrix` and assert the state-type
check rejects it.

- [ ] **Step 6: Run tests and commit**

```bash
julia --project=. test/holstein_current_correlation.jl
julia --project=. -e 'using Pkg; Pkg.test()'
git add examples/holstein_current_correlation/utils.jl test/holstein_current_correlation.jl
git commit -m "feat: validate Holstein equilibrium metadata"
```

Expected: all tests pass before committing.

### Task 3: Fixed-Time Equilibration and Binary Save Executable

**Files:**
- Create: `examples/holstein_current_correlation/equilibrate.jl`
- Modify: `examples/holstein_current_correlation/utils.jl`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: Task 1 propagation/normalization and Task 2 metadata functions, plus existing `decompose_brownian_bcf` and `build_holstein_heomtt`.
- Produces from `utils.jl`: `run_equilibration(config, problem; equilibration_time_fs=1000.0)::NamedTuple`.
- Produces from `utils.jl`: `write_equilibration_csv(path, result)::String`.
- Produces: `equilibrate_main(config=DEFAULT_CONFIG; equilibration_time_fs=1000.0, output_directory=joinpath(@__DIR__,"output"), overwrite=false)::NamedTuple`.

- [ ] **Step 1: Add failing short equilibration tests**

Build a small zero-Liouvillian problem and call `run_equilibration` from
`utils.jl` with `equilibration_time_fs=config.time_step_fs`. Assert two time points,
population/trace/rank arrays of matching length, a returned final `TTTensor`,
and unit final trace after normalization. Assert a non-integral duration/time
step and nonpositive duration throw `ArgumentError`.

- [ ] **Step 2: Run focused tests to see the executable API missing**

```bash
julia --project=. test/holstein_current_correlation.jl
```

Expected: FAIL because `equilibrate.jl` or `run_equilibration` is absent.

- [ ] **Step 3: Implement state-returning equilibration diagnostics**

Implement these functions in `utils.jl`. Use `propagate_fixed_steps` from the
localized state. The observation callback must call `measure_heom_state` from
Task 1 and return its named tuple. Assemble
`times`, `populations`, `trace`, `maximum_rank`, and `mean_rank`, normalize the
final complete state, and return it as `state`. `write_equilibration_csv` uses
the population CSV schema from the existing example and returns its path.

- [ ] **Step 4: Implement guarded `equilibrate_main`**

The script includes `../holstein/holstein_brownian_heomtt.jl` and local
`utils.jl`, defines constants `DEFAULT_EQUILIBRATION_TIME_FS = 1000.0` and the
default output paths, and then:

1. validates config and duration;
2. decomposes the bath and builds the problem;
3. runs equilibration;
4. creates the output directory;
5. writes the diagnostic CSV;
6. calls `save_tt_binary(state_path, result.state; overwrite)`;
7. builds and writes matching metadata only after binary save succeeds; and
8. prints trace drift, saved trace, final ranks, and paths.

Guard execution with `if abspath(PROGRAM_FILE) == @__FILE__`.

If metadata publication fails after a new non-overwrite state save, remove
only that just-created state so the pair is not presented inconsistently. If
`overwrite=true` replaced an existing state, report the metadata error without
deleting the replaced state; document this limitation and recommend writing to
a fresh output directory for production replacement.

- [ ] **Step 5: Test real short save/load orchestration without TPSD**

Expose dependency-injection keywords or a lower-level
`save_equilibration_outputs` helper so the test supplies its small fake
problem/decomposition and one-step result without invoking QFiND or 1000
steps. Assert binary, TOML, and CSV files exist, the loaded state validates,
and a second default call refuses overwrite.

- [ ] **Step 6: Run tests and commit**

```bash
julia --project=. test/holstein_current_correlation.jl
julia --project=. -e 'using Pkg; Pkg.test()'
git add examples/holstein_current_correlation/equilibrate.jl examples/holstein_current_correlation/utils.jl test/holstein_current_correlation.jl
git commit -m "feat: save equilibrated Holstein TT state"
```

### Task 4: Reloaded Current Correlation, Environment, and Documentation

**Files:**
- Create: `examples/holstein_current_correlation/current_correlation.jl`
- Create: `examples/holstein_current_correlation/Project.toml`
- Create: `examples/holstein_current_correlation/README.md`
- Modify: `examples/holstein_current_correlation/utils.jl`
- Modify: `test/holstein_current_correlation.jl`

**Interfaces:**
- Consumes: saved binary/metadata, reconstructed problem/decomposition, current operators, and common propagation.
- Produces: `run_current_correlation(equilibrium_state, config, problem; correlation_time_fs=200.0)::NamedTuple`.
- Produces: `write_current_correlation_csv(path, result)::String`.
- Produces: `current_correlation_main(config=DEFAULT_CONFIG; correlation_time_fs=200.0, output_directory=joinpath(@__DIR__,"output"))::NamedTuple`.

- [ ] **Step 1: Add failing zero-time and CSV tests**

For a small twin-space HEOM root state, build current operators and call
`run_current_correlation(...; correlation_time_fs=0.0)`. Assert one time point
at zero and:

```julia
source = operators.left_action * equilibrium_state
expected = tt_dot(operators.observable, source)
@test result.correlation == ComplexF64[expected]
```

Write a synthetic result containing complex values and assert the CSV header
is exactly
`time_fs,correlation_real_fs^-2,correlation_imag_fs^-2,max_rank,mean_rank`
and both components appear in each row. Both functions live in `utils.jl`, so
this focused root test does not load QFiND or CairoMakie.

- [ ] **Step 2: Run focused tests and confirm missing functions**

```bash
julia --project=. test/holstein_current_correlation.jl
```

Expected: FAIL with missing correlation runner/writer.

- [ ] **Step 3: Implement current-source propagation and CSV**

Validate the duration as a nonnegative integer multiple of the time step.
Build current operators, apply and TT-round the source, then observe
`tt_dot(operators.observable,state)` plus ranks at steps zero through the
requested end. Return `times`, `correlation::Vector{ComplexF64}`,
`maximum_rank`, `mean_rank`, and final source state. The CSV writer must preserve
complex values as separate real/imaginary columns.

- [ ] **Step 4: Implement guarded reload orchestration and plotting**

`current_correlation_main` must:

1. require both default state and metadata paths;
2. decompose and rebuild the problem;
3. load via `load_tt_binary` and `read_equilibrium_metadata`;
4. validate state/metadata against the reconstructed model;
5. run the 200 fs correlation;
6. write CSV; and
7. save one CairoMakie PNG with labeled real and imaginary lines and `fs^-2`
   y-axis.

Guard execution with `if abspath(PROGRAM_FILE) == @__FILE__`. Do not normalize
the current-source state because it is not a density matrix.

- [ ] **Step 5: Add the standalone example environment**

Create `Project.toml` with the same UUIDs and compatibility ranges as the
existing Holstein example for `CairoMakie`, `KaisouEOM`, `LinearAlgebra`,
`QFiND`, `Statistics`, `TTDynamics`, and `TTSolver`, plus stdlib TOML UUID
`fa267f1f-6049-4f14-aa54-33bafae1ed76` and compatibility matching Julia 1.11.
Do not add a root dependency.

- [ ] **Step 6: Write the README**

Document local sibling-checkout setup and pinned remote setup following
`examples/holstein/README.md`, then the required run order:

```bash
julia --project=examples/holstein_current_correlation examples/holstein_current_correlation/equilibrate.jl
julia --project=examples/holstein_current_correlation examples/holstein_current_correlation/current_correlation.jl
```

State the 1000 fs and 200 fs defaults, exact correlation/current definitions,
units, every output filename, explicit overwrite behavior, metadata checks,
and convergence warning. Explain that the saved state is fixed-time
equilibrated, not automatically certified stationary.

- [ ] **Step 7: Run import, focused, environment, and full checks**

```bash
julia --project=. test/holstein_current_correlation.jl
julia --project=examples/holstein_current_correlation -e 'include("examples/holstein_current_correlation/equilibrate.jl"); include("examples/holstein_current_correlation/current_correlation.jl"); @assert DEFAULT_EQUILIBRATION_TIME_FS == 1000.0; @assert DEFAULT_CORRELATION_TIME_FS == 200.0'
julia --project=. -e 'using Pkg; Pkg.test()'
git diff --check
```

Expected: imports do not propagate or write outputs, focused and full tests
pass, the standalone environment resolves with all direct imports, and the
formatting check is clean. Do not run the default trajectories.

- [ ] **Step 8: Commit and perform a final scope audit**

```bash
git add examples/holstein_current_correlation test/holstein_current_correlation.jl
git commit -m "feat: calculate Holstein current correlation"
git status --short
git log --oneline -4
```

Confirm no generated `output/` files, Manifest, or changes to the existing
Holstein population outputs are committed. Run the focused and full tests once
more from the committed tree and record exact pass counts.
