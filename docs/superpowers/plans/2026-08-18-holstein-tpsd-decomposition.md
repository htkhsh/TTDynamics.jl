# Holstein TPSD Bath Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Holstein example's sampled-time ESPRIT fit with QFiND's truncated Padé spectral decomposition using Padé order 8 and truncation tolerance `5e-2`.

**Architecture:** Keep physical and numerical controls in the lightweight `HolsteinConfig`, replacing ESPRIT-specific fields with TPSD and diagnostic-grid fields. Let QFiND `tpsd` produce HEOM decay rates and coefficients directly in femtosecond units; retain numerical `BosonicBCF` sampling only as a reported validation diagnostic.

**Tech Stack:** Julia 1.11, TTDynamics, TTSolver, HEOMKit, QFiND, CairoMakie, LinearAlgebra, Statistics, Test.

## Global Constraints

- Preserve the five-site periodic one-exciton Holstein model and `-J` hopping with `J = 400 cm^-1`.
- Preserve `Omega = 1400 cm^-1`, direct QFiND `Gamma_Q = 200 cm^-1`, `lambda = 600 cm^-1`, and `T = 300 K`.
- Use `tpsd` with default Padé order 8, balanced-truncation tolerance `5e-2`, and `pade_type = :Nm1`.
- Do not convert TPSD outputs again: QFiND returns exponents in `fs^-1` and coefficients in `fs^-2`.
- Continue converting the electronic Hamiltonian from `cm^-1` to `fs^-1` with `icm2ifs`.
- Use numerical `BosonicBCF` sampling only to report the TPSD relative validation error; the grid must not affect decomposition.
- Remove the ExpFit import and example dependency completely.
- Do not run or claim completion of the default 100 fs propagation.
- Preserve unrelated untracked KSL CSV/PNG outputs.

---

## File Structure

- Modify `examples/holstein/utils.jl`: replace ESPRIT controls with TPSD and validation-grid controls.
- Modify `test/runtests.jl`: test TPSD defaults and reject invalid Padé controls.
- Modify `examples/holstein/holstein_brownian_heomtt.jl`: call QFiND `tpsd`, validate its result, and report diagnostic BCF error.
- Modify `examples/holstein/Project.toml`: remove ExpFit from direct dependencies and compat.
- Modify `examples/holstein/README.md`: remove ExpFit setup and document TPSD settings and convergence controls.

### Task 1: TPSD Configuration Contract

**Files:**
- Modify: `examples/holstein/utils.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `HolsteinConfig` fields `pade_order::Int`, `tpsd_tolerance::Float64`, `pade_type::Symbol`, `validation_final_time_fs::Float64`, and `validation_sample_count::Int`.
- Removes: `bcf_final_time_fs`, `bcf_sample_count`, and `bcf_fit_tolerance`.
- Preserves: `bcf_upper_bound_cm` for the numerical validation correlation function.

- [ ] **Step 1: Add failing tests for TPSD defaults and validation**

In the existing `Periodic Holstein model utilities` test set, replace no tests yet and append:

```julia
    @test config.pade_order == 8
    @test config.tpsd_tolerance == 5e-2
    @test config.pade_type == :Nm1
    @test config.validation_final_time_fs == 100.0
    @test config.validation_sample_count == 400
    @test_throws ArgumentError HolsteinConfig(pade_order=0)
    @test_throws ArgumentError HolsteinConfig(tpsd_tolerance=0.0)
    @test_throws ArgumentError HolsteinConfig(pade_type=:invalid)
    @test_throws ArgumentError HolsteinConfig(validation_sample_count=1)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL with `type HolsteinConfig has no field pade_order`.

- [ ] **Step 3: Replace ESPRIT-specific fields and constructor keywords**

Replace these struct fields:

```julia
bcf_final_time_fs::Float64
bcf_sample_count::Int
bcf_fit_tolerance::Float64
```

with:

```julia
pade_order::Int
tpsd_tolerance::Float64
pade_type::Symbol
validation_final_time_fs::Float64
validation_sample_count::Int
```

Use exact constructor defaults:

```julia
pade_order=8,
tpsd_tolerance=5e-2,
pade_type=:Nm1,
validation_final_time_fs=100.0,
validation_sample_count=400,
```

Convert them with `Int(pade_order)`, `Float64(tpsd_tolerance)`,
`Symbol(pade_type)`, `Float64(validation_final_time_fs)`, and
`Int(validation_sample_count)` when constructing the concrete struct.

- [ ] **Step 4: Implement exact validation rules**

Add `tpsd_tolerance` and `validation_final_time_fs` to the finite and positive
scalar checks. Validate:

```julia
config.pade_order > 0 ||
    throw(ArgumentError("pade_order must be positive"))
config.pade_type in (:N, :Nm1) ||
    throw(ArgumentError("pade_type must be :N or :Nm1"))
config.validation_sample_count >= 2 ||
    throw(ArgumentError("validation_sample_count must be at least two"))
```

Remove all validation references to the deleted fields.

- [ ] **Step 5: Run the full library tests and confirm GREEN**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: all existing tests plus the nine new TPSD assertions pass.

- [ ] **Step 6: Commit the configuration contract**

```bash
git add examples/holstein/utils.jl test/runtests.jl
git commit -m "refactor: add Holstein TPSD configuration"
```

### Task 2: QFiND TPSD Decomposition and Dependency Cleanup

**Files:**
- Modify: `examples/holstein/holstein_brownian_heomtt.jl`
- Modify: `examples/holstein/Project.toml`
- Modify: `examples/holstein/README.md`

**Interfaces:**
- Consumes: the Task 1 `HolsteinConfig` TPSD and validation fields.
- Produces: `decompose_brownian_bcf(config::HolsteinConfig) -> NamedTuple` with fields `exponents`, `coefficients`, `relative_error`, `validation_times`, `reference_samples`, and `tpsd_samples`.
- Removes: `fit_brownian_bcf` and all ExpFit usage.
- Preserves: `build_holstein_heomtt(config, decomposition)` and the remaining propagation/output interfaces.

- [ ] **Step 1: Record the pre-change guarded-load failure after removing ExpFit from the environment**

First remove the `ExpFit` entries from `[deps]` and `[compat]` in
`examples/holstein/Project.toml`, then run without editing the executable:

```bash
julia --project=examples/holstein -e '
include("examples/holstein/holstein_brownian_heomtt.jl")
'
```

Expected: FAIL at `using ExpFit`, proving the executable still depends on the
package being removed.

- [ ] **Step 2: Replace the import and decomposition function**

Delete `using ExpFit`. Rename `fit_brownian_bcf` to
`decompose_brownian_bcf`, and replace the sampled ESPRIT call with:

```julia
exponents, coefficients = tpsd(
    spectral_density,
    config.temperature_K,
    config.pade_order,
    config.tpsd_tolerance;
    pade_type=config.pade_type,
)
exponents = ComplexF64.(exponents)
coefficients = ComplexF64.(coefficients)
```

Keep the empty, finite, and strictly-positive-real-decay checks. The positive
decay convention remains `C(t) = sum(c[k] * exp(-exponents[k] * t))`.
Do not multiply or divide `exponents` or `coefficients` by `icm2ifs`.

- [ ] **Step 3: Add the independent numerical BCF diagnostic**

After TPSD validation, construct the reference only for diagnostics:

```julia
reference_bcf = BosonicBCF(
    spectral_density,
    config.temperature_K;
    ub=config.bcf_upper_bound_cm,
)
validation_times = collect(range(
    0.0,
    config.validation_final_time_fs;
    length=config.validation_sample_count,
))
reference_samples = reference_bcf.(validation_times)
tpsd_samples = bcf_approx(validation_times, exponents, coefficients)
reference_norm = norm(reference_samples)
reference_norm > 0 || error("Brownian BCF validation norm is zero")
relative_error = norm(tpsd_samples - reference_samples) / reference_norm
isfinite(relative_error) || error("Brownian TPSD validation error is nonfinite")
```

Return exactly:

```julia
return (;
    exponents,
    coefficients,
    relative_error,
    validation_times,
    reference_samples,
    tpsd_samples,
)
```

- [ ] **Step 4: Update orchestration and console terminology**

In `main`, replace the BCF fit summary with:

```julia
println(
    "  TPSD: pade order=$(config.pade_order), " *
    "tolerance=$(config.tpsd_tolerance), type=$(config.pade_type)",
)
decomposition = decompose_brownian_bcf(config)
println(
    "  TPSD: $(length(decomposition.exponents)) terms, " *
    "validation relative error=" *
    "$(round(decomposition.relative_error, sigdigits=5))",
)
problem = build_holstein_heomtt(config, decomposition)
```

Rename local arguments and messages from `fit`/`fitted` to
`decomposition`/`TPSD`. Update the production warning to converge Padé order
and TPSD tolerance instead of a BCF fit window/tolerance.

- [ ] **Step 5: Remove ExpFit from reproducible setup documentation**

In `examples/holstein/README.md`:

- describe the environment as decomposition and plotting dependencies, not
  fitting dependencies;
- change “four unregistered dependencies” to “three unregistered
  dependencies”;
- remove the pinned ExpFit `PackageSpec` from the remote `Pkg.add` list;
- remove `"$DEV_ROOT/ExpFit"` from the sibling-checkout command;
- replace the final convergence sentence with Padé order, TPSD tolerance,
  hierarchy local size, time step, and TT tolerances;
- document the exact defaults `pade_order=8`, `tpsd_tolerance=5e-2`, and
  `pade_type=:Nm1`.

- [ ] **Step 6: Verify no stale fitting dependency or terminology remains**

Run:

```bash
rg -n -i 'ExpFit|esprit|fit_brownian|bcf_fit|fitted' \
  examples/holstein docs/superpowers/specs/2026-08-17-periodic-holstein-brownian-heomtt-design.md
```

Expected: no matches.

- [ ] **Step 7: Verify the exact TPSD decomposition and units contract**

Run in the configured example environment:

```bash
julia --project=examples/holstein -e '
include("examples/holstein/holstein_brownian_heomtt.jl")
config = HolsteinConfig(
    site_count=2,
    site_energies_cm=zeros(2),
    final_time_fs=1.0,
    time_step_fs=1.0,
    hierarchy_local_size=2,
)
decomposition = decompose_brownian_bcf(config)
@assert !isempty(decomposition.exponents)
@assert all(isfinite, decomposition.exponents)
@assert all(isfinite, decomposition.coefficients)
@assert all(>(0), real.(decomposition.exponents))
@assert isfinite(decomposition.relative_error)
@assert config.pade_order == 8
@assert config.tpsd_tolerance == 5e-2
println("terms=", length(decomposition.exponents))
println("validation_error=", decomposition.relative_error)
'
```

Expected: exit 0 with finite output. Inspect the source to confirm the only
`icm2ifs` multiplication remains the electronic `H_cm * icm2ifs` conversion;
TPSD outputs pass directly into `BathExp`.

- [ ] **Step 8: Run the library suite and guarded load**

Run:

```bash
julia --project=. test/runtests.jl
julia --project=examples/holstein -e '
include("examples/holstein/holstein_brownian_heomtt.jl")
@assert DEFAULT_CONFIG.pade_order == 8
@assert DEFAULT_CONFIG.tpsd_tolerance == 5e-2
'
git diff --check
```

Expected: all library tests pass, guarded load exits silently, and the diff
contains no whitespace errors.

- [ ] **Step 9: Commit the TPSD executable and dependency cleanup**

```bash
git add examples/holstein/holstein_brownian_heomtt.jl \
        examples/holstein/Project.toml \
        examples/holstein/README.md
git commit -m "refactor: use TPSD for Holstein bath decomposition"
```

### Task 3: Final Scientific and Repository Verification

**Files:**
- Verify: `examples/holstein/utils.jl`
- Verify: `examples/holstein/holstein_brownian_heomtt.jl`
- Verify: `examples/holstein/Project.toml`
- Verify: `examples/holstein/README.md`
- Verify: `test/runtests.jl`

**Interfaces:**
- Consumes: the complete TPSD change from Tasks 1-2.
- Produces: evidence that TPSD, units, dependency cleanup, and existing dynamics satisfy the approved design.

- [ ] **Step 1: Verify the committed diff and working tree**

Run:

```bash
git diff --check HEAD~2..HEAD
git status --short
git log -4 --oneline
```

Expected: no whitespace errors; only the pre-existing KSL CSV/PNG files may be
untracked; the two TPSD implementation commits are present.

- [ ] **Step 2: Run fresh tests and TPSD guarded validation**

Run the Task 2 Step 7 TPSD validation command again, followed by:

```bash
julia --project=. test/runtests.jl
```

Expected: positive finite decay rates, finite coefficients and validation
error, followed by a green library suite.

- [ ] **Step 3: Review the scientific data flow**

Confirm from source, with file/line evidence in the handoff:

1. `BrownianSD` receives `cm^-1` physical inputs;
2. QFiND `tpsd` performs its internal `icm2ifs` scaling;
3. returned `fs^-1`/`fs^-2` arrays pass unchanged to each `BathExp`;
4. only `H_cm` is explicitly multiplied by `icm2ifs`;
5. numerical `BosonicBCF` samples affect only `relative_error` reporting;
6. HEOM-TT propagation and population/trace/rank observables are unchanged.

- [ ] **Step 4: Report completion without running the default dynamics**

Report changed files, TPSD defaults, term count, validation error, exact test
results, example setup command, and the explicit fact that the 100 fs default
propagation was not run. Do not commit generated CSV, PNG, or Manifest files.
