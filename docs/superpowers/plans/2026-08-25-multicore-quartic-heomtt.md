# Multi-core Quartic HEOM-TT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic multi-physical-core scaled HEOM-TT engine and use it to simulate one-site quartic-oscillator relaxation and two-site charge transfer.

**Architecture:** Physical density factors are column-major-vectorized TT cores followed by one TT core per hierarchy mode. A generic library layer constructs the scaled HEOM generator from a physical Liouvillian MPO, local coupling specifications, and validated forward/backward exponential correlations; the quartic example independently builds its oscillator basis, system MPO, numerical cBO correlation, ESPRIT fit, propagation, and outputs.

**Tech Stack:** Julia 1.11, TTDynamics, TTSolver TTMatrix/TTTensor/tAMEn, LinearAlgebra, ExpFit, QuadGK, CairoMakie, Test.

**Spec:** `docs/superpowers/specs/2026-08-25-multicore-quartic-heomtt-design.md`

## Global Constraints

- Preserve the existing `HEOMTTSystem` API and its twin-space implementation.
- Production paths must not expand a global dense system Hamiltonian, Liouvillian, or ADO hierarchy.
- Use column-major `vec`: `left(A) = I ⊗ A`, `right(A) = transpose(A) ⊗ I`.
- Core order is all physical Liouville cores followed by site-major, pole-minor hierarchy cores.
- The electronic Hilbert space is one dimension-`N` core, so electron number is one by construction.
- Do not add a bath counter term to the system MPO.
- Numerical quadrature and ESPRIT run only during correlation preparation, never during propagation.
- Keep example-only dependencies out of the root `Project.toml`.
- Follow TDD: observe every new test fail before implementing it, then commit only relevant files.
- Do not overwrite result files unless the caller explicitly passes `overwrite=true`.

## File Map

- `src/multicore_heom.jl`: generic correlation types, multi-core system validation, scaled generator, hierarchy-vacuum state, root observables.
- `src/TTDynamics.jl`: include and export the generic API.
- `examples/quartic/config.jl`: immutable physical, fit, hierarchy, and solver settings plus validation.
- `examples/quartic/oscillator.jl`: raw harmonic basis, quartic eigensystem, projected local operators, thermal density.
- `examples/quartic/correlation.jl`: cBO spectral density, stable finite-temperature quadrature, ESPRIT pole validation and diagnostics.
- `examples/quartic/model.jl`: open-chain electron core, physical Hamiltonian/Liouvillian MPOs, coupling specifications, initial state, observables.
- `examples/quartic/dynamics.jl`: Crank--Nicolson/tAMEn stepping, measurement, convergence diagnostics, CSV writing.
- `examples/quartic/plotting.jl`: optional CairoMakie plots only.
- `examples/quartic/one_site_relaxation.jl`: executable one-site workflow.
- `examples/quartic/two_site_transfer.jl`: executable four-case charge-transfer comparison.
- `examples/quartic/Project.toml`: self-contained example dependencies.
- `examples/quartic/README.md`: setup, commands, units, outputs, convergence, limitations.
- `test/multicore_heom.jl`: generic type validation and component-wise generator tests.
- `test/quartic.jl`: local model, correlation, fitting, propagation, and output tests.
- `test/runtests.jl`: include both new test files and example source files.

---

### Task 1: Quartic Oscillator Basis and Configuration

**Files:**
- Create: `examples/quartic/config.jl`
- Create: `examples/quartic/oscillator.jl`
- Create: `test/quartic.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `QuarticConfig(; kwargs...)`, `validate_quartic_config(::QuarticConfig)`, `QuarticMode`, `build_quartic_mode(config)::QuarticMode`, `quartic_thermal_density(mode, temperature)::Matrix{ComplexF64}`.
- `QuarticMode` fields: `energies`, `hamiltonian`, `q`, `q2`, `p`, `transform`, each with concrete `Float64`/`ComplexF64` arrays.

- [ ] **Step 1: Write the failing harmonic-limit and validation tests**

```julia
include("../examples/quartic/config.jl")
include("../examples/quartic/oscillator.jl")

@testset "quartic local oscillator" begin
    cfg = QuarticConfig(site_count=1, basis_frequency=2.0, omega_p=2.0,
                        K2=2.0, K4=0.0, d_raw=16, d_keep=6)
    mode = build_quartic_mode(cfg)
    @test mode.energies .- mode.energies[1] ≈ 2.0 .* (0:5) atol=1e-10
    @test ishermitian(mode.hamiltonian)
    @test mode.q2 ≈ mode.q * mode.q atol=2e-1
    @test_throws ArgumentError QuarticConfig(omega_p=0.0)
    @test_throws ArgumentError QuarticConfig(K4=-1.0)
    @test_throws ArgumentError QuarticConfig(d_raw=4, d_keep=5)
end
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `julia --project=. test/quartic.jl`

Expected: FAIL with `UndefVarError: QuarticConfig not defined` or a missing include file.

- [ ] **Step 3: Implement validated configuration and projected basis**

Define keyword defaults for every spec control, including `site_count`,
`site_energies`, `hopping`, `g`, `basis_frequency`, `d_raw`, `d_keep`,
`omega_p`, `K2`, `K4`, `temperature`, `bath_lambda`, `bath_gamma`, fit-grid,
quadrature, hierarchy, TT, tAMEn, and progress fields. Implement the operators
with these exact conventions:

```julia
a[n, n + 1] = sqrt(n)                 # n = 1:d_raw-1
q = (a + a') / sqrt(2basis_frequency)
p = -1im * sqrt(basis_frequency / 2) * (a - a')
h = omega_p / 2 * p^2 + K2 / 2 * q^2 + K4 / 4 * q^4
eigenbasis = eigen(Hermitian(h))
U = eigenbasis.vectors[:, 1:d_keep]
project(A) = ComplexF64.(U' * A * U)
```

Store the retained Hamiltonian as `Diagonal(eigenbasis.values[1:d_keep])`.
Compute the thermal density stably by subtracting the lowest retained energy
before exponentiation and normalizing its trace.

- [ ] **Step 4: Add convergence, parity, double-well, and thermal tests**

```julia
cfg12 = QuarticConfig(site_count=1, d_raw=12, d_keep=4, K2=-1.0, K4=0.2)
cfg20 = QuarticConfig(site_count=1, d_raw=20, d_keep=4, K2=-1.0, K4=0.2)
m12, m20 = build_quartic_mode(cfg12), build_quartic_mode(cfg20)
@test m12.energies ≈ m20.energies rtol=2e-3
@test ishermitian(m20.hamiltonian)
@test maximum(abs, diag(m20.q)) < 1e-10
ρth = quartic_thermal_density(m20, cfg20.temperature)
@test ishermitian(ρth)
@test isapprox(real(tr(ρth)), 1.0; atol=1e-12)
@test minimum(eigvals(Hermitian(ρth))) >= -1e-13
```

- [ ] **Step 5: Run and commit**

Run: `julia --project=. test/quartic.jl`

Expected: PASS for configuration and oscillator testsets.

```bash
git add examples/quartic/config.jl examples/quartic/oscillator.jl test/quartic.jl test/runtests.jl
git commit -m "feat: add quartic oscillator basis"
```

### Task 2: Generic Multi-core HEOM Types and Local Liouville Actions

**Files:**
- Create: `src/multicore_heom.jl`
- Create: `test/multicore_heom.jl`
- Modify: `src/TTDynamics.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `CorrelationFitMetadata`, `ExponentialCorrelation`, `LocalBathCoupling`, `MultiCoreHEOMTTSystem`, `liouville_left`, `liouville_right`, `multicore_heom_dimensions`.
- Consumes: `TTMatrix` and `tt_dims`/operator dimensions from TTSolver.

- [ ] **Step 1: Write failing vectorization and constructor tests**

```julia
@testset "multi-core HEOM types" begin
    A = ComplexF64[1 2im; 3 4]
    ρ = ComplexF64[0.4 0.1im; -0.1im 0.6]
    @test liouville_left(A) * vec(ρ) ≈ vec(A * ρ)
    @test liouville_right(A) * vec(ρ) ≈ vec(ρ * A)
    metadata = CorrelationFitMetadata(0, 0, 0, 0, 0.1, (0.0, 1.0),
        [1.0], 1.0, Inf, 1.0, String[])
    fit = ExponentialCorrelation([1 + 0im], [0.2 + 0.1im],
        [0.2 - 0.1im], metadata)
    @test fit.scales ≈ [sqrt(abs(0.2 + 0.1im))]
    @test_throws ArgumentError ExponentialCorrelation([-1 + 0im], [1im], [-1im], metadata)
end
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `julia --project=. test/multicore_heom.jl`

Expected: FAIL because `CorrelationFitMetadata` and Liouville helpers are undefined.

- [ ] **Step 3: Implement concrete immutable types and validation**

Use concrete vector fields from the spec. Provide this convenience constructor:

```julia
function ExponentialCorrelation(rates, forward, backward, metadata;
                                scale_epsilon=eps(Float64),
                                stability_tolerance=1e-12)
    # convert, validate equal nonzero lengths and finiteness
    # reject real(rate) < -stability_tolerance
    # clamp only tolerance-sized negative real parts to zero
    scales = sqrt.(max.(abs.(forward), abs.(backward), scale_epsilon))
    ExponentialCorrelation(rates_fixed, forward_fixed, backward_fixed,
                           scales, metadata)
end
```

Implement `liouville_left(A) = kron(I, ComplexF64.(A))` and
`liouville_right(A) = kron(transpose(ComplexF64.(A)), I)` with explicitly
sized complex identities. `LocalBathCoupling` stores `physical_core::Int`,
`operator::Matrix{ComplexF64}`, and `correlation::ExponentialCorrelation`.
`MultiCoreHEOMTTSystem` validates the physical MPO input/output dimensions,
local operator dimension, and one positive hierarchy size for every expanded
bath pole.

- [ ] **Step 4: Test dimension expansion and invalid inputs**

Build a two-core identity MPO, attach one two-pole coupling to core 2, and
assert:

```julia
@test multicore_heom_dimensions(system) == [4, 9, 3, 3]
@test_throws ArgumentError LocalBathCoupling(3, Matrix{ComplexF64}(I, 3, 3), fit)
@test_throws ArgumentError MultiCoreHEOMTTSystem(physical_L, [4, 9], couplings, [0])
```

- [ ] **Step 5: Run library tests and commit**

Run: `julia --project=. test/multicore_heom.jl`

Expected: PASS.

```bash
git add src/multicore_heom.jl src/TTDynamics.jl test/multicore_heom.jl test/runtests.jl
git commit -m "feat: add multicore HEOM system types"
```

### Task 3: Scaled Multi-core HEOM Generator

**Files:**
- Modify: `src/multicore_heom.jl`
- Modify: `test/multicore_heom.jl`

**Interfaces:**
- Consumes: `MultiCoreHEOMTTSystem`, `LocalBathCoupling`, `ExponentialCorrelation`.
- Produces: `build_multicore_heom_generator(system; tol=1e-12)::TTMatrix`, `build_multicore_heom_initial_state(system, physical_density_cores; tol=1e-12)::TTTensor`, `root_ado`, `root_expectation`.

- [ ] **Step 1: Write the failing dense component-equation test**

Construct one physical Liouville core of size four, one hierarchy core with
occupations `0:2`, a non-Hermitian complex test density vector, and one complex
forward/backward coefficient pair. Independently fill the dense expected
matrix by looping over hierarchy occupation `n` and applying:

```julia
expected[nblock, nblock] .+= physical_L - n * rate * I4
n < 2 && (expected[nblock, upblock] .+= -1im * scale * sqrt(n + 1) * (VL - VR))
n > 0 && (expected[nblock, downblock] .+=
    -1im * sqrt(n) / scale * (forward * VL - backward * VR))
```

Contract the returned two-core TTMatrix using the repository's TT full-format
conversion helper (or a test-local core contraction if none is exported), and
assert `dense_generator ≈ expected atol=1e-11`.

- [ ] **Step 2: Run and verify the operator test fails**

Run: `julia --project=. test/multicore_heom.jl`

Expected: FAIL with `UndefVarError: build_multicore_heom_generator not defined`.

- [ ] **Step 3: Implement hierarchy operators and generator terms**

For local hierarchy dimension `m`, define:

```julia
number = diagm(0 => ComplexF64.(0:m-1))
lower  = diagm(1 => sqrt.(ComplexF64.(1:m-1)))
raise  = diagm(-1 => sqrt.(ComplexF64.(1:m-1)))
```

Build each term with local TT Kronecker products across the physical and
hierarchy dimensions, add it to the accumulator, and `tt_round` after each
bath and once at the end. Select `lower`/`raise` by matching the independent
component test, not by name alone. Do not use a dense global operator.

- [ ] **Step 4: Implement vacuum product state and root contractions**

Vectorize each physical density core with `vec`, append `[1, 0, ...]` for every
hierarchy core, and build a rank-one `TTTensor`. `root_ado` fixes every
hierarchy index at one and returns the remaining physical `TTTensor`.
`root_expectation` accepts a physical observable TT vector and contracts it
with that root without expanding the hierarchy.

- [ ] **Step 5: Test two physical cores and two hierarchy modes**

Assert the dimensions, initial root trace, and a local expectation for a
product density. Also assert generator construction leaves the input MPO and
correlation arrays unchanged.

- [ ] **Step 6: Run and commit**

Run: `julia --project=. test/multicore_heom.jl`

Expected: PASS, including dense component equality.

```bash
git add src/multicore_heom.jl test/multicore_heom.jl
git commit -m "feat: build scaled multicore HEOM generator"
```

### Task 4: cBO Spectral Density and Numerical Correlation

**Files:**
- Create: `examples/quartic/Project.toml`
- Create: `examples/quartic/correlation.jl`
- Modify: `test/quartic.jl`

**Interfaces:**
- Produces: `AbstractSpectralDensity`, `CriticallyDampedBrownian`, `spectral_density`, `bath_correlation`, `sample_bath_correlation`.
- Consumes: temperature and quadrature fields from `QuarticConfig`.

- [ ] **Step 1: Declare the self-contained example environment**

Add exact dependency entries for TTDynamics, TTSolver, ExpFit
`3e354d20-bd3f-44fe-967b-b5188f355960`, QuadGK
`1fd47b50-473d-5c70-9696-f719f8f3bcdc`, CairoMakie, LinearAlgebra, and Printf.
Set `julia = "1.11"` and compatible package ranges without changing the root
environment.

- [ ] **Step 2: Write failing normalization and finite-correlation tests**

```julia
bath = CriticallyDampedBrownian(0.7, 1.3)
normalization, _ = quadgk(ω -> spectral_density(bath, ω) / ω,
                           0.0, Inf; rtol=1e-10)
@test normalization / π ≈ bath.lambda rtol=1e-8
@test isfinite(bath_correlation(bath, 0.0, 0.5; rtol=1e-8, atol=1e-10))
times = collect(0.0:0.1:0.5)
samples = sample_bath_correlation(bath, 0.5, times; rtol=1e-7, atol=1e-9)
@test length(samples) == length(times)
@test all(isfinite, samples)
```

- [ ] **Step 3: Run and verify failure**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: FAIL because `CriticallyDampedBrownian` is undefined.

- [ ] **Step 4: Implement stable integrands and sampling**

Use:

```julia
spectral_density(b::CriticallyDampedBrownian, ω) =
    4b.lambda * b.gamma^3 * ω / (ω^2 + b.gamma^2)^2

xcothx(x) = abs(x) < 1e-4 ? 1 + x^2 / 3 - x^4 / 45 : x * coth(x)
```

Rewrite `J(ω)coth(βω/2)` as `(J(ω)/ω) * (2/β) *
xcothx(βω/2)` so `ω=0` is finite. Integrate real and imaginary parts
separately with `quadgk` over `0..omega_integration_max`, or `0..Inf` when the
option is `nothing`.

- [ ] **Step 5: Add cutoff and tolerance convergence tests**

Compare samples using finite upper bounds `20gamma` and `40gamma` and tighter
tolerances; require relative agreement at the configured smoke-test tolerance.

- [ ] **Step 6: Run and commit**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: PASS for oscillator and correlation testsets.

```bash
git add examples/quartic/Project.toml examples/quartic/correlation.jl test/quartic.jl
git commit -m "feat: sample critically damped bath correlation"
```

### Task 5: Validated Complex ESPRIT Fit

**Files:**
- Modify: `examples/quartic/correlation.jl`
- Modify: `test/quartic.jl`

**Interfaces:**
- Produces: `fit_correlation_esprit(samples, times; kwargs...)::ExponentialCorrelation`, `evaluate_correlation(fit, times; branch=:forward)`, `validate_correlation_fit`.
- Consumes: `ExpFit.esprit`, generic correlation types from Task 2.

- [ ] **Step 1: Write a failing synthetic common-basis test**

```julia
times = collect(0.0:0.05:4.0)
rates = ComplexF64[0.3 + 0.7im, 0.3 - 0.7im, 0.305 + 0.705im]
coeff = ComplexF64[1.0 + 0.2im, 0.4 - 0.1im, 0.03 + 0.02im]
samples = [sum(coeff .* exp.(-rates .* t)) for t in times]
fit = fit_correlation_esprit(samples, times; fit_rank=3,
    pole_stability_tolerance=1e-10, fit_relative_tolerance=1e-8)
@test norm(evaluate_correlation(fit, times) - samples) / norm(samples) < 1e-8
@test norm(evaluate_correlation(fit, times; branch=:backward) - conj.(samples)) /
      norm(samples) < 1e-8
@test all(rate -> any(isapprox(rate, conj(other); atol=1e-8)
                           for other in fit.rates), fit.rates)
```

- [ ] **Step 2: Run and verify failure**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: FAIL because `fit_correlation_esprit` is undefined.

- [ ] **Step 3: Implement pole extraction and validation**

Require a uniformly spaced grid. Call `ExpFit.esprit(samples, dt, fit_rank)`
or its tolerance overload with `ncols=hankel_rows`. Convert the returned rates
to `ComplexF64`, reject nonfinite rates, `abs(imag(rate)) > pi/dt + tol`, and
`real(rate) < -pole_stability_tolerance`. Record tolerance-sized corrections.
Add a conjugate only when no existing rate lies within
`duplicate_pole_tolerance`; never merge two original nearby rates.

- [ ] **Step 4: Fit both coefficient branches and metadata**

Build `V[m,k] = exp(-rates[k] * times[m])`; solve `V \ samples` and
`V \ conj.(samples)` independently. Record `svd(V).S`, `cond(V)`, pairwise
minimum pole distance, maximum coefficient magnitude, training errors, and
the fit window. Evaluate validation data with the fixed rates and reject when
both absolute and relative configured error requirements fail.

- [ ] **Step 5: Add unstable-pole and cBO validation tests**

Assert explicit errors for nonuniform time grids and a growing synthetic
series. Fit a small directly integrated cBO grid, then compare on midpoints
excluded from training and assert finite diagnostics and bounded validation
error. Repeat one smoke fit at adjacent ranks and record both errors without
requiring monotonicity.

- [ ] **Step 6: Run and commit**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: PASS, including common-basis reconstruction.

```bash
git add examples/quartic/correlation.jl test/quartic.jl
git commit -m "feat: validate complex ESPRIT bath fits"
```

### Task 6: Quartic System MPO, Couplings, State, and Observables

**Files:**
- Create: `examples/quartic/model.jl`
- Modify: `test/quartic.jl`

**Interfaces:**
- Produces: `QuarticModel`, `build_system_mpo`, `build_quartic_model`, `build_factorized_initial_tt`, `quartic_observables`.
- Consumes: Tasks 1--5 and generic multi-core HEOM API.

- [ ] **Step 1: Write failing system-MPO and counter-term tests**

For `N=2`, `d_keep=3`, contract only the physical Hamiltonian MPO in the test
and compare it with an independently assembled dense matrix:

```julia
H_expected = kron(H_el, I3, I3) + kron(I2, hloc, I3) +
             kron(I2, I3, hloc) +
             g * kron(projector(1), q, I3) +
             g * kron(projector(2), I3, q)
@test dense(model.system_hamiltonian) ≈ H_expected atol=1e-11
cfg_b = with_bath(cfg; bath_lambda=2cfg.bath_lambda,
                       bath_gamma=3cfg.bath_gamma, temperature=2cfg.temperature)
@test dense(build_system_mpo(cfg_b, mode)) ≈ dense(build_system_mpo(cfg, mode))
```

- [ ] **Step 2: Run and verify failure**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: FAIL because `build_system_mpo` is undefined.

- [ ] **Step 3: Implement physical Hamiltonian and Liouvillian MPOs**

Build the open-chain `N × N` electronic Hamiltonian as the only dense
multi-site object (it is one local TT core). Build oscillator local terms and
`g * projector(n) ⊗ q_n` as separate `tt_mkron` products over Hilbert cores.
Convert each Hilbert-space product term to its column-major left and right
Liouville local factors and accumulate `-1im * (left - right)` directly as a
Liouville TTMatrix. Do not dense-contract this MPO outside tests.

- [ ] **Step 4: Attach one q-coupled bath per oscillator**

For oscillator physical core `n + 1`, construct
`LocalBathCoupling(n + 1, mode.q, fit)`. Reuse the same immutable fit for all
sites. Build `MultiCoreHEOMTTSystem` with uniform
`hierarchy_nmax + 1`, then construct its generator.

- [ ] **Step 5: Build initial state and observables**

Use `|initial_site><initial_site|` for the electron, the Task 1 thermal density
for every oscillator, and hierarchy vacuum. Build physical observable TT
vectors for electron projectors, oscillator `q`, `q2`, `hamiltonian`, physical
identity trace, and electron-number sum.

- [ ] **Step 6: Test limits and commit**

Assert initial trace and population, population sum one, `g=0` factorization at
the generator level, and `bath_lambda=0` produces either zero hierarchy
couplings or the explicitly documented closed-system shortcut without zero
scales.

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: PASS.

```bash
git add examples/quartic/model.jl test/quartic.jl
git commit -m "feat: build quartic electron oscillator MPO"
```

### Task 7: tAMEn Propagation and Diagnostics

**Files:**
- Create: `examples/quartic/dynamics.jl`
- Modify: `test/quartic.jl`

**Interfaces:**
- Produces: `measure_quartic_state`, `propagate_quartic_heom`, `write_quartic_csv`, `quartic_output_paths`.
- Consumes: `QuarticModel`, `tamen`, `extract_snapshot`, TT rank and dot helpers.

- [ ] **Step 1: Write failing initial-measurement and CSV tests**

```julia
measurement = measure_quartic_state(model.initial_state, model)
@test measurement.trace ≈ 1.0 atol=1e-11
@test measurement.electron_number ≈ 1.0 atol=1e-11
@test measurement.hermiticity_error < 1e-11
@test measurement.populations ≈ [1.0, 0.0] atol=1e-11

mktempdir() do dir
    result = propagate_quartic_heom(model, smoke_cfg; final_time=0.0)
    path = write_quartic_csv(joinpath(dir, "quartic.csv"), result)
    @test startswith(readline(path), "time,population_site_1")
    @test_throws ArgumentError write_quartic_csv(path, result)
end
```

- [ ] **Step 2: Run and verify failure**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: FAIL because `measure_quartic_state` is undefined.

- [ ] **Step 3: Implement root measurements without hierarchy expansion**

Contract the hierarchy-vacuum root against physical observable TT vectors.
For Hermiticity, reconstruct a dense root only in the configured smoke-sized
diagnostic path; otherwise contract an MPO norm of `rho-rho†`. Return a named
tuple containing populations, electron number, oscillator arrays, complex
trace, Hermiticity error, maximum rank, and mean rank. Throw on nonfinite data.

- [ ] **Step 4: Implement one-step CN/tAMEn propagation**

Follow the existing Holstein options exactly: append a temporal ones core,
call `tamen(space_time_state, dt * generator, tolerance, options)`, extract the
snapshot at normalized time one with scheme `"CN"`, and round the state.
Record the available residual/truncation arrays returned by `tamen`. Warn when
`maximum(tt_ranks(state)) >= tt_max_rank`; do not silently truncate to a new
physical model.

- [ ] **Step 5: Test zero-bath unitary dynamics and short dissipative smoke run**

For a tiny `N=1`, `d_keep=2` model, compare a single zero-bath root step to
`exp(-1im*H*dt) * rho * exp(1im*H*dt)` within the smoke solver tolerance. Run
two finite-bath steps and assert finite observables, trace tolerance,
Hermiticity tolerance, and electron number one.

- [ ] **Step 6: Implement atomic CSV output and test overwrite behavior**

Write time, all populations, trace real/imaginary, Hermiticity error,
electron number, all oscillator measurements, TT ranks, and residual fields.
Use `mktemp(dirname(path))` followed by rename; reject an existing target when
`overwrite=false`.

- [ ] **Step 7: Run and commit**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: PASS for zero-time, zero-bath, and short dissipative tests.

```bash
git add examples/quartic/dynamics.jl test/quartic.jl
git commit -m "feat: propagate and measure quartic HEOM-TT"
```

### Task 8: Reproducible One-site and Two-site Executables

**Files:**
- Create: `examples/quartic/plotting.jl`
- Create: `examples/quartic/one_site_relaxation.jl`
- Create: `examples/quartic/two_site_transfer.jl`
- Modify: `test/quartic.jl`

**Interfaces:**
- Produces: `one_site_main`, `two_site_main`, `plot_one_site_result`, `plot_two_site_comparison`.
- Consumes: all quartic example components.

- [ ] **Step 1: Write import-safe executable tests**

Include both executable files with an injected zero-duration config and assert
that inclusion creates no output. Call each `main` with test doubles for the
correlation builder, propagator, and plotter; assert the one-site workflow
returns one result and the two-site workflow labels exactly:

```julia
["harmonic_no_bath", "harmonic_bath", "quartic_no_bath", "quartic_bath"]
```

- [ ] **Step 2: Run and verify failure**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: FAIL because the executable files do not exist.

- [ ] **Step 3: Implement the one-site workflow**

Load configuration, build the local mode, sample and fit the bath once, print
rates, both coefficient vectors, errors, condition number, and closest-pole
distance, build the model, optionally replace its oscillator density with a
displaced retained-basis state, propagate, and save CSV plus `q`, `q2`, energy,
trace, and rank figures.

- [ ] **Step 4: Implement the two-site four-case workflow**

Use common electronic and numerical settings. Build one harmonic mode and one
quartic mode; compute one finite-bath fit and reuse it in both finite-bath
cases. Use the closed-system path for both no-bath cases. Save one CSV per case
and a population comparison plot with parameter/fit metadata in a text sidecar.

- [ ] **Step 5: Keep plotting optional and output-safe**

Load CairoMakie lazily only inside plot functions. All main functions accept
`output_directory`, `overwrite=false`, and injectable builders/runners so unit
tests neither plot nor perform expensive quadrature.

- [ ] **Step 6: Run and commit**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: PASS, with no files created by source inclusion.

```bash
git add examples/quartic/plotting.jl examples/quartic/one_site_relaxation.jl examples/quartic/two_site_transfer.jl test/quartic.jl
git commit -m "feat: add quartic HEOM-TT workflows"
```

### Task 9: Documentation, Convergence Harness, and Full Verification

**Files:**
- Create: `examples/quartic/README.md`
- Modify: `examples/quartic/dynamics.jl`
- Modify: `test/quartic.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: documented commands and `run_quartic_convergence(base_config; parameter, values, runner)` returning labeled diagnostics.
- Consumes: completed executables and measurements.

- [ ] **Step 1: Write failing convergence-harness tests**

Inject a runner returning deterministic population arrays and assert sweeps for
`d_raw`, `d_keep`, `basis_frequency`, `hierarchy_nmax`, `fit_rank`, `tt_cutoff`,
and `time_step` preserve labels, parameter values, final populations, trace,
fit error, and maximum rank.

- [ ] **Step 2: Implement the explicit convergence dispatcher**

Accept only the seven symbols above, rebuild the required upstream objects
when the changed parameter affects them, and return a vector of named tuples.
Reject unsupported symbols with `ArgumentError`; do not mutate the base
configuration.

- [ ] **Step 3: Document setup and exact commands**

The README must include local sibling-package development, environment
instantiation, these commands:

```bash
julia --project=examples/quartic examples/quartic/one_site_relaxation.jl
julia --project=examples/quartic examples/quartic/two_site_transfer.jl
julia --project=examples/quartic -e 'include("examples/quartic/two_site_transfer.jl"); two_site_main(QuarticConfig(final_time=0.1, time_step=0.1, d_raw=6, d_keep=2))'
```

Explain units, no-counter-term convention, core ordering, output columns,
fit diagnostics, all seven convergence controls, close-pole limitations,
hierarchy product cutoff, and the optional N=3 extension.

- [ ] **Step 4: Run formatting/static checks**

Run: `git diff --check`

Expected: no output.

Run: `julia --project=. -e 'using TTDynamics'`

Expected: exits zero.

- [ ] **Step 5: Run focused and complete tests**

Run: `julia --project=examples/quartic test/quartic.jl`

Expected: PASS.

Run: `julia --project=. test/runtests.jl`

Expected: all repository tests PASS.

- [ ] **Step 6: Run documented quick examples**

Run the README's zero/short-time commands in a fresh temporary output
directory. Expected: both exit zero; generated CSV values are finite; reported
trace, Hermiticity, electron number, fit errors, and TT ranks satisfy the quick
configuration tolerances.

- [ ] **Step 7: Commit documentation and convergence support**

```bash
git add examples/quartic/README.md examples/quartic/dynamics.jl test/quartic.jl test/runtests.jl
git commit -m "docs: document quartic HEOM-TT convergence"
```

- [ ] **Step 8: Perform final verification before completion**

Invoke `superpowers:verification-before-completion`, rerun `git diff --check`,
the focused generic test, focused quartic test, full suite, and documented quick
executables. Record commands, pass counts, solver parameters, observed TT
ranks, fit errors, trace/Hermiticity/electron-number errors, and any remaining
near-pole conditioning limitation in the handoff.
