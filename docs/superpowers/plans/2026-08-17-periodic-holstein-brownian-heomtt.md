# Periodic Holstein Brownian HEOM-TT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a readable executable example for HEOM-TT dynamics of a five-site periodic real-space Holstein model with independent Brownian baths.

**Architecture:** Keep model/configuration helpers in `examples/holstein/utils.jl`, where they can be tested without loading plotting or fitting packages. Put Brownian BCF fitting, HEOM-TT construction, tAMEn propagation, reporting, and output in `examples/holstein/holstein_brownian_heomtt.jl`, with a guarded `main()` entry point.

**Tech Stack:** Julia 1.11, TTDynamics, TTSolver, KaisouEOM, QFiND, ExpFit, CairoMakie, LinearAlgebra, Statistics, Test.

## Global Constraints

- Use the one-exciton site basis with `N = 5` and periodic boundary conditions.
- Use the hopping term `-J`, with `J = 400 cm^-1`.
- Use independent identical Brownian baths with `Omega = 1400 cm^-1`, `Gamma = 200 cm^-1`, `lambda = 600 cm^-1`, and `T = 300 K`.
- Start from an excitation localized at site 1 and evolve for 100 fs by default.
- Resolve generated output paths relative to `examples/holstein/`.
- Keep the full BCF fit and propagation out of CI.
- Do not modify or remove unrelated untracked example output files.
- Do not claim production convergence from default numerical settings.

---

## File Structure

- Create `examples/holstein/utils.jl`: configuration type, validation, periodic Hamiltonian, and site projectors.
- Create `examples/holstein/holstein_brownian_heomtt.jl`: Brownian fit, HEOM-TT assembly, evolution, CSV writing, plotting, and `main()`.
- Modify `test/runtests.jl`: unit tests for utilities and a small synthetic multi-bath HEOM-TT state.

### Task 1: Model Configuration and Periodic Hamiltonian

**Files:**
- Create: `examples/holstein/utils.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `HolsteinConfig(; kwargs...)` with concrete physical and solver fields.
- Produces: `validate_config(config::HolsteinConfig) -> HolsteinConfig`.
- Produces: `periodic_holstein_hamiltonian(site_energies::AbstractVector, hopping::Real) -> Matrix{ComplexF64}` in the same units as its arguments.
- Produces: `site_projectors(site_count::Integer) -> Vector{Matrix{ComplexF64}}`.

- [ ] **Step 1: Add failing Hamiltonian and projector tests**

Append the include and test set to `test/runtests.jl`:

```julia
include("../examples/holstein/utils.jl")

@testset "Periodic Holstein model utilities" begin
    H = periodic_holstein_hamiltonian(zeros(5), 400.0)
    @test ishermitian(H)
    @test diag(H) == zeros(5)
    @test H[1, 2] == H[2, 3] == H[3, 4] == H[4, 5] == H[5, 1] == -400.0
    @test H[1, 3] == H[2, 4] == H[3, 5] == 0.0

    H2 = periodic_holstein_hamiltonian([10.0, 20.0], 3.0)
    @test H2 == ComplexF64[10 -3; -3 20]

    projectors = site_projectors(5)
    @test all(ishermitian, projectors)
    @test sum(projectors) == Matrix{ComplexF64}(I, 5, 5)
    @test all(iszero(projectors[i] * projectors[j]) for i in 1:5 for j in 1:5 if i != j)
end
```

- [ ] **Step 2: Run the tests and verify the missing-helper failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: FAIL while including `examples/holstein/utils.jl`, because the file or helper definitions do not exist.

- [ ] **Step 3: Implement the minimal model helpers**

Create `examples/holstein/utils.jl` with documented functions. Build the matrix without double-counting the two-site bond:

```julia
using LinearAlgebra

function periodic_holstein_hamiltonian(site_energies::AbstractVector,
                                       hopping::Real)
    site_count = length(site_energies)
    site_count >= 2 || throw(ArgumentError("site_energies must contain at least two sites"))
    hopping >= 0 || throw(ArgumentError("hopping must be nonnegative"))

    H = Matrix(Diagonal(ComplexF64.(site_energies)))
    for site in 1:(site_count - 1)
        H[site, site + 1] = H[site + 1, site] = -hopping
    end
    if site_count > 2
        H[site_count, 1] = H[1, site_count] = -hopping
    end
    return H
end

function site_projectors(site_count::Integer)
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    projectors = Matrix{ComplexF64}[]
    for site in 1:site_count
        projector = zeros(ComplexF64, site_count, site_count)
        projector[site, site] = 1
        push!(projectors, projector)
    end
    return projectors
end
```

- [ ] **Step 4: Run the utility tests and verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: all existing and new tests PASS.

- [ ] **Step 5: Add failing configuration validation tests**

Extend the same test set:

```julia
    config = HolsteinConfig()
    @test validate_config(config) === config
    @test config.site_count == 5
    @test config.hopping_cm == 400.0
    @test config.brownian_frequency_cm == 1400.0
    @test config.reorganization_energy_cm == 600.0
    @test config.brownian_damping_cm == 200.0
    @test config.temperature_K == 300.0
    @test_throws ArgumentError HolsteinConfig(site_count=1)
    @test_throws ArgumentError HolsteinConfig(initial_site=6)
    @test_throws ArgumentError HolsteinConfig(time_step_fs=3.0, final_time_fs=100.0)
    @test_throws ArgumentError HolsteinConfig(hierarchy_local_size=0)
```

- [ ] **Step 6: Run the tests and verify the missing-type failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: FAIL with `UndefVarError: HolsteinConfig not defined`.

- [ ] **Step 7: Implement `HolsteinConfig` and exact validation rules**

Define the concrete storage type without `@kwdef`, then add the keyword
constructor shown below so `site_energies_cm` follows `site_count` by default:

```julia
struct HolsteinConfig
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
    bcf_final_time_fs::Float64
    bcf_sample_count::Int
    bcf_fit_tolerance::Float64
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

function HolsteinConfig(;
    site_count=5,
    site_energies_cm=zeros(site_count),
    hopping_cm=400.0,
    brownian_frequency_cm=1400.0,
    brownian_damping_cm=200.0,
    reorganization_energy_cm=600.0,
    temperature_K=300.0,
    initial_site=1,
    final_time_fs=100.0,
    time_step_fs=1.0,
    bcf_final_time_fs=100.0,
    bcf_sample_count=400,
    bcf_fit_tolerance=1e-6,
    bcf_upper_bound_cm=10_000.0,
    hierarchy_local_size=4,
    temporal_basis_size=3,
    tamen_tolerance=1e-4,
    operator_tolerance=1e-10,
    state_rounding_tolerance=1e-10,
    sweep_count=5,
    local_iterations=10,
    kick_rank=4,
    progress_interval=10,
)
    config = HolsteinConfig(
        Int(site_count), Float64.(site_energies_cm), Float64(hopping_cm),
        Float64(brownian_frequency_cm), Float64(brownian_damping_cm),
        Float64(reorganization_energy_cm), Float64(temperature_K),
        Int(initial_site), Float64(final_time_fs), Float64(time_step_fs),
        Float64(bcf_final_time_fs), Int(bcf_sample_count),
        Float64(bcf_fit_tolerance), Float64(bcf_upper_bound_cm),
        Int(hierarchy_local_size), Int(temporal_basis_size),
        Float64(tamen_tolerance), Float64(operator_tolerance),
        Float64(state_rounding_tolerance), Int(sweep_count),
        Int(local_iterations), Int(kick_rank), Int(progress_interval),
    )
    return validate_config(config)
end
```

In `validate_config`, validate length equality, index bounds, finite values,
nonnegative hopping, positivity for the remaining physical/numerical scales,
sample count >= 2, and
`isapprox(final_time_fs / time_step_fs, round(...); atol=1e-12, rtol=1e-12)`.
Return the same config after validation.

- [ ] **Step 8: Run tests and commit Task 1**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: all tests PASS.

```bash
git add examples/holstein/utils.jl test/runtests.jl
git commit -m "feat: add periodic Holstein model utilities"
```

### Task 2: Multi-Bath HEOM-TT Construction Contract

**Files:**
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `site_projectors(site_count)` from Task 1.
- Consumes: public `HEOMTTSystem`, `build_heom_liouvillian`, and `build_initial_state` from TTDynamics.
- Establishes: one `BathExp` per site with a site-local projector works with the existing HEOM-TT implementation.

- [ ] **Step 1: Add a synthetic two-site multi-bath test**

Append:

```julia
using KaisouEOM

@testset "Holstein multi-bath HEOM-TT initial state" begin
    projectors = site_projectors(2)
    baths = [BathExp(ComplexF64[-0.1], ComplexF64[0.02], V) for V in projectors]
    noise = NoiseExp(baths)
    system = HEOMTTSystem(ComplexF64[0 -1; -1 0], noise, 2)
    _, trace_observable, populations = build_heom_liouvillian(system; tol=1e-12)
    rho0 = build_initial_state(system, 1; tol=1e-12)

    @test real(tt_dot(trace_observable, rho0)) ≈ 1.0
    @test real(tt_dot(populations[1], rho0)) ≈ 1.0
    @test real(tt_dot(populations[2], rho0)) ≈ 0.0 atol=1e-14
end
```

- [ ] **Step 2: Run the focused test and inspect any failure**

Run: `julia --project=. test/runtests.jl`

Expected: PASS. If it fails, treat that as a correctness issue in the existing
multi-bath HEOM-TT path and use `superpowers:systematic-debugging` before any
source change; do not weaken the assertions.

- [ ] **Step 3: Commit the established integration contract**

```bash
git add test/runtests.jl
git commit -m "test: cover multi-bath HEOM-TT initial state"
```

### Task 3: Readable Brownian HEOM-TT Executable

**Files:**
- Create: `examples/holstein/holstein_brownian_heomtt.jl`

**Interfaces:**
- Consumes: all Task 1 helpers and `HolsteinConfig`.
- Produces: `fit_brownian_bcf(config) -> NamedTuple` with `exponents`, `coefficients`, `relative_error`, `sample_times`, `samples`, and `fitted_samples`.
- Produces: `build_holstein_heomtt(config, fit) -> NamedTuple` with `system`, `liouvillian`, `trace_observable`, and `population_observables`.
- Produces: `run_dynamics(config, problem) -> NamedTuple` with `times`, `populations`, `trace`, `maximum_rank`, and `mean_rank`.
- Produces: `write_population_csv(path, result)` and `save_diagnostic_plots(output_directory, config, result)`.
- Produces: guarded `main()` executable entry point.

- [ ] **Step 1: Create imports, documented configuration, and BCF fitting**

Start the script with:

```julia
using LinearAlgebra
using Statistics
using TTSolver
using TTDynamics
using QFiND
using ExpFit
using KaisouEOM
using KaisouEOM: icm2ifs
using CairoMakie

include("utils.jl")

const DEFAULT_CONFIG = HolsteinConfig()
```

Implement `fit_brownian_bcf` by constructing
`BrownianSD(config.brownian_frequency_cm, config.brownian_damping_cm,
config.reorganization_energy_cm)`, then
`BosonicBCF(spectral_density, config.temperature_K;
ub=config.bcf_upper_bound_cm)`. Sample the inclusive range from 0 to
`bcf_final_time_fs`, call `ExpFit.esprit(samples, sample_step,
bcf_fit_tolerance)`, and calculate `norm(fitted_samples - samples) /
norm(samples)`. Reject an empty fit, nonfinite error, nonfinite coefficients,
or any exponent with nonnegative real part.

- [ ] **Step 2: Add HEOM-TT problem construction**

Implement `build_holstein_heomtt` using:

```julia
H_cm = periodic_holstein_hamiltonian(config.site_energies_cm,
                                     config.hopping_cm)
H_fs = H_cm * icm2ifs
baths = [BathExp(fit.exponents, fit.coefficients, projector)
         for projector in site_projectors(config.site_count)]
noise = NoiseExp(baths)
system = HEOMTTSystem(H_fs, noise, config.hierarchy_local_size)
liouvillian, trace_observable, population_observables =
    build_heom_liouvillian(system; tol=config.operator_tolerance)
```

Return the four objects in a named tuple. Print the number of fitted terms per
site, total hierarchy cores, their local sizes, and initial TT ranks.

- [ ] **Step 3: Add measurement and tAMEn propagation functions**

Implement one focused `measure_state(rho, trace_observable,
population_observables)` helper. In `run_dynamics`, allocate arrays of length
`round(Int, final_time_fs / time_step_fs) + 1`, construct the initial state,
measure index 1, and for every subsequent time index use:

```julia
space_time_state = tkron(rho, tt_ones(config.temporal_basis_size;
                                      T=ComplexF64))
space_time_state, time_grid, _, _ = tamen(
    space_time_state,
    config.time_step_fs * problem.liouvillian,
    config.tamen_tolerance,
    options,
)
rho = extract_snapshot(space_time_state, time_grid, 1.0, "CN")
rho = tt_round(rho, config.state_rounding_tolerance)
```

Use the options `:verb => 0`, configured sweeps/local iterations/kick rank,
`:time_scheme => "CN"`, `:time_error_damp => 100.0`, and
`:obs => [problem.trace_observable]`. Record real populations and trace plus
maximum/mean ranks. Print progress at step zero, every `progress_interval`, and
the final step.

- [ ] **Step 4: Add dependency-free CSV output**

Implement `write_population_csv` using `open(path, "w") do io` and `join` so
no CSV/DataFrames dependency is introduced. The exact header is:

```text
time_fs,population_site_1,...,population_site_N,trace,max_rank,mean_rank
```

Write one row for each saved time and return the path.

- [ ] **Step 5: Add the three diagnostic plots**

Implement `save_diagnostic_plots` with CairoMakie:

- plot every site population against time with labels `site 1`, ..., `site N`;
- plot trace and a dashed horizontal reference at 1;
- plot maximum and mean TT rank;
- include the physical parameters in the population title;
- save the three exact filenames from the design to the supplied directory;
- close no global state and return the three paths.

- [ ] **Step 6: Add `main()` and the direct-execution guard**

`main(config=DEFAULT_CONFIG)` must validate the config, print a concise model
and numerical summary, call the fit/build/run/output functions, report maximum
absolute trace drift, and return the result. Use:

```julia
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

Add a prominent comment above numerical defaults saying users must converge
the BCF window/tolerance, hierarchy local size, time step, and TT tolerance for
production calculations.

- [ ] **Step 7: Verify loading does not launch the simulation**

Run:

```bash
julia --project=. -e 'include("examples/holstein/holstein_brownian_heomtt.jl"); @assert DEFAULT_CONFIG.site_count == 5'
```

Expected: exit code 0, with no propagation progress and no generated output.
If example-only packages are absent from the package environment, run with the
same Julia environment used by the existing `examples/heom/sb_ohmic_heomtt.jl`
and document that command in the final handoff.

- [ ] **Step 8: Run formatting/static checks and the test suite**

Run:

```bash
git diff --check
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: no whitespace errors and all tests PASS.

- [ ] **Step 9: Run a deliberately small smoke calculation**

Load the script and call `main` with a two-site, short-time configuration:

```bash
julia --project=. -e 'include("examples/holstein/holstein_brownian_heomtt.jl"); main(HolsteinConfig(site_count=2, site_energies_cm=zeros(2), final_time_fs=1.0, time_step_fs=1.0, bcf_final_time_fs=20.0, bcf_sample_count=80, hierarchy_local_size=2, progress_interval=1))'
```

Expected: exit code 0; finite fit error; two finite populations; finite trace;
and four output files under `examples/holstein/`. Do not commit generated CSV
or PNG files.

- [ ] **Step 10: Commit the executable example**

```bash
git add examples/holstein/holstein_brownian_heomtt.jl
git commit -m "feat: add Brownian Holstein HEOM-TT example"
```

### Task 4: Final Verification and Documentation Check

**Files:**
- Verify: `examples/holstein/utils.jl`
- Verify: `examples/holstein/holstein_brownian_heomtt.jl`
- Verify: `test/runtests.jl`

**Interfaces:**
- Consumes: the complete example and tests from Tasks 1-3.
- Produces: evidence that the committed implementation satisfies the design.

- [ ] **Step 1: Inspect repository status and ensure only expected generated files remain untracked**

Run: `git status --short`

Expected: no tracked modifications. Existing user-owned KSL outputs and any
new Holstein smoke-test CSV/PNG files may be untracked and must not be added.

- [ ] **Step 2: Run fresh verification**

Run:

```bash
git diff --check HEAD^
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'include("examples/holstein/holstein_brownian_heomtt.jl"); @assert periodic_holstein_hamiltonian(zeros(5), 400.0)[1, 5] == -400.0'
```

Expected: all commands exit 0.

- [ ] **Step 3: Review readability and design coverage**

Confirm each function has one responsibility, all public example helpers have
docstrings, variable names include units where ambiguity exists, the `-J`
sign and periodic bond are visible in code/comments, defaults match the
approved values, outputs use `@__DIR__`, and the convergence warning is next
to the numerical defaults.

- [ ] **Step 4: Record final commit and handoff**

Run: `git log -4 --oneline`

Report the created files, physical defaults, exact test/smoke commands and
results, generated untracked files, and any environment limitation. Do not
claim that the default 100 fs calculation was run unless it actually was.
