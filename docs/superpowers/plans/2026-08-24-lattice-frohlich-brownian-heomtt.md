# Lattice Fröhlich Brownian HEOM-TT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently runnable periodic lattice Fröhlich HEOM-TT example whose normalized long-range diagonal coupling is directly comparable with the existing Holstein example.

**Architecture:** Create `examples/lattice_frohlich` and reuse the existing Holstein configuration, Brownian TPSD, propagation, measurement, and CSV machinery by including the Holstein example definitions. Keep the model-specific code isolated in Fröhlich kernel/coupling utilities and a distinct problem builder and executable; do not change existing Holstein behavior.

**Tech Stack:** Julia 1.11+, TTDynamics, TTSolver, HEOMKit, QFiND, CairoMakie, LinearAlgebra, Statistics, Test.

**Spec:** `docs/superpowers/specs/2026-08-24-lattice-frohlich-brownian-heomtt-design.md`

## Global Constraints

- Use periodic distance `min(abs(m-n), N-abs(m-n))` and the default raw kernel `(d^2 + 1)^(-3/2)`.
- Normalize every electronic-site column so `sum(abs2, f[:, n]) == 1`, preserving the site-local meaning of `reorganization_energy_cm`.
- Keep the electronic Hamiltonian, independent identical Brownian baths, TPSD decomposition, initial state, HEOM-TT construction, and tAMEn propagation consistent with the Holstein example.
- Do not alter the behavior of files under `examples/holstein`.
- Prefix all generated data and plot basenames with `lattice_frohlich_brownian_`.
- Including the executable must neither start dynamics nor write files.
- Preserve all unrelated uncommitted files and generated outputs already present in the worktree.

---

## File Structure

- Create `examples/lattice_frohlich/utils.jl`: periodic distance, raw/normalized kernel matrices, and diagonal coupling operators.
- Create `examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl`: reuse Holstein machinery, build the Fröhlich HEOM-TT problem, plot diagnostics, and provide the guarded executable.
- Create `examples/lattice_frohlich/Project.toml`: isolated example dependencies and compatibility bounds.
- Create `examples/lattice_frohlich/README.md`: equations, normalization convention, setup, execution, outputs, and convergence warning.
- Create `test/lattice_frohlich.jl`: utility, local-limit, problem-construction, output-name, and include-guard tests.
- Modify `test/runtests.jl`: include the new focused test file.

### Task 1: Periodic Fröhlich kernel and coupling operators

**Files:**
- Create: `examples/lattice_frohlich/utils.jl`
- Create: `test/lattice_frohlich.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `LinearAlgebra` standard library.
- Produces: `periodic_lattice_distance(m::Integer, n::Integer, site_count::Integer)::Int`.
- Produces: `default_frohlich_kernel(distance::Integer)::Float64`.
- Produces: `normalized_frohlich_kernel(site_count::Integer; kernel=default_frohlich_kernel)::Matrix{Float64}`, indexed as `[bath_site, electronic_site]`.
- Produces: `frohlich_coupling_operators(site_count::Integer; kernel=default_frohlich_kernel)::Vector{Matrix{ComplexF64}}`.

- [ ] **Step 1: Add the new test file to the suite and write failing utility tests**

Append this include after `include("heom_twin_space.jl")` in `test/runtests.jl`:

```julia
include("lattice_frohlich.jl")
```

Start `test/lattice_frohlich.jl` with:

```julia
using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

include("../examples/holstein/utils.jl")
include("../examples/lattice_frohlich/utils.jl")

@testset "Lattice Frohlich kernel utilities" begin
    @test periodic_lattice_distance(1, 1, 5) == 0
    @test periodic_lattice_distance(1, 5, 5) == 1
    @test periodic_lattice_distance(2, 5, 6) == 3
    @test periodic_lattice_distance(5, 1, 5) == 1
    @test_throws ArgumentError periodic_lattice_distance(0, 1, 5)
    @test_throws ArgumentError periodic_lattice_distance(1, 6, 5)
    @test_throws ArgumentError periodic_lattice_distance(1, 1, 1)

    @test default_frohlich_kernel(0) == 1.0
    @test default_frohlich_kernel(1) == 2.0^(-1.5)
    @test default_frohlich_kernel(2) == 5.0^(-1.5)
    @test_throws ArgumentError default_frohlich_kernel(-1)

    weights = normalized_frohlich_kernel(5)
    @test size(weights) == (5, 5)
    @test all(isfinite, weights)
    @test all(>(0), weights)
    @test weights == transpose(weights)
    @test all(isapprox(sum(abs2, weights[:, n]), 1.0; atol=1e-14) for n in 1:5)
    @test all(weights[:, mod1(n + 1, 5)] == circshift(weights[:, n], 1) for n in 1:5)

    operators = frohlich_coupling_operators(5)
    @test length(operators) == 5
    @test all(ishermitian, operators)
    @test all(isdiag, operators)
    @test all(diag(operators[m]) == weights[m, :] for m in 1:5)

    local_kernel = d -> d == 0 ? 1.0 : 0.0
    @test frohlich_coupling_operators(5; kernel=local_kernel) == site_projectors(5)
    @test_throws ArgumentError normalized_frohlich_kernel(1)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> NaN)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> -1.0)
    @test_throws ArgumentError normalized_frohlich_kernel(3; kernel=d -> 0.0)
end
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
julia --project=examples/holstein --startup-file=no -e 'using Test, LinearAlgebra; include("test/lattice_frohlich.jl")'
```

Expected: FAIL because `examples/lattice_frohlich/utils.jl` does not exist.

- [ ] **Step 3: Implement the minimal validated utilities**

Create `examples/lattice_frohlich/utils.jl`:

```julia
using LinearAlgebra

function periodic_lattice_distance(m::Integer, n::Integer, site_count::Integer)::Int
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    1 <= m <= site_count || throw(ArgumentError("m must index a lattice site"))
    1 <= n <= site_count || throw(ArgumentError("n must index a lattice site"))
    separation = abs(Int(m) - Int(n))
    return min(separation, Int(site_count) - separation)
end

function default_frohlich_kernel(distance::Integer)::Float64
    distance >= 0 || throw(ArgumentError("distance must be nonnegative"))
    return (Float64(distance)^2 + 1.0)^(-1.5)
end

function normalized_frohlich_kernel(site_count::Integer;
                                     kernel=default_frohlich_kernel)::Matrix{Float64}
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    count = Int(site_count)
    raw = Matrix{Float64}(undef, count, count)
    for bath_site in 1:count, electronic_site in 1:count
        value = Float64(kernel(periodic_lattice_distance(bath_site, electronic_site, count)))
        isfinite(value) || throw(ArgumentError("kernel values must be finite"))
        value >= 0 || throw(ArgumentError("kernel values must be nonnegative"))
        raw[bath_site, electronic_site] = value
    end
    weights = similar(raw)
    for electronic_site in 1:count
        scale = sqrt(sum(abs2, @view raw[:, electronic_site]))
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("each electronic site must have nonzero finite coupling"))
        weights[:, electronic_site] .= raw[:, electronic_site] ./ scale
    end
    all(n -> isapprox(sum(abs2, @view weights[:, n]), 1.0; atol=1e-14), 1:count) ||
        error("normalized Frohlich kernel violates unit column norm")
    return weights
end

function frohlich_coupling_operators(site_count::Integer;
                                     kernel=default_frohlich_kernel)
    weights = normalized_frohlich_kernel(site_count; kernel)
    return [Matrix(Diagonal(ComplexF64.(weights[m, :]))) for m in axes(weights, 1)]
end
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command again.

Expected: all tests in `Lattice Frohlich kernel utilities` PASS.

- [ ] **Step 5: Commit the utility slice**

```bash
git add examples/lattice_frohlich/utils.jl test/lattice_frohlich.jl test/runtests.jl
git commit -m "feat: add normalized lattice Frohlich couplings"
```

### Task 2: Fröhlich HEOM-TT problem builder

**Files:**
- Create: `examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl`
- Modify: `test/lattice_frohlich.jl`

**Interfaces:**
- Consumes: `HolsteinConfig`, `periodic_holstein_hamiltonian`, `decompose_brownian_bcf`, `run_dynamics`, and `write_population_csv` from the Holstein example.
- Consumes: `frohlich_coupling_operators` from Task 1.
- Produces: `build_lattice_frohlich_heomtt(config::HolsteinConfig, decomposition; kernel=default_frohlich_kernel)::NamedTuple` with fields `system`, `liouvillian`, `trace_observable`, and `population_observables`.

- [ ] **Step 1: Write failing construction and local-limit tests**

Append to `test/lattice_frohlich.jl`:

```julia
include("../examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl")

@testset "Lattice Frohlich HEOM-TT construction" begin
    config = HolsteinConfig(
        site_count=3,
        site_energies_cm=[0.0, 10.0, -5.0],
        hierarchy_local_size=2,
        final_time_fs=1.0,
        time_step_fs=1.0,
    )
    decomposition = (
        exponents=ComplexF64[0.25],
        coefficients=ComplexF64[0.03 - 0.01im],
    )
    problem = build_lattice_frohlich_heomtt(config, decomposition)
    expected = frohlich_coupling_operators(3)

    @test problem.system.H_sys ≈ periodic_holstein_hamiltonian(
        config.site_energies_cm,
        config.hopping_cm,
    ) * icm2ifs
    @test problem.system.noise.V == expected
    @test heom_tt_dimensions(problem.system) == [3, 3, 2, 2, 2]
    @test length(problem.population_observables) == 3
    initial = build_initial_state(problem.system, config.initial_site; tol=1e-12)
    @test tt_dims(initial) == [3, 3, 2, 2, 2]
    @test isapprox(real(tt_dot(problem.trace_observable, initial)), 1.0; atol=1e-12)

    local_kernel = d -> d == 0 ? 1.0 : 0.0
    local_problem = build_lattice_frohlich_heomtt(config, decomposition; kernel=local_kernel)
    @test local_problem.system.noise.V == site_projectors(3)
end
```

- [ ] **Step 2: Run the construction test and verify RED**

Run:

```bash
julia --project=examples/holstein --startup-file=no -e 'include("test/lattice_frohlich.jl")'
```

Expected: FAIL because `build_lattice_frohlich_heomtt` is not defined.

- [ ] **Step 3: Add imports/includes and the minimal problem builder**

Create `examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl` with this initial content:

```julia
using LinearAlgebra
using Statistics
using TTSolver
using TTDynamics
using QFiND
using HEOMKit
using HEOMKit: icm2ifs
using CairoMakie

if !isdefined(@__MODULE__, :HolsteinConfig)
    include("../holstein/utils.jl")
end
if !isdefined(@__MODULE__, :decompose_brownian_bcf)
    include("../holstein/holstein_brownian_heomtt.jl")
end
if !isdefined(@__MODULE__, :normalized_frohlich_kernel)
    include("utils.jl")
end

const DEFAULT_LATTICE_FROHLICH_CONFIG = HolsteinConfig()

function build_lattice_frohlich_heomtt(config::HolsteinConfig, decomposition;
                                        kernel=default_frohlich_kernel)
    validate_config(config)
    H_fs = periodic_holstein_hamiltonian(
        config.site_energies_cm,
        config.hopping_cm,
    ) * icm2ifs
    couplings = frohlich_coupling_operators(config.site_count; kernel)
    baths = [
        BathExp(decomposition.exponents, decomposition.coefficients, coupling)
        for coupling in couplings
    ]
    system = HEOMTTSystem(H_fs, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command again.

Expected: utility and construction testsets PASS.

- [ ] **Step 5: Commit the builder slice**

```bash
git add examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl test/lattice_frohlich.jl
git commit -m "feat: build lattice Frohlich HEOM-TT systems"
```

### Task 3: Fröhlich diagnostics and guarded executable

**Files:**
- Modify: `examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl`
- Modify: `test/lattice_frohlich.jl`

**Interfaces:**
- Consumes: `build_lattice_frohlich_heomtt` from Task 2 and shared Holstein `run_dynamics`/`write_population_csv`.
- Produces: `save_lattice_frohlich_diagnostic_plots(output_directory, config, result)::NTuple{3,String}`.
- Produces: `lattice_frohlich_main(config=DEFAULT_LATTICE_FROHLICH_CONFIG; output_directory=@__DIR__)`.

- [ ] **Step 1: Write failing output-name and include-guard tests**

Append to `test/lattice_frohlich.jl`:

```julia
@testset "Lattice Frohlich executable contract" begin
    @test DEFAULT_LATTICE_FROHLICH_CONFIG isa HolsteinConfig
    @test lattice_frohlich_output_paths("/tmp/example") == (
        csv=joinpath("/tmp/example", "lattice_frohlich_brownian_populations.csv"),
        populations=joinpath("/tmp/example", "lattice_frohlich_brownian_populations.png"),
        trace=joinpath("/tmp/example", "lattice_frohlich_brownian_trace.png"),
        rank=joinpath("/tmp/example", "lattice_frohlich_brownian_rank.png"),
    )

    example = abspath(joinpath(
        @__DIR__, "..", "examples", "lattice_frohlich",
        "lattice_frohlich_brownian_heomtt.jl",
    ))
    expression = "include($(repr(example))); print(\"lattice-frohlich-import-ok\")"
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(example)) -e $expression`
    mktempdir() do directory
        output = IOBuffer()
        process = run(pipeline(Cmd(command; dir=directory); stdout=output, stderr=output); wait=false)
        wait(process)
        text = String(take!(output))
        success(process) || println(stderr, text)
        @test success(process)
        @test occursin("lattice-frohlich-import-ok", text)
        @test isempty(readdir(directory))
    end
end
```

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```bash
julia --project=examples/holstein --startup-file=no -e 'include("test/lattice_frohlich.jl")'
```

Expected: FAIL because `lattice_frohlich_output_paths` is not defined.

- [ ] **Step 3: Implement exact paths, plots, main, and executable guard**

Append these functions to `lattice_frohlich_brownian_heomtt.jl`:

```julia
function lattice_frohlich_output_paths(output_directory::AbstractString)
    return (
        csv=joinpath(output_directory, "lattice_frohlich_brownian_populations.csv"),
        populations=joinpath(output_directory, "lattice_frohlich_brownian_populations.png"),
        trace=joinpath(output_directory, "lattice_frohlich_brownian_trace.png"),
        rank=joinpath(output_directory, "lattice_frohlich_brownian_rank.png"),
    )
end

function save_lattice_frohlich_diagnostic_plots(output_directory, config::HolsteinConfig, result)
    mkpath(output_directory)
    paths = lattice_frohlich_output_paths(output_directory)
    population_figure = Figure(size=(900, 600))
    population_axis = Axis(population_figure[1, 1]; xlabel="Time [fs]",
        ylabel="Population", title="Periodic lattice Frohlich HEOM-TT populations")
    for site in 1:config.site_count
        lines!(population_axis, result.times, result.populations[site, :];
               linewidth=2, label="site $site")
    end
    axislegend(population_axis; position=:rt)
    save(paths.populations, population_figure)

    trace_figure = Figure(size=(700, 450))
    trace_axis = Axis(trace_figure[1, 1]; xlabel="Time [fs]", ylabel="Tr[rho]",
                      title="Lattice Frohlich HEOM-TT trace conservation")
    lines!(trace_axis, result.times, result.trace; linewidth=2, label="trace")
    hlines!(trace_axis, [1.0]; linewidth=1, linestyle=:dash, color=:gray)
    axislegend(trace_axis; position=:rt)
    save(paths.trace, trace_figure)

    rank_figure = Figure(size=(700, 450))
    rank_axis = Axis(rank_figure[1, 1]; xlabel="Time [fs]", ylabel="TT rank",
                     title="Lattice Frohlich HEOM-TT rank evolution")
    lines!(rank_axis, result.times, result.maximum_rank; linewidth=2,
           label="maximum rank")
    lines!(rank_axis, result.times, result.mean_rank; linewidth=2,
           label="mean rank")
    axislegend(rank_axis; position=:rt)
    save(paths.rank, rank_figure)
    return (paths.populations, paths.trace, paths.rank)
end

function lattice_frohlich_main(config=DEFAULT_LATTICE_FROHLICH_CONFIG;
                                output_directory=@__DIR__)
    validate_config(config)
    println("Periodic lattice Frohlich model with normalized long-range diagonal coupling")
    decomposition = decompose_brownian_bcf(config)
    println("  TPSD terms per bath: $(length(decomposition.exponents))")
    println("  TPSD validation relative error: $(decomposition.relative_error)")
    problem = build_lattice_frohlich_heomtt(config, decomposition)
    result = run_dynamics(config, problem)
    paths = lattice_frohlich_output_paths(output_directory)
    mkpath(output_directory)
    write_population_csv(paths.csv, result)
    plot_paths = save_lattice_frohlich_diagnostic_plots(output_directory, config, result)
    println("  Wrote: $(paths.csv)")
    foreach(path -> println("  Wrote: $path"), plot_paths)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    lattice_frohlich_main()
end
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command again.

Expected: all Fröhlich testsets PASS and the subprocess prints the import marker without creating files.

- [ ] **Step 5: Commit the executable slice**

```bash
git add examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl test/lattice_frohlich.jl
git commit -m "feat: add lattice Frohlich HEOM-TT executable"
```

### Task 4: Standalone environment and documentation

**Files:**
- Create: `examples/lattice_frohlich/Project.toml`
- Create: `examples/lattice_frohlich/README.md`
- Modify: `test/lattice_frohlich.jl`

**Interfaces:**
- Consumes: the Task 3 executable and sibling development packages.
- Produces: a Julia environment loadable with `--project=examples/lattice_frohlich` and documented local/remote setup commands.

- [ ] **Step 1: Write a failing environment metadata test**

Append to the executable-contract testset:

```julia
project_text = read(joinpath(@__DIR__, "..", "examples", "lattice_frohlich", "Project.toml"), String)
@test occursin("HEOMKit = \"e865c079-6a0d-426d-afc2-450809b0c699\"", project_text)
@test occursin("TTDynamics = \"98f59f52-e016-4068-afe5-90126cbc5c1c\"", project_text)
@test occursin("QFiND = \"af16a7c1-792b-4481-9b88-c9c438329a9c\"", project_text)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
julia --project=examples/holstein --startup-file=no -e 'include("test/lattice_frohlich.jl")'
```

Expected: ERROR opening the absent `examples/lattice_frohlich/Project.toml`.

- [ ] **Step 3: Create the environment declaration**

Create `examples/lattice_frohlich/Project.toml` with:

```toml
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
HEOMKit = "e865c079-6a0d-426d-afc2-450809b0c699"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
QFiND = "af16a7c1-792b-4481-9b88-c9c438329a9c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TTDynamics = "98f59f52-e016-4068-afe5-90126cbc5c1c"
TTSolver = "6fd66e5e-56e0-488c-9f68-e25b72b7e6b0"

[compat]
CairoMakie = "0.12 - 0.15"
HEOMKit = "1"
LinearAlgebra = "1.11"
QFiND = "1"
Statistics = "1.11"
TTDynamics = "1"
TTSolver = "1"
julia = "1.11"
```

- [ ] **Step 4: Write the README with exact model and commands**

Create `examples/lattice_frohlich/README.md` containing:

```markdown
# Periodic lattice Fröhlich Brownian HEOM-TT example

This example replaces each local Holstein coupling projector with
`S_m = sum_n f_mn |n><n|`, where the periodic-distance kernel is
`k_mn = (d_mn^2 + 1)^(-3/2)` and each electronic-site column is normalized so
`sum_m f_mn^2 = 1`. Therefore `reorganization_energy_cm` retains the same
site-local meaning as in the Holstein example.

For a remote setup, add the known-compatible revisions and instantiate:

```bash
julia --project=examples/lattice_frohlich -e '
using Pkg
Pkg.add([
    Pkg.PackageSpec(url="https://github.com/htkhsh/TTDynamics.jl.git", rev="ace749d14ed8dd7eeef74cd21ba00dc11a4ade7b"),
    Pkg.PackageSpec(url="https://github.com/htkhsh/TTSolver.jl.git", rev="7a2eb169648257b9619c14bab349d63798bd220a"),
    Pkg.PackageSpec(url="https://github.com/DOC-Package/HEOMKit.jl.git", rev="06f7939557bce802ea967b6028e9899f55a120dd"),
    Pkg.PackageSpec(url="https://github.com/DOC-Package/QFiND.jl.git", rev="e5d1186cb2f07195c8dd07429861cbe898c9b4da"),
])
Pkg.instantiate()
'
```

From the TTDynamics repository root, develop the sibling working copies:

```bash
DEV_ROOT="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
julia --project=examples/lattice_frohlich -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path=path) for path in ARGS]); Pkg.instantiate()' "$PWD" "$DEV_ROOT/TTSolver" "$DEV_ROOT/HEOMKit" "$DEV_ROOT/QFiND"
```

Run the calculation with:

```bash
julia --project=examples/lattice_frohlich examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl
```

The calculation writes one CSV and three PNG diagnostics with the
`lattice_frohlich_brownian_` prefix. Converge the Padé order, TPSD tolerance,
hierarchy local size, time step, and TT tolerances before using results as
production data.
```

- [ ] **Step 5: Run the focused test and verify GREEN**

Run the Step 2 command again.

Expected: all Fröhlich tests PASS.

- [ ] **Step 6: Commit environment and documentation**

```bash
git add examples/lattice_frohlich/Project.toml examples/lattice_frohlich/README.md test/lattice_frohlich.jl
git commit -m "docs: add lattice Frohlich example workflow"
```

### Task 5: Package reload and full verification

**Files:**
- Potential generated ignored file: `examples/lattice_frohlich/Manifest.toml` (do not commit).
- Verify only: all Task 1–4 files and the existing TTDynamics test suite.

**Interfaces:**
- Consumes: completed example, tests, and sibling working copies.
- Produces: fresh evidence that the environment imports and the feature introduces no regression.

- [ ] **Step 1: Develop local packages into the new environment**

Run from `/home/takahashi/.julia/dev`:

```bash
julia --project=TTDynamics/examples/lattice_frohlich --startup-file=no -e '
using Pkg
paths = abspath.(["TTDynamics", "TTSolver", "HEOMKit", "QFiND"])
Pkg.develop([Pkg.PackageSpec(path=path) for path in paths])
Pkg.resolve()
'
```

Expected: the generated Manifest records local paths for all four packages and names `HEOMKit`, not `KaisouEOM`.

- [ ] **Step 2: Run the focused test from the TTDynamics environment**

```bash
julia --project=TTDynamics --startup-file=no -e 'include("TTDynamics/test/lattice_frohlich.jl")'
```

Expected: all Fröhlich testsets PASS.

- [ ] **Step 3: Verify the standalone example import**

```bash
julia --project=TTDynamics/examples/lattice_frohlich --startup-file=no -e '
include("TTDynamics/examples/lattice_frohlich/lattice_frohlich_brownian_heomtt.jl")
@assert DEFAULT_LATTICE_FROHLICH_CONFIG.site_count == 5
@assert !ispath("TTDynamics/examples/lattice_frohlich/lattice_frohlich_brownian_populations.csv")
println("lattice Frohlich import OK")
'
```

Expected: `lattice Frohlich import OK`, with no simulation or output files.

- [ ] **Step 4: Run the full package suite**

```bash
julia --project=TTDynamics --startup-file=no -e 'using Pkg; Pkg.test()'
```

Expected: all tests PASS. If the known current-correlation executable guard fails solely because the pre-existing untracked `examples/holstein_current_correlation/output` directory exists, preserve that user output, record the exact failure, and run every other test plus the focused Fröhlich test rather than deleting or moving the directory.

- [ ] **Step 5: Check names, diffs, and generated files**

```bash
rg -n 'KaisouEOM|kaisoueom' examples/lattice_frohlich test/lattice_frohlich.jl
git diff --check
git status --short
```

Expected: `rg` prints nothing; `git diff --check` exits zero; status contains only intended feature changes plus the preserved pre-existing user changes. Do not add the environment-local Manifest if it is ignored.

- [ ] **Step 6: Commit any verification-only corrections**

If verification required source or test corrections, stage only the files belonging to this feature and commit them:

```bash
git add examples/lattice_frohlich test/lattice_frohlich.jl test/runtests.jl
git commit -m "fix: complete lattice Frohlich example verification"
```

If no corrections were needed, do not create an empty commit.
