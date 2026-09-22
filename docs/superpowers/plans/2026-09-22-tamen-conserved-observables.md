# tAMEn Conserved-Observable API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public conserved-observable API to `TTSolver.tamen` and use it for opt-in trace preservation in standard HEOM, MultiCore HEOM, and MultiCore Redfield propagation.

**Architecture:** TTSolver validates and normalizes public conserved observables into its existing AMEn auxiliary-vector path without projecting the output state. TTDynamics adds one shared interval propagator, system-specific generator/trace construction, and three consistent public wrappers whose `preserve_trace=false` default retains current behavior.

**Tech Stack:** Julia 1.11, TTSolver `TTTensor`/`TTMatrix` and tAMEn, TTDynamics HEOM/Redfield types, HEOMKit, `LinearAlgebra`, `Random`, and `Test`.

**Spec:** `docs/superpowers/specs/2026-09-22-tamen-conserved-observables-design.md`

## Global Constraints

- Do not rescale, normalize, or project a propagated state after tAMEn.
- Conservation is tolerance-based AMEn basis retention, not a machine-precision guarantee.
- Keep `opts[:obs]` and positional `tamen` calls backward compatible.
- Reject simultaneous legacy and new spellings of the same tAMEn option.
- `preserve_trace` defaults to `false` in every TTDynamics wrapper.
- When `preserve_trace=true`, density-matrix 2-norm correction must be disabled.
- MultiCore HEOM preserves only `Tr(rho_root)`; it does not constrain purity, population, Frobenius norm, or higher-tier ADO traces.
- The selected tAMEn time scheme controls endpoint extraction; the default is `"cheb"`.
- Add no package dependency.

## Review Focus

- A vector containing two conserved observables must retain both; it must not be interpreted as the current `size(obs, 2) == 1` legacy shape. Task 2 pins this with two independent block invariants.
- Empty conserved-observable collections must act like no constraint without indexing `obs[1]`. Task 1 adds a public-call regression test.
- A MultiCore HEOM system with zero hierarchy modes must still produce a full-state trace observable with exactly the physical dimensions. Task 5 tests both open and closed systems.
- A one-point or repeated-value time grid must return the requested trajectory without entering tAMEn, while still validating option conflicts. Task 3 tests both paths.
- TTDynamics must not mutate the caller's `options` dictionary while adding trace and norm settings per interval. Task 3 snapshots and compares the input dictionary.

---

### Task 1: Add and validate the TTSolver keyword API

**Files:**
- Modify: `/home/takahashi/.julia/dev/TTSolver/src/tamen.jl:165-230`
- Modify: `/home/takahashi/.julia/dev/TTSolver/test/runtests.jl:160-205`

**Interfaces:**
- Consumes: existing `tamen(X, A, tol, opts=Dict{Symbol,Any}(), z=nothing)` and `tt_dims(::TTTensor)`.
- Produces: `tamen(X, A, tol, opts=Dict{Symbol,Any}(), z=nothing; conserved_observables=nothing, correct_2_norm=nothing)` and private `_normalize_conserved_observables(observables, dimensions)`.

- [ ] **Step 1: Write failing validation and compatibility tests**

Add this helper and testset after the existing `extract snapshot` tests in `TTSolver/test/runtests.jl`:

```julia
function keyword_tamen_problem()
    state = TTTensor([reshape([1.0, 0.0], 1, 2, 1)])
    space_time = tkron(state, tt_ones(3))
    generator = TTMatrix([reshape(zeros(2, 2), 1, 2, 2, 1)])
    observable = tt_ones(2)
    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => 1,
        :local_iters => 2,
        :time_scheme => "CN",
    )
    return space_time, generator, observable, options
end

@testset "tamen conserved-observable keyword validation" begin
    X, A, observable, options = keyword_tamen_problem()

    result = tamen(
        X, A, 1e-5, copy(options);
        conserved_observables=TTTensor[],
        correct_2_norm=false,
    )
    @test length(result) == 5
    legacy_empty = merge(copy(options), Dict(:obs => [observable]))
    @test length(tamen(
        X, A, 1e-5, legacy_empty;
        conserved_observables=TTTensor[],
    )) == 5

    @test_throws ArgumentError tamen(
        X, A, 1e-5, copy(options);
        conserved_observables=observable,
    )
    @test_throws ArgumentError tamen(
        X, A, 1e-5, copy(options);
        conserved_observables=[ones(2)],
    )
    @test_throws DimensionMismatch tamen(
        X, A, 1e-5, copy(options);
        conserved_observables=[tt_ones(3)],
    )
    @test_throws ArgumentError tamen(
        X, A, 1e-5, merge(copy(options), Dict(:obs => [observable]));
        conserved_observables=[observable],
    )
    @test_throws ArgumentError tamen(
        X, A, 1e-5,
        merge(copy(options), Dict(:correct_2_norm => false));
        correct_2_norm=false,
    )
    @test_throws ArgumentError tamen(
        X, A, 1e-5, copy(options);
        correct_2_norm=1,
    )
end
```

- [ ] **Step 2: Run the focused test and verify the keyword is missing**

Run:

```bash
cd /home/takahashi/.julia/dev/TTSolver
julia --project=. test/runtests.jl
```

Expected: FAIL in `tamen conserved-observable keyword validation` with an unsupported `conserved_observables` keyword.

- [ ] **Step 3: Implement observable normalization and keyword conflict checks**

Add this helper immediately before `tamen` in `TTSolver/src/tamen.jl`:

```julia
function _normalize_conserved_observables(observables, dimensions::Vector{Int})
    observables isa AbstractArray || throw(ArgumentError(
        "conserved_observables must be an array of TTTensor values",
    ))
    normalized = TTTensor[]
    for (index, observable) in enumerate(observables)
        observable isa TTTensor || throw(ArgumentError(
            "conserved observable $index must be a TTTensor",
        ))
        tt_dims(observable) == dimensions || throw(DimensionMismatch(
            "conserved observable $index dimensions must be $dimensions; got $(tt_dims(observable))",
        ))
        push!(normalized, observable)
    end
    reshape(normalized, 1, :)
end
```

Change the method header and insert keyword resolution immediately after `d -= 1`:

```julia
function tamen(X, A, tol, opts=Dict{Symbol,Any}(), z=nothing;
               conserved_observables=nothing,
               correct_2_norm=nothing)
    d, n, _, rx, vectype = check_vector(X, "x")
    d -= 1

    if conserved_observables !== nothing
        normalized_observables = _normalize_conserved_observables(
            conserved_observables, collect(Int, n[1:d]),
        )
        if !isempty(normalized_observables)
            haskey(opts, :obs) && throw(ArgumentError(
                "specify conserved observables through either conserved_observables or opts[:obs], not both",
            ))
            opts[:obs] = normalized_observables
        end
    end
    if correct_2_norm !== nothing
        correct_2_norm isa Bool || throw(ArgumentError(
            "correct_2_norm must be true, false, or nothing",
        ))
        haskey(opts, :correct_2_norm) && throw(ArgumentError(
            "specify 2-norm correction through either correct_2_norm or opts[:correct_2_norm], not both",
        ))
        opts[:correct_2_norm] = correct_2_norm
    end
```

Keep the remainder of `tamen` unchanged. The normalized `1 × N` matrix makes `size(obs, 2)` count a vector's observables correctly and leaves an empty array safe under the existing `isempty` guard.

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
cd /home/takahashi/.julia/dev/TTSolver
julia --project=. test/runtests.jl
```

Expected: PASS, including the new validation testset and the existing positional `tamen` tests.

- [ ] **Step 5: Commit the public API shell**

```bash
cd /home/takahashi/.julia/dev/TTSolver
git add src/tamen.jl test/runtests.jl
git commit -m "feat: add tAMEn conservation keywords"
```

### Task 2: Verify TTSolver conservation semantics and document them

**Files:**
- Modify: `/home/takahashi/.julia/dev/TTSolver/src/tamen.jl:165`
- Modify: `/home/takahashi/.julia/dev/TTSolver/test/runtests.jl`

**Interfaces:**
- Consumes: the Task 1 `tamen` keyword API.
- Produces: tested Chebyshev/CN conserved-linear-functional behavior, multiple-observable coverage, legacy `opts[:obs]` coverage, and the public `tamen` docstring.

- [ ] **Step 1: Write failing numerical conservation tests**

Add `using Random` beside the existing test imports, then add the following
after Task 1's validation testset:

```julia
function conservative_tamen_problem()
    generator_dense = [
        -1.0  0.5  0.0  0.0;
         1.0 -0.5  0.0  0.0;
         0.0  0.0 -0.2  0.4;
         0.0  0.0  0.2 -0.4
    ]
    initial_dense = [0.8, 0.2, 0.3, 0.7]
    initial = TTTensor([reshape(initial_dense, 1, 4, 1)])
    generator = TTMatrix([reshape(generator_dense, 1, 4, 4, 1)])
    first_mass = TTTensor([reshape([1.0, 1.0, 0.0, 0.0], 1, 4, 1)])
    second_mass = TTTensor([reshape([0.0, 0.0, 1.0, 1.0], 1, 4, 1)])
    return initial, generator, [first_mass, second_mass]
end

@testset "tamen conserved linear functionals" begin
    initial, generator, observables = conservative_tamen_problem()
    initial_values = [tt_dot(observable, initial) for observable in observables]

    for scheme in ("cheb", "CN")
        Random.seed!(20260922)
        X = tkron(initial, tt_ones(5))
        options = Dict{Symbol,Any}(
            :verb => 0,
            :nswp => 4,
            :local_iters => 8,
            :time_scheme => scheme,
            :time_error_damp => 1e3,
        )
        U, time_grid, _, _, _ = tamen(
            X, generator, 1e-6, options;
            conserved_observables=observables,
            correct_2_norm=false,
        )
        snapshots = extract_snapshot(U, time_grid, [0.0, 0.5, 1.0], scheme)
        for snapshot in snapshots, (observable, initial_value) in zip(observables, initial_values)
            @test tt_dot(observable, snapshot) ≈ initial_value atol=2e-5
        end
        @test abs(norm(last(snapshots)) - norm(initial)) > 1e-3
    end
end

@testset "tamen legacy observable option" begin
    initial, generator, observables = conservative_tamen_problem()
    X = tkron(initial, tt_ones(5))
    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => 4,
        :local_iters => 8,
        :time_scheme => "CN",
        :time_error_damp => 1e3,
        :obs => [first(observables)],
        :correct_2_norm => false,
    )
    U, time_grid, _, _, _ = tamen(X, generator, 1e-6, options)
    final_state = extract_snapshot(U, time_grid, 1.0, "CN")
    @test tt_dot(first(observables), final_state) ≈
          tt_dot(first(observables), initial) atol=2e-5
end
```

- [ ] **Step 2: Run the numerical tests and record actual stable tolerances**

Run:

```bash
cd /home/takahashi/.julia/dev/TTSolver
julia --project=. test/runtests.jl
```

Expected: PASS. If platform-dependent tAMEn error exceeds `2e-5`, increase
only the assertion `atol` to no more than `100 * 1e-6`; do not add state
normalization or loosen the solver tolerance.

- [ ] **Step 3: Add the public docstring**

Place this docstring immediately before `tamen`:

```julia
"""
    tamen(X, A, tol, opts=Dict{Symbol,Any}(), z=nothing;
          conserved_observables=nothing, correct_2_norm=nothing)

Integrate a linear evolution in tensor-train space with tAMEn.

`conserved_observables` is an array of `TTTensor` linear functionals whose
mode dimensions match the spatial modes of `X`. tAMEn retains their directions
in its AMEn approximation spaces so their contractions remain at their initial
values within solver accuracy. It does not project or rescale returned states.

`correct_2_norm` exposes the existing 2-norm correction as a keyword. Leave it
disabled for density-matrix dynamics unless the Frobenius norm is physically
conserved. The legacy `opts[:obs]` and `opts[:correct_2_norm]` spellings remain
supported, but a setting may not be supplied through both APIs.
"""
```

- [ ] **Step 4: Run the complete TTSolver suite**

Run:

```bash
cd /home/takahashi/.julia/dev/TTSolver
julia --project=. test/runtests.jl
git diff --check
```

Expected: all testsets PASS and `git diff --check` prints nothing.

- [ ] **Step 5: Commit tested TTSolver conservation behavior**

```bash
cd /home/takahashi/.julia/dev/TTSolver
git add src/tamen.jl test/runtests.jl
git commit -m "test: verify tAMEn conserved observables"
```

### Task 3: Centralize TTDynamics tAMEn propagation and migrate Redfield

**Files:**
- Create: `/home/takahashi/.julia/dev/TTDynamics/src/tamen_propagation.jl`
- Modify: `/home/takahashi/.julia/dev/TTDynamics/src/multicore_redfield.jl:94-136`
- Modify: `/home/takahashi/.julia/dev/TTDynamics/src/TTDynamics.jl:64-78`
- Create: `/home/takahashi/.julia/dev/TTDynamics/test/tamen_propagation.jl`
- Modify: `/home/takahashi/.julia/dev/TTDynamics/test/runtests.jl:8-12`

**Interfaces:**
- Consumes: Task 1's `tamen(X, A, tol, opts, z; conserved_observables, correct_2_norm)`, `extract_snapshot`, `tt_round`, and existing Redfield generator/observable builders.
- Produces: private `_propagate_tamen_trajectory(generator, state, times; kwargs)`, private `_validated_tamen_observables(observables, dimensions)`, and the extended `propagate_multicore_redfield_tamen(system, state, times; preserve_trace=false, conserved_observables=nothing)`.

- [ ] **Step 1: Write failing Redfield propagation and edge-case tests**

Create `TTDynamics/test/tamen_propagation.jl` with imports plus this first testset:

```julia
using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

function propagation_redfield_system()
    h = ComplexF64[0.2 0.3; 0.3 -0.1]
    v = ComplexF64[1 0; 0 -1]
    metadata = CorrelationFitMetadata(
        0, 0, 0, 0, 0.1, (0.0, 1.0), [1.0], 1.0, Inf, 1.0, String[],
    )
    correlation = ExponentialCorrelation(
        ComplexF64[0.8], ComplexF64[0.25 + 0.04im],
        ComplexF64[0.15 + 0.02im], metadata,
    )
    MultiCoreRedfieldTTSystem(
        tt_matrix(h), [h], [MultiCoreBathCoupling(1, v, correlation)],
    )
end

@testset "shared Redfield tAMEn propagation" begin
    system = propagation_redfield_system()
    state = product_tt_state([
        ComplexF64[1, 0], ComplexF64[1, 0],
    ])
    trace_observable = build_multicore_trace_observable(
        MultiCoreHEOMTTSystem(
            system.hamiltonian, MultiCoreBathCoupling[], Int[],
        ); tol=1e-14,
    )
    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => 2,
        :local_iters => 4,
        :time_scheme => "CN",
    )
    original_options = copy(options)
    trajectory = propagate_multicore_redfield_tamen(
        system, state, [0.0, 0.0, 0.01];
        temporal_basis_size=3,
        tolerance=1e-4,
        state_tolerance=1e-10,
        preserve_trace=true,
        options=options,
    )
    @test length(trajectory) == 3
    @test trajectory[2] === trajectory[1]
    @test options == original_options
    initial_trace = tt_dot(trace_observable, state)
    @test all(
        snapshot -> isapprox(
            tt_dot(trace_observable, snapshot), initial_trace; atol=5e-4,
        ),
        trajectory,
    )

    explicit_trajectory = propagate_multicore_redfield_tamen(
        system, state, [0.0, 0.01];
        temporal_basis_size=3,
        tolerance=1e-4,
        state_tolerance=1e-10,
        conserved_observables=[trace_observable],
        options=options,
    )
    @test tt_dot(trace_observable, last(explicit_trajectory)) ≈
          initial_trace atol=5e-4

    one_point = propagate_multicore_redfield_tamen(
        system, state, [0.0]; preserve_trace=true, options=options,
    )
    @test length(one_point) == 1
    @test only(one_point) === state
    @test_throws ArgumentError propagate_multicore_redfield_tamen(
        system, state, [0.0];
        preserve_trace=true,
        options=merge(copy(options), Dict(:correct_2_norm => true)),
    )
    @test_throws DimensionMismatch propagate_multicore_redfield_tamen(
        system, state, [0.0];
        conserved_observables=[tt_ones(3)],
        options=options,
    )
end
```

Append `include("tamen_propagation.jl")` to `TTDynamics/test/runtests.jl` after the existing MultiCore test includes.

- [ ] **Step 2: Run the focused TTDynamics suite and verify the new keywords fail**

Run:

```bash
cd /home/takahashi/.julia/dev/TTDynamics
julia --project=. test/runtests.jl
```

Expected: FAIL because `propagate_multicore_redfield_tamen` does not accept `preserve_trace`.

- [ ] **Step 3: Implement shared validation and interval propagation**

Create `TTDynamics/src/tamen_propagation.jl`. Implement these focused units:

```julia
function _validated_tamen_observables(observables, dimensions)
    observables === nothing && return TTTensor[]
    observables isa AbstractArray || throw(ArgumentError(
        "conserved_observables must be an array of TTTensor values",
    ))
    validated = TTTensor[]
    for (index, observable) in enumerate(observables)
        observable isa TTTensor || throw(ArgumentError(
            "conserved observable $index must be a TTTensor",
        ))
        tt_dims(observable) == dimensions || throw(DimensionMismatch(
            "conserved observable $index dimensions must be $dimensions; got $(tt_dims(observable))",
        ))
        push!(validated, observable)
    end
    validated
end

function _propagate_tamen_trajectory(
    generator::TTMatrix,
    state::TTTensor,
    times;
    expected_dimensions,
    temporal_basis_size::Integer=3,
    tolerance::Real=1e-8,
    state_tolerance::Real=1e-12,
    options=Dict{Symbol,Any}(),
    conserved_observables=nothing,
    preserve_trace::Bool=false,
    trace_observable=nothing,
)
    temporal_basis_size >= 2 || throw(ArgumentError(
        "temporal_basis_size must be at least two",
    ))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError(
        "tolerance must be finite and positive",
    ))
    isfinite(state_tolerance) && state_tolerance >= 0 || throw(ArgumentError(
        "state_tolerance must be finite and nonnegative",
    ))
    tt_dims(state) == expected_dimensions || throw(DimensionMismatch(
        "initial state dimensions must be $expected_dimensions; got $(tt_dims(state))",
    ))
    time_grid = collect(Float64, times)
    !isempty(time_grid) || throw(ArgumentError("times must not be empty"))
    all(isfinite, time_grid) && all(diff(time_grid) .>= 0) || throw(ArgumentError(
        "times must be finite and nondecreasing",
    ))

    effective = _validated_tamen_observables(
        conserved_observables, expected_dimensions,
    )
    if preserve_trace
        trace_observable isa TTTensor || error(
            "trace preservation requires a full-state trace observable",
        )
        tt_dims(trace_observable) == expected_dimensions || error(
            "internal trace observable dimensions do not match the state",
        )
        pushfirst!(effective, trace_observable)
    end

    local_options = Dict{Symbol,Any}(options)
    !isempty(effective) && haskey(local_options, :obs) && throw(ArgumentError(
        "new conservation keywords cannot be combined with options[:obs]",
    ))
    correct_2_norm = nothing
    if preserve_trace
        if haskey(local_options, :correct_2_norm)
            local_options[:correct_2_norm] === false || throw(ArgumentError(
                "preserve_trace=true is incompatible with 2-norm correction",
            ))
        else
            correct_2_norm = false
        end
    end

    trajectory = TTTensor[state]
    current = state
    for interval in 2:length(time_grid)
        step = time_grid[interval] - time_grid[interval - 1]
        step == 0 && (push!(trajectory, current); continue)
        interval_options = copy(local_options)
        scheme = String(get(interval_options, :time_scheme, "cheb"))
        space_time = tkron(
            current, tt_ones(Int(temporal_basis_size); T=ComplexF64),
        )
        space_time, local_time_grid, _, _, _ = tamen(
            space_time, step * generator, tolerance, interval_options;
            conserved_observables=isempty(effective) ? nothing : effective,
            correct_2_norm=correct_2_norm,
        )
        current = tt_round(
            extract_snapshot(space_time, local_time_grid, 1.0, scheme),
            Float64(state_tolerance),
        )
        push!(trajectory, current)
    end
    trajectory
end
```

The error checks occur before the interval loop so one-point grids cannot bypass configuration validation.

- [ ] **Step 4: Move the Redfield wrapper onto the shared implementation**

Remove the existing propagation method from `src/multicore_redfield.jl`. Add this method to `src/tamen_propagation.jl`:

```julia
function propagate_multicore_redfield_tamen(
    system::MultiCoreRedfieldTTSystem,
    state::TTTensor,
    times;
    temporal_basis_size::Integer=3,
    tolerance::Real=1e-8,
    state_tolerance::Real=1e-12,
    generator_tolerance::Real=1e-12,
    options=Dict{Symbol,Any}(),
    conserved_observables=nothing,
    preserve_trace::Bool=false,
)
    generator = build_multicore_redfield_generator(
        system; tol=Float64(generator_tolerance),
    )
    dimensions = reduce(
        vcat,
        ([dimension, dimension] for dimension in system.hilbert_dimensions);
        init=Int[],
    )
    trace_system = MultiCoreHEOMTTSystem(
        system.hamiltonian, MultiCoreBathCoupling[], Int[],
    )
    trace_observable = preserve_trace ?
        build_multicore_trace_observable(trace_system) : nothing
    _propagate_tamen_trajectory(
        generator, state, times;
        expected_dimensions=dimensions,
        temporal_basis_size,
        tolerance,
        state_tolerance,
        options,
        conserved_observables,
        preserve_trace,
        trace_observable,
    )
end
```

Include `tamen_propagation.jl` after `multicore_redfield.jl` in `src/TTDynamics.jl`. Keep the existing export name.

- [ ] **Step 5: Run focused and complete TTDynamics tests**

Run:

```bash
cd /home/takahashi/.julia/dev/TTDynamics
julia --project=. test/runtests.jl
```

Expected: PASS, including repeated-time identity, one-point preflight checks, trace preservation, caller-option immutability, and all pre-existing Redfield tests.

- [ ] **Step 6: Commit the shared Redfield propagation layer**

```bash
cd /home/takahashi/.julia/dev/TTDynamics
git add src/TTDynamics.jl src/multicore_redfield.jl src/tamen_propagation.jl test/runtests.jl test/tamen_propagation.jl
git commit -m "refactor: centralize tAMEn propagation"
```

### Task 4: Add standard HEOM trace-preserving propagation

**Files:**
- Modify: `/home/takahashi/.julia/dev/TTDynamics/src/tamen_propagation.jl`
- Modify: `/home/takahashi/.julia/dev/TTDynamics/src/TTDynamics.jl:42-68`
- Modify: `/home/takahashi/.julia/dev/TTDynamics/test/tamen_propagation.jl`

**Interfaces:**
- Consumes: `_propagate_tamen_trajectory`, `build_heom_liouvillian`, `heom_tt_dimensions`, and `build_initial_state`.
- Produces: `propagate_heom_tamen(system::HEOMTTSystem, state::TTTensor, times; temporal_basis_size=3, tolerance=1e-8, state_tolerance=1e-12, generator_tolerance=1e-12, options=Dict{Symbol,Any}(), conserved_observables=nothing, preserve_trace=false)`.

- [ ] **Step 1: Write the failing standard HEOM test**

Append this constructor and testset to `test/tamen_propagation.jl`:

```julia
function propagation_heom_system()
    h = ComplexF64[0.3 0.2; 0.2 -0.1]
    v = ComplexF64[1 0; 0 -1]
    bath = BathExp(
        ComplexF64[0.6], ComplexF64[0.04 - 0.01im], v,
    )
    HEOMTTSystem(h, NoiseExp([bath]), 2)
end

@testset "standard HEOM tAMEn trace preservation" begin
    system = propagation_heom_system()
    state = build_initial_state(system, 1; tol=1e-14)
    _, trace_observable, _ = build_heom_liouvillian(system; tol=1e-14)
    options = Dict{Symbol,Any}(
        :verb => 0,
        :nswp => 2,
        :local_iters => 4,
        :time_scheme => "CN",
    )
    trajectory = propagate_heom_tamen(
        system, state, [0.0, 0.01];
        temporal_basis_size=3,
        tolerance=1e-4,
        state_tolerance=1e-10,
        preserve_trace=true,
        options=options,
    )
    initial_trace = tt_dot(trace_observable, state)
    @test length(trajectory) == 2
    @test all(
        snapshot -> isapprox(
            tt_dot(trace_observable, snapshot), initial_trace; atol=5e-4,
        ),
        trajectory,
    )
    @test all(snapshot -> tt_dims(snapshot) == heom_tt_dimensions(system), trajectory)
end
```

- [ ] **Step 2: Run the suite and verify `propagate_heom_tamen` is undefined**

Run:

```bash
cd /home/takahashi/.julia/dev/TTDynamics
julia --project=. test/runtests.jl
```

Expected: FAIL with `UndefVarError: propagate_heom_tamen not defined`.

- [ ] **Step 3: Implement and export the standard HEOM wrapper**

Add this method to `src/tamen_propagation.jl`:

```julia
function propagate_heom_tamen(
    system::HEOMTTSystem,
    state::TTTensor,
    times;
    temporal_basis_size::Integer=3,
    tolerance::Real=1e-8,
    state_tolerance::Real=1e-12,
    generator_tolerance::Real=1e-12,
    options=Dict{Symbol,Any}(),
    conserved_observables=nothing,
    preserve_trace::Bool=false,
)
    generator, trace_observable, _ = build_heom_liouvillian(
        system; tol=Float64(generator_tolerance),
    )
    _propagate_tamen_trajectory(
        generator, state, times;
        expected_dimensions=heom_tt_dimensions(system),
        temporal_basis_size,
        tolerance,
        state_tolerance,
        options,
        conserved_observables,
        preserve_trace,
        trace_observable=preserve_trace ? trace_observable : nothing,
    )
end
```

Add `propagate_heom_tamen` to the HEOM export block in `src/TTDynamics.jl`.

- [ ] **Step 4: Run all TTDynamics tests**

Run:

```bash
cd /home/takahashi/.julia/dev/TTDynamics
julia --project=. test/runtests.jl
```

Expected: PASS, including the standard HEOM root-trace assertion.

- [ ] **Step 5: Commit standard HEOM propagation**

```bash
cd /home/takahashi/.julia/dev/TTDynamics
git add src/TTDynamics.jl src/tamen_propagation.jl test/tamen_propagation.jl
git commit -m "feat: propagate HEOM with tAMEn constraints"
```

### Task 5: Add MultiCore HEOM full-state trace preservation

**Files:**
- Modify: `/home/takahashi/.julia/dev/TTDynamics/src/tamen_propagation.jl`
- Modify: `/home/takahashi/.julia/dev/TTDynamics/src/TTDynamics.jl:52-68`
- Modify: `/home/takahashi/.julia/dev/TTDynamics/test/tamen_propagation.jl`

**Interfaces:**
- Consumes: `_hierarchy_vacuum_cores`, `build_multicore_trace_observable`, `multicore_heom_dimensions`, `build_multicore_heom_generator`, and `_propagate_tamen_trajectory`.
- Produces: private `_full_multicore_heom_trace_observable(system; tol)` and public `propagate_multicore_heom_tamen(system::MultiCoreHEOMTTSystem, state::TTTensor, times; temporal_basis_size=3, tolerance=1e-8, state_tolerance=1e-12, generator_tolerance=1e-12, options=Dict{Symbol,Any}(), conserved_observables=nothing, preserve_trace=false)`.

- [ ] **Step 1: Write failing full-observable and MultiCore propagation tests**

Append these helpers and testset to `test/tamen_propagation.jl`:

```julia
function propagation_multicore_heom_system(; open_system=true)
    h1 = ComplexF64[0.2 0.1; 0.1 -0.2]
    h2 = ComplexF64[0.0 0.3; 0.3 0.4]
    hamiltonian = tt_mkron(h1, h2)
    open_system || return MultiCoreHEOMTTSystem(
        hamiltonian, MultiCoreBathCoupling[], Int[],
    )
    v = ComplexF64[1 0; 0 -1]
    metadata = CorrelationFitMetadata(
        0, 0, 0, 0, 0.1, (0.0, 1.0), [1.0], 1.0, Inf, 1.0, String[],
    )
    correlation = ExponentialCorrelation(
        ComplexF64[0.7], ComplexF64[0.2 - 0.03im],
        ComplexF64[0.15 + 0.04im], metadata,
    )
    MultiCoreHEOMTTSystem(
        hamiltonian, [MultiCoreBathCoupling(1, v, correlation)], [2],
    )
end

function partial_trace_two_subsystems(density, dimensions, keep)
    d1, d2 = dimensions
    if keep == 1
        reduced = zeros(ComplexF64, d1, d1)
        for ket in 1:d1, bra in 1:d1, traced in 1:d2
            reduced[ket, bra] += density[
                (ket - 1) * d2 + traced,
                (bra - 1) * d2 + traced,
            ]
        end
        return reduced
    end
    reduced = zeros(ComplexF64, d2, d2)
    for ket in 1:d2, bra in 1:d2, traced in 1:d1
        reduced[ket, bra] += density[
            (traced - 1) * d2 + ket,
            (traced - 1) * d2 + bra,
        ]
    end
    reduced
end

@testset "MultiCore HEOM tAMEn trace preservation" begin
    rho1 = ComplexF64[0.7 0.1; 0.1 0.3]
    rho2 = ComplexF64[0.4 0.05im; -0.05im 0.6]

    for open_system in (false, true)
        system = propagation_multicore_heom_system(; open_system)
        state = build_multicore_heom_initial_state(
            system, [rho1, rho2]; tol=1e-14,
        )
        full_trace = TTDynamics._full_multicore_heom_trace_observable(
            system; tol=1e-14,
        )
        @test tt_dims(full_trace) == multicore_heom_dimensions(system)
        physical_trace = build_multicore_trace_observable(system; tol=1e-14)
        @test tt_dot(full_trace, state) ≈
              multicore_expectation(physical_trace, state, system) atol=1e-12

        trajectory = propagate_multicore_heom_tamen(
            system, state, [0.0, 0.01];
            temporal_basis_size=3,
            tolerance=1e-4,
            state_tolerance=1e-10,
            preserve_trace=true,
            options=Dict{Symbol,Any}(
                :verb => 0,
                :nswp => 2,
                :local_iters => 4,
                :time_scheme => "CN",
            ),
        )
        initial_trace = tt_dot(full_trace, state)
        @test all(
            snapshot -> isapprox(
                tt_dot(full_trace, snapshot), initial_trace; atol=5e-4,
            ),
            trajectory,
        )
        final_density = root_density_matrix(last(trajectory), system)
        for subsystem in 1:2
            reduced = partial_trace_two_subsystems(
                final_density, system.hilbert_dimensions, subsystem,
            )
            @test tr(reduced) ≈ initial_trace atol=5e-4
        end
    end
end
```

- [ ] **Step 2: Run the suite and verify the helper and wrapper are undefined**

Run:

```bash
cd /home/takahashi/.julia/dev/TTDynamics
julia --project=. test/runtests.jl
```

Expected: FAIL because `_full_multicore_heom_trace_observable` and `propagate_multicore_heom_tamen` do not exist.

- [ ] **Step 3: Implement the full-state MultiCore HEOM trace observable**

Add to `src/tamen_propagation.jl`:

```julia
function _full_multicore_heom_trace_observable(
    system::MultiCoreHEOMTTSystem;
    tol::Float64=1e-12,
)
    physical = build_multicore_trace_observable(system; tol)
    cores = [ComplexF64.(copy(core)) for core in physical.cores]
    append!(cores, _hierarchy_vacuum_cores(system.hierarchy_sizes))
    full = TTTensor(cores)
    tt_dims(full) == multicore_heom_dimensions(system) || error(
        "internal MultiCore HEOM trace observable dimensions are inconsistent",
    )
    full
end
```

Do not change `build_multicore_trace_observable`; its physical-only contract remains public and is used by `multicore_expectation`.

- [ ] **Step 4: Implement and export the MultiCore HEOM wrapper**

Add to `src/tamen_propagation.jl`:

```julia
function propagate_multicore_heom_tamen(
    system::MultiCoreHEOMTTSystem,
    state::TTTensor,
    times;
    temporal_basis_size::Integer=3,
    tolerance::Real=1e-8,
    state_tolerance::Real=1e-12,
    generator_tolerance::Real=1e-12,
    options=Dict{Symbol,Any}(),
    conserved_observables=nothing,
    preserve_trace::Bool=false,
)
    generator = build_multicore_heom_generator(
        system; tol=Float64(generator_tolerance),
    )
    trace_observable = preserve_trace ?
        _full_multicore_heom_trace_observable(system) : nothing
    _propagate_tamen_trajectory(
        generator, state, times;
        expected_dimensions=multicore_heom_dimensions(system),
        temporal_basis_size,
        tolerance,
        state_tolerance,
        options,
        conserved_observables,
        preserve_trace,
        trace_observable,
    )
end
```

Export `propagate_multicore_heom_tamen` from `src/TTDynamics.jl`.

- [ ] **Step 5: Run all TTDynamics tests**

Run:

```bash
cd /home/takahashi/.julia/dev/TTDynamics
julia --project=. test/runtests.jl
```

Expected: PASS for open and zero-hierarchy MultiCore systems, root trace, both reduced traces, and all earlier testsets.

- [ ] **Step 6: Commit MultiCore HEOM conservation**

```bash
cd /home/takahashi/.julia/dev/TTDynamics
git add src/TTDynamics.jl src/tamen_propagation.jl test/tamen_propagation.jl
git commit -m "feat: preserve MultiCore HEOM root trace"
```

### Task 6: Document the unified APIs and run cross-repository verification

**Files:**
- Modify: `/home/takahashi/.julia/dev/TTDynamics/docs/src/api.md:185-282`
- Verify: `/home/takahashi/.julia/dev/TTSolver/src/tamen.jl`
- Verify: `/home/takahashi/.julia/dev/TTSolver/test/runtests.jl`
- Verify: `/home/takahashi/.julia/dev/TTDynamics/src/tamen_propagation.jl`
- Verify: `/home/takahashi/.julia/dev/TTDynamics/test/tamen_propagation.jl`

**Interfaces:**
- Consumes: all public APIs delivered by Tasks 1-5.
- Produces: user-facing examples and final evidence that both repositories pass together through TTDynamics' local `../TTSolver` manifest path.

- [ ] **Step 1: Add HEOM propagation documentation**

After `build_initial_state` in `docs/src/api.md`, add:

````markdown
#### Trace-preserving tAMEn propagation

```julia
trajectory = propagate_heom_tamen(
    system,
    state,
    times;
    preserve_trace=true,
    tolerance=1e-6,
)
```

`preserve_trace=true` adds the root-density trace observable to tAMEn's
approximation spaces and disables density-matrix 2-norm correction. It does
not normalize or project snapshots after propagation, so trace conservation
follows the configured solver accuracy.
````

- [ ] **Step 2: Add MultiCore HEOM and Redfield documentation**

After the MultiCore example in `docs/src/api.md`, add:

````markdown
Both MultiCore density-matrix propagators use the same opt-in interface:

```julia
heom_trajectory = propagate_multicore_heom_tamen(
    heom_system, heom_state, times; preserve_trace=true,
)
redfield_trajectory = propagate_multicore_redfield_tamen(
    redfield_system, redfield_state, times; preserve_trace=true,
)
```

For MultiCore HEOM the conserved value is `Tr(rho_root)`: every physical
ket/bra pair is contracted with its identity and every hierarchy mode is
selected at its vacuum index. This also gives unit trace for every reduced
physical density matrix, but it does not preserve subsystem purity, local
populations, the Frobenius norm, or individual higher-tier ADO traces.

Advanced callers can pass full-state TT linear functionals through
`conserved_observables`. The legacy `options[:obs]` spelling remains
available, but it cannot be mixed with the new conservation keywords.
````

- [ ] **Step 3: Run the complete TTSolver suite from a clean process**

Run:

```bash
cd /home/takahashi/.julia/dev/TTSolver
julia --project=. test/runtests.jl
git status --short
git diff --check
```

Expected: all tests PASS; only intended committed or documentation changes appear; `git diff --check` is silent.

- [ ] **Step 4: Run the complete TTDynamics suite against the local TTSolver**

Confirm `TTDynamics/Manifest.toml` still contains `path = "../TTSolver"`, then run:

```bash
cd /home/takahashi/.julia/dev/TTDynamics
julia --project=. test/runtests.jl
git diff --check
```

Expected: all tests PASS against the newly modified local TTSolver and `git diff --check` is silent.

- [ ] **Step 5: Inspect the final public diff for accidental normalization**

Run:

```bash
cd /home/takahashi/.julia/dev
git -C TTSolver diff HEAD~2 -- src/tamen.jl test/runtests.jl
git -C TTDynamics diff HEAD~3 -- src test docs/src/api.md
rg -n "normalize|/.*trace|trace.*\/" TTSolver/src/tamen.jl TTDynamics/src/tamen_propagation.jl
```

Expected: conservation is implemented only through AMEn observables; no endpoint division, rescaling, or projection appears.

- [ ] **Step 6: Commit the TTDynamics API documentation**

```bash
cd /home/takahashi/.julia/dev/TTDynamics
git add docs/src/api.md
git commit -m "docs: explain tAMEn trace preservation"
```

- [ ] **Step 7: Record final repository states**

Run:

```bash
git -C /home/takahashi/.julia/dev/TTSolver status --short --branch
git -C /home/takahashi/.julia/dev/TTDynamics status --short --branch
git -C /home/takahashi/.julia/dev/TTSolver log -4 --oneline
git -C /home/takahashi/.julia/dev/TTDynamics log -5 --oneline
```

Expected: both worktrees are clean, TTSolver contains the two conservation commits, and TTDynamics contains the shared propagation, standard HEOM, MultiCore HEOM, and documentation commits.
