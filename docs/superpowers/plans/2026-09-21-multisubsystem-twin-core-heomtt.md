# Multi-Subsystem Twin-Core HEOM-TT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible HEOM-TT system whose multi-subsystem density operator uses adjacent, separate ket and bra TT cores for every physical subsystem.

**Architecture:** A Hilbert-space Hamiltonian MPO is embedded twice into an interleaved twin-space chain, once on ket cores and once transposed on bra cores. Local bath couplings add hierarchy terms without forming `d^2` physical modes; state and observable helpers factor each local matrix across its adjacent ket-bra pair.

**Tech Stack:** Julia 1.11, TTDynamics, TTSolver `TTMatrix`/`TTTensor`, HEOMKit exponential correlations, `LinearAlgebra`, `Test`, Documenter.

**Spec:** `docs/superpowers/specs/2026-09-21-multisubsystem-twin-core-heomtt-design.md`

## Global Constraints

- Preserve `HEOMTTSystem` and `MultiCoreHEOMTTSystem` behavior and layouts.
- Use the physical core order `[ket_1, bra_1, ..., ket_M, bra_M]`, followed by hierarchy cores.
- Never create a fused physical mode of dimension `d_j^2`.
- Reuse `ExponentialCorrelation` and its scaled HEOM convention.
- Keep charge-transport model knowledge outside TTDynamics.
- Do not expand an MPO merely to validate global Hermiticity.
- Dense root reconstruction defaults to `max_dimension=2_000` and checks the limit before expansion.
- All new public builders reject nonfinite or negative tolerances before allocating TT operators.

## Review Focus

- Hamiltonian MPOs with non-unit outer TT ranks must be rejected because they do not represent one global operator; Task 1 pins this validation.
- Multiple couplings may arrive in any subsystem order and may contain different pole counts; Task 3 pins the caller-order hierarchy mapping.
- A closed system with no couplings or hierarchy modes must remain valid; Tasks 1, 3, and 4 pin this zero-length path.
- Nearly positive density matrices may contain eigenvalues down to `-density_tolerance`, while genuinely negative states must fail; Task 4 pins both cases and the SVD rank cutoff.
- Dense root reconstruction must enforce its dimension guard before `tt_full` or matrix allocation and must preserve complex off-diagonal entries under index permutation; Task 5 pins both behaviors.

---

## File Structure

- Create `src/twin_core_heom.jl`: public system/coupling types, validation, twin MPO embedding, and scaled HEOM generator.
- Create `src/twin_core_states.jl`: product/correlated initial densities, hierarchy-root extraction, observables, expectations, and guarded dense reconstruction.
- Create `test/twin_core_heom.jl`: layout, validation, Hamiltonian embedding, and HEOM generator equivalence tests.
- Create `test/twin_core_states.jl`: density-state, observable, root extraction, and dense permutation tests.
- Create `test/twin_core_test_utils.jl`: dense index permutation helpers shared by both focused test files.
- Modify `src/TTDynamics.jl`: export and include the two focused source files after `multicore_heom.jl`.
- Modify `test/runtests.jl`: include both focused test files after `multicore_heom.jl`.
- Modify `docs/src/api.md`: document the new types, ordering, builders, state helpers, and migration boundaries.

### Task 1: Twin-Core System Types and Layout Validation

**Files:**
- Create: `src/twin_core_heom.jl`
- Create: `test/twin_core_heom.jl`
- Modify: `src/TTDynamics.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `TTMatrix`, `tt_dims`, `_complex_tt_matrix`, `_validate_exponential_correlation`, and `ExponentialCorrelation` from existing TTDynamics/TTSolver code.
- Produces: `TwinCoreBathCoupling`, `TwinCoreHEOMTTSystem`, `twin_core_physical_dimensions(system)`, and `twin_core_heom_dimensions(system)`.

- [ ] **Step 1: Write the failing type and validation tests**

Create `test/twin_core_heom.jl` with reusable fixtures and the first testset:

```julia
using LinearAlgebra
using Test
using TTSolver
using HEOMKit

if !isdefined(@__MODULE__, :TTDynamics)
    include(joinpath(@__DIR__, "..", "src", "TTDynamics.jl"))
end
using .TTDynamics

function twin_test_metadata()
    CorrelationFitMetadata(
        0, 0, 0, 0, 0.1, (0.0, 1.0), [1.0], 1.0, Inf, 1.0, String[],
    )
end

function twin_test_correlation(; rates=ComplexF64[0.7 + 0.1im],
                               forward=ComplexF64[0.2 - 0.03im],
                               backward=ComplexF64[0.15 + 0.04im])
    ExponentialCorrelation(rates, forward, backward, twin_test_metadata())
end

@testset "twin-core HEOM system validation" begin
    H1 = ComplexF64[0.2 0.4; 0.4 -0.1]
    H2 = ComplexF64[0.0 0.1 0.0; 0.1 0.3 0.2; 0.0 0.2 0.5]
    H = tt_mkron(H1, H2)
    fit = twin_test_correlation()
    coupling = TwinCoreBathCoupling(2, Diagonal(ComplexF64[1, 0, -1]), fit)
    system = TwinCoreHEOMTTSystem(H, [coupling], [4])

    @test system.hilbert_dimensions == [2, 3]
    @test twin_core_physical_dimensions(system) == [2, 2, 3, 3]
    @test twin_core_heom_dimensions(system) == [2, 2, 3, 3, 4]
    @test TwinCoreHEOMTTSystem(H, [coupling], 5).hierarchy_sizes == [5]

    closed = TwinCoreHEOMTTSystem(H, TwinCoreBathCoupling[], Int[])
    @test twin_core_heom_dimensions(closed) == [2, 2, 3, 3]

    @test_throws ArgumentError TwinCoreBathCoupling(
        0, Matrix{ComplexF64}(I, 2, 2), fit,
    )
    @test_throws ArgumentError TwinCoreBathCoupling(
        1, ComplexF64[0 1; 0 0], fit,
    )
    @test_throws ArgumentError TwinCoreHEOMTTSystem(
        H, [TwinCoreBathCoupling(3, Matrix{ComplexF64}(I, 2, 2), fit)], [3],
    )
    @test_throws DimensionMismatch TwinCoreHEOMTTSystem(
        H, [TwinCoreBathCoupling(2, Matrix{ComplexF64}(I, 2, 2), fit)], [3],
    )
    @test_throws ArgumentError TwinCoreHEOMTTSystem(H, [coupling], Int[])
    @test_throws ArgumentError TwinCoreHEOMTTSystem(H, [coupling], [0])

    nonsquare = TTMatrix([ones(ComplexF64, 1, 2, 3, 1)])
    @test_throws DimensionMismatch TwinCoreHEOMTTSystem(
        nonsquare, TwinCoreBathCoupling[], Int[],
    )
    bad_boundary = TTMatrix([ones(ComplexF64, 2, 2, 2, 1)])
    @test_throws ArgumentError TwinCoreHEOMTTSystem(
        bad_boundary, TwinCoreBathCoupling[], Int[],
    )
    nonfinite = TTMatrix([fill(ComplexF64(NaN), 1, 2, 2, 1)])
    @test_throws ArgumentError TwinCoreHEOMTTSystem(
        nonfinite, TwinCoreBathCoupling[], Int[],
    )
end
```

- [ ] **Step 2: Run the focused test to verify the public types are missing**

Run:

```bash
julia --project=. test/twin_core_heom.jl
```

Expected: FAIL with `UndefVarError: TwinCoreBathCoupling not defined`.

- [ ] **Step 3: Implement types, dimension helpers, and complete validation**

Create `src/twin_core_heom.jl` with these definitions and equivalent docstrings:

```julia
struct TwinCoreBathCoupling
    subsystem::Int
    operator::Matrix{ComplexF64}
    correlation::ExponentialCorrelation

    function TwinCoreBathCoupling(subsystem::Int,
                                  operator::Matrix{ComplexF64},
                                  correlation::ExponentialCorrelation)
        subsystem > 0 || throw(ArgumentError("subsystem must be positive"))
        _validate_local_operator(operator)
        ishermitian(operator) ||
            throw(ArgumentError("twin-core bath operator must be Hermitian"))
        new(subsystem, operator, correlation)
    end
end

TwinCoreBathCoupling(subsystem::Integer, operator::AbstractMatrix,
                     correlation::ExponentialCorrelation) =
    TwinCoreBathCoupling(Int(subsystem), Matrix{ComplexF64}(operator), correlation)

TwinCoreBathCoupling(coupling::TwinCoreBathCoupling) = coupling

struct TwinCoreHEOMTTSystem
    hamiltonian::TTMatrix{ComplexF64}
    hilbert_dimensions::Vector{Int}
    couplings::Vector{TwinCoreBathCoupling}
    hierarchy_sizes::Vector{Int}
end

function _validate_twin_hamiltonian(hamiltonian::TTMatrix)
    output_dims, input_dims = tt_dims(hamiltonian)
    output_dims == input_dims ||
        throw(DimensionMismatch("Hamiltonian MPO must have square local dimensions"))
    size(first(hamiltonian.cores), 1) == 1 ||
        throw(ArgumentError("Hamiltonian MPO left boundary rank must be one"))
    size(last(hamiltonian.cores), 4) == 1 ||
        throw(ArgumentError("Hamiltonian MPO right boundary rank must be one"))
    for (index, core) in pairs(hamiltonian.cores)
        all(isfinite, core) ||
            throw(ArgumentError("Hamiltonian MPO core $index must be finite"))
    end
    all(>(0), output_dims) ||
        throw(ArgumentError("Hamiltonian local dimensions must be positive"))
    output_dims
end

function TwinCoreHEOMTTSystem(hamiltonian::TTMatrix, couplings, hierarchy_sizes)
    H = _complex_tt_matrix(hamiltonian)
    local_couplings = [TwinCoreBathCoupling(coupling) for coupling in couplings]
    hierarchy = collect(Int, hierarchy_sizes)
    output_dims = _validate_twin_hamiltonian(H)
    expanded_poles = sum(length(c.correlation.rates) for c in local_couplings; init=0)
    length(hierarchy) == expanded_poles || throw(ArgumentError(
        "hierarchy_sizes must provide one positive local size per expanded bath pole",
    ))
    all(>(0), hierarchy) ||
        throw(ArgumentError("hierarchy_sizes must be positive"))
    for coupling in local_couplings
        1 <= coupling.subsystem <= length(output_dims) || throw(ArgumentError(
            "coupling subsystem $(coupling.subsystem) must be in 1:$(length(output_dims))",
        ))
        expected = output_dims[coupling.subsystem]
        size(coupling.operator) == (expected, expected) || throw(DimensionMismatch(
            "coupling subsystem $(coupling.subsystem) expects a $expected x $expected operator",
        ))
    end
    TwinCoreHEOMTTSystem(H, output_dims, local_couplings, hierarchy)
end

function TwinCoreHEOMTTSystem(hamiltonian::TTMatrix, couplings,
                              hierarchy_size::Integer)
    hierarchy_size > 0 ||
        throw(ArgumentError("hierarchy_size must be positive"))
    local_couplings = [TwinCoreBathCoupling(coupling) for coupling in couplings]
    pole_count = sum(length(c.correlation.rates) for c in local_couplings; init=0)
    TwinCoreHEOMTTSystem(hamiltonian, local_couplings,
                        fill(Int(hierarchy_size), pole_count))
end

twin_core_physical_dimensions(system::TwinCoreHEOMTTSystem) =
    reduce(vcat, ([d, d] for d in system.hilbert_dimensions); init=Int[])

twin_core_heom_dimensions(system::TwinCoreHEOMTTSystem) =
    [twin_core_physical_dimensions(system); system.hierarchy_sizes]
```

Write the boundary-rank checks as ordinary `if` statements if the pipeline form is less readable; preserve the exact errors and behavior.

- [ ] **Step 4: Wire the source and tests into the package**

In `src/TTDynamics.jl`, export the four interfaces and include the source after `multicore_heom.jl`:

```julia
export TwinCoreBathCoupling,
       TwinCoreHEOMTTSystem,
       twin_core_physical_dimensions,
       twin_core_heom_dimensions

include("twin_core_heom.jl")
```

In `test/runtests.jl`, add:

```julia
include("twin_core_heom.jl")
```

immediately after `include("multicore_heom.jl")`.

- [ ] **Step 5: Run focused validation tests**

Run:

```bash
julia --project=. test/twin_core_heom.jl
```

Expected: PASS, including closed-system, non-unit boundary-rank, non-Hermitian coupling, and dimension-mismatch cases.

- [ ] **Step 6: Commit the type and layout foundation**

```bash
git add src/TTDynamics.jl src/twin_core_heom.jl test/runtests.jl test/twin_core_heom.jl
git commit -m "feat: add twin-core HEOM system types"
```

### Task 2: Hamiltonian MPO Twin-Space Embedding

**Files:**
- Modify: `src/twin_core_heom.jl`
- Modify: `src/TTDynamics.jl`
- Create: `test/twin_core_test_utils.jl`
- Modify: `test/twin_core_heom.jl`

**Interfaces:**
- Consumes: `TwinCoreHEOMTTSystem`, `TTMatrix.cores`, `tt_round`, and TTSolver's MPO arithmetic.
- Produces: `build_twin_liouvillian(hamiltonian::TTMatrix; tol=1e-12)` returning a `TTMatrix{ComplexF64}` over `[d1,d1,...,dM,dM]`.

- [ ] **Step 1: Add dense permutation and Liouvillian equivalence tests**

Create `test/twin_core_test_utils.jl` with:

```julia
function kron_linear_index(indices, dimensions)
    index = 1
    for (local_index, dimension) in zip(indices, dimensions)
        index = (index - 1) * dimension + local_index
    end
    index
end

function twin_permutation(dimensions)
    total = prod(dimensions)
    twin_dimensions = reduce(vcat, ([d, d] for d in dimensions); init=Int[])
    permutation = zeros(ComplexF64, total^2, total^2)
    ranges = (Base.OneTo(d) for d in dimensions)
    for ket in Iterators.product(ranges...)
        for bra in Iterators.product((Base.OneTo(d) for d in dimensions)...)
            ket_index = kron_linear_index(ket, dimensions)
            bra_index = kron_linear_index(bra, dimensions)
            interleaved = Int[]
            for subsystem in eachindex(dimensions)
                push!(interleaved, ket[subsystem], bra[subsystem])
            end
            twin_index = kron_linear_index(interleaved, twin_dimensions)
            vectorized_index = ket_index + (bra_index - 1) * total
            permutation[twin_index, vectorized_index] = 1
        end
    end
    permutation
end
```

At the top of `test/twin_core_heom.jl`, after its imports, add:

```julia
include(joinpath(@__DIR__, "twin_core_test_utils.jl"))
```

Then append the following testset:

```julia

@testset "Hamiltonian MPO twin embedding" begin
    sx = ComplexF64[0 1; 1 0]
    sz = ComplexF64[1 0; 0 -1]
    q3 = ComplexF64[0 1 0; 1 0 sqrt(2); 0 sqrt(2) 0]
    n3 = ComplexF64.(Diagonal(0:2))
    I2 = Matrix{ComplexF64}(I, 2, 2)
    I3 = Matrix{ComplexF64}(I, 3, 3)
    H = 0.3 * tt_mkron(sz, I3) +
        0.7 * tt_mkron(I2, n3) +
        0.2 * tt_mkron(sx, q3)
    before = deepcopy(H.cores)

    L = build_twin_liouvillian(H; tol=1e-14)
    @test tt_dims(L) == ([2, 2, 3, 3], [2, 2, 3, 3])
    @test H.cores == before

    H_dense = tt_full(H)
    dimension = size(H_dense, 1)
    identity_full = Matrix{ComplexF64}(I, dimension, dimension)
    vectorized = -1im * (
        kron(identity_full, H_dense) - kron(transpose(H_dense), identity_full)
    )
    P = twin_permutation([2, 3])
    @test tt_full(L) ≈ P * vectorized * P' atol=1e-11

    @test_throws ArgumentError build_twin_liouvillian(H; tol=-1.0)
    @test_throws ArgumentError build_twin_liouvillian(H; tol=NaN)
    nonfinite = TTMatrix([fill(ComplexF64(NaN), 1, 2, 2, 1)])
    @test_throws ArgumentError build_twin_liouvillian(nonfinite)
end
```

- [ ] **Step 2: Run the test to verify the builder is missing**

Run:

```bash
julia --project=. test/twin_core_heom.jl
```

Expected: FAIL with `UndefVarError: build_twin_liouvillian not defined`.

- [ ] **Step 3: Implement rank-preserving identity cores and both embeddings**

Add these focused helpers to `src/twin_core_heom.jl`:

```julia
function _twin_bond_identity(rank::Int, dimension::Int)
    core = zeros(ComplexF64, rank, dimension, dimension, rank)
    for bond in 1:rank, local_index in 1:dimension
        core[bond, local_index, local_index, bond] = 1
    end
    core
end

function _twin_left_embedding(hamiltonian::TTMatrix)
    cores = Array{ComplexF64,4}[]
    for core in hamiltonian.cores
        dimension = size(core, 2)
        push!(cores, ComplexF64.(core))
        push!(cores, _twin_bond_identity(size(core, 4), dimension))
    end
    TTMatrix(cores)
end

function _twin_right_embedding(hamiltonian::TTMatrix)
    cores = Array{ComplexF64,4}[]
    for core in hamiltonian.cores
        dimension = size(core, 2)
        push!(cores, _twin_bond_identity(size(core, 1), dimension))
        push!(cores, ComplexF64.(permutedims(core, (1, 3, 2, 4))))
    end
    TTMatrix(cores)
end

function build_twin_liouvillian(hamiltonian::TTMatrix; tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    _validate_twin_hamiltonian(hamiltonian)
    left = _twin_left_embedding(hamiltonian)
    right = _twin_right_embedding(hamiltonian)
    tt_round(-1im * (left - right), tol)
end
```

The ket-side pass core must use the outgoing rank `size(core, 4)`; the bra-side pass core must use the incoming rank `size(core, 1)`. Swapping these passes is incorrect for nontrivial MPO rank and is caught by the dense test.

- [ ] **Step 4: Export and run the focused test**

Export `build_twin_liouvillian` from `src/TTDynamics.jl`, then run:

```bash
julia --project=. test/twin_core_heom.jl
```

Expected: PASS with dense agreement for the rank-greater-than-one Hamiltonian MPO.

- [ ] **Step 5: Commit the twin-space embedding**

```bash
git add src/TTDynamics.jl src/twin_core_heom.jl test/twin_core_test_utils.jl test/twin_core_heom.jl
git commit -m "feat: embed Hamiltonian MPOs in twin space"
```

### Task 3: Scaled Twin-Core HEOM Generator

**Files:**
- Modify: `src/twin_core_heom.jl`
- Modify: `src/TTDynamics.jl`
- Modify: `test/twin_core_heom.jl`

**Interfaces:**
- Consumes: `build_twin_liouvillian`, `TwinCoreHEOMTTSystem`, `TwinCoreBathCoupling`, and `ExponentialCorrelation`.
- Produces: `build_twin_heom_generator(system; tol=1e-12)` with caller-order pole-to-hierarchy mapping.

- [ ] **Step 1: Add one-subsystem compatibility and two-subsystem component tests**

Append these helpers and tests to `test/twin_core_heom.jl`:

```julia
function dense_local_product(dimensions, replacements...)
    factors = [Matrix{ComplexF64}(I, d, d) for d in dimensions]
    for (index, matrix) in replacements
        factors[index] = ComplexF64.(matrix)
    end
    reduce(kron, factors)
end

@testset "scaled twin-core HEOM generator" begin
    H1 = ComplexF64[0.2 0.3; 0.3 -0.1]
    V1 = ComplexF64[1 0; 0 -1]
    exponent = ComplexF64[0.8]
    coefficient = ComplexF64[0.25 + 0.04im]
    noise = NoiseExp(BathExp(exponent, coefficient, V1))
    metadata = twin_test_metadata()
    fit = ExponentialCorrelation(
        ComplexF64.(noise.γ), ComplexF64.(noise.c1), ComplexF64.(noise.c2),
        sqrt.(noise.abs_coeff), metadata,
    )
    old_system = HEOMTTSystem(H1, noise, [3])
    old_generator, _, _ = build_heom_liouvillian(old_system; tol=1e-14)
    new_system = TwinCoreHEOMTTSystem(
        tt_matrix(H1), [TwinCoreBathCoupling(1, V1, fit)], [3],
    )
    new_generator = build_twin_heom_generator(new_system; tol=1e-14)
    @test tt_dims(new_generator) == ([2, 2, 3], [2, 2, 3])
    @test tt_full(new_generator) ≈ tt_full(old_generator) atol=1e-11

    H2 = ComplexF64[0.0 0.2 0.0; 0.2 0.4 0.1; 0.0 0.1 0.7]
    H = tt_mkron(H1, H2)
    V2 = ComplexF64[0 1 0; 1 0 0; 0 0 -0.5]
    fit2 = twin_test_correlation(
        rates=ComplexF64[0.6 + 0.2im],
        forward=ComplexF64[0.3 - 0.1im],
        backward=ComplexF64[-0.04 + 0.2im],
    )
    system = TwinCoreHEOMTTSystem(
        H, [TwinCoreBathCoupling(2, V2, fit2)], [2],
    )
    generator = build_twin_heom_generator(system; tol=1e-14)
    dimensions = [2, 2, 3, 3, 2]
    hierarchy_core = 5
    ket_core = 3
    bra_core = 4
    number = ComplexF64[0 0; 0 1]
    lower = ComplexF64[0 1; 0 0]
    raise = ComplexF64[0 0; 1 0]
    rate = only(fit2.rates)
    scale = only(fit2.scales)
    forward = only(fit2.coeff_forward)
    backward = only(fit2.coeff_backward)
    physical = build_twin_liouvillian(H; tol=1e-14)
    expected = kron(tt_full(physical), Matrix{ComplexF64}(I, 2, 2))
    expected -= rate * dense_local_product(dimensions, hierarchy_core => number)
    expected -= 1im * scale * (
        dense_local_product(dimensions, ket_core => V2, hierarchy_core => lower) -
        dense_local_product(dimensions, bra_core => transpose(V2), hierarchy_core => lower)
    )
    expected -= 1im / scale * (
        forward * dense_local_product(
            dimensions, ket_core => V2, hierarchy_core => raise,
        ) -
        backward * dense_local_product(
            dimensions, bra_core => transpose(V2), hierarchy_core => raise,
        )
    )
    @test tt_full(generator) ≈ expected atol=1e-10

    closed = TwinCoreHEOMTTSystem(H, TwinCoreBathCoupling[], Int[])
    @test tt_full(build_twin_heom_generator(closed; tol=1e-14)) ≈
          tt_full(build_twin_liouvillian(H; tol=1e-14)) atol=1e-11
end
```

In the same testset, add a caller-order test with two couplings, the first on
subsystem 2 with two poles and the second on subsystem 1 with one pole:

```julia
    two_pole_fit = twin_test_correlation(
        rates=ComplexF64[0.4, 0.9],
        forward=ComplexF64[0.2, 0.05],
        backward=ComplexF64[0.18, 0.04],
    )
    reverse_order = [
        TwinCoreBathCoupling(2, Matrix{ComplexF64}(I, 3, 3), two_pole_fit),
        TwinCoreBathCoupling(1, Matrix{ComplexF64}(I, 2, 2),
                             twin_test_correlation()),
    ]
    ordered_system = TwinCoreHEOMTTSystem(H, reverse_order, [2, 3, 4])
    @test getfield.(ordered_system.couplings, :subsystem) == [2, 1]
    @test ordered_system.hierarchy_sizes == [2, 3, 4]
    @test twin_core_heom_dimensions(ordered_system)[end-2:end] == [2, 3, 4]
```

- [ ] **Step 2: Run the focused test to verify the generator is missing**

Run:

```bash
julia --project=. test/twin_core_heom.jl
```

Expected: FAIL with `UndefVarError: build_twin_heom_generator not defined`.

- [ ] **Step 3: Implement local-product terms and hierarchy extension**

Add these helpers and the generator to `src/twin_core_heom.jl`:

```julia
function _twin_core_local_matrix(system::TwinCoreHEOMTTSystem, replacements...)
    dimensions = twin_core_heom_dimensions(system)
    factors = [Matrix{ComplexF64}(I, d, d) for d in dimensions]
    for (index, factor) in replacements
        checkbounds(Bool, factors, index) ||
            throw(ArgumentError("twin-core local replacement $index is out of range"))
        size(factor) == size(factors[index]) || throw(DimensionMismatch(
            "twin-core replacement $index has size $(size(factor)); " *
            "expected $(size(factors[index]))",
        ))
        factors[index] = ComplexF64.(factor)
    end
    tt_mkron(factors)
end

function _extend_twin_liouvillian(system::TwinCoreHEOMTTSystem; tol::Float64)
    physical = build_twin_liouvillian(system.hamiltonian; tol)
    cores = [copy(core) for core in physical.cores]
    for dimension in system.hierarchy_sizes
        identity_matrix = Matrix{ComplexF64}(I, dimension, dimension)
        push!(cores, reshape(identity_matrix, 1, dimension, dimension, 1))
    end
    TTMatrix(cores)
end

function build_twin_heom_generator(system::TwinCoreHEOMTTSystem;
                                   tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    generator = _extend_twin_liouvillian(system; tol)
    hierarchy_index = 0
    physical_core_count = 2 * length(system.hilbert_dimensions)
    for coupling in system.couplings
        ket_core = 2 * coupling.subsystem - 1
        bra_core = 2 * coupling.subsystem
        correlation = coupling.correlation
        for pole in eachindex(correlation.rates)
            hierarchy_index += 1
            hierarchy_core = physical_core_count + hierarchy_index
            dimension = system.hierarchy_sizes[hierarchy_index]
            number = diagm(0 => ComplexF64.(0:(dimension - 1)))
            lower = diagm(1 => sqrt.(ComplexF64.(1:(dimension - 1))))
            raise = diagm(-1 => sqrt.(ComplexF64.(1:(dimension - 1))))
            V = coupling.operator
            decay = _twin_core_local_matrix(system, hierarchy_core => number)
            upward = _twin_core_local_matrix(
                system, ket_core => V, hierarchy_core => lower,
            ) - _twin_core_local_matrix(
                system, bra_core => transpose(V), hierarchy_core => lower,
            )
            downward = correlation.coeff_forward[pole] * _twin_core_local_matrix(
                system, ket_core => V, hierarchy_core => raise,
            ) - correlation.coeff_backward[pole] * _twin_core_local_matrix(
                system, bra_core => transpose(V), hierarchy_core => raise,
            )
            generator -= correlation.rates[pole] * decay
            generator -= 1im * correlation.scales[pole] * upward
            generator -= 1im / correlation.scales[pole] * downward
            generator = tt_round(generator, tol)
        end
    end
    tt_round(generator, tol)
end
```

- [ ] **Step 4: Export and run compatibility tests**

Export `build_twin_heom_generator`, then run:

```bash
julia --project=. test/twin_core_heom.jl
```

Expected: PASS for existing one-system equivalence, explicit two-subsystem dense terms, reverse subsystem order, and the closed-system path.

- [ ] **Step 5: Commit the HEOM generator**

```bash
git add src/TTDynamics.jl src/twin_core_heom.jl test/twin_core_heom.jl
git commit -m "feat: build twin-core HEOM generators"
```

### Task 4: Factorized and Correlated Initial Density States

**Files:**
- Create: `src/twin_core_states.jl`
- Create: `test/twin_core_states.jl`
- Modify: `src/TTDynamics.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `TwinCoreHEOMTTSystem`, `twin_core_physical_dimensions`, `TTTensor`, and `tt_round`.
- Produces: `build_twin_heom_initial_state(system, local_density_matrices; tol, density_tolerance)` and `build_twin_heom_initial_state(system, physical_state::TTTensor; tol)`.

- [ ] **Step 1: Write failing product and correlated-state tests**

Create `test/twin_core_states.jl`:

```julia
using LinearAlgebra
using Test
using TTSolver

if !isdefined(@__MODULE__, :TTDynamics)
    include(joinpath(@__DIR__, "..", "src", "TTDynamics.jl"))
end
using .TTDynamics
if !isdefined(@__MODULE__, :twin_permutation)
    include(joinpath(@__DIR__, "twin_core_test_utils.jl"))
end

function state_test_correlation()
    metadata = CorrelationFitMetadata(
        0, 0, 0, 0, 0.1, (0.0, 1.0), [1.0], 1.0, Inf, 1.0, String[],
    )
    ExponentialCorrelation(
        ComplexF64[0.7], ComplexF64[0.2], ComplexF64[0.2], metadata,
    )
end

@testset "twin-core HEOM initial density states" begin
    H = tt_mkron(ComplexF64[0 1; 1 0], ComplexF64.(Diagonal(0:2)))
    system = TwinCoreHEOMTTSystem(H, TwinCoreBathCoupling[], Int[])
    rho1 = ComplexF64[0.7 0.1im; -0.1im 0.3]
    rho2 = ComplexF64[0.2 0 0; 0 0.5 0.1im; 0 -0.1im 0.3]
    state = build_twin_heom_initial_state(
        system, [rho1, rho2]; tol=1e-14, density_tolerance=1e-12,
    )
    @test tt_dims(state) == [2, 2, 3, 3]
    @test tt_ranks(state)[2] == rank(rho1)
    @test tt_ranks(state)[4] == rank(rho2)

    dimensions = [2, 3]
    P = twin_permutation(dimensions)
    expected_interleaved = P * vec(kron(rho1, rho2))
    @test vec(tt_full(state)) ≈ expected_interleaved atol=1e-11

    correlated = tt_svd(expected_interleaved, [2, 2, 3, 3]; tol=1e-14)
    rebuilt = build_twin_heom_initial_state(system, correlated; tol=1e-14)
    @test vec(tt_full(rebuilt)) ≈ expected_interleaved atol=1e-11

    near_psd = ComplexF64[1 + 1e-13 0; 0 -1e-13]
    @test tt_dims(build_twin_heom_initial_state(
        TwinCoreHEOMTTSystem(tt_matrix(zeros(2, 2)), TwinCoreBathCoupling[], Int[]),
        [near_psd]; density_tolerance=1e-12,
    )) == [2, 2]
    @test_throws ArgumentError build_twin_heom_initial_state(system, [rho1])
    @test_throws DimensionMismatch build_twin_heom_initial_state(
        system, [rho1, Matrix{ComplexF64}(I, 2, 2) / 2],
    )
    @test_throws ArgumentError build_twin_heom_initial_state(
        system, [rho1, ComplexF64[1.1 0 0; 0 -0.1 0; 0 0 0]],
        density_tolerance=1e-12,
    )
    @test_throws ArgumentError build_twin_heom_initial_state(
        system, [rho1, rho2]; density_tolerance=-1,
    )

    open_system = TwinCoreHEOMTTSystem(
        H,
        [TwinCoreBathCoupling(1, Matrix{ComplexF64}(I, 2, 2),
                              state_test_correlation())],
        [2],
    )
    open_state = build_twin_heom_initial_state(open_system, [rho1, rho2])
    @test tt_dims(open_state) == [2, 2, 3, 3, 2]
    hierarchy_view = reshape(tt_full(open_state), 2, :)
    @test hierarchy_view[2, :] ≈ zeros(size(hierarchy_view, 2)) atol=1e-12
    correlated_open = build_twin_heom_initial_state(
        open_system, correlated; tol=1e-14,
    )
    @test tt_dims(correlated_open) == [2, 2, 3, 3, 2]
    correlated_hierarchy = reshape(tt_full(correlated_open), 2, :)
    @test correlated_hierarchy[2, :] ≈
          zeros(size(correlated_hierarchy, 2)) atol=1e-12

    wrong_physical_state = product_tt_state([
        ComplexF64[1, 0], ComplexF64[1, 0],
        ComplexF64[1, 0], ComplexF64[1, 0],
    ])
    @test_throws DimensionMismatch build_twin_heom_initial_state(
        system, wrong_physical_state,
    )
end
```

The hierarchy assertion checks the full state rather than an individual core,
so it remains valid after TT rounding changes the core gauge.

- [ ] **Step 2: Run the focused state test and verify failure**

Run:

```bash
julia --project=. test/twin_core_states.jl
```

Expected: FAIL with `UndefVarError: build_twin_heom_initial_state not defined`.

- [ ] **Step 3: Implement density validation and ket-bra SVD factorization**

Create `src/twin_core_states.jl` with:

```julia
function _validate_density_matrix(density, dimension, index, tolerance)
    density isa AbstractMatrix ||
        throw(ArgumentError("local density $index must be a matrix"))
    size(density) == (dimension, dimension) || throw(DimensionMismatch(
        "local density $index must have size ($dimension, $dimension)",
    ))
    all(isfinite, density) ||
        throw(ArgumentError("local density $index must be finite"))
    isapprox(density, adjoint(density); atol=tolerance, rtol=0) ||
        throw(ArgumentError("local density $index must be Hermitian"))
    isapprox(real(tr(density)), 1; atol=tolerance, rtol=0) ||
        throw(ArgumentError("local density $index must have unit trace"))
    minimum(eigvals(Hermitian(ComplexF64.(density)))) >= -tolerance ||
        throw(ArgumentError("local density $index must be positive semidefinite"))
end

function _matrix_to_twin_cores(matrix::AbstractMatrix; relative_tolerance::Float64)
    decomposition = svd(ComplexF64.(matrix))
    cutoff = relative_tolerance * maximum(decomposition.S)
    bond_rank = max(count(>(cutoff), decomposition.S), 1)
    ket = reshape(decomposition.U[:, 1:bond_rank], 1, size(matrix, 1), bond_rank)
    bra_matrix = Diagonal(decomposition.S[1:bond_rank]) *
                 decomposition.Vt[1:bond_rank, :]
    bra = reshape(bra_matrix, bond_rank, size(matrix, 2), 1)
    ket, bra
end

function _hierarchy_vacuum_cores(hierarchy_sizes)
    [reshape(ComplexF64[1; zeros(ComplexF64, d - 1)], 1, d, 1)
     for d in hierarchy_sizes]
end

function build_twin_heom_initial_state(
    system::TwinCoreHEOMTTSystem,
    local_density_matrices;
    tol::Float64=1e-12,
    density_tolerance::Float64=1e-10,
)
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    isfinite(density_tolerance) && density_tolerance >= 0 ||
        throw(ArgumentError("density_tolerance must be finite and nonnegative"))
    length(local_density_matrices) == length(system.hilbert_dimensions) ||
        throw(ArgumentError("one local density matrix is required per subsystem"))
    cores = Array{ComplexF64,3}[]
    for (index, (density, dimension)) in enumerate(
        zip(local_density_matrices, system.hilbert_dimensions)
    )
        _validate_density_matrix(density, dimension, index, density_tolerance)
        ket, bra = _matrix_to_twin_cores(
            density; relative_tolerance=density_tolerance,
        )
        push!(cores, ket, bra)
    end
    append!(cores, _hierarchy_vacuum_cores(system.hierarchy_sizes))
    tt_round(TTTensor(cores), tol)
end
```

Implement the correlated physical-state overload in the same file:

```julia
function build_twin_heom_initial_state(
    system::TwinCoreHEOMTTSystem,
    physical_state::TTTensor;
    tol::Float64=1e-12,
)
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    expected = twin_core_physical_dimensions(system)
    tt_dims(physical_state) == expected || throw(DimensionMismatch(
        "physical twin state dimensions must be $expected; got $(tt_dims(physical_state))",
    ))
    size(first(physical_state.cores), 1) == 1 ||
        throw(ArgumentError("physical twin state left boundary rank must be one"))
    size(last(physical_state.cores), 3) == 1 ||
        throw(ArgumentError("physical twin state right boundary rank must be one"))
    for (index, core) in pairs(physical_state.cores)
        all(isfinite, core) ||
            throw(ArgumentError("physical twin state core $index must be finite"))
    end
    cores = [ComplexF64.(copy(core)) for core in physical_state.cores]
    append!(cores, _hierarchy_vacuum_cores(system.hierarchy_sizes))
    tt_round(TTTensor(cores), tol)
end
```

- [ ] **Step 4: Wire exports, includes, and direct-test utilities**

In `src/TTDynamics.jl`, export `build_twin_heom_initial_state` and include `twin_core_states.jl` immediately after `twin_core_heom.jl`.

In `test/runtests.jl`, include `twin_core_states.jl` after `twin_core_heom.jl`. Ensure the state test can run directly without relying on a test file that has not been included.

- [ ] **Step 5: Run focused HEOM and state tests**

Run:

```bash
julia --project=. test/twin_core_heom.jl
julia --project=. test/twin_core_states.jl
```

Expected: both PASS; the mixed local state has the expected ket-bra ranks, the small negative eigenvalue within tolerance is accepted, and an actual negative density is rejected.

- [ ] **Step 6: Commit initial density support**

```bash
git add src/TTDynamics.jl src/twin_core_states.jl test/runtests.jl test/twin_core_states.jl
git commit -m "feat: build twin-core HEOM initial states"
```

### Task 5: Root Extraction, Observables, and Dense Reconstruction

**Files:**
- Modify: `src/twin_core_states.jl`
- Modify: `src/TTDynamics.jl`
- Modify: `test/twin_core_states.jl`

**Interfaces:**
- Consumes: initial full HEOM `TTTensor`s and the private `_matrix_to_twin_cores` helper.
- Produces: `root_twin_state`, `build_twin_observable`, `build_twin_trace_observable`, `build_twin_local_observable`, `twin_expectation`, and the `root_density_matrix(::TTTensor, ::TwinCoreHEOMTTSystem; max_dimension)` overload.

- [ ] **Step 1: Add failing root, observable, and permutation tests**

Append to `test/twin_core_states.jl`:

```julia
@testset "twin-core roots and observables" begin
    H = tt_mkron(zeros(ComplexF64, 2, 2), zeros(ComplexF64, 3, 3))
    system = TwinCoreHEOMTTSystem(H, TwinCoreBathCoupling[], Int[])
    rho1 = ComplexF64[0.6 0.2im; -0.2im 0.4]
    rho2 = ComplexF64[0.3 0.1 0; 0.1 0.4 -0.05im; 0 0.05im 0.3]
    state = build_twin_heom_initial_state(system, [rho1, rho2]; tol=1e-14)
    root = root_twin_state(state, system)
    @test tt_dims(root) == [2, 2, 3, 3]

    trace_observable = build_twin_trace_observable(system; tol=1e-14)
    @test twin_expectation(trace_observable, state, system) ≈ 1 atol=1e-11

    nonhermitian = ComplexF64[0 1; 0 0]
    observable = build_twin_local_observable(
        system, 1, nonhermitian; tol=1e-14,
    )
    expected = tr(kron(nonhermitian, Matrix{ComplexF64}(I, 3, 3)) *
                  kron(rho1, rho2))
    @test twin_expectation(observable, state, system) ≈ expected atol=1e-11

    dense = root_density_matrix(state, system; max_dimension=10)
    @test dense ≈ kron(rho1, rho2) atol=1e-11
    @test dense[1, 2] ≈ kron(rho1, rho2)[1, 2]
    @test_throws ArgumentError root_density_matrix(state, system; max_dimension=5)

    wrong_observable = product_tt_state([
        ComplexF64[1, 0], ComplexF64[1, 0], ComplexF64[1, 0], ComplexF64[1, 0],
    ])
    @test_throws DimensionMismatch twin_expectation(
        wrong_observable, state, system,
    )

    open_system = TwinCoreHEOMTTSystem(
        tt_matrix(zeros(ComplexF64, 2, 2)),
        [TwinCoreBathCoupling(1, Matrix{ComplexF64}(I, 2, 2),
                              state_test_correlation())],
        [2],
    )
    open_state = build_twin_heom_initial_state(
        open_system, [ComplexF64[0.7 0.1; 0.1 0.3]]; tol=1e-14,
    )
    modified_cores = deepcopy(open_state.cores)
    modified_cores[end][:, 1, :] .*= 2
    modified_cores[end][:, 2, :] .*= 3
    modified_state = TTTensor(modified_cores)
    @test vec(tt_full(root_twin_state(modified_state, open_system))) ≈
          2 .* vec(tt_full(root_twin_state(open_state, open_system))) atol=1e-11
end
```

- [ ] **Step 2: Run the focused state test to verify the APIs are missing**

Run:

```bash
julia --project=. test/twin_core_states.jl
```

Expected: FAIL with `UndefVarError: root_twin_state not defined`.

- [ ] **Step 3: Implement hierarchy-root extraction and observable builders**

Add to `src/twin_core_states.jl`:

```julia
function _validate_twin_heom_state(state::TTTensor,
                                   system::TwinCoreHEOMTTSystem)
    expected = twin_core_heom_dimensions(system)
    tt_dims(state) == expected || throw(DimensionMismatch(
        "twin-core HEOM state dimensions must be $expected; got $(tt_dims(state))",
    ))
end

function root_twin_state(state::TTTensor,
                         system::TwinCoreHEOMTTSystem)::TTTensor
    _validate_twin_heom_state(state, system)
    physical_count = 2 * length(system.hilbert_dimensions)
    environment = ones(ComplexF64, 1)
    for core in Iterators.reverse(state.cores[(physical_count + 1):end])
        environment = ComplexF64.(core[:, 1, :]) * environment
    end
    cores = [ComplexF64.(copy(core)) for core in state.cores[1:physical_count]]
    last_core = cores[end]
    left_rank, dimension, right_rank = size(last_core)
    contracted = reshape(last_core, left_rank * dimension, right_rank) * environment
    cores[end] = reshape(contracted, left_rank, dimension, 1)
    TTTensor(cores)
end

function build_twin_observable(local_operators; tol::Float64=1e-12)
    isfinite(tol) && tol >= 0 ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    isempty(local_operators) &&
        throw(ArgumentError("at least one local observable is required"))
    cores = Array{ComplexF64,3}[]
    for (index, operator) in pairs(local_operators)
        operator isa AbstractMatrix ||
            throw(ArgumentError("local observable $index must be a matrix"))
        size(operator, 1) == size(operator, 2) || throw(DimensionMismatch(
            "local observable $index must be square",
        ))
        all(isfinite, operator) ||
            throw(ArgumentError("local observable $index must be finite"))
        ket, bra = _matrix_to_twin_cores(
            adjoint(operator); relative_tolerance=tol,
        )
        push!(cores, ket, bra)
    end
    tt_round(TTTensor(cores), tol)
end

function build_twin_trace_observable(system::TwinCoreHEOMTTSystem;
                                     tol::Float64=1e-12)
    identities = [Matrix{ComplexF64}(I, d, d) for d in system.hilbert_dimensions]
    build_twin_observable(identities; tol)
end

function build_twin_local_observable(system::TwinCoreHEOMTTSystem,
                                     subsystem::Integer,
                                     operator::AbstractMatrix;
                                     tol::Float64=1e-12)
    1 <= subsystem <= length(system.hilbert_dimensions) ||
        throw(ArgumentError("subsystem must index a physical Hilbert core"))
    expected = system.hilbert_dimensions[subsystem]
    size(operator) == (expected, expected) || throw(DimensionMismatch(
        "local observable must have size ($expected, $expected)",
    ))
    factors = [Matrix{ComplexF64}(I, d, d) for d in system.hilbert_dimensions]
    factors[subsystem] = ComplexF64.(operator)
    build_twin_observable(factors; tol)
end

function twin_expectation(observable::TTTensor, state::TTTensor,
                          system::TwinCoreHEOMTTSystem)
    expected = twin_core_physical_dimensions(system)
    tt_dims(observable) == expected || throw(DimensionMismatch(
        "twin observable dimensions must be $expected; got $(tt_dims(observable))",
    ))
    tt_dot(observable, root_twin_state(state, system))
end
```

- [ ] **Step 4: Implement guarded dense root reconstruction without a permutation matrix**

Add this overload to `src/twin_core_states.jl`:

```julia
function _bounded_hilbert_dimension(dimensions, maximum)
    maximum > 0 || throw(ArgumentError("max_dimension must be positive"))
    total = 1
    for dimension in dimensions
        total <= div(maximum, dimension) || throw(ArgumentError(
            "Hilbert dimension exceeds max_dimension=$maximum",
        ))
        total *= dimension
    end
    total
end

function _kron_linear_index(indices, dimensions)
    index = 1
    for (local_index, dimension) in zip(indices, dimensions)
        index = (index - 1) * dimension + local_index
    end
    index
end

function root_density_matrix(state::TTTensor,
                             system::TwinCoreHEOMTTSystem;
                             max_dimension::Integer=2_000)
    dimensions = system.hilbert_dimensions
    total = _bounded_hilbert_dimension(dimensions, Int(max_dimension))
    root_values = vec(tt_full(root_twin_state(state, system)))
    twin_dimensions = twin_core_physical_dimensions(system)
    density = zeros(ComplexF64, total, total)
    for ket in Iterators.product((Base.OneTo(d) for d in dimensions)...)
        for bra in Iterators.product((Base.OneTo(d) for d in dimensions)...)
            interleaved = Int[]
            for subsystem in eachindex(dimensions)
                push!(interleaved, ket[subsystem], bra[subsystem])
            end
            ket_index = _kron_linear_index(ket, dimensions)
            bra_index = _kron_linear_index(bra, dimensions)
            twin_index = _kron_linear_index(interleaved, twin_dimensions)
            density[ket_index, bra_index] = root_values[twin_index]
        end
    end
    density
end
```

The dimension check must execute before `root_twin_state`, `tt_full`, or dense allocation. This is observable in the guard test and prevents accidental exponential work.

- [ ] **Step 5: Export and run all focused tests**

Export the six Task 5 public functions from `src/TTDynamics.jl`, then run:

```bash
julia --project=. test/twin_core_heom.jl
julia --project=. test/twin_core_states.jl
```

Expected: PASS, including non-Hermitian observable dualization, complex off-diagonal reconstruction, hierarchy-root contraction, and the pre-allocation dimension guard.

- [ ] **Step 6: Commit root and observable support**

```bash
git add src/TTDynamics.jl src/twin_core_states.jl test/twin_core_states.jl
git commit -m "feat: add twin-core root observables"
```

### Task 6: Public Documentation and Regression Verification

**Files:**
- Modify: `docs/src/api.md`
- Verify: `src/TTDynamics.jl`
- Verify: `test/runtests.jl`

**Interfaces:**
- Consumes: every public type and function from Tasks 1-5.
- Produces: documented tensor ordering, runnable API examples, and full-suite evidence that existing HEOM layouts remain unchanged.

- [ ] **Step 1: Add the public API documentation**

Add a `Multi-Subsystem Twin-Core HEOM-TT` section after the existing HEOM section in `docs/src/api.md`. Include these exact points and a runnable small example:

```julia
H = tt_mkron(
    ComplexF64[0 1; 1 0],
    ComplexF64.(Diagonal(0:2)),
)
correlation = ExponentialCorrelation(rates, forward, backward, metadata)
coupling = TwinCoreBathCoupling(2, q, correlation)
system = TwinCoreHEOMTTSystem(H, [coupling], [3])

generator = build_twin_heom_generator(system)
state = build_twin_heom_initial_state(system, [rho_charge, rho_mode])
trace_observable = build_twin_trace_observable(system)
trace_value = twin_expectation(trace_observable, state, system)
```

Document:

- the distinction between one Hilbert subsystem and one HEOM hierarchy mode;
- the layout `[ket_1, bra_1, ..., ket_M, bra_M, hierarchy...]`;
- why physical dimensions are never squared;
- one hierarchy size per expanded exponential pole in coupling-list order;
- factorized local density matrices versus the correlated physical-state overload;
- `root_twin_state` and observable contractions as the scalable path;
- `root_density_matrix` as a guarded diagnostic path;
- unchanged legacy `HEOMTTSystem` and `MultiCoreHEOMTTSystem` layouts.

- [ ] **Step 2: Run format and focused verification**

Run:

```bash
git diff --check
julia --project=. test/twin_core_heom.jl
julia --project=. test/twin_core_states.jl
```

Expected: no whitespace errors and both focused files PASS.

- [ ] **Step 3: Run the full TTDynamics test suite**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all existing and new tests PASS, including `heom_twin_space.jl`, `multicore_heom.jl`, and `quartic.jl`.

- [ ] **Step 4: Build the documentation**

Run:

```bash
julia --project=docs docs/make.jl
```

Expected: Documenter completes without missing binding, syntax, or doctest errors.

- [ ] **Step 5: Inspect the final diff and public surface**

Run:

```bash
git status --short
git diff --stat 1fa9058
rg -n "TwinCore|build_twin|root_twin|twin_expectation" src test docs/src/api.md
```

Expected: only the planned source, test, and API documentation files differ
from the approved-spec commit `1fa9058`; every documented public name is
exported and covered by a focused test.

- [ ] **Step 6: Commit documentation and final integration**

```bash
git add docs/src/api.md
git commit -m "docs: document twin-core HEOM-TT"
```

If Step 5 reveals an omitted export or test-runner include, stage that exact `src/TTDynamics.jl` or `test/runtests.jl` correction in the same commit after rerunning Steps 2-4.
