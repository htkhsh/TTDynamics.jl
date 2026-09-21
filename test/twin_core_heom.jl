using LinearAlgebra
using Test
using TTSolver
using HEOMKit
include(joinpath(@__DIR__, "twin_core_test_utils.jl"))

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

@testset "Hamiltonian MPO twin embedding" begin
    sx = ComplexF64[0 1; 1 0]
    sz = ComplexF64[1 0; 0 -1]
    q3 = ComplexF64[0 1 0; 1 0 sqrt(2); 0 sqrt(2) 0]
    n3 = ComplexF64.(Diagonal(0:2))
    I2 = Matrix{ComplexF64}(I, 2, 2)
    I3 = Matrix{ComplexF64}(I, 3, 3)
    H = 0.3 * tt_mkron(sz, I3) + 0.7 * tt_mkron(I2, n3) + 0.2 * tt_mkron(sx, q3)
    before = deepcopy(H.cores)
    L = build_twin_liouvillian(H; tol=1e-14)
    @test tt_dims(L) == ([2, 2, 3, 3], [2, 2, 3, 3])
    @test H.cores == before
    H_dense = tt_full(H)
    dimension = size(H_dense, 1)
    identity_full = Matrix{ComplexF64}(I, dimension, dimension)
    vectorized = -1im * (kron(identity_full, H_dense) - kron(transpose(H_dense), identity_full))
    P = twin_permutation([2, 3])
    @test tt_full(L) ≈ P * vectorized * P' atol=1e-11
    @test_throws ArgumentError build_twin_liouvillian(H; tol=-1.0)
    @test_throws ArgumentError build_twin_liouvillian(H; tol=NaN)
    nonfinite = TTMatrix([fill(ComplexF64(NaN), 1, 2, 2, 1)])
    @test_throws ArgumentError build_twin_liouvillian(nonfinite)
end

@testset "scaled twin-core HEOM generator" begin
    H1 = ComplexF64[0.2 0.3; 0.3 -0.1]
    V1 = ComplexF64[1 0; 0 -1]
    fit = twin_test_correlation(rates=ComplexF64[0.8], forward=ComplexF64[0.25 + 0.04im], backward=ComplexF64[0.15 + 0.02im])
    system = TwinCoreHEOMTTSystem(tt_matrix(H1), [TwinCoreBathCoupling(1, V1, fit)], [3])
    generator = build_twin_heom_generator(system; tol=1e-14)
    @test tt_dims(generator) == ([2, 2, 3], [2, 2, 3])
    closed = TwinCoreHEOMTTSystem(tt_matrix(H1), TwinCoreBathCoupling[], Int[])
    @test tt_full(build_twin_heom_generator(closed; tol=1e-14)) ≈
          tt_full(build_twin_liouvillian(closed.hamiltonian; tol=1e-14)) atol=1e-11
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
