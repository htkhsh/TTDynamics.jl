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
