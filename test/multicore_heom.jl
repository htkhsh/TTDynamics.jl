using LinearAlgebra
using Test
using TTSolver

if !isdefined(@__MODULE__, :TTDynamics)
    include(joinpath(@__DIR__, "..", "src", "TTDynamics.jl"))
end
using .TTDynamics

@testset "multi-core HEOM types" begin
    A = ComplexF64[1 2im; 3 4]
    rho = ComplexF64[0.4 0.1im; -0.1im 0.6]
    @test liouville_left(A) * vec(rho) ≈ vec(A * rho)
    @test liouville_right(A) * vec(rho) ≈ vec(rho * A)

    metadata = CorrelationFitMetadata(
        0,
        0,
        0,
        0,
        0.1,
        (0.0, 1.0),
        [1.0],
        1.0,
        Inf,
        1.0,
        String[],
    )
    fit = ExponentialCorrelation([1 + 0im], [0.2 + 0.1im], [0.2 - 0.1im], metadata)
    @test fit.scales ≈ [sqrt(abs(0.2 + 0.1im))]
    @test_throws ArgumentError ExponentialCorrelation(
        [-1 + 0im],
        [1im],
        [-1im],
        metadata,
    )
end

@testset "multi-core HEOM dimensions and validation" begin
    metadata = CorrelationFitMetadata(
        0,
        0,
        0,
        0,
        0.1,
        (0.0, 1.0),
        [1.0],
        1.0,
        Inf,
        1.0,
        String[],
    )
    fit = ExponentialCorrelation(
        [1 + 0im, 2 + 0im],
        [0.2 + 0.1im, 0.05 + 0.02im],
        [0.2 - 0.1im, 0.05 - 0.02im],
        metadata,
    )
    physical_L = TTMatrix([
        reshape(Matrix{ComplexF64}(I, 4, 4), 1, 4, 4, 1),
        reshape(Matrix{ComplexF64}(I, 9, 9), 1, 9, 9, 1),
    ])
    couplings = [LocalBathCoupling(2, Matrix{ComplexF64}(I, 3, 3), fit)]
    system = MultiCoreHEOMTTSystem(physical_L, [4, 9], couplings, [3, 3])

    @test multicore_heom_dimensions(system) == [4, 9, 3, 3]
    @test_throws ArgumentError LocalBathCoupling(0, Matrix{ComplexF64}(I, 3, 3), fit)
    @test_throws ArgumentError MultiCoreHEOMTTSystem(
        physical_L,
        [4, 9],
        [LocalBathCoupling(3, Matrix{ComplexF64}(I, 3, 3), fit)],
        [3, 3],
    )
    @test_throws ArgumentError MultiCoreHEOMTTSystem(physical_L, [4, 9], couplings, [0])
end
