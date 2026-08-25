using LinearAlgebra
using QuadGK
using TTDynamics
using Test

include("../config.jl")
include("../oscillator.jl")
include("../correlation.jl")

@testset "quartic bath correlation sampling" begin
    cfg = QuarticConfig(
        temperature=0.5,
        bath_lambda=0.7,
        bath_gamma=1.3,
        quadrature_atol=1e-10,
        quadrature_rtol=1e-8,
    )
    bath = CriticallyDampedBrownian(cfg.bath_lambda, cfg.bath_gamma)
    normalization, _ = quadgk(
        ω -> spectral_density(bath, ω) / ω,
        0.0,
        Inf;
        rtol=1e-10,
    )
    @test normalization / π ≈ bath.lambda rtol=1e-8
    @test isfinite(bath_correlation(bath, 0.0, cfg.temperature; rtol=1e-8, atol=1e-10))

    times = collect(0.0:0.1:0.5)
    samples = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        rtol=cfg.quadrature_rtol,
        atol=cfg.quadrature_atol,
    )
    @test length(samples) == length(times)
    @test all(isfinite, samples)

    cutoff_20 = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        omega_integration_max=20 * bath.gamma,
        rtol=1e-9,
        atol=1e-11,
    )
    cutoff_40 = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-9,
        atol=1e-11,
    )
    @test cutoff_20 ≈ cutoff_40 rtol=2e-3

    looser = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-7,
        atol=1e-9,
    )
    tighter = sample_bath_correlation(
        bath,
        cfg.temperature,
        times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-9,
        atol=1e-11,
    )
    @test looser ≈ tighter rtol=5e-4
end

@testset "quartic bath correlation rejects invalid quadrature controls" begin
    bath = CriticallyDampedBrownian(0.7, 1.3)

    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; rtol=Inf)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; rtol=NaN)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; atol=Inf)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; atol=NaN)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; maxevals=0)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; maxevals=-1)
    @test_throws ArgumentError bath_correlation(bath, 0.0, 0.5; maxevals=1.5)
end
