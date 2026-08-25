using LinearAlgebra
using Test
using QuadGK

include("../examples/quartic/config.jl")
include("../examples/quartic/oscillator.jl")
let correlation_path = joinpath(@__DIR__, "..", "examples", "quartic", "correlation.jl")
    isfile(correlation_path) && include(correlation_path)
end

function raw_quartic_operators(config::QuarticConfig)
    a = zeros(ComplexF64, config.d_raw, config.d_raw)
    for n in 1:(config.d_raw - 1)
        a[n, n + 1] = sqrt(n)
    end
    q = (a + a') / sqrt(2 * config.basis_frequency)
    p = -1im * sqrt(config.basis_frequency / 2) * (a - a')
    h = config.omega_p / 2 * (p * p) +
        config.K2 / 2 * (q * q) +
        config.K4 / 4 * (q * q * q * q)
    return (; q, p, h)
end

@testset "quartic local oscillator" begin
    cfg = QuarticConfig(
        site_count=1,
        basis_frequency=1.0,
        omega_p=2.0,
        K2=2.0,
        K4=0.0,
        d_raw=16,
        d_keep=6,
    )
    mode = build_quartic_mode(cfg)
    @test mode.energies .- mode.energies[1] ≈ 2.0 .* (0:5) atol=1e-10
    @test ishermitian(mode.hamiltonian)
    raw = raw_quartic_operators(cfg)
    @test mode.q2 ≈ mode.transform' * (raw.q * raw.q) * mode.transform atol=1e-12
    @test_throws ArgumentError QuarticConfig(omega_p=0.0)
    @test_throws ArgumentError QuarticConfig(K4=-1.0)
    @test_throws ArgumentError QuarticConfig(d_raw=4, d_keep=5)
end

@testset "quartic harmonic basis uses projected eigensystem" begin
    cfg = QuarticConfig(
        site_count=1,
        basis_frequency=2.0,
        omega_p=2.0,
        K2=2.0,
        K4=0.0,
        d_raw=16,
        d_keep=6,
    )
    raw = raw_quartic_operators(cfg)
    eigenbasis = eigen(Hermitian(raw.h))
    mode = build_quartic_mode(cfg)

    @test mode.energies ≈ eigenbasis.values[1:cfg.d_keep] atol=1e-12
    @test mode.hamiltonian ≈ Matrix(Diagonal(ComplexF64.(eigenbasis.values[1:cfg.d_keep]))) atol=1e-12
end

@testset "quartic double well convergence and thermal density" begin
    cfg12 = QuarticConfig(site_count=1, d_raw=12, d_keep=4, K2=-1.0, K4=0.2)
    cfg20 = QuarticConfig(site_count=1, d_raw=20, d_keep=4, K2=-1.0, K4=0.2)
    m12 = build_quartic_mode(cfg12)
    m20 = build_quartic_mode(cfg20)
    @test m12.energies ≈ m20.energies rtol=2e-3
    @test ishermitian(m20.hamiltonian)
    @test maximum(abs, diag(m20.q)) < 1e-10
    ρth = quartic_thermal_density(m20, cfg20.temperature)
    @test ishermitian(ρth)
    @test isapprox(real(tr(ρth)), 1.0; atol=1e-12)
    @test minimum(eigvals(Hermitian(ρth))) >= -1e-13
end

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
