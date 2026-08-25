using LinearAlgebra
using Test

include("../examples/quartic/config.jl")
include("../examples/quartic/oscillator.jl")

@testset "quartic local oscillator" begin
    cfg = QuarticConfig(
        site_count=1,
        basis_frequency=2.0,
        omega_p=2.0,
        K2=2.0,
        K4=0.0,
        d_raw=16,
        d_keep=6,
    )
    mode = build_quartic_mode(cfg)
    @test mode.energies .- mode.energies[1] ≈ 2.0 .* (0:5) atol=1e-10
    @test ishermitian(mode.hamiltonian)
    @test mode.q2 ≈ mode.q * mode.q atol=2e-1
    @test_throws ArgumentError QuarticConfig(omega_p=0.0)
    @test_throws ArgumentError QuarticConfig(K4=-1.0)
    @test_throws ArgumentError QuarticConfig(d_raw=4, d_keep=5)
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
