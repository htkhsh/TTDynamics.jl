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

@testset "complex ESPRIT uses a conjugate-closed common pole basis" begin
    times = collect(0.0:0.05:4.0)
    rates = ComplexF64[0.3 + 0.7im, 0.3 - 0.7im, 0.305 + 0.705im]
    coefficients = ComplexF64[1.0 + 0.2im, 0.4 - 0.1im, 0.03 + 0.02im]
    samples = ComplexF64[
        sum(coefficients[index] * exp(-rates[index] * time) for index in eachindex(rates))
        for time in times
    ]

    fit = fit_correlation_esprit(
        samples,
        times;
        fit_rank=3,
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
        fit_relative_tolerance=1e-8,
    )

    @test norm(evaluate_correlation(fit, times) - samples) / norm(samples) < 1e-8
    @test norm(evaluate_correlation(fit, times; branch=:backward) - conj.(samples)) /
          norm(samples) < 1e-8
    @test length(fit.rates) == 4
    @test all(
        rate -> any(
            other -> isapprox(rate, conj(other); atol=1e-8, rtol=0),
            fit.rates,
        ),
        fit.rates,
    )
end

@testset "complex ESPRIT rejects invalid grids and unstable poles" begin
    uniform_times = collect(0.0:0.1:2.0)
    decaying_samples = ComplexF64[exp(-(0.4 + 0.2im) * time) for time in uniform_times]
    nonuniform_times = copy(uniform_times)
    nonuniform_times[8] += 0.01
    @test_throws ArgumentError fit_correlation_esprit(
        decaying_samples,
        nonuniform_times;
        fit_rank=1,
    )

    growing_samples = ComplexF64[exp(0.2 * time) for time in uniform_times]
    @test_throws ArgumentError fit_correlation_esprit(
        growing_samples,
        uniform_times;
        fit_rank=1,
    )

    dt = uniform_times[2] - uniform_times[1]
    @test_throws ArgumentError _validated_conjugate_closed_rates(
        ComplexF64[0.2 + (π / dt + 1e-6) * im],
        dt;
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
    )
    corrected_rates, corrections = _validated_conjugate_closed_rates(
        ComplexF64[-5e-11 + 0.3im],
        dt;
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
    )
    @test real(first(corrected_rates)) == 0
    @test any(contains("corrected to zero"), corrections)
end

@testset "complex ESPRIT validates cBO fits away from training points" begin
    bath = CriticallyDampedBrownian(0.2, 1.1)
    temperature = 0.7
    training_times = collect(0.0:0.2:2.0)
    validation_times = collect(0.1:0.2:1.9)
    training_samples = sample_bath_correlation(
        bath,
        temperature,
        training_times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-8,
        atol=1e-10,
    )
    validation_samples = sample_bath_correlation(
        bath,
        temperature,
        validation_times;
        omega_integration_max=40 * bath.gamma,
        rtol=1e-8,
        atol=1e-10,
    )

    fit = fit_correlation_esprit(
        training_samples,
        training_times;
        fit_rank=4,
        fit_absolute_tolerance=1e-4,
        fit_relative_tolerance=0.1,
        validation_samples=validation_samples,
        validation_times=validation_times,
    )
    holdout_error = norm(evaluate_correlation(fit, validation_times) - validation_samples) /
                    norm(validation_samples)
    @test holdout_error < 0.1
    @test fit.metadata.validation_relative_error ≈ holdout_error rtol=1e-12
    @test all(isfinite, fit.metadata.singular_values)
    @test all(
        isfinite,
        (
            fit.metadata.training_absolute_error,
            fit.metadata.training_relative_error,
            fit.metadata.validation_absolute_error,
            fit.metadata.validation_relative_error,
            fit.metadata.sampling_interval,
            fit.metadata.vandermonde_condition,
            fit.metadata.minimum_pole_separation,
            fit.metadata.maximum_coefficient_magnitude,
        ),
    )

    rank_errors = Float64[]
    for rank in (3, 4)
        rank_fit = fit_correlation_esprit(
            training_samples,
            training_times;
            fit_rank=rank,
            fit_absolute_tolerance=1e-4,
            fit_relative_tolerance=0.5,
        )
        diagnostics = validate_correlation_fit(
            rank_fit,
            validation_samples,
            validation_times;
            fit_absolute_tolerance=1e-4,
            fit_relative_tolerance=0.5,
        )
        push!(rank_errors, diagnostics.relative_error)
    end
    @test length(rank_errors) == 2
    @test all(isfinite, rank_errors)
    @test all(<(0.5), rank_errors)
end
