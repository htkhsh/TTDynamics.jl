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
    validation_times = collect(0.025:0.05:3.975)
    validation_samples = ComplexF64[
        sum(coefficients[index] * exp(-rates[index] * time) for index in eachindex(rates))
        for time in validation_times
    ]

    fit = fit_correlation_esprit(
        samples,
        times;
        fit_rank=3,
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
        fit_relative_tolerance=1e-8,
        validation_samples=validation_samples,
        validation_times=validation_times,
    )

    @test norm(evaluate_correlation(fit, times) - samples) / norm(samples) < 1e-8
    @test norm(evaluate_correlation(fit, times; branch=:backward) - conj.(samples)) /
          norm(samples) < 1e-8
    @test length(fit.rates) == 4
    @test 0 < fit.metadata.minimum_pole_separation < 0.01
    @test all(
        rate -> any(
            other -> isapprox(rate, conj(other); atol=1e-8, rtol=0),
            fit.rates,
        ),
        fit.rates,
    )
end

@testset "complex ESPRIT requires disjoint validation data" begin
    training_times = collect(0.0:0.1:2.0)
    samples = ComplexF64[exp(-(0.4 + 0.2im) * time) for time in training_times]

    @test_throws ArgumentError fit_correlation_esprit(
        samples,
        training_times;
        fit_rank=1,
    )
    @test_throws ArgumentError fit_correlation_esprit(
        samples,
        training_times;
        fit_rank=1,
        validation_samples=samples,
        validation_times=training_times,
    )
end

@testset "complex ESPRIT rejects invalid grids and unstable poles" begin
    uniform_times = collect(0.0:0.1:2.0)
    decaying_samples = ComplexF64[exp(-(0.4 + 0.2im) * time) for time in uniform_times]
    validation_times = collect(0.05:0.1:1.95)
    decaying_validation =
        ComplexF64[exp(-(0.4 + 0.2im) * time) for time in validation_times]
    nonuniform_times = copy(uniform_times)
    nonuniform_times[8] += 0.01
    @test_throws ArgumentError fit_correlation_esprit(
        decaying_samples,
        nonuniform_times;
        fit_rank=1,
        validation_samples=decaying_validation,
        validation_times=validation_times,
    )

    growing_samples = ComplexF64[exp(0.2 * time) for time in uniform_times]
    growing_validation = ComplexF64[exp(0.2 * time) for time in validation_times]
    @test_throws ArgumentError fit_correlation_esprit(
        growing_samples,
        uniform_times;
        fit_rank=1,
        validation_samples=growing_validation,
        validation_times=validation_times,
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

    @test_throws ArgumentError _validated_conjugate_closed_rates(
        ComplexF64[-5e-11 + 0im, 0.0 + 0im],
        dt;
        pole_stability_tolerance=1e-10,
        duplicate_pole_tolerance=1e-8,
    )
end

@testset "correlation validation rejects either inaccurate branch" begin
    rates = ComplexF64[0.4 + 0.2im, 0.4 - 0.2im]
    coeff_forward = ComplexF64[0.7 + 0.1im, 0.2 - 0.05im]
    coeff_backward = zeros(ComplexF64, 2)
    metadata = CorrelationFitMetadata(
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        (0.0, 1.0),
        [1.0, 0.5],
        2.0,
        0.4,
        maximum(abs, coeff_forward),
        String[],
    )
    fit = ExponentialCorrelation(rates, coeff_forward, coeff_backward, metadata)
    validation_times = collect(0.05:0.1:0.95)
    validation_samples = ComplexF64[
        sum(
            coeff_forward[index] * exp(-rates[index] * time)
            for index in eachindex(rates)
        ) for time in validation_times
    ]

    @test_throws ArgumentError validate_correlation_fit(
        fit,
        validation_samples,
        validation_times;
        fit_absolute_tolerance=1e-10,
        fit_relative_tolerance=1e-10,
    )
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
    holdout_forward_error =
        norm(evaluate_correlation(fit, validation_times) - validation_samples) /
        norm(validation_samples)
    holdout_backward_error = norm(
        evaluate_correlation(fit, validation_times; branch=:backward) -
        conj.(validation_samples),
    ) / norm(validation_samples)
    training_forward_error =
        norm(evaluate_correlation(fit, training_times) - training_samples) /
        norm(training_samples)
    training_backward_error = norm(
        evaluate_correlation(fit, training_times; branch=:backward) -
        conj.(training_samples),
    ) / norm(training_samples)
    @test max(holdout_forward_error, holdout_backward_error) < 0.1
    @test fit.metadata.validation_relative_error ≈
          max(holdout_forward_error, holdout_backward_error) rtol=1e-10
    @test fit.metadata.training_relative_error ≈
          max(training_forward_error, training_backward_error) rtol=1e-10
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
            validation_samples=validation_samples,
            validation_times=validation_times,
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
