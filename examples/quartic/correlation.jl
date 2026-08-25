using ExpFit
using LinearAlgebra
using QuadGK

abstract type AbstractSpectralDensity end

struct CriticallyDampedBrownian <: AbstractSpectralDensity
    lambda::Float64
    gamma::Float64

    function CriticallyDampedBrownian(lambda, gamma)
        λ = Float64(lambda)
        Γ = Float64(gamma)
        isfinite(λ) || throw(ArgumentError("lambda must be finite"))
        isfinite(Γ) || throw(ArgumentError("gamma must be finite"))
        λ >= 0 || throw(ArgumentError("lambda must be nonnegative"))
        Γ > 0 || throw(ArgumentError("gamma must be positive"))
        return new(λ, Γ)
    end
end

spectral_density(bath::CriticallyDampedBrownian, ω) =
    4 * bath.lambda * bath.gamma^3 * ω / (ω^2 + bath.gamma^2)^2

xcothx(x) = abs(x) < 1e-4 ? 1 + x^2 / 3 - x^4 / 45 : x * coth(x)

function validate_quadrature_controls(rtol, atol, maxevals)
    rtol_value = Float64(rtol)
    atol_value = Float64(atol)
    isfinite(rtol_value) && rtol_value > 0 ||
        throw(ArgumentError("rtol must be finite and positive"))
    isfinite(atol_value) && atol_value > 0 ||
        throw(ArgumentError("atol must be finite and positive"))
    maxevals isa Integer || throw(ArgumentError("maxevals must be a positive integer"))
    maxevals > 0 || throw(ArgumentError("maxevals must be a positive integer"))
    return rtol_value, atol_value, Int(maxevals)
end

function spectral_density_over_omega(bath::CriticallyDampedBrownian, ω)
    if iszero(ω)
        return 4 * bath.lambda / bath.gamma
    end
    return spectral_density(bath, ω) / ω
end

function bath_correlation(
    bath::CriticallyDampedBrownian,
    t,
    temperature;
    omega_integration_max=nothing,
    rtol=1e-8,
    atol=1e-10,
    maxevals=100_000,
)
    τ = Float64(t)
    βinv = Float64(temperature)
    isfinite(τ) || throw(ArgumentError("time must be finite"))
    isfinite(βinv) || throw(ArgumentError("temperature must be finite"))
    βinv > 0 || throw(ArgumentError("temperature must be positive"))
    rtol_value, atol_value, maxevals_value = validate_quadrature_controls(rtol, atol, maxevals)

    upper = if omega_integration_max === nothing
        Inf
    else
        ωmax = Float64(omega_integration_max)
        isfinite(ωmax) || throw(ArgumentError("omega_integration_max must be finite or nothing"))
        ωmax > 0 || throw(ArgumentError("omega_integration_max must be positive"))
        ωmax
    end

    β = inv(βinv)
    thermal_integrand(ω) = begin
        x = β * ω / 2
        prefactor = (2 / β) * spectral_density_over_omega(bath, ω) * xcothx(x)
        return prefactor * cos(ω * τ) / π
    end
    dissipative_integrand(ω) = -spectral_density(bath, ω) * sin(ω * τ) / π

    real_part, _ = quadgk(
        thermal_integrand,
        0.0,
        upper;
        rtol=rtol_value,
        atol=atol_value,
        maxevals=maxevals_value,
    )
    imag_part, _ = quadgk(
        dissipative_integrand,
        0.0,
        upper;
        rtol=rtol_value,
        atol=atol_value,
        maxevals=maxevals_value,
    )
    return ComplexF64(real_part, imag_part)
end

function sample_bath_correlation(
    bath::CriticallyDampedBrownian,
    temperature,
    times;
    omega_integration_max=nothing,
    rtol=1e-8,
    atol=1e-10,
    maxevals=100_000,
)
    return ComplexF64[
        bath_correlation(
            bath,
            t,
            temperature;
            omega_integration_max=omega_integration_max,
            rtol=rtol,
            atol=atol,
            maxevals=maxevals,
        ) for t in times
    ]
end

function _validated_uniform_grid(times; uniform_grid_tolerance)
    length(times) >= 2 || throw(ArgumentError("at least two sample times are required"))
    time_values = Float64.(times)
    all(isfinite, time_values) || throw(ArgumentError("sample times must be finite"))
    dt = time_values[2] - time_values[1]
    dt > 0 || throw(ArgumentError("sample times must be strictly increasing"))
    for index in 3:length(time_values)
        step = time_values[index] - time_values[index - 1]
        isapprox(step, dt; atol=uniform_grid_tolerance, rtol=uniform_grid_tolerance) ||
            throw(ArgumentError("sample times must be uniformly spaced"))
    end
    return time_values, dt
end

function _validate_fit_tolerance(value, name)
    tolerance = Float64(value)
    isfinite(tolerance) && tolerance > 0 ||
        throw(ArgumentError("$name must be finite and positive"))
    return tolerance
end

function _validate_nonnegative_tolerance(value, name)
    tolerance = Float64(value)
    isfinite(tolerance) && tolerance >= 0 ||
        throw(ArgumentError("$name must be finite and nonnegative"))
    return tolerance
end

function _extract_esprit_candidates(
    samples,
    dt;
    fit_rank,
    fit_rank_max,
    fit_sval_rtol,
    hankel_rows,
)
    rank = Int(fit_rank)
    rank_max = Int(fit_rank_max)
    rank >= 0 || throw(ArgumentError("fit_rank must be nonnegative"))
    rank_max > 0 || throw(ArgumentError("fit_rank_max must be positive"))
    rank == 0 || rank <= rank_max ||
        throw(ArgumentError("fit_rank must not exceed fit_rank_max"))

    ncols = if hankel_rows === nothing
        nothing
    else
        hankel_rows isa Integer ||
            throw(ArgumentError("hankel_rows must be an integer or nothing"))
        Int(hankel_rows)
    end

    candidate = if rank > 0
        ExpFit.esprit(samples, dt, rank; ncols=ncols)
    else
        tolerance = _validate_fit_tolerance(fit_sval_rtol, "fit_sval_rtol")
        initial = ExpFit.esprit(samples, dt, tolerance; ncols=ncols)
        length(initial.expon) <= rank_max ?
            initial : ExpFit.esprit(samples, dt, rank_max; ncols=ncols)
    end
    isempty(candidate.expon) && throw(ArgumentError("ESPRIT returned no pole candidates"))
    return ComplexF64.(candidate.expon)
end

function _validated_conjugate_closed_rates(
    candidate_rates,
    dt;
    pole_stability_tolerance,
    duplicate_pole_tolerance,
)
    stability_tolerance = _validate_nonnegative_tolerance(
        pole_stability_tolerance,
        "pole_stability_tolerance",
    )
    duplicate_tolerance = _validate_nonnegative_tolerance(
        duplicate_pole_tolerance,
        "duplicate_pole_tolerance",
    )
    nyquist = π / dt
    rates = ComplexF64[]
    corrections = String[]

    for (index, candidate) in pairs(candidate_rates)
        isfinite(candidate) || throw(ArgumentError("ESPRIT pole $index is nonfinite"))
        real_part = real(candidate)
        imaginary_part = imag(candidate)
        real_part < -stability_tolerance && throw(ArgumentError(
            "ESPRIT pole $index is unstable: real(rate) = $real_part",
        ))
        abs(imaginary_part) > nyquist + stability_tolerance && throw(ArgumentError(
            "ESPRIT pole $index exceeds the Nyquist rate $nyquist",
        ))
        if real_part < 0
            push!(corrections, "pole $index real part $real_part corrected to zero")
            real_part = 0.0
        end
        if abs(imaginary_part) > nyquist
            corrected = copysign(nyquist, imaginary_part)
            push!(
                corrections,
                "pole $index imaginary part $imaginary_part corrected to Nyquist rate $corrected",
            )
            imaginary_part = corrected
        end
        push!(rates, ComplexF64(real_part, imaginary_part))
    end

    _reject_exact_duplicate_rates(rates)

    original_count = length(rates)
    for index in 1:original_count
        partner = conj(rates[index])
        if !any(rate -> isapprox(rate, partner; atol=duplicate_tolerance, rtol=0), rates)
            push!(rates, partner)
            push!(corrections, "added conjugate partner for pole $index")
        end
    end
    _reject_exact_duplicate_rates(rates)
    return rates, corrections
end

function _reject_exact_duplicate_rates(rates)
    for first in 1:(length(rates) - 1), second in (first + 1):length(rates)
        rates[first] == rates[second] && throw(ArgumentError(
            "pole basis contains an exact duplicate after numerical corrections",
        ))
    end
    return nothing
end

_correlation_vandermonde(rates, times) =
    ComplexF64[exp(-rate * time) for time in times, rate in rates]

function _correlation_errors(predicted, expected)
    absolute_error = Float64(norm(predicted - expected))
    reference_norm = Float64(norm(expected))
    relative_error = absolute_error / max(reference_norm, eps(Float64))
    all(isfinite, (absolute_error, relative_error)) ||
        throw(ArgumentError("correlation fit produced nonfinite errors"))
    return absolute_error, relative_error
end

function _enforce_correlation_error(
    absolute_error,
    relative_error;
    fit_absolute_tolerance,
    fit_relative_tolerance,
    label,
)
    absolute_tolerance = _validate_fit_tolerance(
        fit_absolute_tolerance,
        "fit_absolute_tolerance",
    )
    relative_tolerance = _validate_fit_tolerance(
        fit_relative_tolerance,
        "fit_relative_tolerance",
    )
    absolute_error <= absolute_tolerance || relative_error <= relative_tolerance ||
        throw(ArgumentError(
            "$label correlation error exceeds both tolerances: " *
            "absolute=$absolute_error, relative=$relative_error",
        ))
    return nothing
end

function _minimum_pole_separation(rates)
    length(rates) < 2 && return 0.0
    return minimum(
        abs(rates[first] - rates[second])
        for first in 1:(length(rates) - 1) for second in (first + 1):length(rates)
    )
end

"""
    evaluate_correlation(fit, times; branch=:forward)

Evaluate an exponential correlation on its common pole basis.
"""
function evaluate_correlation(fit::ExponentialCorrelation, time::Real; branch=:forward)
    isfinite(time) || throw(ArgumentError("evaluation times must be finite"))
    coefficients = if branch === :forward
        fit.coeff_forward
    elseif branch === :backward
        fit.coeff_backward
    else
        throw(ArgumentError("branch must be :forward or :backward"))
    end
    return sum(
        coefficients[index] * exp(-fit.rates[index] * time)
        for index in eachindex(fit.rates)
    )
end

function evaluate_correlation(fit::ExponentialCorrelation, times; branch=:forward)
    return ComplexF64[evaluate_correlation(fit, time; branch=branch) for time in times]
end

function _validation_errors(rates, coefficients, samples, times)
    sample_values = ComplexF64.(samples)
    time_values = Float64.(times)
    length(sample_values) == length(time_values) ||
        throw(ArgumentError("validation samples and times must have matching lengths"))
    isempty(sample_values) && throw(ArgumentError("validation data must be nonempty"))
    all(isfinite, sample_values) || throw(ArgumentError("validation samples must be finite"))
    all(isfinite, time_values) || throw(ArgumentError("validation times must be finite"))
    predicted = _correlation_vandermonde(rates, time_values) * coefficients
    return _correlation_errors(predicted, sample_values)
end

function _validated_independent_data(
    samples,
    times,
    training_times;
    grid_tolerance,
)
    samples === nothing && throw(ArgumentError("validation_samples must be provided"))
    times === nothing && throw(ArgumentError("validation_times must be provided"))
    sample_values = ComplexF64.(samples)
    time_values = Float64.(times)
    length(sample_values) == length(time_values) ||
        throw(ArgumentError("validation samples and times must have matching lengths"))
    isempty(sample_values) && throw(ArgumentError("validation data must be nonempty"))
    all(isfinite, sample_values) || throw(ArgumentError("validation samples must be finite"))
    all(isfinite, time_values) || throw(ArgumentError("validation times must be finite"))
    for validation_time in time_values, training_time in training_times
        isapprox(
            validation_time,
            training_time;
            atol=grid_tolerance,
            rtol=grid_tolerance,
        ) && throw(ArgumentError(
            "validation times must not reuse training times",
        ))
    end
    return sample_values, time_values
end

function _both_branch_errors(rates, coeff_forward, coeff_backward, samples, times)
    sample_values = ComplexF64.(samples)
    forward = _validation_errors(rates, coeff_forward, sample_values, times)
    backward = _validation_errors(rates, coeff_backward, conj.(sample_values), times)
    return (forward=forward, backward=backward)
end

function _enforce_both_branch_errors(
    errors;
    fit_absolute_tolerance,
    fit_relative_tolerance,
    label,
)
    _enforce_correlation_error(
        errors.forward...;
        fit_absolute_tolerance=fit_absolute_tolerance,
        fit_relative_tolerance=fit_relative_tolerance,
        label="$label forward",
    )
    _enforce_correlation_error(
        errors.backward...;
        fit_absolute_tolerance=fit_absolute_tolerance,
        fit_relative_tolerance=fit_relative_tolerance,
        label="$label backward",
    )
    return (
        absolute_error=max(errors.forward[1], errors.backward[1]),
        relative_error=max(errors.forward[2], errors.backward[2]),
    )
end

"""
    validate_correlation_fit(fit, samples, times; kwargs...)

Evaluate independent samples with fixed fitted rates and coefficients, returning
the absolute and relative two-norm errors. Reject the validation only when both
configured error limits are exceeded.
"""
function validate_correlation_fit(
    fit::ExponentialCorrelation,
    samples,
    times;
    fit_absolute_tolerance=1e-8,
    fit_relative_tolerance=1e-6,
)
    errors = _both_branch_errors(
        fit.rates,
        fit.coeff_forward,
        fit.coeff_backward,
        samples,
        times,
    )
    conservative = _enforce_both_branch_errors(
        errors;
        fit_absolute_tolerance=fit_absolute_tolerance,
        fit_relative_tolerance=fit_relative_tolerance,
        label="validation",
    )
    return (;
        conservative...,
        forward_absolute_error=errors.forward[1],
        forward_relative_error=errors.forward[2],
        backward_absolute_error=errors.backward[1],
        backward_relative_error=errors.backward[2],
    )
end

"""
    fit_correlation_esprit(samples, times; kwargs...) -> ExponentialCorrelation

Use ExpFit's ESPRIT implementation for candidate extraction, validate and close
the poles under conjugation, then independently refit the forward and backward
coefficients on that fixed common basis.
"""
function fit_correlation_esprit(
    samples,
    times;
    fit_rank=0,
    fit_rank_max=12,
    fit_sval_rtol=1e-10,
    hankel_rows=nothing,
    pole_stability_tolerance=1e-10,
    duplicate_pole_tolerance=1e-8,
    fit_absolute_tolerance=1e-8,
    fit_relative_tolerance=1e-6,
    uniform_grid_tolerance=1e-10,
    validation_samples=nothing,
    validation_times=nothing,
)
    grid_tolerance = _validate_fit_tolerance(
        uniform_grid_tolerance,
        "uniform_grid_tolerance",
    )
    time_values, dt = _validated_uniform_grid(
        times;
        uniform_grid_tolerance=grid_tolerance,
    )
    sample_values = ComplexF64.(samples)
    length(sample_values) == length(time_values) ||
        throw(ArgumentError("samples and times must have matching lengths"))
    all(isfinite, sample_values) || throw(ArgumentError("samples must be finite"))
    validation_sample_values, validation_time_values = _validated_independent_data(
        validation_samples,
        validation_times,
        time_values;
        grid_tolerance=grid_tolerance,
    )

    candidate_rates = _extract_esprit_candidates(
        sample_values,
        dt;
        fit_rank=fit_rank,
        fit_rank_max=fit_rank_max,
        fit_sval_rtol=fit_sval_rtol,
        hankel_rows=hankel_rows,
    )
    rates, corrections = _validated_conjugate_closed_rates(
        candidate_rates,
        dt;
        pole_stability_tolerance=pole_stability_tolerance,
        duplicate_pole_tolerance=duplicate_pole_tolerance,
    )

    vandermonde = _correlation_vandermonde(rates, time_values)
    decomposition = svd(vandermonde)
    singular_values = Float64.(decomposition.S)
    all(isfinite, singular_values) ||
        throw(ArgumentError("common pole basis has nonfinite conditioning diagnostics"))
    column_count = size(vandermonde, 2)
    column_count <= size(vandermonde, 1) || throw(ArgumentError(
        "common pole basis has more columns than training samples",
    ))
    rank_threshold = maximum(size(vandermonde)) * eps(Float64) * first(singular_values)
    count(>(rank_threshold), singular_values) == column_count || throw(ArgumentError(
        "common pole basis is numerically column-rank deficient",
    ))
    condition_number = first(singular_values) / last(singular_values)
    isfinite(condition_number) ||
        throw(ArgumentError("common pole basis has nonfinite conditioning diagnostics"))

    coeff_forward = ComplexF64.(vandermonde \ sample_values)
    coeff_backward = ComplexF64.(vandermonde \ conj.(sample_values))
    all(isfinite, coeff_forward) && all(isfinite, coeff_backward) ||
        throw(ArgumentError("common-basis coefficient fit is nonfinite"))

    training_branch_errors = _both_branch_errors(
        rates,
        coeff_forward,
        coeff_backward,
        sample_values,
        time_values,
    )
    training_errors = _enforce_both_branch_errors(
        training_branch_errors;
        fit_absolute_tolerance=fit_absolute_tolerance,
        fit_relative_tolerance=fit_relative_tolerance,
        label="training",
    )

    validation_branch_errors = _both_branch_errors(
        rates,
        coeff_forward,
        coeff_backward,
        validation_sample_values,
        validation_time_values,
    )
    validation_errors = _enforce_both_branch_errors(
        validation_branch_errors;
        fit_absolute_tolerance=fit_absolute_tolerance,
        fit_relative_tolerance=fit_relative_tolerance,
        label="validation",
    )

    minimum_separation = Float64(_minimum_pole_separation(rates))
    maximum_coefficient = Float64(maximum(abs, [coeff_forward; coeff_backward]))
    all(isfinite, (minimum_separation, maximum_coefficient)) ||
        throw(ArgumentError("common pole basis has nonfinite diagnostics"))
    metadata = CorrelationFitMetadata(
        training_errors.absolute_error,
        training_errors.relative_error,
        validation_errors.absolute_error,
        validation_errors.relative_error,
        dt,
        (first(time_values), last(time_values)),
        singular_values,
        condition_number,
        minimum_separation,
        maximum_coefficient,
        corrections,
    )
    return ExponentialCorrelation(
        rates,
        coeff_forward,
        coeff_backward,
        metadata;
        stability_tolerance=Float64(pole_stability_tolerance),
    )
end
